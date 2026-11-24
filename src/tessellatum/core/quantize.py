"""Color quantization: reduce an image to a small palette of flat colors.

K-means in Lab space, then a merge of the colors a painter could neither tell
apart nor mix -- closer than ``color.MIN_PALETTE_DE00`` -- so the palette comes
back with every color clearly different from every other.

Line art gets its palette a different way (``flat_color_palette``). Its fills
are flat colors the artist chose, so the page should offer those colors rather
than the means k-means lands on, and the margin should be kept while the colors
are picked rather than by merging afterwards, which spends colors and leaves
shades the artwork does not have.

With the painter's own paints given, the palette is chosen from them instead
(``paint_palette``): the ones that paint the picture best; and, if asked, colors
mixed from two of them where a mix earns its place (``mixed_palette``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tessellatum.core import kernels, mixing, parallel
from tessellatum.core.color import MIN_PALETTE_DE00, bgr_to_lab, ciede2000, pairwise_de00, printed_de00
from tessellatum.core.ink import near

# The sparse bilateral filter samples its window every sigma_space /
# _TAPS_PER_SIGMA pixels along each axis.
_TAPS_PER_SIGMA = 6

# One k-means++ run over every pixel. On the sample images, three attempts
# took 3x as long for a finished painting that was on average no closer to
# the source (worst case: flat-color artwork at Easy, +3.7% color error, or
# 0.18 CIEDE2000). Fitting on a sample of pixels instead -- even half of them
# -- occasionally lost a distinct color entirely.
_KMEANS_ATTEMPTS = 1
_KMEANS_MAX_ITERATIONS = 30
_KMEANS_EPSILON = 0.5
# Colors the paper's black merges are asked of k-means again (see ``_kmeans_colors``): at most this many more times,
# and never for more than this many times the colors asked for.
_REFILLS = 2
_MOST_CLUSTERS = 2
# A color more than this share of whose pixels lie where the picture is looked at (its faces and subject) keeps the
# screen's margin, whatever the paper's black.
_KEPT_SHARE = 0.5

# The line-art palette (``flat_color_palette``) reads the fills' colors as a
# histogram of OpenCV's 8-bit Lab, where a step of 1 is 0.39 L* or one unit of
# a or b. A bin is _BIN units across, and a color's own pixels are the bin and
# the 26 around it -- a neighborhood of one bin, about 2 to 4 CIEDE2000.
_BIN = 4.0
_BINS_PER_AXIS = 64  # 256 / _BIN
# The painter's own paints are chosen over the picture's colors in bins this wide, about 3-6 CIEDE2000: a few thousand
# colors to weigh every paint against rather than tens of thousands, for a choice made in a tenth of a second.
_PAINT_BIN = 8.0
# How many times a pixel in the faces or the subject counts when the paints are chosen, as their details count more.
_KEPT_WEIGHT = 2.0
# The most swaps of a paint taken for one left over, after the greedy choice.
_SWAP_ROUNDS = 8
# A palette's mixed colors (``mixed_palette``) are mixed from this many of the paints at most -- all of them if there
# are no more, else those taken and then those nearest the most of the picture: 276 pairs, about 1,400 mixes.
MIX_MOST_PAINTS = 24
# When a mixed color earns its place in a palette, by default (see ``MixRule``): the pixels it is used for get this
# much nearer, in CIEDE2000 on average -- clearly more than the ~2.3 a side-by-side difference starts to be seen at --
# and they are at least this share of the picture.
MIX_GAIN_DE00 = 3.0
MIX_MIN_SHARE = 0.01
# How many times the mixes taken are added to and checked again (see ``mixed_palette``).
_MIX_ROUNDS = 4


@dataclass(frozen=True)
class MixRule:
    """When a color mixed from two paints earns its place in a palette (see ``mixed_palette``): the pixels it would be
    used for -- those nearer it than every other color taken -- get ``gain_de00`` nearer on average, in CIEDE2000,
    and they are at least ``min_share`` of the picture (0-1), the faces and the subject counted twice."""

    gain_de00: float = MIX_GAIN_DE00
    min_share: float = MIX_MIN_SHARE
# Two colors farther apart than this many times the margin in plain Lab distance
# clear it without CIEDE2000 being computed: it is never that much larger.
_SURELY_APART = 3.0
# How often a color settles again onto the pixels that ended up nearest it, and
# how far those reach: the neighborhood it was read from, whose corner is
# _BIN * sqrt(3) away.
_SETTLE_STEPS = 2
_SETTLE_RADIUS = 1.5 * _BIN
# A pixel takes the color nearest it in L*a*b*, which is 8-bit Lab with its L
# scaled back by 100/255: squared, each channel's difference counts this much.
_LAB_WEIGHT = np.array([(100 / 255) ** 2, 1.0, 1.0])
# How much a difference in lightness counts in k-means against one in a* or b*, both in L*a*b* units. K-means runs in
# OpenCV's 8-bit Lab, whose L is L* stretched by 255/100, so that is the weight it has unless this says otherwise.
_KMEANS_LIGHTNESS_WEIGHT = 255 / 100
# K-means asked again for the colors paper merges runs on every this many-th pixel, and every pixel then takes the
# nearest of its colors: a full run each time would make a page a third slower.
_REFILL_STRIDE = 4
_OPENCV_LIGHTNESS_SCALE = 255 / 100
# How many pixels are given their color at a time: the pixels-by-colors scores
# of one block stay a few MB instead of the whole page's.
_NEAREST_BLOCK = 1 << 16


def quantize(
    image_bgr: np.ndarray,
    num_colors: int,
    blur_sigma: float,
    seed: int = 0,
    min_de00: float = MIN_PALETTE_DE00,
    ink: np.ndarray | None = None,
    halo_px: float = 0.0,
    paper_black_lstar: float = 0.0,
    keep: np.ndarray | None = None,
    paints_bgr: np.ndarray | None = None,
    mix: MixRule | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, tuple]:
    """Reduce ``image_bgr`` to at most ``num_colors`` flat, clearly different colors.

    K-means minimizes distance in Lab, where equal distances are not equally
    visible: a photograph of fur or stone fills its clusters with browns a
    painter can neither tell apart nor mix. So colors closer than ``min_de00``
    CIEDE2000 are merged afterwards (``_merge_close_colors``), which is why
    the palette can come back shorter than asked for.

    Line art's ink is printed, not painted (see ``ink``), so with ``ink`` given
    its pixels take no color at all, and the colors are found without them.
    Nor are they found from the pixels within ``halo_px`` of the ink: those are
    the ink's anti-aliased edge, a mix of the ink and the fill beside it, whose
    in-between colors would otherwise become colors of their own. Each of them
    takes the color of the nearest pixel the colors were found from, which is
    the fill it edges.

    With ``ink``, the colors are the fills' own (``flat_color_palette``) rather
    than k-means': line art is painted in the colors the artist chose, and
    picking them apart by CIEDE2000 from the start keeps colors a merge
    afterwards would spend. Nothing is random on that path.

    Without it, the margin can also be kept as the colors print
    (``paper_black_lstar``): paper and paint can't go darker than their own
    black, so near-blacks a screen shows apart print as one. Those are merged
    too, but for the colors mostly in ``keep`` (the faces and the subject,
    whose shading is what the page is looked at for), and the colors that
    frees are asked of k-means again, so the page keeps as many as the
    screen's margin would (see ``_kmeans_colors``).

    Args:
        image_bgr: HxWx3 uint8 image in BGR order (OpenCV convention).
        num_colors: how many colors to look for.
        blur_sigma: bilateral-filter smoothing strength (0 disables smoothing).
        seed: RNG seed so results are reproducible for the same inputs.
        min_de00: the clear margin every two palette colors keep, in CIEDE2000.
            0 leaves k-means' colors as they are, and lets line art's colors
            stand as close together as the picture puts them.
        ink: HxW bool, the pixels printed as ink, or None.
        halo_px: how far from the ink its anti-aliased edge reaches.
        paper_black_lstar: the darkest black paper and paint show, in L*; 0
            keeps the margin as the screen shows the colors.
        keep: HxW bool, where the colors keep the screen's margin whatever
            the paper's black, or None.
        paints_bgr: Mx3 uint8, the painter's own paints, or None. Given, the
            colors are chosen from them (``paint_palette``), on line art as on
            any other picture, and are exactly theirs.
        mix: with ``paints_bgr``, colors mixed from two of the paints are
            taken too where they meet this rule (``mixed_palette``), and the
            result has a third item, the recipes.

    Returns:
        (label_map, palette_bgr):
            label_map: HxW int32 array, each pixel's color index; ``K``, one
                past the palette, on the ink.
            palette_bgr: Kx3 uint8 array of BGR colors, K <= num_colors,
                ordered from darkest to lightest (by perceptual lightness)
                so numbering reads naturally; with ``mix``, the paints so,
                then the mixed colors so.
            recipes (only with ``mix``): each palette color's, None for a
                paint (see ``mixed_palette``).
    """
    smoothed = _smooth(image_bgr, blur_sigma)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    fitted = None  # the pixels the colors are found from: all of them, unless there is ink and something besides it
    if ink is not None and ink.any() and not ink.all():
        fitted = ~near(ink, halo_px)
        if not fitted.any():  # nothing but ink and its edge: fit what there is off the ink
            fitted = ~ink
    samples = lab.reshape(-1, 3).astype(np.float32)
    if paints_bgr is not None and len(paints_bgr) > 0:
        paints_bgr = np.asarray(paints_bgr, dtype=np.uint8).reshape(-1, 3)
        chosen_from = slice(None) if fitted is None else fitted.reshape(-1)
        kept = None if keep is None else np.asarray(keep, dtype=bool).reshape(-1)[chosen_from]
        chosen_samples = np.ascontiguousarray(samples[chosen_from])
        extra: tuple = ()
        if mix is not None:
            labels, palette_bgr, recipes = mixed_palette(
                chosen_samples, paints_bgr, num_colors, min_de00, mix, paper_black_lstar, kept
            )
            extra = (recipes,)
        else:
            labels, chosen = paint_palette(chosen_samples, paints_bgr, num_colors, min_de00, paper_black_lstar, kept)
            palette_bgr = paints_bgr[chosen]
        if fitted is None:
            return (labels.reshape(lab.shape[:2]), palette_bgr, *extra)
        label_map = np.full(lab.shape[:2], len(palette_bgr), dtype=np.int32)
        label_map[fitted] = labels
        label_map[~fitted & ~ink] = _nearest_fitted(label_map, fitted)[~fitted & ~ink]
        return (label_map, palette_bgr, *extra)
    if fitted is None:
        kept = None if keep is None else np.asarray(keep, dtype=bool).reshape(-1)
        labels, palette_bgr = _kmeans_colors(samples, num_colors, seed, min_de00, paper_black_lstar, kept)
        return labels.reshape(lab.shape[:2]), palette_bgr

    samples = np.ascontiguousarray(samples[fitted.reshape(-1)])
    centers_lab = flat_color_palette(samples, num_colors, min_de00)
    palette_bgr = _lab_centers_to_bgr(centers_lab)
    label_map = np.full(lab.shape[:2], len(palette_bgr), dtype=np.int32)
    label_map[fitted] = _nearest_color(samples, centers_lab)
    label_map[~fitted & ~ink] = _nearest_fitted(label_map, fitted)[~fitted & ~ink]
    return label_map, palette_bgr


def _kmeans_colors(
    samples: np.ndarray,
    num_colors: int,
    seed: int,
    min_de00: float,
    paper_black_lstar: float = 0.0,
    kept: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means over ``samples`` (Nx3 float32 Lab), then the merge: each sample's color, and the palette.

    With ``paper_black_lstar``, the merge goes on as the colors print (see
    ``_merge_close_colors``), and when that merges colors the screen's margin
    keeps, k-means is asked for that many more and their colors merged the same
    way, up to ``_REFILLS`` times: the slots the near-blacks gave up go where
    the picture has colors left to tell apart. The page never gets more colors
    than the screen's margin keeps, and where printing merges nothing it is the
    same page as without it.
    """
    labels, centers_lab, group, on_screen = _fit_colors(samples, num_colors, seed, min_de00, paper_black_lstar, kept)
    colors = len(np.unique(group))
    asked = num_colors
    most = min(_MOST_CLUSTERS * num_colors, len(samples))
    for _ in range(_REFILLS):
        if colors >= on_screen or asked >= most:
            break
        asked = min(asked + on_screen - colors, most)
        refit = _fit_colors(
            samples, asked, seed, min_de00, paper_black_lstar, kept, down_to=on_screen, stride=_REFILL_STRIDE
        )
        if len(np.unique(refit[2])) <= colors:  # asking for more freed nothing
            break
        labels, centers_lab, group, _ = refit
        colors = len(np.unique(group))

    survivors = np.unique(group)  # the colors that survived the merge, in cluster order
    kept_centers = centers_lab[survivors]
    order = np.argsort(kept_centers[:, 0])  # sort by L (lightness), darkest first
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)

    final_index = np.zeros(len(centers_lab), dtype=np.int32)  # cluster -> its color's place on the legend
    final_index[survivors] = rank
    return final_index[group][labels].astype(np.int32), _lab_centers_to_bgr(kept_centers)[order]


def _fit_colors(
    samples: np.ndarray,
    num_colors: int,
    seed: int,
    min_de00: float,
    paper_black_lstar: float,
    kept: np.ndarray | None,
    down_to: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One k-means run and its merge: (each sample's cluster, the centers, each cluster's color, the screen's count).

    The last is how many colors the merge kept at the screen's margin, before
    it went on as the colors print. With ``stride``, k-means runs on every
    ``stride``-th sample, and every sample then takes the nearest center.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITERATIONS, _KMEANS_EPSILON)
    stretch = _KMEANS_LIGHTNESS_WEIGHT / _OPENCV_LIGHTNESS_SCALE
    if stretch != 1.0:
        samples = samples.copy()
        samples[:, 0] *= stretch
    fitted = samples if stride <= 1 else np.ascontiguousarray(samples[::stride])
    cv2.setRNGSeed(seed)
    _compactness, labels, centers_lab = cv2.kmeans(
        fitted, min(num_colors, len(fitted)), None, criteria, attempts=_KMEANS_ATTEMPTS, flags=cv2.KMEANS_PP_CENTERS
    )
    if fitted is not samples:
        labels = _nearest_center(samples, centers_lab)
        used = np.bincount(labels, minlength=len(centers_lab)) > 0
        if not used.all():  # a center no sample is nearest: not a color of the picture, so not one of the page's
            labels = (np.cumsum(used) - 1).astype(np.int32)[labels]
            centers_lab = np.ascontiguousarray(centers_lab[used])
    if stretch != 1.0:
        centers_lab[:, 0] /= stretch

    labels = labels.reshape(-1).astype(np.int32)
    pixels_per_color = np.bincount(labels, minlength=len(centers_lab))
    kept_pixels = None
    if paper_black_lstar > 0 and kept is not None:
        kept_pixels = np.bincount(labels, weights=kept.astype(np.float64), minlength=len(centers_lab))
    centers_lab, group, on_screen = _merge_close_colors(
        centers_lab, pixels_per_color, min_de00, paper_black_lstar, kept_pixels, down_to
    )
    return labels, centers_lab, group, on_screen


def _nearest_center(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Each sample's nearest of ``centers``, in the space k-means ran in (plain Euclidean)."""
    weighted = np.ascontiguousarray((2 * centers).T, dtype=np.float32)
    offset = (centers * centers).sum(axis=1).astype(np.float32)
    nearest = np.empty(len(samples), dtype=np.int32)
    for start in range(0, len(samples), _NEAREST_BLOCK):
        block = samples[start : start + _NEAREST_BLOCK]
        nearest[start : start + _NEAREST_BLOCK] = np.argmin(offset - block @ weighted, axis=1)
    return nearest


def flat_color_palette(samples_lab: np.ndarray, num_colors: int, min_de00: float = MIN_PALETTE_DE00) -> np.ndarray:
    """The flat colors of ``samples_lab``, at most ``num_colors``, every two at least ``min_de00`` apart.

    Line art is painted in the colors the artist filled it with, so the page
    should offer those colors. K-means offers the mean of each group of pixels
    it makes, which on a cartoon carries the blends along every fill's edge and
    lands between two of the artwork's colors; the margin is then reached by
    merging the closest pair over and over (``_merge_close_colors``), which
    spends colors the artwork could have had. Here a color is a place the
    picture's pixels crowd around, and the margin is kept while the colors are
    picked, so nothing is merged away afterwards.

    The fills' colors are counted into a histogram of bins ``_BIN`` units
    across, and each bin stands for a color: the color its own pixels and those
    of the 26 bins around it average to, weighted by how many of them there
    are. Colors are then taken one at a time -- the one with the most pixels
    crowding around it and lying farthest from every color taken so far, never
    within ``min_de00`` of one of them -- until ``num_colors`` are taken or
    every color left is within the margin of one. Each then settles twice onto
    the pixels that ended up nearest it, unless the move would break the
    margin.

    Args:
        samples_lab: Nx3 float32, the fills' pixels in OpenCV's 8-bit Lab.
        num_colors: how many colors to take at most.
        min_de00: the clear margin between two of them, in CIEDE2000; 0 or less
            lets them stand as close as the picture puts them.

    Returns:
        Kx3 float64 colors in the same 8-bit Lab scale, K <= num_colors,
        darkest first. Nothing is random: the same pixels give the same colors.
    """
    crowd_color, crowd_weight, bin_color, bin_weight = _color_histogram(samples_lab)
    taken = _take_colors(crowd_color, crowd_weight, num_colors, min_de00)
    centers = _settle_colors(taken, bin_color, bin_weight, min_de00)
    return centers[np.argsort(centers[:, 0])]  # darkest first, as the k-means palette is sorted


def _color_histogram(samples_lab: np.ndarray) -> tuple[np.ndarray, ...]:
    """The picture's colors as a histogram: per non-empty bin, its own color and count, and its neighborhood's.

    A bin's own color is the mean of the pixels in it; its neighborhood's is
    the mean over the bin and the 26 around it, which is what a color the
    pixels crowd around looks like when it falls between two bins.
    """
    # One empty bin of margin on every side, so that the roll in _neighborhood
    # cannot carry white round to black: both ends of the scale are full.
    side = _BINS_PER_AXIS + 2
    bins = np.clip((samples_lab / _BIN).astype(np.int32), 0, _BINS_PER_AXIS - 1) + 1
    index = (bins[:, 0] * side + bins[:, 1]) * side + bins[:, 2]
    counts = np.bincount(index, minlength=side**3).astype(np.float64).reshape(side, side, side)
    sums = np.stack(
        [np.bincount(index, samples_lab[:, channel], side**3).reshape(side, side, side) for channel in range(3)], -1
    )
    crowd = _neighborhood(counts)
    crowd_color = np.stack([_neighborhood(sums[..., channel]) for channel in range(3)], -1)
    crowd_color /= np.maximum(crowd, 1)[..., None]
    live = counts > 0
    return crowd_color[live], crowd[live], sums[live] / counts[live][:, None], counts[live]


def _neighborhood(volume: np.ndarray) -> np.ndarray:
    """Each cell plus the 26 around it, summed. The roll wraps through the empty margin, so no two ends meet."""
    total = volume
    for axis in range(3):
        total = total + np.roll(total, 1, axis) + np.roll(total, -1, axis)
    return total


def _take_colors(colors: np.ndarray, weight: np.ndarray, num_colors: int, min_de00: float) -> np.ndarray:
    """Take the color the most pixels crowd around and the farthest from those already taken, over and over."""
    lab = bgr_to_lab(_lab_centers_to_bgr(colors))
    allowed = np.ones(len(colors), dtype=bool)
    unserved = np.ones(len(colors))  # squared Lab distance to the nearest color taken; 1 before any is
    taken: list[int] = []
    while len(taken) < num_colors:
        want = np.where(allowed, weight * unserved, -1.0)
        best = int(np.argmax(want))
        if want[best] <= 0:  # every color left is within the margin of one already taken
            break
        allowed &= _clear_of(lab, lab[best], min_de00)
        distance = ((colors - colors[best]) ** 2).sum(axis=1)
        unserved = distance if not taken else np.minimum(unserved, distance)
        taken.append(best)
    return colors[taken]


def _clear_of(lab: np.ndarray, one_lab: np.ndarray, min_de00: float) -> np.ndarray:
    """Which of ``lab`` stand at least ``min_de00`` from ``one_lab``.

    CIEDE2000 is only computed where it can come out below the margin: it never
    exceeds ``_SURELY_APART`` times the plain Lab distance, so colors farther
    than that are clear of it without the exact number.
    """
    clear = np.ones(len(lab), dtype=bool)
    if min_de00 <= 0:
        return clear
    near_enough = ((lab - one_lab) ** 2).sum(axis=1) < (_SURELY_APART * min_de00) ** 2
    clear[near_enough] = ciede2000(lab[near_enough], one_lab) >= min_de00
    return clear


def _settle_colors(
    centers: np.ndarray, bin_color: np.ndarray, bin_weight: np.ndarray, min_de00: float
) -> np.ndarray:
    """Move each color onto the pixels that ended up nearest it, as long as the margin holds.

    A color taken from one bin's neighborhood can sit a little off the color
    the fill really is, because the neighborhood reaches into the blends along
    the fill's edges. Settling it on the bins nearest it that are within one
    neighborhood brings it back onto the fill. A move that would put two colors
    closer than the margin is not made, and settling then stops.
    """
    if len(centers) < 2:
        return centers
    for _ in range(_SETTLE_STEPS):
        nearest = _nearest_color(bin_color, centers)
        moved = centers.copy()
        for index in range(len(centers)):
            mine = (nearest == index) & (((bin_color - centers[index]) ** 2).sum(axis=1) <= _SETTLE_RADIUS**2)
            if mine.any():
                moved[index] = (bin_weight[mine] @ bin_color[mine]) / bin_weight[mine].sum()
        if pairwise_de00(_lab_centers_to_bgr(moved)).min() < min_de00:
            break
        centers = moved
    return centers


def _nearest_color(samples_lab: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    """Each sample's nearest center by distance in L*a*b*, a block of samples at a time.

    Both are in OpenCV's 8-bit Lab, whose L is L* stretched by 255/100. Measured
    there, a step in lightness would count two and a half times what the same
    step in a or b does, and a dark gray whose own color is not on the legend
    would go to a brown of its lightness rather than the near-black beside it:
    #2d2d2d is 6.9 CIEDE2000 from #171717 and 15.4 from #482c1a, but nearer the
    brown in 8-bit Lab. Each block is one matrix product, the samples' squared
    norms cancelling.
    """
    centers = np.asarray(centers_lab, dtype=np.float64)
    weighted = centers * _LAB_WEIGHT
    scale = np.ascontiguousarray((2 * weighted).T, dtype=samples_lab.dtype)
    offset = (weighted * centers).sum(axis=1).astype(samples_lab.dtype)
    nearest = np.empty(len(samples_lab), dtype=np.int32)
    for start in range(0, len(samples_lab), _NEAREST_BLOCK):
        block = samples_lab[start : start + _NEAREST_BLOCK]
        nearest[start : start + len(block)] = np.argmax(block @ scale - offset, axis=1)
    return nearest


def _nearest_fitted(label_map: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """For every pixel, the label of the nearest ``fitted`` pixel."""
    _distance, nearest = cv2.distanceTransformWithLabels(
        (~fitted).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    label_of = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    label_of[nearest[fitted]] = label_map[fitted]
    return label_of[nearest]


def _lab_centers_to_bgr(centers_lab: np.ndarray) -> np.ndarray:
    """Cluster centers in OpenCV's 8-bit Lab scale -> the Kx3 uint8 BGR colors the page is painted in."""
    centers_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(centers_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def _merge_close_colors(
    centers_lab: np.ndarray,
    pixels_per_color: np.ndarray,
    min_de00: float,
    paper_black_lstar: float = 0.0,
    kept_pixels: np.ndarray | None = None,
    down_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge the two closest colors over and over, until every two differ by ``min_de00``.

    Distance is CIEDE2000 between the colors as they will be printed and
    painted -- the 8-bit sRGB the centers convert to -- so it is the same
    number the finished legend is judged on.

    Two merged colors become the one their pixels average to, which is the
    color the painter would have mixed for both of them. Which of the two
    keeps the slot makes no difference: the merged color is the same either
    way, and the palette is sorted by lightness afterwards. Merging the
    *closest* pair first, rather than the pair that costs the least error, is
    what keeps the most colors and the closest painting: measured against five
    other mechanisms before it was chosen.

    With ``paper_black_lstar``, the merge then goes on with the colors as paper
    shows them (``color.printed_de00``), the closest pair first: near-blacks
    the screen tells apart but paper doesn't. A color more than ``_KEPT_SHARE``
    of whose pixels are in ``kept_pixels`` keeps the screen's distance to
    every other. Last, while more than ``down_to`` colors are left, the pair
    closest by that test merges.

    Args:
        centers_lab: Kx3 cluster centers in OpenCV's 8-bit Lab scale.
        pixels_per_color: how many pixels each cluster has.
        min_de00: the margin to reach; 0 or less leaves the centers alone.
        paper_black_lstar: the paper's darkest black, in L*; 0 or less keeps the
            screen's distances.
        kept_pixels: how many of each cluster's pixels lie where colors keep the
            screen's distances, or None for none.
        down_to: the most colors to keep, or None.

    Returns:
        (centers_lab, group, on_screen): the centers, with every survivor moved
        to its merged color; for each original cluster the index of the center
        it ended up in (``group[i] == i`` for a color that survived); and how
        many colors the screen's margin kept.
    """
    centers = np.array(centers_lab, dtype=np.float64)
    weights = np.asarray(pixels_per_color, dtype=np.float64).copy()
    kept = None if kept_pixels is None else np.asarray(kept_pixels, dtype=np.float64).copy()
    group = np.arange(len(centers), dtype=np.int32)
    if min_de00 <= 0:
        return centers, group, len(centers)

    alive = list(range(len(centers)))

    def merge(first: int, second: int) -> None:
        survivor, absorbed = alive[first], alive[second]
        total = weights[survivor] + weights[absorbed]
        if total > 0:
            centers[survivor] = (
                centers[survivor] * weights[survivor] + centers[absorbed] * weights[absorbed]
            ) / total
        weights[survivor] = total
        if kept is not None:
            kept[survivor] += kept[absorbed]
        group[group == absorbed] = survivor
        alive.remove(absorbed)

    while len(alive) > 1:
        distance = pairwise_de00(_lab_centers_to_bgr(centers[alive]))
        first, second = divmod(int(np.argmin(distance)), len(alive))
        if distance[first, second] >= min_de00:
            break
        merge(first, second)
    on_screen = len(alive)

    if paper_black_lstar > 0 or down_to is not None:
        while len(alive) > 1:
            apart = _printed_distances(centers[alive], weights[alive], paper_black_lstar, None if kept is None else kept[alive])
            first, second = divmod(int(np.argmin(apart)), len(alive))
            if apart[first, second] >= min_de00 and (down_to is None or len(alive) <= down_to):
                break
            merge(first, second)

    return centers, group, on_screen


def _printed_distances(
    centers_lab: np.ndarray, weights: np.ndarray, paper_black_lstar: float, kept_pixels: np.ndarray | None
) -> np.ndarray:
    """CIEDE2000 between every two colors as paper shows them, but the screen's for a pair with a kept color.

    Infinity down the diagonal, as ``pairwise_de00``.
    """
    lab = bgr_to_lab(_lab_centers_to_bgr(centers_lab))
    distance = printed_de00(lab[:, None, :], lab[None, :, :], paper_black_lstar)
    if kept_pixels is not None:
        kept = kept_pixels > _KEPT_SHARE * np.maximum(weights, 1)
        either = kept[:, None] | kept[None, :]
        distance = np.where(either, ciede2000(lab[:, None, :], lab[None, :, :]), distance)
    np.fill_diagonal(distance, np.inf)
    return distance


def _smooth(image_bgr: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma <= 0:
        return image_bgr
    return bilateral_filter(image_bgr, sigma_color=blur_sigma * 8, sigma_space=blur_sigma * 4)


def bilateral_filter(image_bgr: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Edge-preserving smoothing: ``cv2.bilateralFilter(image, 0, sigma_color, sigma_space)``, sampled sparsely.

    The exact filter visits every pixel within radius 1.5 * sigma_space, so
    its cost grows with sigma_space squared: ~9,000 taps per pixel at the Easy
    preset. Here the same window and weights are sampled on a lattice about
    sigma_space / 6 pixels apart -- a few hundred taps -- with rows filtered in
    parallel. The result differs from the exact filter by a small fraction of
    a just-noticeable color difference. When the lattice would be every pixel
    anyway, OpenCV's filter is used as is.
    """
    spacing = max(1, round(sigma_space / _TAPS_PER_SIGMA))
    if spacing == 1:
        return cv2.bilateralFilter(image_bgr, 0, sigma_color, sigma_space)
    return _sparse_bilateral(image_bgr, sigma_color, sigma_space, spacing)


def _sparse_bilateral(image_bgr: np.ndarray, sigma_color: float, sigma_space: float, spacing: int) -> np.ndarray:
    """Bilateral filter using window taps ``spacing`` pixels apart (1 = every pixel)."""
    h, w = image_bgr.shape[:2]
    radius = max(1, round(sigma_space * 1.5))

    steps = np.arange(-(radius // spacing), radius // spacing + 1) * spacing
    dy, dx = np.meshgrid(steps, steps, indexing="ij")
    inside = dy * dy + dx * dx <= radius * radius
    dy, dx = dy[inside].astype(np.int64), dx[inside].astype(np.int64)
    space_weights = np.exp(-0.5 * (dy * dy + dx * dx) / (sigma_space * sigma_space)).astype(np.float32)
    color_weights = np.exp(-0.5 * (np.arange(3 * 255 + 1) / sigma_color) ** 2).astype(np.float32)
    offsets = (dy * (w + 2 * radius) + dx) * 3

    padded = cv2.copyMakeBorder(image_bgr, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    padded_flat = np.ascontiguousarray(padded).reshape(-1)
    out = np.empty((h, w, 3), dtype=np.uint8)
    out_flat = out.reshape(-1)

    def filter_rows(y_start: int, y_stop: int) -> None:
        kernels.bilateral_rows(
            padded_flat, out_flat, y_start, y_stop, w, radius, offsets, space_weights, color_weights
        )

    parallel.for_each_stripe(filter_rows, h)
    return out


def paint_palette(
    samples_lab: np.ndarray,
    paints_bgr: np.ndarray,
    num_colors: int,
    min_de00: float,
    paper_black_lstar: float = 0.0,
    kept: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The painter's own paints that paint ``samples_lab`` best: each sample's paint, and the paints taken.

    Rather than colors found in the picture, colors chosen from a fixed set: up to ``num_colors`` of ``paints_bgr``
    (the paints' colors), each sample then painted in the nearest of them by CIEDE2000. Which ones is a facility
    location problem: the set whose nearest paint is closest to the picture's pixels, by CIEDE2000 summed over them,
    the pixels in ``kept`` (the faces and the subject) counted ``_KEPT_WEIGHT`` times. It is solved greedily -- the
    paint that lowers that sum the most, over and over, while one still lowers it, which is within 1 - 1/e of the best
    there is -- and then improved by the best swap of a paint taken for one left over, while a swap helps
    (``_SWAP_ROUNDS`` at most).

    Every two paints taken keep the margin, ``min_de00``, as the screen shows them and, with ``paper_black_lstar``, as
    paper does, unless one of them is mostly used in ``kept``, as the picture's own colors do (see
    ``_merge_close_colors``): a legend has no two colors a painter can't tell apart, even if they own both.

    The paints are chosen over the picture's colors counted into bins ``_PAINT_BIN`` units across, and each sample
    then takes the paint nearest its ``_BIN``-wide bin's color, so the work goes by the picture's colors, not its
    pixels. Nothing is random.

    Args:
        samples_lab: Nx3 float32, the pixels in OpenCV's 8-bit Lab.
        paints_bgr: Mx3 uint8, the paints' colors, BGR.
        num_colors: how many paints to take at most.
        min_de00: the margin between any two taken, in CIEDE2000; 0 or less lets any two be taken together.
        paper_black_lstar: the paper's darkest black, in L*; 0 keeps the screen's distances.
        kept: N bool, the samples that count more and keep the screen's margin, or None.

    Returns:
        (labels, chosen): each sample's index into ``chosen``, and ``chosen``, the indices into ``paints_bgr`` of the
        paints taken, darkest first.
    """
    paints_bgr = np.asarray(paints_bgr, dtype=np.uint8).reshape(-1, 3)
    if len(paints_bgr) == 0:
        raise ValueError("no paints to choose from")
    paint_lab = bgr_to_lab(paints_bgr)
    coarse, coarse_lab, weight = _binned(samples_lab, kept, _PAINT_BIN)
    distance = ciede2000(coarse_lab[:, None, :], paint_lab[None, :, :]).astype(np.float32)  # bins x paints
    clash = _clashes(paint_lab, distance, coarse, kept, min_de00, paper_black_lstar)
    taken = _take_paints(distance, weight, clash, num_colors)
    chosen = np.array(sorted(taken, key=lambda p: (paint_lab[p, 0], p)), dtype=np.int64)
    fine, fine_lab, _ = _binned(samples_lab, None, _BIN)
    bin_paint = np.argmin(ciede2000(fine_lab[:, None, :], paint_lab[chosen][None, :, :]), axis=1).astype(np.int32)
    return bin_paint[fine], chosen


def _clashes(
    lab: np.ndarray,
    distance: np.ndarray,
    coarse: np.ndarray,
    kept: np.ndarray | None,
    min_de00: float,
    paper_black_lstar: float,
) -> np.ndarray:
    """Which two of the colors at ``lab`` are closer than ``min_de00``: as the screen shows them and, with
    ``paper_black_lstar``, as paper does, unless one of them is mostly used in ``kept`` (see ``paint_palette``).
    ``distance`` is bins x colors, ``coarse`` each sample's bin."""
    clash = np.zeros((len(lab), len(lab)), dtype=bool)
    if min_de00 > 0:
        apart = ciede2000(lab[:, None, :], lab[None, :, :])
        if paper_black_lstar > 0:
            printed = printed_de00(lab[:, None, :], lab[None, :, :], paper_black_lstar)
            if kept is not None:
                mostly_kept = _kept_share(coarse, kept, distance) > _KEPT_SHARE
                printed = np.where(mostly_kept[:, None] | mostly_kept[None, :], apart, printed)
            apart = np.minimum(apart, printed)
        clash = apart < min_de00
        np.fill_diagonal(clash, False)
    return clash


def _take_paints(distance: np.ndarray, weight: np.ndarray, clash: np.ndarray, num_colors: int) -> list[int]:
    """The paints ``paint_palette`` takes: greedily, then improved by swaps."""
    taken: list[int] = []
    allowed = np.ones(distance.shape[1], dtype=bool)
    nearest = np.zeros(len(distance), dtype=np.float32)
    while len(taken) < min(num_colors, distance.shape[1]):
        # What each paint would save: how much nearer it brings each bin than its nearest paint so far, weighted.
        if taken:
            gain = weight @ np.maximum(nearest[:, None] - distance, 0)
        else:
            gain = -(weight @ distance)
        gain = np.where(allowed, gain, -np.inf)
        best = int(np.argmax(gain))
        if not np.isfinite(gain[best]) or (taken and gain[best] <= 0):
            break
        taken.append(best)
        allowed &= ~clash[best]
        allowed[best] = False
        nearest = distance[:, best] if len(taken) == 1 else np.minimum(nearest, distance[:, best])

    return _swap_paints(taken, distance, weight, clash)


def mixed_palette(
    samples_lab: np.ndarray,
    paints_bgr: np.ndarray,
    num_colors: int,
    min_de00: float,
    rule: MixRule,
    paper_black_lstar: float = 0.0,
    kept: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """The painter's paints that paint ``samples_lab`` best, and then colors mixed from two of them that earn their
    place: each sample's color, the colors, and each color's recipe.

    Paints first: they are chosen exactly as ``paint_palette`` chooses them. Then, while ``num_colors`` leaves room, a
    mix is added -- two of the paints in 1 + 1 to 3 + 1 parts (``mixing.pair_mixes``, its color predicted as real
    paint mixes) -- if it meets ``rule``: the pixels it would take over, those nearer it than every color taken, get
    ``rule.gain_de00`` nearer on average, and are ``rule.min_share`` of the picture, the faces and the subject counted
    twice (``_KEPT_WEIGHT``). Of those that do, the one that brings the picture nearest is added, and so on. A mix
    added later can take pixels from one added earlier, so every mix is then checked again against all the colors
    taken, and one that no longer meets the rule is dropped (the one that does least first) and not offered again;
    the room it leaves is offered to the others, ``_MIX_ROUNDS`` times at most.

    A mix is never within the margin of any paint, chosen or not (the paint does it without mixing), nor nearer one
    than ``rule.gain_de00``, nor within the margin of another color taken; the margin is kept as ``paint_palette``
    keeps it. Nothing is random.

    Returns:
        (labels, palette_bgr, recipes): each sample's index into the palette; the palette, the paints darkest first
        and then the mixes darkest first; and for each color None (a paint) or its recipe, ((paint, parts), (paint,
        parts)), the paint with more parts first.
    """
    paints_bgr = np.asarray(paints_bgr, dtype=np.uint8).reshape(-1, 3)
    if len(paints_bgr) == 0:
        raise ValueError("no paints to choose from")
    paint_lab = bgr_to_lab(paints_bgr)
    coarse, coarse_lab, weight = _binned(samples_lab, kept, _PAINT_BIN)
    distance = ciede2000(coarse_lab[:, None, :], paint_lab[None, :, :]).astype(np.float32)  # bins x paints
    clash = _clashes(paint_lab, distance, coarse, kept, min_de00, paper_black_lstar)
    paints_taken = sorted(_take_paints(distance, weight, clash, num_colors), key=lambda p: (paint_lab[p, 0], p))
    mixes: list[int] = []
    if len(paints_taken) < num_colors and len(paints_bgr) > 1:
        weighed = _mix_paints(paints_taken, distance, weight)
        pairs, parts, mix_rgb = mixing.pair_mixes(paints_bgr[:, ::-1], weighed)
        mix_lab = bgr_to_lab(np.ascontiguousarray(mix_rgb[:, ::-1]))
        mixes = _take_mixes(
            paints_taken, paint_lab, mix_lab, distance, coarse, coarse_lab, weight, kept, min_de00, paper_black_lstar,
            num_colors - len(paints_taken), rule,
        )
        mixes.sort(key=lambda m: (mix_lab[m, 0], m))
    palette_bgr = paints_bgr[paints_taken]
    palette_lab = paint_lab[paints_taken]
    recipes: tuple = (None,) * len(paints_taken)
    if mixes:
        palette_bgr = np.concatenate([palette_bgr, np.ascontiguousarray(mix_rgb[mixes][:, ::-1])])
        palette_lab = np.concatenate([palette_lab, mix_lab[mixes]])
        recipes += tuple(
            tuple(sorted(((int(p), int(n)) for p, n in zip(pairs[m], parts[m])), key=lambda pn: (-pn[1], pn[0])))
            for m in mixes
        )
    fine, fine_lab, _ = _binned(samples_lab, None, _BIN)
    bin_color = np.argmin(ciede2000(fine_lab[:, None, :], palette_lab[None, :, :]), axis=1).astype(np.int32)
    return bin_color[fine], palette_bgr, recipes


def _mix_paints(taken: list[int], distance: np.ndarray, weight: np.ndarray) -> list[int]:
    """The paints a palette's mixes are made from, in the order listed: all of them, or ``MIX_MOST_PAINTS`` -- those
    taken, then those nearest the most of the picture."""
    count = distance.shape[1]
    if count <= MIX_MOST_PAINTS:
        return list(range(count))
    share = np.bincount(np.argmin(distance, axis=1), weights=weight, minlength=count)
    picked = list(taken)[:MIX_MOST_PAINTS]
    for paint in np.lexsort((np.arange(count), -share)):
        if len(picked) == MIX_MOST_PAINTS:
            break
        if int(paint) not in picked:
            picked.append(int(paint))
    return sorted(picked)


def _take_mixes(
    paints_taken: list[int],
    paint_lab: np.ndarray,
    mix_lab: np.ndarray,
    distance: np.ndarray,
    coarse: np.ndarray,
    coarse_lab: np.ndarray,
    weight: np.ndarray,
    kept: np.ndarray | None,
    min_de00: float,
    paper_black_lstar: float,
    room: int,
    rule: MixRule,
) -> list[int]:
    """The mixes ``mixed_palette`` takes, ``room`` at most, by index into ``mix_lab``."""
    if len(mix_lab) == 0 or room <= 0:
        return []
    total = float(weight.sum())
    nearest = distance[:, paints_taken].min(axis=1)
    mix_distance = _distances_within(coarse_lab, mix_lab, nearest)  # bins x mixes
    # The most a mix can do is what it does against the paints alone: it only does less as colors are added. One that
    # can't meet the rule even so is not weighed.
    closer = np.maximum(nearest[:, None] - mix_distance, 0)
    hopeful = _earns_place(weight @ closer, weight @ (closer > 0), total, rule, upper_bound=True)
    hopeful &= ciede2000(mix_lab[:, None, :], paint_lab[None, :, :]).min(axis=1) >= max(min_de00, rule.gain_de00)
    candidates = np.flatnonzero(hopeful)
    if len(candidates) == 0:
        return []
    # The margin as ``_clashes`` keeps it, worked out only where it is needed: each mix against every paint, and a mix
    # taken against the other mixes.
    mostly_kept = None
    if min_de00 > 0 and paper_black_lstar > 0 and kept is not None:
        joint_distance = np.concatenate([distance, mix_distance[:, candidates]], axis=1)
        mostly_kept = _kept_share(coarse, kept, joint_distance) > _KEPT_SHARE
    paint_kept = None if mostly_kept is None else mostly_kept[: len(paint_lab)]
    mix_kept = None if mostly_kept is None else mostly_kept[len(paint_lab) :]
    to_paints = _clash_rows(mix_lab[candidates], paint_lab, mix_kept, paint_kept, min_de00, paper_black_lstar)
    apart = ~to_paints.any(axis=1)  # within the margin of no paint at all
    candidates = candidates[apart]
    mix_kept = None if mix_kept is None else mix_kept[apart]
    candidate_lab = mix_lab[candidates]
    rows: dict[int, np.ndarray] = {}

    def clash(t: int) -> np.ndarray:
        if t not in rows:
            own = None if mix_kept is None else mix_kept[t : t + 1]
            rows[t] = _clash_rows(candidate_lab[t : t + 1], candidate_lab, own, mix_kept, min_de00, paper_black_lstar)[0]
        return rows[t]

    columns = mix_distance[:, candidates]  # bins x candidates

    banned = np.zeros(len(candidates), dtype=bool)
    taken: list[int] = []  # into candidates
    for _ in range(_MIX_ROUNDS):
        now = nearest.copy()
        allowed = ~banned
        for t in taken:
            now = np.minimum(now, columns[:, t])
            allowed &= ~clash(t)
            allowed[t] = False
        while len(taken) < room and allowed.any():
            live = np.flatnonzero(allowed)
            gain = np.maximum(now[:, None] - columns[:, live], 0)
            saving = weight @ gain
            ok = _earns_place(saving, weight @ (gain > 0), total, rule)
            if not ok.any():
                break
            best = int(live[np.flatnonzero(ok)[np.argmax(saving[ok])]])
            taken.append(best)
            allowed &= ~clash(best)
            allowed[best] = False
            now = np.minimum(now, columns[:, best])
        dropped = False
        while taken:
            # Each mix against every other color taken: how much nearer it brings the pixels it is nearest.
            savings, used = [], []
            for t in taken:
                without = nearest
                for other in taken:
                    if other != t:
                        without = np.minimum(without, columns[:, other])
                gain = np.maximum(without - columns[:, t], 0)
                savings.append(float(weight @ gain))
                used.append(float(weight @ (gain > 0)))
            ok = _earns_place(np.array(savings), np.array(used), total, rule)
            if ok.all():
                break
            worst = min(np.flatnonzero(~ok), key=lambda i: (savings[i], i))
            banned[taken[worst]] = True
            del taken[worst]
            dropped = True
        if not dropped:
            break
    return [int(candidates[t]) for t in taken]


def _clash_rows(
    lab_a: np.ndarray,
    lab_b: np.ndarray,
    kept_a: np.ndarray | None,
    kept_b: np.ndarray | None,
    min_de00: float,
    paper_black_lstar: float,
) -> np.ndarray:
    """Which of the colors ``lab_a`` (n) is closer than ``min_de00`` to which of ``lab_b`` (m), n x m, as ``_clashes``
    judges it: on screen and, with ``paper_black_lstar``, as printed, unless either is mostly used where the picture
    is looked at (``kept_a``, ``kept_b``)."""
    if min_de00 <= 0:
        return np.zeros((len(lab_a), len(lab_b)), dtype=bool)
    apart = ciede2000(lab_a[:, None, :], lab_b[None, :, :])
    if paper_black_lstar > 0:
        printed = printed_de00(lab_a[:, None, :], lab_b[None, :, :], paper_black_lstar)
        if kept_a is not None and kept_b is not None:
            printed = np.where(kept_a[:, None] | kept_b[None, :], apart, printed)
        apart = np.minimum(apart, printed)
    return apart < min_de00


def _distances_within(lab: np.ndarray, colors_lab: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """CIEDE2000 from each of ``lab`` (n x 3) to each of ``colors_lab`` (m x 3), n x m float32, where it can be less
    than ``reach`` (n) and infinite elsewhere: a color more than ``_SURELY_APART`` times that from a bin in plain Lab
    distance is surely farther, and is never computed."""
    squared = (lab**2).sum(axis=1)[:, None] + (colors_lab**2).sum(axis=1)[None, :] - 2 * lab @ colors_lab.T
    near = squared < (_SURELY_APART * reach.astype(np.float64))[:, None] ** 2
    distance = np.full(near.shape, np.inf, dtype=np.float32)
    rows, columns = np.nonzero(near)
    distance[rows, columns] = ciede2000(lab[rows], colors_lab[columns])
    return distance


def _earns_place(
    saving: np.ndarray, used: np.ndarray, total: float, rule: MixRule, upper_bound: bool = False
) -> np.ndarray:
    """Whether mixes that would bring pixels weighing ``used`` (of ``total``) nearer by ``saving`` in all meet
    ``rule``. With ``upper_bound``, whether they might, given that both can only shrink as colors are added: their
    saving reaches the gain times the share, and their pixels the share."""
    enough = (used >= rule.min_share * total) & (used > 0)
    if upper_bound:
        return enough & (saving >= rule.gain_de00 * rule.min_share * total)
    return enough & (saving >= rule.gain_de00 * np.maximum(used, 1e-12))


def _swap_paints(taken: list[int], distance: np.ndarray, weight: np.ndarray, clash: np.ndarray) -> list[int]:
    """``taken`` improved by the best swap of one paint taken for one left over, over and over, while one helps."""
    taken = list(taken)
    if not taken or len(taken) == distance.shape[1]:
        return taken
    for _ in range(_SWAP_ROUNDS):
        columns = distance[:, taken]
        order = np.argsort(columns, axis=1, kind="stable")
        first = np.take_along_axis(columns, order[:, :1], axis=1)[:, 0]
        if len(taken) > 1:
            second = np.take_along_axis(columns, order[:, 1:2], axis=1)[:, 0]
        else:
            second = np.full_like(first, np.inf)
        cost = float(weight @ first)
        best_saving, best_slot, best_paint = 1e-6 * cost, -1, -1
        for slot in range(len(taken)):
            without = np.where(order[:, 0] == slot, second, first)  # each bin's nearest among the others
            others = taken[:slot] + taken[slot + 1 :]
            candidates = ~clash[others].any(axis=0) if others else np.ones(distance.shape[1], dtype=bool)
            candidates[taken] = False
            if not candidates.any():
                continue
            columns_left = np.flatnonzero(candidates)
            if len(taken) == 1:
                costs = weight @ distance[:, columns_left]
            else:
                costs = weight @ np.minimum(without[:, None], distance[:, columns_left])
            pick = int(np.argmin(costs))
            if cost - float(costs[pick]) > best_saving:
                best_saving, best_slot, best_paint = cost - float(costs[pick]), slot, int(columns_left[pick])
        if best_slot < 0:
            break
        taken[best_slot] = best_paint
    return taken


def _binned(samples_lab: np.ndarray, kept: np.ndarray | None, size: float) -> tuple[np.ndarray, ...]:
    """The samples counted into bins ``size`` units of OpenCV's 8-bit Lab across: each sample's bin, each bin's mean
    color as CIE Lab, and its weight, a sample in ``kept`` counting ``_KEPT_WEIGHT`` times."""
    per_axis = int(round(256 / size))
    bins = np.clip((samples_lab / size).astype(np.int32), 0, per_axis - 1)
    flat = (bins[:, 0] * per_axis + bins[:, 1]) * per_axis + bins[:, 2]
    live, index = np.unique(flat, return_inverse=True)
    index = index.reshape(-1).astype(np.int32)
    counts = np.bincount(index, minlength=len(live)).astype(np.float64)
    mean = np.stack([np.bincount(index, samples_lab[:, c], len(live)) for c in range(3)], -1) / counts[:, None]
    lab = np.stack([mean[:, 0] / _OPENCV_LIGHTNESS_SCALE, mean[:, 1] - 128, mean[:, 2] - 128], -1)
    weight = counts
    if kept is not None:
        weight = weight + (_KEPT_WEIGHT - 1) * np.bincount(index, np.asarray(kept, dtype=np.float64), len(live))
    return index, lab, weight.astype(np.float32)


def _kept_share(index: np.ndarray, kept: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """For each paint, the share in ``kept`` of the samples whose nearest paint of all it is."""
    nearest = np.argmin(distance, axis=1)[index]
    total = np.bincount(nearest, minlength=distance.shape[1]).astype(np.float64)
    inside = np.bincount(nearest, np.asarray(kept, dtype=np.float64), distance.shape[1])
    return inside / np.maximum(total, 1)

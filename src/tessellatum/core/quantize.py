"""Color quantization: reduce an image to a small palette of flat colors.

K-means in Lab space, then a merge of the colors a painter could neither tell
apart nor mix -- closer than ``color.MIN_PALETTE_DE00`` -- so the palette comes
back with every color clearly different from every other.

Line art gets its palette a different way (``flat_color_palette``). Its fills
are flat colors the artist chose, so the page should offer those colors rather
than the means k-means lands on, and the margin should be kept while the colors
are picked rather than by merging afterwards, which spends colors and leaves
shades the artwork does not have.
"""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.color import MIN_PALETTE_DE00, bgr_to_lab, ciede2000, pairwise_de00
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

# The line-art palette (``flat_color_palette``) reads the fills' colors as a
# histogram of OpenCV's 8-bit Lab, where a step of 1 is 0.39 L* or one unit of
# a or b. A bin is _BIN units across, and a color's own pixels are the bin and
# the 26 around it -- a neighborhood of one bin, about 2 to 4 CIEDE2000.
_BIN = 4.0
_BINS_PER_AXIS = 64  # 256 / _BIN
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
) -> tuple[np.ndarray, np.ndarray]:
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

    Returns:
        (label_map, palette_bgr):
            label_map: HxW int32 array, each pixel's color index; ``K``, one
                past the palette, on the ink.
            palette_bgr: Kx3 uint8 array of BGR colors, K <= num_colors,
                ordered from darkest to lightest (by perceptual lightness)
                so numbering reads naturally.
    """
    smoothed = _smooth(image_bgr, blur_sigma)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    fitted = None  # the pixels the colors are found from: all of them, unless there is ink and something besides it
    if ink is not None and ink.any() and not ink.all():
        fitted = ~near(ink, halo_px)
        if not fitted.any():  # nothing but ink and its edge: fit what there is off the ink
            fitted = ~ink
    samples = lab.reshape(-1, 3).astype(np.float32)
    if fitted is None:
        labels, palette_bgr = _kmeans_colors(samples, num_colors, seed, min_de00)
        return labels.reshape(lab.shape[:2]), palette_bgr

    samples = np.ascontiguousarray(samples[fitted.reshape(-1)])
    centers_lab = flat_color_palette(samples, num_colors, min_de00)
    palette_bgr = _lab_centers_to_bgr(centers_lab)
    label_map = np.full(lab.shape[:2], len(palette_bgr), dtype=np.int32)
    label_map[fitted] = _nearest_color(samples, centers_lab)
    label_map[~fitted & ~ink] = _nearest_fitted(label_map, fitted)[~fitted & ~ink]
    return label_map, palette_bgr


def _kmeans_colors(
    samples: np.ndarray, num_colors: int, seed: int, min_de00: float
) -> tuple[np.ndarray, np.ndarray]:
    """K-means over ``samples`` (Nx3 float32 Lab), then the merge: each sample's color, and the palette."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITERATIONS, _KMEANS_EPSILON)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers_lab = cv2.kmeans(
        samples, min(num_colors, len(samples)), None, criteria, attempts=_KMEANS_ATTEMPTS, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(-1).astype(np.int32)
    pixels_per_color = np.bincount(labels, minlength=len(centers_lab))
    centers_lab, group = _merge_close_colors(centers_lab, pixels_per_color, min_de00)

    kept = np.unique(group)  # the colors that survived the merge, in cluster order
    kept_centers = centers_lab[kept]
    order = np.argsort(kept_centers[:, 0])  # sort by L (lightness), darkest first
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)

    final_index = np.zeros(len(centers_lab), dtype=np.int32)  # cluster -> its color's place on the legend
    final_index[kept] = rank
    return final_index[group][labels].astype(np.int32), _lab_centers_to_bgr(kept_centers)[order]


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
    centers_lab: np.ndarray, pixels_per_color: np.ndarray, min_de00: float
) -> tuple[np.ndarray, np.ndarray]:
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

    Args:
        centers_lab: Kx3 cluster centers in OpenCV's 8-bit Lab scale.
        pixels_per_color: how many pixels each cluster has.
        min_de00: the margin to reach; 0 or less leaves the centers alone.

    Returns:
        (centers_lab, group): the centers, with every survivor moved to its
        merged color, and for each original cluster the index of the center it
        ended up in (``group[i] == i`` for a color that survived).
    """
    centers = np.array(centers_lab, dtype=np.float64)
    weights = np.asarray(pixels_per_color, dtype=np.float64).copy()
    group = np.arange(len(centers), dtype=np.int32)
    if min_de00 <= 0:
        return centers, group

    alive = list(range(len(centers)))
    while len(alive) > 1:
        distance = pairwise_de00(_lab_centers_to_bgr(centers[alive]))
        first, second = divmod(int(np.argmin(distance)), len(alive))
        if distance[first, second] >= min_de00:
            break
        survivor, absorbed = alive[first], alive[second]
        total = weights[survivor] + weights[absorbed]
        if total > 0:
            centers[survivor] = (
                centers[survivor] * weights[survivor] + centers[absorbed] * weights[absorbed]
            ) / total
        weights[survivor] = total
        group[group == absorbed] = survivor
        alive.remove(absorbed)

    return centers, group


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

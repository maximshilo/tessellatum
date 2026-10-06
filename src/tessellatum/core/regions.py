"""Connected-component region extraction, region merging, and contour extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.color import MIN_PALETTE_DE00, bgr_to_lab, ciede2000

# How often line art's regions are merged and split again before they settle (see ``split_areas``). On the benchmark
# pages they settle after one or two.
_SETTLE_ROUNDS = 4

# Regions with a smaller bounding box than this are extracted on the calling
# thread: for them, handing off to a worker (and contending for the GIL) costs
# more than the OpenCV work itself.
_MIN_POOLED_BOX_PX = 32_000

# A round brush can't reach into a corner: it stops at the circle touching both sides, and the brush rule
# (``absorb_thin_parts``) gives the point beyond -- the corner's *tip* -- to the region beside it, rounding every
# corner to the brush. Asked to, it keeps the tips of corners no sharper than a given angle instead (see
# ``CornerRule``): a painter fills them with the brush's point. By default it doesn't: 180 degrees rounds every corner,
# as pages always were. 20 degrees keeps a spire's point, and a star's, while a drainpipe on a wall, a few times longer
# than wide, is a needle and still goes.
SHARPEST_CORNER_DEG = 180.0

# A tip is kept only where the picture shows it plainly: where its pixels lie, on average, this much closer by
# CIEDE2000 to their region's color than to the color the brush rule would give them, by default (see
# ``CornerRule``). A roof against the sky, or a flat fill's point, stands that far apart; the spikes of fur and
# foliage, whose pixels lie between two neighboring colors of the palette, a step of it apart, do not, and are rounded
# off as before.
CORNER_CONTRAST_DE00 = 20.0

# A tip meets the rest of its region along a base at least this many brush radii long -- half the brush. A part
# narrower where it leaves its region is a strip, however short, and goes as any thin part does.
_TIP_BASE_RADII = 1.0

# How far round a tip the pixels its brush can't reach are part of it (see ``corner_tips``): the band along its sides
# and its base, a pixel or so wide, that lies within a pixel of a brush.
_TIP_RIM_PX = 2

# How much deeper or longer than its shape says a corner's tip may measure on the pixel grid: a corner as sharp as
# the setting measured up to 17% deeper at a preview's 6 px brush radius (a wedge's tip is a few hundred pixels), 6%
# at an export's.
_TIP_SLACK = 1.2

# A shape's corner stands alone: a tip with another corner's tip -- of any region -- within this many brush radii of it
# (a brush's width, between their middles) is the edge of fur, foliage or a ragged silhouette, whose spikes and notches
# alternate, and is rounded off as any thin part is. A triangle's corners lie further apart than that, but for a
# triangle smaller than the brush is wide. Kept, a ragged edge's tips would make its lines zigzag (on the benchmark's
# castle 4% longer than smoothed, against 1% with them rounded).
_TIP_APART_RADII = 2.0


@dataclass(frozen=True, eq=False)
class CornerRule:
    """Which corners' tips the brush rule keeps (see ``absorb_thin_parts`` and ``corner_tips``).

    ``sharpest_deg`` is the sharpest corner whose tip is kept, in degrees; 180
    keeps none. A tip is kept only if its pixels in ``image_bgr`` lie, on
    average, at least ``contrast_de00`` closer (CIEDE2000) to their region's
    color in ``palette_bgr`` than to the color that would take them; without a
    picture, or at 0, the corner's shape is enough.
    """

    sharpest_deg: float
    contrast_de00: float = CORNER_CONTRAST_DE00
    image_bgr: np.ndarray | None = None
    palette_bgr: np.ndarray | None = None


def tip_length(sharpest_deg: float) -> float:
    """How far beyond the brush the point of a corner ``sharpest_deg`` sharp lies, in brush radii.

    The circle a brush of radius r stops at, in a corner of angle a, has its
    middle r / sin(a/2) from the point, so the point lies r (1 / sin(a/2) - 1)
    beyond it: 0.41 at 90 degrees, 2.9 at 30, 4.8 at 20. 0 at 180 degrees and
    over.
    """
    if not sharpest_deg > 0:
        raise ValueError(f"a corner is sharper than 0 degrees, not {sharpest_deg}")
    if sharpest_deg >= 180:
        return 0.0
    return 1 / math.sin(math.radians(sharpest_deg) / 2) - 1


def tip_depth(sharpest_deg: float) -> float:
    """How deep a corner ``sharpest_deg`` sharp has its tip, in brush radii: the tip's area over the length of its base.

    A round brush of radius r in a corner of angle a reaches no nearer the
    point than the circle touching both sides. What lies beyond, the tip, has
    the area r² (cot(a/2) - (π - a)/2) and meets the rest of the region along
    an arc r (π - a) long, so the ratio of the two grows as the corner
    sharpens: 0.14 at 90 degrees, 0.93 at 30, 2.1 at 15. A strip's is its
    length. 0 at 180 degrees and over, where there is no corner.
    """
    if not sharpest_deg > 0:
        raise ValueError(f"a corner is sharper than 0 degrees, not {sharpest_deg}")
    if sharpest_deg >= 180:
        return 0.0
    angle = math.radians(sharpest_deg)
    return (1 / math.tan(angle / 2) - (math.pi - angle) / 2) / (math.pi - angle)


def corner_tips(region_id_map: np.ndarray, num_regions: int, min_width_px: float, sharpest_deg: float) -> np.ndarray:
    """The pixels a brush ``min_width_px`` wide can't reach that are corners' tips all the same: HxW bool.

    What a region's brush can't reach falls into pieces (8-connected). A
    piece is a corner's tip if it meets the part of its region the brush
    reaches along one stretch, its base, at least half the brush long (pairs
    of pixels side by side, ``_TIP_BASE_RADII``), and is neither deeper nor
    longer than the tip of a corner ``sharpest_deg`` sharp: its area over its
    base no more than ``tip_depth``, its farthest pixel no farther from the
    rest than ``tip_length``, both give or take the pixel grid
    (``_TIP_SLACK``). A strip narrower than half the brush where it leaves its
    region, longer than such a tip is deep, or a needle reaching further than
    its point, is no tip; nor is a neck between two parts the brush reaches, a
    channel meeting it in two places, or a region with none. And a tip stands
    alone: with another's middle within a brush's width of its own
    (``_TIP_APART_RADII``), it is a spike of a ragged edge.

    A piece is made of the pixels more than a pixel beyond the brush and
    those beside them. The rest lie within a pixel of it: along a slanted
    edge, the grid's staircase leaves single pixels there, which are no
    corner, and would join a tip to whatever lies along the edge. A tip then
    takes in what the brush can't reach of its region within ``_TIP_RIM_PX``
    of it: the pixels along its sides and its base the piece leaves out, but
    not a thin part that runs on from it.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    return _tips(ids, num_regions, min_width_px / 2, sharpest_deg, None)[0] >= 0


def _tips(
    ids: np.ndarray, num_regions: int, radius: float, sharpest_deg: float, fits: np.ndarray | None
) -> tuple[np.ndarray, int]:
    """(each pixel's corner tip, numbered from 0, -1 for none; how many numbers there are), see ``corner_tips``.

    ``fits`` is where a brush of ``radius`` fits (``_brush_fits``), if known.
    """
    h, w = ids.shape
    pieces = np.full((h, w), -1, dtype=np.int32)
    depth, length = tip_depth(sharpest_deg), tip_length(sharpest_deg)
    if depth <= 0 or num_regions == 0:
        return pieces, 0
    if fits is None:
        fits = _brush_fits(ids, num_regions, radius)
    if not fits.any():
        return pieces, 0
    to_brush = cv2.distanceTransform((~fits).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    flat = ids.reshape(-1)
    unreached = np.empty((h, w), dtype=bool)
    area, box = kernels.brush_pieces(
        flat, to_brush.reshape(-1), radius, h, w, unreached.reshape(-1), pieces.reshape(-1)
    )
    if area.size == 0:
        return pieces, 0
    base_of = np.empty((h, w), dtype=np.int32)
    base = kernels.piece_bases(pieces.reshape(-1), flat, unreached.reshape(-1), area.size, h, w, base_of.reshape(-1))
    # The stretches along which each piece meets what its brush reaches: 8-connected runs of the pixels it meets.
    stretches = kernels.count_stretches(base_of.reshape(-1), area.size, h, w)
    tip = (base >= _TIP_BASE_RADII * radius) & (stretches == 1) & (area <= (_TIP_SLACK * depth * radius + 0.5) * base)
    if tip.any():
        reach = kernels.piece_lengths(pieces.reshape(-1), flat, unreached.reshape(-1), box, tip, h, w)
        tip &= reach <= _TIP_SLACK * length * radius + 1  # measured from the middle of the last pixel reached
    if not tip.any():
        return np.full((h, w), -1, dtype=np.int32), 0
    on = pieces >= 0
    ys, xs = np.nonzero(on)
    which = np.flatnonzero(tip)
    if which.size > 1:
        middle_y = (np.bincount(pieces[ys, xs], ys, minlength=area.size) / area)[which]
        middle_x = (np.bincount(pieces[ys, xs], xs, minlength=area.size) / area)[which]
        order = np.argsort(middle_x, kind="stable")
        crowded = np.empty(which.size, dtype=bool)
        crowded[order] = kernels.near_another(middle_y[order], middle_x[order], _TIP_APART_RADII * radius)
        tip[which[crowded]] = False
    if not tip.any():
        return np.full((h, w), -1, dtype=np.int32), 0
    tips = np.full((h, w), -1, dtype=np.int32)
    kept = tip[pieces[ys, xs]]
    tips[ys[kept], xs[kept]] = pieces[ys[kept], xs[kept]]
    # The rim: what the brush can't reach of a tip's region beside it, which takes the tip's number.
    pixels = (ys[kept] * w + xs[kept]).astype(np.int64)
    kernels.tip_rims(tips.reshape(-1), flat, unreached.reshape(-1), pixels, _TIP_RIM_PX, h, w)
    return tips, int(area.size)


def _keep_corners(widened, ids, region_color, radius, fits, corners: CornerRule) -> None:
    """Give, in place, every corner's tip ``corners`` keeps its own region's color back in ``widened``."""
    tips, count = _tips(ids, int(region_color.size), radius, corners.sharpest_deg, fits)
    if count == 0:
        return
    ys, xs = np.nonzero(tips >= 0)
    own = region_color[ids[ys, xs]]
    taken = widened[ys, xs] != own
    ys, xs, own = ys[taken], xs[taken], own[taken]
    if not ys.size:
        return
    if corners.contrast_de00 > 0 and corners.image_bgr is not None:
        # How much closer each pixel is to its own color than to the one it was given, averaged over its tip.
        picture = bgr_to_lab(np.asarray(corners.image_bgr)[ys, xs])
        palette = bgr_to_lab(np.asarray(corners.palette_bgr, dtype=np.uint8).reshape(-1, 3))
        closer = ciede2000(picture, palette[widened[ys, xs]]) - ciede2000(picture, palette[own])
        which = tips[ys, xs]
        mean = np.bincount(which, closer, minlength=count) / np.maximum(np.bincount(which, minlength=count), 1)
        plain = mean[which] >= corners.contrast_de00
        ys, xs, own = ys[plain], xs[plain], own[plain]
    widened[ys, xs] = own


@dataclass
class Region:
    """One contiguous, single-colored region of the final coloring page."""

    region_id: int
    color_index: int
    contour: np.ndarray  # Nx1x2 int32 points (OpenCV contour format), outer boundary
    interior_point: tuple[int, int]  # (x, y) safe point for a number label
    interior_radius: float  # px clearance at interior_point, used for font sizing
    area: int


def build_regions(
    labels: np.ndarray,
    num_colors: int,
    min_area_px: int,
    min_width_px: float = 0.0,
    detail: np.ndarray | None = None,
    detail_weight: int = 2,
    corners: CornerRule | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``labels`` into connected regions and merge away what can't be painted.

    Regions are 8-connected runs of one color, numbered by color and then by
    raster position. Regions below ``min_area_px`` are merged, smallest first,
    into the neighbor sharing the most boundary (see
    ``kernels.merge_small_regions``). That merge ignores color, so it can
    leave two neighbors sharing one: those are then unioned into a single
    region (``kernels.merge_same_color_neighbors``), so no boundary on the
    page separates two areas the painter fills with the same color.

    ``detail`` (HxW bool) marks where the page spends more detail -- the
    faces in the picture. A region lying there may be ``detail_weight`` times
    smaller (half, by default): each of its pixels in ``detail`` counts that
    many times towards ``min_area_px``, so a region partly in it needs
    somewhere between that share and all of it. Sizes are counted so for which
    region is smallest, too.

    With ``min_width_px`` set, every part of a region narrower than that --
    which a brush that wide cannot paint without crossing a line -- is then
    given away to the region whose paint reaches it first, and the regions
    are rebuilt from the result (see ``absorb_thin_parts``), but for the
    corners' tips ``corners`` keeps. A region thinner than the brush
    everywhere disappears into its neighbors, in ``detail`` too.

    Returns:
        (region_id_map, region_color): region_id_map is HxW int32 (each pixel's
        region id), region_color is region_count-length int32 array mapping a
        region id to its color index in the palette.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    if detail is not None and not detail.any():
        detail = None
    if detail_weight < 1:
        raise ValueError(f"a pixel in the detail counts at least once, not {detail_weight} times")
    if detail is not None and detail_weight == 1:
        detail = None
    region_id_map, region_color = _regions_from_labels(labels, num_colors, min_area_px, detail, detail_weight)
    if min_width_px > 0 and region_color.size:
        widened = absorb_thin_parts(region_id_map, region_color, labels, min_width_px, corners)
        if widened is not None:
            region_id_map, region_color = _regions_from_labels(widened, num_colors, min_area_px, detail, detail_weight)
    return region_id_map, region_color


def _regions_from_labels(
    labels: np.ndarray, num_colors: int, min_area_px: int, detail: np.ndarray | None = None, detail_weight: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``labels``, with the two merges that always apply."""
    h, w = labels.shape
    region_id_map = np.empty((h, w), dtype=np.int32)
    flat_ids = region_id_map.reshape(-1)

    region_color, areas = kernels.label_components(labels.reshape(-1), h, w, int(num_colors), flat_ids)
    if region_color.size:
        if detail is not None:
            # A pixel in the detail counts detail_weight times: once in its region's pixel count, the rest here.
            ids_in_detail = flat_ids[np.asarray(detail, dtype=bool).reshape(-1) & (flat_ids >= 0)]
            areas += (detail_weight - 1) * np.bincount(ids_in_detail, minlength=areas.size)
        kernels.merge_small_regions(flat_ids, h, w, areas, int(min_area_px), True)
        kernels.merge_same_color_neighbors(flat_ids, h, w, region_color, areas, True)
    return region_id_map, region_color


def absorb_thin_parts(
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    labels: np.ndarray,
    min_width_px: float,
    corners: CornerRule | None = None,
) -> np.ndarray | None:
    """Give every pixel the color of the region whose core lies nearest, or None if there is no core.

    A region's *core* is the pixels where a round brush ``min_width_px``
    across fits inside the region: the places a painter can put the brush
    without crossing into a neighbor. Every pixel then takes the color of the
    nearest core pixel, which leaves each region its core plus everything the
    brush sweeps around it, and hands the parts too thin to paint -- and
    regions with no core at all -- to whichever neighbor reaches them first.

    A pixel its own region's brush can reach keeps its color: no other
    region's core can be nearer than its own. So only thin parts move, and a
    thin part between two regions is split down its middle rather than given
    to one side.

    Pixels in no region (``labels`` outside the palette) keep their label and
    take no part. They are line art's ink, which paint does not cross, so a
    thin part is only ever given to a core it can reach without crossing
    them: where the nearest core lies on the far side, the nearest one on its
    own side is found instead (``kernels.nearest_seed_within``), and a thin
    part with no core on its side at all keeps its color.

    Given ``corners``, the tips of corners it keeps keep their color too,
    rather than every corner being rounded to the brush (see ``CornerRule``).

    Returns the new HxW int32 label map, or None when the brush fits nowhere
    on the page, leaving nothing to grow from.
    """
    fits = _brush_fits(region_id_map, int(region_color.size), min_width_px / 2)
    if not fits.any():
        return None

    # For every pixel, the label of the nearest zero pixel: of the nearest core pixel.
    _distance, nearest = cv2.distanceTransformWithLabels(
        (~fits).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    nearest = np.ascontiguousarray(nearest, dtype=np.int32)
    widened = np.empty(nearest.shape, dtype=np.int32)
    kernels.nearest_core_color(
        nearest.reshape(-1),
        fits.reshape(-1),
        np.ascontiguousarray(region_id_map, dtype=np.int32).reshape(-1),
        np.asarray(region_color, dtype=np.int32),
        widened.reshape(-1),
    )

    outside = region_id_map < 0
    if outside.any():
        _keep_to_own_side(widened, nearest, fits, outside, region_id_map, region_color, labels)
        widened[outside] = labels[outside]
    if corners is not None:
        ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
        _keep_corners(widened, ids, np.asarray(region_color, dtype=np.int32), min_width_px / 2, fits, corners)
    return widened


def _keep_to_own_side(widened, nearest, fits, outside, region_id_map, region_color, labels) -> None:
    """Undo, in place, every pixel of ``widened`` given a core that lies across ``outside`` from it.

    The pixels in no region split the rest of the page into compartments
    (8-connected, as regions are). A pixel whose nearest core is in another
    compartment takes the color of the nearest core in its own instead, by
    distance along a path that stays in it; with no core in its compartment,
    it keeps its own color from ``labels``.
    """
    h, w = region_id_map.shape
    _count, compartment = cv2.connectedComponents((~outside).view(np.uint8), connectivity=8)
    compartment = np.ascontiguousarray(compartment, dtype=np.int32)
    across = np.empty((h, w), dtype=bool)
    # Only the compartments holding such pixels need searching, and only as far as they reach.
    wanted = kernels.across_to_nearest_core(
        nearest.reshape(-1), fits.reshape(-1), compartment.reshape(-1), outside.reshape(-1), across.reshape(-1)
    )
    if not wanted.any():
        return
    # A shortest path from the cores leaves them at their edge, so their insides need no searching.
    core_edge = fits & ~cv2.erode(fits.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
    searched = wanted[compartment] & ~outside & (~fits | core_edge)
    rows, columns = np.flatnonzero(searched.any(axis=1)), np.flatnonzero(searched.any(axis=0))
    box = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))
    passable = np.ascontiguousarray(searched[box])
    seeds = np.full(passable.shape, -1, dtype=np.int32)
    seeded = fits[box] & passable
    seeds[seeded] = region_color[region_id_map[box][seeded]]
    bh, bw = passable.shape
    reached = np.full((h, w), -1, dtype=np.int32)
    reached[box] = kernels.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), bh, bw).reshape(bh, bw)
    widened[across] = np.where(reached[across] >= 0, reached[across], labels[across])


def join_ink(
    labels: np.ndarray, num_colors: int, image_bgr: np.ndarray, ink_bgr, min_area_px: int, own: np.ndarray | None = None
) -> np.ndarray:
    """``labels`` with the patches too small to keep that are in the ink's own color, and mostly edged by it, made ink.

    ``labels`` marks line art's ink with ``num_colors``, one past the palette.
    A patch of one palette color (8-connected) smaller than ``min_area_px``
    merges into the neighbor it shares the most boundary with (see
    ``build_regions``). Where that neighbor is the ink, and the patch's pixels
    in ``image_bgr`` average to the ink's own color -- closer to ``ink_bgr``
    than two palette colors may be (``color.MIN_PALETTE_DE00``) -- the patch
    is the ink running wider than a line, where a disk as wide as the widest
    line fits, and it joins the ink rather than be painted the color of the
    fill beside it. A larger one, such as a black face, stays a region to
    paint, and so does a small dark fill the ink only edges in part.

    The patch's own pixels decide its color, not its palette color: a coarse
    palette lumps dark grays and browns in with black. Only those in ``own``
    (HxW bool) count where it has any: off the ink's anti-aliased edge, whose
    mix with the ink would pull a dark fill's color towards it. Returns a new
    label map.
    """
    ink = labels == num_colors
    if not ink.any():
        return labels
    h, w = labels.shape
    ids = np.empty((h, w), dtype=np.int32)
    _component_color, areas = kernels.label_components(np.ascontiguousarray(labels).reshape(-1), h, w, num_colors, ids.reshape(-1))
    beside_ink = cv2.dilate(ink.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool) & (ids >= 0)
    candidates = np.zeros(areas.size, dtype=bool)
    candidates[ids[beside_ink]] = True
    candidates &= areas < min_area_px
    if not candidates.any():
        return labels
    inside = ids >= 0
    joining = np.zeros(areas.size, dtype=bool)
    colors = _mean_colors(ids, image_bgr, candidates, own)
    joining[candidates] = _de00_to(colors[candidates], ink_bgr) < MIN_PALETTE_DE00
    joining &= _ink_is_main_neighbor(ids, joining)
    if not joining.any():
        return labels
    return np.where(inside & joining[np.where(inside, ids, 0)], num_colors, labels).astype(np.int32)


def _ink_is_main_neighbor(ids: np.ndarray, asked: np.ndarray) -> np.ndarray:
    """For each patch id ``asked`` about, whether the ink (-1) owns at least as much of its outer ring as any one patch.

    A patch's ring is the pixels 8-adjacent to it that aren't its own, each
    counted once, as ``kernels.merge_small_regions`` counts it; a tie goes to
    the ink.
    """
    h, w = ids.shape
    if not asked.any():
        return asked
    # Off the page is nobody's.
    return kernels.ink_is_main_neighbor(np.ascontiguousarray(ids, dtype=np.int32).reshape(-1), asked, h, w)


def settle_enclosed(
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    image_bgr: np.ndarray,
    ink_bgr,
    min_width_px: float,
    own: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """What becomes of the regions line art's ink encloses on its own, with no other region beside them.

    Such a region has nothing to merge into -- the ink lies between it and
    everything else -- so the difficulty's smallest area doesn't apply to it:
    the artwork's own shapes, a finger or a button, keep their number however
    small, as long as a brush ``min_width_px`` wide fits in them. Two kinds
    don't:

    - **one in the ink's own color**, whose pixels in ``image_bgr`` average to
      closer to ``ink_bgr`` than two palette colors may be
      (``color.MIN_PALETTE_DE00``), is the ink itself where it runs wider than
      a line: it is printed with it, as two neighbors of one color are one
      region. As in ``join_ink``, only its pixels in ``own`` count where it
      has any;
    - **one no brush fits in anywhere** is left as bare paper: too small to
      paint, and not part of the drawing's ink either.

    Pixels in no region (-1) are the ink. Returns the region map with both
    kinds taken out (-1), and an HxW bool mask of the pixels printed with the
    ink.
    """
    ids = region_id_map
    count = int(region_color.size)
    inked = np.zeros(ids.shape, dtype=bool)
    if count == 0:
        return ids, inked
    inside = ids >= 0
    areas = np.bincount(ids[inside].ravel(), minlength=count)
    alone = (areas > 0) & ~_has_neighbor(ids, count)
    if not alone.any():
        return ids, inked
    region_of = np.where(inside, ids, 0)
    enclosed = inside & alone[region_of]
    # No two of them touch, so a brush fits in one where the nearest pixel outside all of them is far enough.
    distance = cv2.distanceTransform(np.pad(enclosed, 1).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    fits = enclosed & (distance > min_width_px / 2)
    cored = np.bincount(ids[fits].ravel(), minlength=count) > 0
    like_ink = np.zeros(count, dtype=bool)
    like_ink[alone] = _de00_to(_mean_colors(ids, image_bgr, alone, own)[alone], ink_bgr) < MIN_PALETTE_DE00
    to_ink = alone & like_ink
    to_paper = alone & ~to_ink & ~cored
    inked = inside & to_ink[region_of]
    gone = inked | (inside & to_paper[region_of])
    return np.where(gone, -1, ids).astype(np.int32), inked


def look_through_hatching(
    labels: np.ndarray, num_colors: int, thin_px: float, brush_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """``labels`` as the region stage should see line art: through its hatching, but not through a line between two areas.

    ``labels`` marks line art's ink with ``num_colors``, one past the
    palette. The white pieces of the page -- the 4-connected runs of pixels
    off the ink -- are what a painter sees as areas. A piece a brush
    ``brush_px`` wide fits in is an area to paint; the rest are the gaps
    between strokes. Each paintable piece claims the gaps and the thin ink
    (thinner than ``thin_px``, see ``paint_over_thin_ink``) it reaches first
    without crossing bold ink, a gap going wholly to the piece that reaches
    most of it; what no paintable piece reaches, hatching that bold ink walls
    in, is claimed by nobody but itself. Where two claims meet, the ink stays
    ink: a wall, so no region can join two areas a painter sees as two, and
    each keeps its own number. Where two pieces of different claims touch at
    a corner, across a one-pixel diagonal line, one of the two pixels is taken
    for ink too, and returned so that it is printed with it.

    Every other thin ink pixel takes the label of the nearest pixel off the
    ink in its claim, as if the ink weren't there. So a hatched patch is one
    run of its gaps' colors, which the region stage makes into areas the way
    it does on any picture -- the patch is colored from its own gaps, not
    from whatever lies beyond its strokes. The ink is printed either way.

    Returns the labels the region stage should build regions from, and an HxW
    bool mask of the pixels taken for ink at the corners.
    """
    ink = labels == num_colors
    corners = np.zeros(labels.shape, dtype=bool)
    if not ink.any() or ink.all():
        return labels, corners
    bold = ink & ~_thin_part(ink, thin_px)
    white = ~ink
    count, piece = cv2.connectedComponents(white.view(np.uint8), connectivity=4)
    piece = piece.astype(np.int32) - 1  # -1 on the ink
    # Off the page is not white, so a brush fits in a piece along the page's edge only as far in as it does on paper.
    distance = cv2.distanceTransform(np.pad(white, 1).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    paintable = np.zeros(count - 1, dtype=bool)
    paintable[piece[white & (distance > brush_px / 2)]] = True
    claim = _claims(piece, paintable, ~bold)
    wall = _claim_walls(claim, ink)
    corners = wall & white
    passable = (claim >= 0) & ~wall & ~bold
    through = _nearest_seed_within(np.where(white & passable, labels, -1), passable)
    seen = ink & (through >= 0)
    region_labels = np.where(seen, through, labels)
    region_labels[corners] = num_colors
    return region_labels.astype(np.int32), corners


def _claims(piece: np.ndarray, paintable: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Which paintable piece claims each pixel of ``passable``: the one a path through ``passable`` reaches it from first.

    A piece no brush fits in goes wholly to the claim most of its pixels are
    in. Each 8-connected run of ``passable`` that no paintable piece reaches is
    a claim of its own, numbered after the pieces. -1 off ``passable``.
    """
    white = piece >= 0
    seeds = np.where(white & paintable[np.where(white, piece, 0)], piece, -1)
    claim = _nearest_seed_within(seeds, passable)
    kernels.settle_gaps(np.ascontiguousarray(piece, dtype=np.int32).reshape(-1), claim.reshape(-1), paintable)
    unclaimed = passable & (claim < 0)
    if unclaimed.any():
        _count, runs = cv2.connectedComponents(unclaimed.view(np.uint8), connectivity=8)
        claim = np.where(unclaimed, paintable.size + runs.astype(np.int32) - 1, claim).astype(np.int32)
    return claim


def _claim_walls(claim: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """The pixels that keep two claims apart: of every two 8-adjacent pixels of different claims, one.

    The ink one if only one of them is ink, else the one of the higher claim.
    Afterwards no two pixels of different claims touch, diagonals included.
    """
    h, w = claim.shape
    wall = np.empty((h, w), dtype=bool)
    kernels.claim_walls(
        np.ascontiguousarray(claim, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(ink, dtype=bool).reshape(-1),
        np.zeros(0, dtype=np.int32),  # one group: every claim against every other
        h,
        w,
        wall.reshape(-1),
    )
    return wall


def _nearest_seed_within(seeds: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """``kernels.nearest_seed_within`` on HxW arrays."""
    h, w = passable.shape
    flat_seeds = np.ascontiguousarray(seeds, dtype=np.int32).reshape(-1)
    flat_passable = np.ascontiguousarray(passable, dtype=bool).reshape(-1)
    return kernels.nearest_seed_within(flat_seeds, flat_passable, h, w).reshape(h, w)


def leave_pockets(region_id_map: np.ndarray, region_color: np.ndarray, printed: np.ndarray, min_width_px: float) -> np.ndarray:
    """``region_id_map`` with the pockets no brush reaches that ink walls in left as bare paper (-1).

    A brush ``min_width_px`` wide reaches the part of a region it can sweep
    without leaving it. On line art, what it can't reach lies mostly in
    pockets against ink no paint goes over (``printed``, in no region): the
    tips a fill makes against a bold outline, the channels between dark
    blobs. Such a pocket -- an 8-connected run of unreached pixels, at least
    half of whose contacts with pixels outside it and its region are that ink
    -- is left unpainted. The unreached corners two regions make against each
    other, or against the page's edge, stay.

    Taking pockets out can cut a region in two, or leave a bit of one too
    small to keep: ``split_areas`` sorts that out.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    count = int(region_color.size)
    inside = ids >= 0
    if count == 0 or not inside.any():
        return ids
    radius = min_width_px / 2
    fits = _brush_fits(ids, count, radius)
    if not fits.any():
        return ids
    to_brush = cv2.distanceTransform((~fits).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    h, w = ids.shape
    unreached = np.empty((h, w), dtype=bool)
    kernels.unreached_by_brush(ids.reshape(-1), to_brush.reshape(-1), radius, unreached.reshape(-1))
    if not unreached.any():
        return ids
    walls = printed & ~inside
    n, pocket = cv2.connectedComponents(unreached.view(np.uint8), connectivity=8)
    pocket = np.ascontiguousarray(pocket, dtype=np.int32)
    # Off the page is a contact, not ink.
    contacts, on_walls = kernels.pocket_contacts(pocket.reshape(-1), ids.reshape(-1), walls.reshape(-1), n, h, w)
    left = (contacts > 0) & (2 * on_walls >= contacts)
    left[0] = False  # 0 is every reached pixel
    return np.where(left[pocket], -1, ids).astype(np.int32)


def split_areas(
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    printed: np.ndarray,
    min_width_px: float,
    min_area_px: int,
    num_colors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Line art's regions, one per area a painter sees, each area a brush fits in with a number of its own.

    A region's paint goes over the thin ink in it (``printed``), so a region
    can hold several white areas the ink or its neighbors keep apart -- the
    sky on either side of a figure, joined through hatching, or two parts a
    one-pixel diagonal line divides. The page numbers a region once, so each
    white area a brush ``min_width_px`` wide fits in (a 4-connected run of the
    region's unprinted pixels) becomes a region of its own, in the same
    color, with the rest of the region -- the ink it paints over, the gaps no
    brush fits in -- going to the area that reaches it first through the
    region, a gap wholly to the one that reaches most of it. Where two meet,
    the ink between them stays out of both, as a seam; where two white areas
    touch at a corner, one of the two pixels is taken for ink, and returned
    so that it is printed.

    The regions are then rebuilt from their colors (``num_colors`` for none),
    so parts a pocket cut off (see ``leave_pockets``) become regions of their
    own, and those smaller than ``min_area_px`` merge as ``build_regions``
    merges them -- but only into a region their white shares an edge with, so
    that they join its area rather than become another area of it (see
    ``_rebuilt_by_white``). A merge can still make a region two areas, when
    the white it shares an edge with is a gap of that region rather than its
    area: those are split again, and what is left merged again, until nothing
    splits (at most ``_SETTLE_ROUNDS`` times). A small region left over with
    no room for a brush in its white is no area of its own, and merges as
    ``build_regions`` merges, through its ink too (see ``_merged_bits``). A
    small region whose white does hold a brush but shares no edge with
    another region's is an area the ink encloses on its own, and keeps its
    number as those do (see ``settle_enclosed``): where its ink touches
    another region, it gets a seam, as a thin line between two regions does
    (see ``paint_over_thin_ink``), and where its white touches another's at a
    corner, one of the two pixels is taken for ink. A region all of ink, or
    alone with no room for a brush, is dropped (see ``_without_unpaintable``).

    Returns the region map, the region colors and an HxW bool mask of the
    pixels taken for ink.
    """
    radius = min_width_px / 2
    ids, colors, corners = _split_all(region_id_map, region_color, printed, radius)
    inked = printed | corners
    for _round in range(_SETTLE_ROUNDS):
        ids, colors, grown = _rebuilt_by_white(ids, colors, num_colors, min_area_px, inked)
        # A merge through a gap of the region it joins -- white it shares an edge with, but that the region's own area
        # doesn't -- makes that region two areas again: split, and merge what is left over, until nothing splits.
        count = colors.size
        ids, colors, more = _split_all(ids, colors, inked, radius, grown)
        corners |= more
        inked |= more
        if colors.size == count:
            break
    ids, colors, grown = _merged_bits(ids, colors, inked, radius, min_area_px, num_colors)
    # That merge can join two areas too: split once more, and settle without merging.
    ids, colors, more = _split_all(ids, colors, inked, radius, grown)
    corners |= more
    inked |= more
    h, w = ids.shape
    rebuilt = np.empty((h, w), dtype=np.int32)
    colors, areas = kernels.label_components(_painted(ids, colors, num_colors).reshape(-1), h, w, num_colors, rebuilt.reshape(-1))
    small = (areas > 0) & (areas < min_area_px)
    if small.any():
        rebuilt, more = _seams_round(rebuilt, inked, small)
        corners |= more
        inked |= more
    return _without_unpaintable(rebuilt, colors, inked, min_width_px, min_area_px), colors, corners


def _merged_bits(
    ids: np.ndarray, colors: np.ndarray, printed: np.ndarray, radius: float, min_area_px: int, num_colors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The regions rebuilt, with each one smaller than ``min_area_px`` that has no area of its own merged away.

    A region whose white holds no brush of ``radius`` -- a bit of hatching, or
    of a region's edge cut off -- is not an area the painter sees on its own,
    so it merges as ``build_regions`` merges, into the neighbor owning the
    most of its 8-connected ring, through the ink its paint goes over too. A
    small region whose white holds a brush is an area, and stays. Returns the
    region map, the region colors, and which regions grew -- by a merge, or by
    two regions of one color that touch becoming one when the colors are
    labeled again.
    """
    h, w = ids.shape
    rebuilt = np.empty((h, w), dtype=np.int32)
    colors, areas = kernels.label_components(_painted(ids, colors, num_colors).reshape(-1), h, w, num_colors, rebuilt.reshape(-1))
    joined = _joined(ids, rebuilt, int(colors.size))  # two regions of one color that touch are one again
    small = (areas > 0) & (areas < min_area_px)
    if not small.any():
        return rebuilt, colors, joined
    roomy = _room_for_a_brush(rebuilt, ~printed, small, radius)
    # An area too small for the difficulty keeps its number all the same: it counts as large enough not to merge.
    areas = np.where(small & roomy, np.maximum(areas, min_area_px), areas).astype(areas.dtype)
    before = areas.copy()
    flat = rebuilt.reshape(-1)
    kernels.merge_small_regions(flat, h, w, areas, int(min_area_px), True)
    kernels.merge_same_color_neighbors(flat, h, w, colors, areas, True)
    # A merge keeps the id it merges into, and one merged away has no area left, whose target grew.
    return rebuilt, colors, (areas > before) | (joined & (areas > 0))


def _split_all(
    region_id_map: np.ndarray, region_color: np.ndarray, printed: np.ndarray, radius: float, only: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every region with more than one white area a brush of ``radius`` fits in, split into one region per area.

    ``only`` (a bool per region), when given, says which regions can have
    gained a second area -- those a merge grew -- and no other is looked at.
    Returns the region map, the region colors (new regions appended) and the
    pixels taken for ink where two areas touched at a corner.
    """
    ids = np.array(region_id_map, dtype=np.int32)
    colors = np.asarray(region_color, dtype=np.int32)
    corners = np.zeros(ids.shape, dtype=bool)
    asked = np.ones(colors.size, dtype=bool)
    if only is not None:
        asked[:] = False
        asked[: len(only)] = only
    if not colors.size or not asked.any() or not (ids >= 0).any():
        return ids, colors, corners
    h, w = ids.shape
    flat_ids = ids.reshape(-1)
    flat_printed = np.ascontiguousarray(printed, dtype=bool).reshape(-1)
    # The white areas of the regions asked about: the 4-connected runs of their unprinted pixels, in raster order.
    white = np.empty((h, w), dtype=np.int32)
    kernels.white_of(flat_ids, flat_printed, asked, white.reshape(-1))
    piece = np.empty((h, w), dtype=np.int32)
    piece_region = kernels.white_pieces(white.reshape(-1), h, w, piece.reshape(-1))
    if only is None:
        # Where a brush fits in a region's white: farther than its radius from anything else, which one distance
        # transform per class of regions measures for them all (see ``_brush_fits``). It is the distance measuring each
        # region in its own box gives (``_paintable_pieces``): the nearest pixel outside a region's white lies beside it.
        paintable = np.zeros(piece_region.size, dtype=bool)
        paintable[piece[_brush_fits(white, int(colors.size), radius)]] = True
    else:
        asked &= np.bincount(piece_region, minlength=colors.size) >= 2  # one white area can't split
        paintable = _paintable_pieces(ids, printed, piece, piece_region.size, asked, radius)
    areas_of = np.bincount(piece_region[paintable], minlength=colors.size)
    splitting = areas_of >= 2
    if not splitting.any():
        return ids, colors, corners
    groups = np.empty(h * w, dtype=np.int32)
    seeds = np.empty(h * w, dtype=np.int32)
    kernels.split_seeds(flat_ids, piece.reshape(-1), paintable, splitting, groups, seeds)
    # Each area claims what of its region it reaches first along a path through the region, a gap no brush fits in
    # going wholly to the area that reaches most of it.
    claim = kernels.nearest_seed_in_groups(seeds, groups, h, w)
    kernels.settle_gaps(piece.reshape(-1), claim, paintable)
    # Numbered 0, 1, ... in each region in the order of its areas. A part of the region no area reaches, cut off by a
    # pocket, is claimed by nobody (-1): it keeps the region's id, and the rebuild makes it a region of its own.
    ranked = np.flatnonzero(paintable & splitting[piece_region])
    order = np.argsort(piece_region[ranked], kind="stable")
    first = np.searchsorted(piece_region[ranked][order], piece_region[ranked][order])
    rank = np.full(piece_region.size, -1, dtype=np.int32)
    rank[ranked[order]] = np.arange(ranked.size) - first
    # The first area keeps the region's id; the others get new ones, region by region, after every id there is.
    extra = np.where(splitting, areas_of - 1, 0)
    base = (colors.size + np.cumsum(extra) - extra).astype(np.int64)
    kernels.apply_split(flat_ids, claim, rank, flat_printed, groups, base, h, w, corners.reshape(-1))
    return ids, np.concatenate([colors, np.repeat(colors, extra)]).astype(np.int32), corners


def _paintable_pieces(
    ids: np.ndarray, printed: np.ndarray, piece: np.ndarray, num_pieces: int, asked: np.ndarray, radius: float
) -> np.ndarray:
    """For each piece of white (``piece``, see ``kernels.white_pieces``), whether a brush of ``radius`` fits in it.

    Only the pieces of the regions ``asked`` marks are measured, each region in its own box, where everything but its
    unprinted pixels is in the brush's way; the others are False.
    """
    paintable = np.zeros(num_pieces, dtype=bool)
    if not asked.any():
        return paintable
    h, w = ids.shape
    bounds, areas = kernels.region_bounds(ids.reshape(-1), h, w, int(asked.size))
    for rid in np.flatnonzero(asked & (areas > 0)).tolist():
        x0, y0, x1, y1 = bounds[rid].tolist()
        box = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        white = (ids[box] == rid) & ~printed[box]
        distance = cv2.distanceTransform(np.pad(white, 1).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        paintable[piece[box][white & (distance > radius)]] = True
    return paintable


def _painted(ids: np.ndarray, colors: np.ndarray, num_colors: int) -> np.ndarray:
    """Each pixel's color, ``num_colors`` where it is in no region: labels to rebuild regions from."""
    labels = np.empty(ids.shape, dtype=np.int32)
    kernels.painted_labels(
        np.ascontiguousarray(ids, dtype=np.int32).reshape(-1), np.asarray(colors, dtype=np.int32), num_colors, labels.reshape(-1)
    )
    return labels


def _rebuilt_by_white(
    region_id_map: np.ndarray, region_color: np.ndarray, num_colors: int, min_area_px: int, printed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_regions_from_labels`` on the regions' colors, with the merges seeing only their unprinted pixels, edge to edge.

    A region smaller than ``min_area_px`` -- counting the ink its paint goes
    over -- merges into the neighbor whose white owns the most of its white's
    4-connected ring, and two regions of one color whose white shares an edge
    become one. The ink a region's paint goes over is not where two areas meet
    on the page, and two white areas touching at a corner are two areas, so
    neither makes two regions neighbors. A small region whose white shares no
    edge with another region's is left as it is. Returns the region map, the
    region colors, and which regions grew -- by a merge, or by two regions of
    one color that touch becoming one when the colors are labeled again.
    """
    h, w = region_id_map.shape
    ids = np.empty((h, w), dtype=np.int32)
    labels = _painted(region_id_map, region_color, num_colors)
    region_color, areas = kernels.label_components(labels.reshape(-1), h, w, num_colors, ids.reshape(-1))
    if not region_color.size:
        return ids, region_color, np.zeros(0, dtype=bool)
    joined = _joined(region_id_map, ids, int(region_color.size))
    white = np.where(printed, -1, ids).astype(np.int32)
    merged = white.reshape(-1).copy()
    kernels.merge_small_regions(merged, h, w, areas, int(min_area_px), False)
    kernels.merge_same_color_neighbors(merged, h, w, region_color, areas, False)
    # Every region goes where its white went; one with no white stays itself.
    followed = np.empty((h, w), dtype=np.int32)
    target = kernels.follow_white(ids.reshape(-1), white.reshape(-1), merged, int(region_color.size), followed.reshape(-1))
    grown = np.zeros(region_color.size, dtype=bool)
    grown[target[(target != np.arange(region_color.size)) | joined]] = True
    return followed, region_color, grown


def _joined(old: np.ndarray, new: np.ndarray, count: int) -> np.ndarray:
    """For each of ``count`` regions of ``new``, whether it holds pixels of two regions of ``old`` or more."""
    return kernels.regions_joined(
        np.ascontiguousarray(old, dtype=np.int32).reshape(-1), np.ascontiguousarray(new, dtype=np.int32).reshape(-1), count
    )


def _seams_round(ids: np.ndarray, printed: np.ndarray, which: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ids`` with the regions ``which`` marks kept from touching any other, and the pixels taken for ink to do it.

    Of two 8-adjacent pixels of different regions, one of them a region
    ``which`` marks, one leaves its region: the ``printed`` one if only one
    is, else the one with the higher id. Where neither is printed -- two
    white areas touching at a corner -- the one that leaves is taken for ink,
    and returned so that it is printed. Two white pixels sharing an edge are
    a boundary the page draws, and are left as they are.
    """
    # Two white pixels across a corner are a contact of their own only where neither pixel beside them joins the two
    # regions edge to edge: along a drawn boundary every pixel has the other region at its corners.
    h, w = ids.shape
    kept = np.empty((h, w), dtype=np.int32)
    taken = np.empty((h, w), dtype=bool)
    kernels.seams_round(
        np.ascontiguousarray(ids, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(printed, dtype=bool).reshape(-1),
        np.asarray(which, dtype=bool),
        h,
        w,
        kept.reshape(-1),
        taken.reshape(-1),
    )
    return kept, taken


def _without_unpaintable(
    ids: np.ndarray, region_color: np.ndarray, printed: np.ndarray, min_width_px: float, min_area_px: int
) -> np.ndarray:
    """``ids`` without the regions nothing can be painted in: all printed ink, or alone with no room for a brush.

    A bit of ink a seam cut off from its area is printed, not painted, like
    any ink in no region. A bit of a region no brush fits in, which no other
    region touches for it to merge into, is left as bare paper, as the shapes
    the ink encloses alone are (see ``settle_enclosed``). So is one smaller
    than ``min_area_px`` whose white holds no brush, even if its ink does: a
    lobe a pocket cut off, mostly ink, with no room for its number and
    nothing to merge into, is part of the pocket.
    """
    count = int(region_color.size)
    inside = ids >= 0
    if count == 0 or not inside.any():
        return ids
    region_of = np.where(inside, ids, 0)
    h, w = ids.shape
    areas, unprinted, touched = kernels.region_census(
        np.ascontiguousarray(ids, dtype=np.int32).reshape(-1), np.ascontiguousarray(printed, dtype=bool).reshape(-1), count, h, w
    )
    alone = ~touched & (areas > 0)
    gone = unprinted == 0
    if alone.any():
        fits = _room_for_a_brush(ids, np.ones(ids.shape, dtype=bool), alone, min_width_px / 2)
        white_fits = _room_for_a_brush(ids, ~printed, alone & (areas < min_area_px), min_width_px / 2)
        gone |= alone & (~fits | ((areas < min_area_px) & ~white_fits))
    if not gone.any():
        return ids
    return np.where(inside & gone[region_of], -1, ids).astype(np.int32)


def _room_for_a_brush(ids: np.ndarray, keep: np.ndarray, which: np.ndarray, radius: float) -> np.ndarray:
    """For each region ``which`` marks, whether a brush of ``radius`` fits in its pixels ``keep`` (HxW bool) keeps.

    What ``_brush_fits`` finds for the whole page, for a few regions only: each is measured in its own box, where
    everything but its kept pixels -- other regions, ink, paper, off the page -- is in the brush's way.
    """
    out = np.zeros(which.size, dtype=bool)
    wanted = np.flatnonzero(which)
    if not wanted.size:
        return out
    h, w = ids.shape
    bounds, areas = kernels.region_bounds(np.ascontiguousarray(ids).reshape(-1), h, w, int(which.size))
    for rid in wanted.tolist():
        if areas[rid] == 0:
            continue
        x0, y0, x1, y1 = bounds[rid].tolist()
        kept = (ids[y0 : y1 + 1, x0 : x1 + 1] == rid) & keep[y0 : y1 + 1, x0 : x1 + 1]
        if kept.any():
            distance = cv2.distanceTransform(np.pad(kept, 1).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            out[rid] = bool(distance.max() > radius)
    return out


def merge_cramped(
    region_id_map: np.ndarray, region_color: np.ndarray, printed: np.ndarray, cramped, min_width_px: float
) -> np.ndarray | None:
    """``region_id_map`` with each region in ``cramped`` merged into the neighbor its white shares the most edge with.

    A region whose number found no room anywhere -- not inside it, not beside
    it with a leader, not on its own hatching cleared (see
    ``labels.place_labels``) -- is smaller, in the room it has, than the page
    can number, as a region below the difficulty's smallest area is: it joins
    the area beside it. A neighbor whose unprinted pixels share an edge with
    its own, so that the two become one white area rather than one region of
    two. Failing that, if its white has no room for a brush ``min_width_px``
    wide -- it is hatching, painted across its strokes, and no area of its
    own -- the neighbor it touches most, through its ink too. Otherwise it
    stays. Ties go to the lowest id. A region that comes to share an edge of
    white with one of its own color (``region_color``) becomes one with it.
    Returns None when no region moved.
    """
    ids = np.asarray(region_id_map)
    white = np.where(printed, -1, ids)
    asked = np.zeros(int(ids.max()) + 1, dtype=bool)
    asked[[r for r in cramped if 0 <= r < asked.size]] = True
    regions, neighbors = _most_touched(white, asked, diagonals=False)
    through_ink = asked.copy()
    through_ink[regions] = False
    if through_ink.any():
        roomy = np.bincount(white[_brush_fits(white, asked.size, min_width_px / 2)], minlength=asked.size) > 0
        through_ink &= ~roomy
        if through_ink.any():
            more_regions, more_neighbors = _most_touched(ids, through_ink, diagonals=True)
            regions = np.concatenate([regions, more_regions])
            neighbors = np.concatenate([neighbors, more_neighbors])
    if not regions.size:
        return None
    # A region may join one that joins another in turn; of two that would join each other, the first asked goes.
    target = np.arange(asked.size, dtype=np.int64)

    def root(r: int) -> int:
        while target[r] != r:
            r = int(target[r])
        return r

    for region, neighbor in zip(regions.tolist(), neighbors.tolist()):
        joined = root(neighbor)
        if joined != region:
            target[region] = joined
    for region in regions.tolist():
        target[region] = root(region)
    if np.array_equal(target, np.arange(asked.size)):
        return None
    inside = ids >= 0
    merged = np.where(inside, target[np.where(inside, ids, 0)], -1).astype(np.int32)
    h, w = merged.shape
    white = np.where(printed, -1, merged).astype(np.int32)
    joined = white.reshape(-1).copy()
    colors = np.asarray(region_color, dtype=np.int32)
    areas = np.bincount(merged[inside], minlength=colors.size).astype(np.int64)
    kernels.merge_same_color_neighbors(joined, h, w, colors, areas, False)
    same = np.arange(colors.size, dtype=np.int32)  # every region goes where its white went
    has_white = white >= 0
    same[white[has_white]] = joined.reshape(h, w)[has_white]
    return np.where(inside, same[np.where(inside, merged, 0)], -1).astype(np.int32)


def _most_touched(ids: np.ndarray, asked: np.ndarray, diagonals: bool) -> tuple[np.ndarray, np.ndarray]:
    """For each region ``asked`` about that touches another in ``ids``, the one it touches along the most pixel pairs.

    Pixels touch across an edge, and across a corner too with ``diagonals``.
    Ties go to the lowest id. Returns the regions, in id order, and their
    neighbors.
    """
    shifts = [(ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])]
    if diagonals:
        shifts += [(ids[:-1, :-1], ids[1:, 1:]), (ids[:-1, 1:], ids[1:, :-1])]
    first = np.concatenate([a.ravel() for a, _b in shifts]).astype(np.int64)
    second = np.concatenate([b.ravel() for _a, b in shifts]).astype(np.int64)
    meet = (first >= 0) & (second >= 0) & (first != second)
    mine = np.concatenate([first[meet], second[meet]])
    theirs = np.concatenate([second[meet], first[meet]])
    keep = asked[mine]
    if not keep.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    stride = np.int64(asked.size)
    pairs, counts = np.unique(mine[keep] * stride + theirs[keep], return_counts=True)
    regions, neighbors = np.divmod(pairs, stride)
    order = np.lexsort((neighbors, -counts, regions))  # per region, the most pairs first, then the lowest id
    regions, neighbors = regions[order], neighbors[order]
    first_of_each = np.r_[True, regions[1:] != regions[:-1]]
    return regions[first_of_each], neighbors[first_of_each]


def paint_over_thin_ink(
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    printed: np.ndarray,
    max_width_px: float,
    image_bgr: np.ndarray,
    palette_bgr: np.ndarray,
    own: np.ndarray | None = None,
) -> np.ndarray:
    """``region_id_map`` with the thin parts of line art's ink given to the regions whose paint goes over them.

    ``printed`` (HxW bool) is the ink the page prints, in no region (-1). A
    brush cannot keep off a stroke narrower than it: the gaps between
    hatching strokes are too narrow to paint around them. So the ink in parts
    narrower than ``max_width_px`` is painted over, and each of its pixels is
    given to the region nearest it:

    - a stroke with one region all round it -- hatching, shading, a fold --
      becomes part of that region;
    - a line between two regions is shared down its middle, as a page's own
      lines are on any other picture. A seam one pixel wide stays out of both,
      so no two regions touch across the ink: they are still two areas, each
      with its own number.

    Ink as wide as ``max_width_px`` or wider -- an outline, a black shape --
    stays out of every region: it is never painted, but for its sharp corners,
    where a disk that wide does not reach either.

    Bare paper the thin ink encloses inside one region -- a gap between
    crossing strokes -- becomes part of it too, if its own pixels in
    ``image_bgr`` are that region's color, within ``color.MIN_PALETTE_DE00``
    of it in ``palette_bgr``: the fill showing between the strokes. A white
    highlight in a colored shape stays paper. As in ``join_ink``, only the
    pixels in ``own`` count where it has any.

    The ink is printed either way; only where paint may go changes. Returns
    the new region map.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    count = int(region_color.size)
    outside = ids < 0
    thin = _thin_part(printed & outside, max_width_px)
    if count == 0 or not thin.any() or outside.all():
        return ids
    distance, nearest = cv2.distanceTransformWithLabels(
        outside.view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    # Thin ink lies within half its width of something that is not ink; a stroke beside bare paper, farther. Each
    # joined pixel touching another region leaves again, as a seam.
    h, w = ids.shape
    painted = np.empty((h, w), dtype=np.int32)
    joined = np.empty((h, w), dtype=bool)
    kernels.thin_ink_to_nearest(
        ids.reshape(-1),
        np.ascontiguousarray(thin).reshape(-1),
        np.ascontiguousarray(nearest, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(distance).reshape(-1),
        max_width_px,
        h,
        w,
        painted.reshape(-1),
        joined.reshape(-1),
    )
    painted = _connected_to_own(painted, ~outside, count)
    return _join_enclosed_paper(painted, region_color, printed, image_bgr, palette_bgr, own)


def detail_ink(region_id_map: np.ndarray, printed: np.ndarray, reach_px: float, apart_px: float) -> np.ndarray:
    """HxW bool: the printed ink inside a region that is neither part of a line nor beside one.

    That is ink a region's paint goes over whole (see ``paint_over_thin_ink``):
    hatching, shading. A line between two regions is shared down its middle
    with a seam in no region, so with ``reach_px`` half the thin ink's width,
    leaving out the ink with anything in no region within ``reach_px`` leaves
    out the halves of those lines, and the ink round bare paper or beside
    bold ink. Hatching runs on across a change of color inside a hatched
    patch, where the two regions meet under the strokes with no line between
    them: the ink within ``apart_px`` of another region is left out too, so
    that what stays printed there is the line. It is what a number may clear
    behind it (see ``labels.place_labels``).
    """
    ids = np.asarray(region_id_map)
    candidates = printed & (ids >= 0)
    if not candidates.any():
        return candidates
    near_nothing = cv2.dilate((ids < 0).view(np.uint8), _disk(reach_px)).view(bool)  # off the page is not in no region
    values = (ids + 1).astype(np.float32)
    disk = _disk(apart_px)
    alone = cv2.dilate(values, disk) == cv2.erode(values, disk)  # off the page is nothing, not another region
    return candidates & ~near_nothing & alone


def _thin_part(mask: np.ndarray, width_px: float) -> np.ndarray:
    """The pixels of ``mask`` in parts of it narrower than ``width_px``: those its opening by a disk that wide leaves out.

    Off the page is outside the mask, so a band along the page's edge is judged by its own width.
    """
    if not mask.any():
        return mask.copy()
    disk = _disk(width_px / 2)
    pad = disk.shape[0]
    padded = cv2.copyMakeBorder(mask.view(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, disk)[pad:-pad, pad:-pad].view(bool)
    return mask & ~opened


def _disk(radius: float) -> np.ndarray:
    """The pixels within ``radius`` of a center pixel, as a structuring element."""
    n = int(np.floor(radius))
    y, x = np.mgrid[-n : n + 1, -n : n + 1]
    return ((x * x + y * y) <= radius * radius).astype(np.uint8)


def _connected_to_own(ids: np.ndarray, own_pixels: np.ndarray, count: int) -> np.ndarray:
    """``ids`` without the bits of a region (8-connected) that hold none of its ``own_pixels``: cut off by a seam."""
    h, w = ids.shape
    flat = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
    pieces = np.empty(h * w, dtype=np.int32)
    _region_of_piece, _areas = kernels.label_components(flat, h, w, count, pieces)
    kept = np.empty((h, w), dtype=np.int32)
    kernels.keep_anchored(flat, pieces, np.ascontiguousarray(own_pixels, dtype=bool).reshape(-1), kept.reshape(-1))
    return kept


def _join_enclosed_paper(ids, region_color, printed, image_bgr, palette_bgr, own) -> np.ndarray:
    """``ids`` with each piece of bare paper whose ring is one region, in that region's color, made part of it.

    A piece's ring is the pixels 8-adjacent to it, off the page left out. Its
    color is its own pixels' mean (in ``own`` where it has any), within
    ``color.MIN_PALETTE_DE00`` of the region's color in ``palette_bgr``.
    """
    paper = (ids < 0) & ~printed
    if not paper.any():
        return ids
    h, w = ids.shape
    count, piece = cv2.connectedComponents(paper.view(np.uint8), connectivity=8)
    piece = np.ascontiguousarray(piece, dtype=np.int32) - 1  # -1 off the paper
    # A paper pixel 8-adjacent to a piece's is in the piece; what else it touches is a region, or ink (-1).
    lowest, highest = kernels.paper_rings(
        piece.reshape(-1), np.ascontiguousarray(ids, dtype=np.int32).reshape(-1), paper.reshape(-1), count - 1, h, w
    )
    single = (lowest == highest) & (highest >= 0)
    if not single.any():
        return ids
    means = _mean_colors(piece, image_bgr, single, own)
    target = np.where(single, highest, 0)
    colors = palette_bgr[region_color[target]]
    de00 = ciede2000(bgr_to_lab(np.rint(means).clip(0, 255).astype(np.uint8)), bgr_to_lab(colors))
    joining = single & (de00 < MIN_PALETTE_DE00)
    if not joining.any():
        return ids
    on_paper = piece >= 0
    moves = on_paper & joining[np.where(on_paper, piece, 0)]
    return np.where(moves, target[np.where(on_paper, piece, 0)], ids).astype(np.int32)


def _mean_colors(ids: np.ndarray, image_bgr: np.ndarray, asked: np.ndarray, own: np.ndarray | None) -> np.ndarray:
    """The mean BGR color in ``image_bgr`` of each region id ``asked`` about: of its pixels in ``own`` where it has any.

    One row per region id; the rows of the others are 0.
    """
    h, w = ids.shape
    own = np.zeros(h * w, dtype=bool) if own is None else np.ascontiguousarray(own).reshape(-1)
    sums, counts, own_sums, own_counts = kernels.region_color_sums(
        np.ascontiguousarray(ids).reshape(-1), np.ascontiguousarray(image_bgr).reshape(-1, 3), asked, own
    )
    has_own = own_counts > 0
    sums[has_own], counts[has_own] = own_sums[has_own], own_counts[has_own]
    return sums / np.maximum(counts, 1)[:, None]


def _de00_to(colors_bgr: np.ndarray, target_bgr) -> np.ndarray:
    """CIEDE2000 from each of ``colors_bgr`` (Kx3, rounded to 8-bit sRGB) to ``target_bgr``."""
    colors = np.rint(np.asarray(colors_bgr, dtype=np.float64)).clip(0, 255).astype(np.uint8).reshape(-1, 3)
    target = bgr_to_lab(np.asarray(target_bgr, dtype=np.uint8).reshape(1, 3))
    return ciede2000(bgr_to_lab(colors), target)


def _has_neighbor(ids: np.ndarray, count: int) -> np.ndarray:
    """For every region id below ``count``, whether another region touches it, diagonals included."""
    h, w = ids.shape
    flat = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
    return kernels.region_census(flat, np.zeros(flat.size, dtype=bool), count, h, w)[2]


def _brush_fits(region_id_map: np.ndarray, num_regions: int, radius: float) -> np.ndarray:
    """Pixels where a round brush of ``radius`` sits inside one region.

    A brush fits where the nearest pixel of another region is farther away
    than its radius, so one distance transform per class of
    ``kernels.edge_adjacency_classes`` measures that for every region in the
    class at once. Pixels off the page count as another region: the page edge
    is a line like any other.
    """
    h, w = region_id_map.shape
    class_map = np.empty((h, w), dtype=np.int8)
    num_classes = kernels.edge_adjacency_classes(
        region_id_map.reshape(-1), h, w, num_regions, class_map.reshape(-1)
    )

    in_class = np.zeros((h + 2, w + 2), dtype=np.uint8)  # the border stays 0: off the page is another region
    interior = in_class[1:-1, 1:-1]
    distance = np.empty((h + 2, w + 2), dtype=np.float32)
    far_enough = np.empty((h, w), dtype=bool)
    fits = np.zeros((h, w), dtype=bool)
    for region_class in range(num_classes):
        np.equal(class_map, region_class, out=interior, casting="unsafe")
        cv2.distanceTransform(in_class, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=distance)
        np.greater(distance[1:-1, 1:-1], radius, out=far_enough)
        far_enough &= interior.view(bool)  # only pixels of this class: the rest measure another distance
        fits |= far_enough
    return fits


def extract_regions(
    region_id_map: np.ndarray, region_color: np.ndarray, min_contour_area: float = 1.0, printed: np.ndarray | None = None
) -> list[Region]:
    """Extract one outer contour + label point per surviving region id.

    Each region is processed inside its own one-pixel-padded bounding box
    rather than across the whole image (same result: nothing outside the box
    can affect its contour or distance transform), spread over worker threads.
    Regions come back in region-id order.

    ``printed`` (HxW bool), the ink the page prints -- line art's own, or the
    detail marks in a face (see ``marks``) -- is where no number can go,
    even where it lies in a region (see ``paint_over_thin_ink``): a
    region's label point is the one of its unprinted pixels farthest from
    everything else, as if the ink were in no region.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    h, w = ids.shape
    num_ids = int(ids.max()) + 1 if ids.size else 0
    if num_ids <= 0:
        return []
    bounds_array, areas_array = kernels.region_bounds(ids.reshape(-1), h, w, num_ids)
    # Plain lists: much cheaper than NumPy scalar indexing in the per-region loop.
    bounds = bounds_array.tolist()
    areas = areas_array.tolist()
    colors = np.asarray(region_color).tolist()
    present = np.flatnonzero(areas_array > 0).tolist()
    box_px = [(bounds[r][2] - bounds[r][0] + 1) * (bounds[r][3] - bounds[r][1] + 1) for r in present]

    def extract(rid: int) -> Region | None:
        x0, y0, x1, y1 = bounds[rid]
        mask = np.zeros((y1 - y0 + 3, x1 - x0 + 3), dtype=np.uint8)
        np.equal(ids[y0 : y1 + 1, x0 : x1 + 1], rid, out=mask[1:-1, 1:-1], casting="unsafe")
        origin = (x0 - 1, y0 - 1)  # image coordinates of mask[0, 0]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=origin)
        if not contours:
            return None
        contour = contours[0] if len(contours) == 1 else max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_contour_area:
            return None
        contour = cv2.approxPolyDP(contour, epsilon=1.2, closed=True)

        # The zero padding also makes the distance transform treat the image
        # edge as a boundary, so a region touching the edge can't report a
        # "safe" label point right at the canvas corner.
        room = mask
        if printed is not None:
            unprinted = mask[1:-1, 1:-1] & ~printed[y0 : y1 + 1, x0 : x1 + 1]
            if unprinted.any():
                room = np.zeros_like(mask)
                room[1:-1, 1:-1] = unprinted
        dist = cv2.distanceTransform(room, cv2.DIST_L2, 5)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)

        return Region(
            region_id=rid,
            color_index=colors[rid],
            contour=contour,
            interior_point=(origin[0] + max_loc[0], origin[1] + max_loc[1]),
            interior_radius=float(max_val),
            area=areas[rid],
        )

    extracted = parallel.map_balanced(extract, present, box_px, min_pooled_cost=_MIN_POOLED_BOX_PX)
    return [region for region in extracted if region is not None]

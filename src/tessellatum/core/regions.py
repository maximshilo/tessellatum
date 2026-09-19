"""Connected-component region extraction, region merging, and contour extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.color import MIN_PALETTE_DE00, bgr_to_lab, ciede2000

# Regions with a smaller bounding box than this are extracted on the calling
# thread: for them, handing off to a worker (and contending for the GIL) costs
# more than the OpenCV work itself.
_MIN_POOLED_BOX_PX = 32_000


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
    labels: np.ndarray, num_colors: int, min_area_px: int, min_width_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``labels`` into connected regions and merge away what can't be painted.

    Regions are 8-connected runs of one color, numbered by color and then by
    raster position. Regions below ``min_area_px`` are merged, smallest first,
    into the neighbor sharing the most boundary (see
    ``kernels.merge_small_regions``). That merge ignores color, so it can
    leave two neighbors sharing one: those are then unioned into a single
    region (``kernels.merge_same_color_neighbors``), so no boundary on the
    page separates two areas the painter fills with the same color.

    With ``min_width_px`` set, every part of a region narrower than that --
    which a brush that wide cannot paint without crossing a line -- is then
    given away to the region whose paint reaches it first, and the regions
    are rebuilt from the result (see ``absorb_thin_parts``). A region thinner
    than the brush everywhere disappears into its neighbors.

    Returns:
        (region_id_map, region_color): region_id_map is HxW int32 (each pixel's
        region id), region_color is region_count-length int32 array mapping a
        region id to its color index in the palette.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    region_id_map, region_color = _regions_from_labels(labels, num_colors, min_area_px)
    if min_width_px > 0 and region_color.size:
        widened = absorb_thin_parts(region_id_map, region_color, labels, min_width_px)
        if widened is not None:
            region_id_map, region_color = _regions_from_labels(widened, num_colors, min_area_px)
    return region_id_map, region_color


def _regions_from_labels(labels: np.ndarray, num_colors: int, min_area_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``labels``, with the two merges that always apply."""
    h, w = labels.shape
    region_id_map = np.empty((h, w), dtype=np.int32)
    flat_ids = region_id_map.reshape(-1)

    region_color, areas = kernels.label_components(labels.reshape(-1), h, w, int(num_colors), flat_ids)
    if region_color.size:
        kernels.merge_small_regions(flat_ids, h, w, areas, int(min_area_px))
        kernels.merge_same_color_neighbors(flat_ids, h, w, region_color, areas)
    return region_id_map, region_color


def absorb_thin_parts(
    region_id_map: np.ndarray, region_color: np.ndarray, labels: np.ndarray, min_width_px: float
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
    color_of_label = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    color_of_label[nearest[fits]] = region_color[region_id_map[fits]]
    widened = color_of_label[nearest]

    outside = region_id_map < 0
    if outside.any():
        _keep_to_own_side(widened, nearest, fits, outside, region_id_map, region_color, labels)
        widened[outside] = labels[outside]
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
    compartment_of_label = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    compartment_of_label[nearest[fits]] = compartment[fits]
    across = ~outside & (compartment_of_label[nearest] != compartment)
    if not across.any():
        return
    # Only the compartments holding such pixels need searching, and only as far as they reach.
    wanted = np.zeros(int(compartment.max()) + 1, dtype=bool)
    wanted[compartment[across]] = True
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
    y, x = np.nonzero((ids >= 0) & asked[np.where(ids >= 0, ids, 0)])
    own = ids[y, x].astype(np.int64)
    keys = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            on_page = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)  # off the page is nobody's
            yy, xx = yy[on_page], xx[on_page]
            other = ids[yy, xx]
            differs = other != own[on_page]
            keys.append(own[on_page][differs] * (h * w) + (yy[differs].astype(np.int64) * w + xx[differs]))
    ring = np.unique(np.concatenate(keys))
    patch, pixel = np.divmod(ring, h * w)
    owner = ids.reshape(-1)[pixel]
    by_ink = np.bincount(patch[owner < 0], minlength=asked.size)
    others = owner >= 0
    pairs, counts = np.unique(patch[others] * (int(ids.max()) + 1) + owner[others], return_counts=True)
    most_by_one = np.zeros(asked.size, dtype=np.int64)
    np.maximum.at(most_by_one, pairs // (int(ids.max()) + 1), counts)
    return asked & (by_ink > 0) & (by_ink >= most_by_one)


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
    has = np.zeros(count, dtype=bool)
    pairs = ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :]), (ids[:-1, :-1], ids[1:, 1:]), (ids[:-1, 1:], ids[1:, :-1]))
    for a, b in pairs:
        differ = (a != b) & (a >= 0) & (b >= 0)
        has[a[differ]] = True
        has[b[differ]] = True
    return has


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


def extract_regions(region_id_map: np.ndarray, region_color: np.ndarray, min_contour_area: float = 1.0) -> list[Region]:
    """Extract one outer contour + label point per surviving region id.

    Each region is processed inside its own one-pixel-padded bounding box
    rather than across the whole image (same result: nothing outside the box
    can affect its contour or distance transform), spread over worker threads.
    Regions come back in region-id order.
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
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
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

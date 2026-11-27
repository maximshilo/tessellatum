"""Measurements of a generated page, from what it is made of: the regions, their colors and the numbers.

What the app shows of a page (the palette inspector's coverage, the region inspector's sizes) is measured here, on the
region map at the page's own size, and turned into millimeters on paper by the page's scale. The module depends on
NumPy and OpenCV only, never on the rest of the package, so that the benchmark can load this file by its path and judge
every version of the pipeline with the same code (see ``benchmarks/bench_metrics.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorCoverage:
    """How much of a page each palette color paints: ``pixels[i]`` of the page's pixels and ``regions[i]`` regions in
    color ``i``, for every color (the legend's first). ``page_pixels`` is the page's size in pixels; ``painted`` the
    pixels in any region, which ``pixels`` sums to."""

    pixels: np.ndarray  # int64, one per color
    regions: np.ndarray  # int64, one per color
    page_pixels: int
    painted: int

    def share(self, color: int) -> float:
        """Color ``color``'s share of the page, 0 to 1."""
        return float(self.pixels[color]) / self.page_pixels if self.page_pixels else 0.0

    def area_mm2(self, color: int, px_per_mm: float) -> float:
        """Color ``color``'s area on paper, in square millimeters, on a page of ``px_per_mm``."""
        return float(self.pixels[color]) / (px_per_mm * px_per_mm)


def region_areas(region_id_map: np.ndarray) -> np.ndarray:
    """Each region id's area in pixels (0 for an id merged away); the pixels in no region (-1) are left out."""
    ids = np.asarray(region_id_map)
    inside = ids[ids >= 0]
    return np.bincount(inside.ravel(), minlength=0).astype(np.int64)


def color_coverage(region_id_map: np.ndarray, region_color: np.ndarray, colors: int) -> ColorCoverage:
    """The pixels and the regions of each of ``colors`` palette colors on a page: a region id's color is
    ``region_color[id]``. A color past ``colors`` (a region keeping a color of its own) widens the counts to it, so
    they always sum to the painted pixels and the regions there are."""
    ids = np.asarray(region_id_map)
    areas = region_areas(ids)
    present = np.flatnonzero(areas)
    color_of = np.asarray(region_color, dtype=np.int64)[present]
    count = max(int(colors), int(color_of.max()) + 1 if color_of.size else 0)
    pixels = np.bincount(color_of, weights=areas[present], minlength=count).astype(np.int64)
    regions = np.bincount(color_of, minlength=count).astype(np.int64)
    return ColorCoverage(pixels=pixels, regions=regions, page_pixels=int(ids.size), painted=int(areas.sum()))


def regions_of_color(region_id_map: np.ndarray, region_color: np.ndarray, color: int) -> np.ndarray:
    """The ids of the regions on the page in ``color``, largest first (ties by id)."""
    areas = region_areas(region_id_map)
    present = np.flatnonzero(areas)
    ids = present[np.asarray(region_color, dtype=np.int64)[present] == color]
    order = np.lexsort((ids, -areas[ids]))
    return ids[order]


def color_mask(region_id_map: np.ndarray, region_color: np.ndarray, color: int) -> np.ndarray:
    """HxW bool: the pixels painted in ``color``."""
    ids = np.asarray(region_id_map)
    lookup = np.asarray(region_color, dtype=np.int64) == color
    mask = np.zeros(ids.shape, dtype=bool)
    inside = ids >= 0
    mask[inside] = lookup[ids[inside]]
    return mask


@dataclass(frozen=True)
class RegionStats:
    """Every region's area and bounding box on a page, by region id: ``areas[id]`` pixels within the box
    ``x0[id] <= x < x1[id]``, ``y0[id] <= y < y1[id]``. An id merged away has area 0 and an empty box."""

    areas: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray

    def box(self, region_id: int) -> tuple[int, int, int, int]:
        return int(self.x0[region_id]), int(self.y0[region_id]), int(self.x1[region_id]), int(self.y1[region_id])

    def __contains__(self, region_id: int) -> bool:
        return 0 <= region_id < len(self.areas) and self.areas[region_id] > 0


def region_stats(region_id_map: np.ndarray) -> RegionStats:
    """Each region's area and bounding box (see ``RegionStats``), in one pass over the page."""
    ids = np.asarray(region_id_map)
    height, width = ids.shape
    flat = ids.ravel()
    where = np.flatnonzero(flat >= 0)
    count = int(flat[where].max()) + 1 if where.size else 0
    areas = np.bincount(flat[where], minlength=count).astype(np.int64)
    x0 = np.zeros(count, dtype=np.int64)
    y0 = np.zeros(count, dtype=np.int64)
    x1 = np.zeros(count, dtype=np.int64)
    y1 = np.zeros(count, dtype=np.int64)
    if where.size:
        order = where[np.argsort(flat[where], kind="stable")]
        present = np.flatnonzero(areas)
        starts = np.concatenate(([0], np.cumsum(areas[present])[:-1]))
        ys, xs = np.divmod(order, width)
        y0[present] = ys[starts]  # the pixels are in reading order within a region: its first is in its top row
        y1[present] = np.maximum.reduceat(ys, starts) + 1
        x0[present] = np.minimum.reduceat(xs, starts)
        x1[present] = np.maximum.reduceat(xs, starts) + 1
    return RegionStats(areas=areas, x0=x0, y0=y0, x1=x1, y1=y1)


def widest_brush_px(region_id_map: np.ndarray, region_id: int, box: tuple[int, int, int, int]) -> float:
    """The width, in pixels, of the widest round brush that fits somewhere in region ``region_id`` (whose bounding
    box is ``box``): twice the farthest any pixel of it is from outside it, less the pixel the brush is centered on."""
    x0, y0, x1, y1 = box
    crop = np.asarray(region_id_map)[y0:y1, x0:x1] == region_id
    if not crop.any():
        return 0.0
    padded = np.pad(crop, 1).astype(np.uint8)  # the page's edge bounds a region as its neighbors do
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return max(2.0 * float(distance.max()) - 1.0, 1.0)


def region_mask(region_id_map: np.ndarray, region_id: int) -> np.ndarray:
    """HxW bool: the pixels of region ``region_id``."""
    return np.asarray(region_id_map) == region_id

# -- The page's paintability: as the benchmark judges it ---------------------------------------------------------------
# Moved here from the benchmark's metrics, unchanged, so that the app's analysis and the benchmark measure a page by
# one implementation: ``benchmarks/bench_metrics.py`` loads this file by its path and uses these.

# The sharpest corner whose tip -- the point a round brush can't reach -- counts as paintable: a painter fills it with
# the brush's point, so slivers leave such points out (see ``corner_tips``), wherever a page has them. The pipeline
# keeps them when its sharpest corner setting asks it to (``regions.CornerRule``); by default it rounds every corner.
CORNER_SHARPEST_DEG = 20.0
# A tip meets its region's reach along a base at least this many brush radii long, may measure this much deeper or
# longer than its corner's shape says on the pixel grid, and takes in the pixels within this many of it: as the
# pipeline's ``regions._TIP_BASE_RADII``, ``regions._TIP_SLACK`` and ``regions._TIP_RIM_PX``.
_TIP_BASE_RADII = 1.0
_TIP_SLACK = 1.2
_TIP_RIM_PX = 2
# A tip with another tip's middle within this many brush radii of its own is a ragged edge's, as the pipeline's
# ``regions._TIP_APART_RADII``.
_TIP_APART_RADII = 2.0


def count_undersized(region_id_map: np.ndarray, min_area_px: int, detail: np.ndarray | None = None) -> int:
    """Regions still smaller than the merge threshold (should be none) that another region touches.

    A region no other region touches, diagonals included, has nothing to merge
    into: on line art, a shape the ink encloses on its own, whose size is the
    artwork's.

    ``detail`` (HxW bool) is where the page lets a region be half as large, its
    faces (``PageAnalysis.detail``, from 0.1.34): a pixel there counts twice
    towards the threshold, as the pipeline counts it.
    """
    ids = np.asarray(region_id_map)
    inside = ids >= 0
    pixels = np.bincount(ids[inside].ravel())
    sizes = pixels
    if detail is not None:
        sizes = pixels + np.bincount(ids[inside & np.asarray(detail, dtype=bool)].ravel(), minlength=pixels.size)
    touched = np.zeros(pixels.size, dtype=bool)
    for a, b, _spacing in _neighbor_pairs(ids):
        differ = (a != b) & (a >= 0) & (b >= 0)
        touched[a[differ]] = True
        touched[b[differ]] = True
    return int(((pixels > 0) & (sizes < min_area_px) & touched).sum())


def unlabeled_regions(region_id_map: np.ndarray, labeled_region_ids) -> dict[str, float | int]:
    """Regions without a number, counted from the region map so that regions too small to draw count too."""
    areas = np.bincount(region_id_map[region_id_map >= 0].ravel())
    present = np.flatnonzero(areas)
    labeled = np.fromiter(labeled_region_ids, dtype=np.int64, count=len(labeled_region_ids))
    unlabeled = present[~np.isin(present, labeled)]
    return {
        "unlabeled_regions": int(unlabeled.size),
        "unlabeled_area_fraction": float(areas[unlabeled].sum() / region_id_map.size) if region_id_map.size else 0.0,
    }


def label_sizes(font_sizes_px, scale, min_pt: float) -> dict[str, float | None]:
    """How large the numbers print: the smallest, in points, and the share below ``min_pt``, the page's smallest.

    ``font_sizes_px`` are the numbers' em sizes on the page; ``scale`` is its
    ``print_size.PrintScale``. Both results are None on a page without numbers.
    """
    if len(font_sizes_px) == 0:
        return {"small_label_fraction": None, "min_label_pt": None}
    sizes_pt = np.asarray(font_sizes_px, dtype=np.float64) / scale.px_per_pt
    return {
        "small_label_fraction": float((sizes_pt < min_pt).mean()),
        "min_label_pt": float(sizes_pt.min()),
    }


def sliver_mask(region_id_map: np.ndarray, min_width_px: float) -> np.ndarray:
    """Region pixels a round brush ``min_width_px`` wide can't paint without crossing into another region.

    The brush is the disk of pixels within ``min_width_px / 2`` of a pixel
    center. A pixel is paintable if some brush position that lies entirely
    inside its region covers it (the region's morphological opening by that
    disk). Pixels outside the page count as another region. Slivers are thin
    parts of regions, and also the corners a round brush can't reach.
    """
    ids, areas = _renumbered_regions(region_id_map)
    if areas.size == 0:
        return np.zeros(ids.shape, dtype=bool)
    radius_sq = (min_width_px / 2) ** 2

    # A brush fits where the nearest pixel of another region is farther than its radius.
    fits = (ids >= 0) & (_out_of_region_distance_sq(ids, areas.size) > radius_sq)
    if not fits.any():
        return ids >= 0
    to_brush = cv2.distanceTransform((~fits).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (ids >= 0) & (_squared_distance(to_brush) > radius_sq)


def sliver_share(region_id_map: np.ndarray, min_width_px: float) -> float:
    """Share of page area in slivers (see ``sliver_mask``)."""
    return float(sliver_mask(region_id_map, min_width_px).mean()) if region_id_map.size else 0.0


def corner_tips(
    region_id_map: np.ndarray, min_width_px: float, sharpest_deg: float = CORNER_SHARPEST_DEG
) -> np.ndarray:
    """The pixels a round brush ``min_width_px`` wide can't reach (``sliver_mask``) that are corners' tips: HxW bool.

    In a corner of angle a, a brush of radius r stops at the circle touching
    both sides; the point beyond, the corner's tip, a painter fills with the
    brush's point. Its area over the length of its base, where it meets what
    the brush reaches, is r (cot(a/2) - (π - a)/2) / (π - a), and its point
    lies r (1 / sin(a/2) - 1) beyond the brush.

    What a region's brush can't reach falls into pieces: the pixels more than
    a pixel beyond the brush and those beside them, 8-connected within a
    region. (Single pixels within a pixel of the brush are the grid's
    staircase along a slanted edge.) A piece is a tip if its base -- pairs of
    pixels side by side, one in it and one of its region in no piece -- is at
    least ``_TIP_BASE_RADII`` radii long and meets what the brush reaches
    along one stretch (an 8-connected run), not none, nor two as a neck or a
    channel does; if it is neither deeper (area over base) nor longer (its
    farthest pixel's straight distance from its region's reach) than the tip
    of a corner ``sharpest_deg`` sharp, give or take ``_TIP_SLACK``; and if no
    other tip's middle lies within ``_TIP_APART_RADII`` radii of its own (tips
    crowded so are the spikes of a ragged edge, not a shape's corners). A tip
    then takes in what the brush can't reach of its region within
    ``_TIP_RIM_PX`` of it: the pixels along its sides and its base the piece
    leaves out.

    The pipeline, asked to, keeps corners' tips by this rule
    (``regions.corner_tips``), measuring a piece's length along a path through
    it rather than straight.
    """
    ids, areas = _renumbered_regions(region_id_map)
    none = np.zeros(ids.shape, dtype=bool)
    if areas.size == 0 or sharpest_deg >= 180:
        return none
    count = areas.size
    radius = min_width_px / 2
    fits = (ids >= 0) & (_out_of_region_distance_sq(ids, count) > radius**2)
    if not fits.any():
        return none
    to_brush_sq = _squared_distance(cv2.distanceTransform((~fits).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
    unreached = (ids >= 0) & (to_brush_sq > radius**2)
    beyond = (ids >= 0) & (to_brush_sq > (radius + 1) ** 2)
    if not beyond.any():
        return none
    near = beyond | _beside_own(beyond, ids, unreached, 1)
    pieces, piece_count = _pieces_by_region(near, ids, count)
    area = np.bincount(pieces[near], minlength=piece_count)
    reached = (ids >= 0) & ~unreached
    base = np.zeros(piece_count, dtype=np.int64)
    base_of = np.full(ids.shape, -1, dtype=np.int64)  # each reached pixel a piece meets, by the piece
    pairs = (
        (slice(None), slice(None, -1), slice(None), slice(1, None)),
        (slice(None, -1), slice(None), slice(1, None), slice(None)),
    )
    for rows_a, columns_a, rows_b, columns_b in pairs:
        a, b = (rows_a, columns_a), (rows_b, columns_b)
        same = ids[a] == ids[b]
        for inside, outside in ((a, b), (b, a)):
            meets = same & (pieces[inside] >= 0) & (pieces[outside] < 0)
            base += np.bincount(pieces[inside][meets], minlength=piece_count)
            view = base_of[outside]
            reaching = meets & reached[outside]
            view[reaching] = pieces[inside][reaching]
    # The stretches along which each piece meets what its brush reaches: 8-connected runs of those pixels.
    met = base_of >= 0
    runs, run_count = _pieces_by_region(met, base_of, piece_count)
    run_piece = np.zeros(run_count, dtype=np.int64)
    run_piece[runs[met]] = base_of[met]
    stretches = np.bincount(run_piece, minlength=piece_count)
    angle = math.radians(sharpest_deg)
    depth = (1 / math.tan(angle / 2) - (math.pi - angle) / 2) / (math.pi - angle)
    length = 1 / math.sin(angle / 2) - 1
    tip = (base >= _TIP_BASE_RADII * radius) & (stretches == 1) & (area <= (_TIP_SLACK * depth * radius + 0.5) * base)
    reach = _TIP_SLACK * length * radius + 1  # measured from the middle of the last pixel reached
    for k in np.flatnonzero(tip):
        tip[k] = _farthest_from_reach(ids, pieces, unreached, k, reach) <= reach
    which = np.flatnonzero(tip)
    if which.size > 1:
        ys, xs = np.nonzero(near)
        middles = np.stack(
            [np.bincount(pieces[near], weights, minlength=piece_count)[which] / area[which] for weights in (ys, xs)], 1
        )
        crowded = np.zeros(which.size, dtype=bool)
        for start in range(0, which.size, 1024):  # a block of tips against all of them, to keep the arrays small
            apart = np.hypot(*(middles[start : start + 1024, None, :] - middles[None, :, :]).transpose(2, 0, 1))
            apart[np.arange(apart.shape[0]), np.arange(start, start + apart.shape[0])] = np.inf
            crowded[start : start + 1024] = apart.min(axis=1) < _TIP_APART_RADII * radius
        tip[which[crowded]] = False
    if not tip.any():
        return none
    tips = near & tip[np.maximum(pieces, 0)]
    return tips | _beside_own(tips, ids, unreached, _TIP_RIM_PX)


def _beside_own(mask: np.ndarray, ids: np.ndarray, within: np.ndarray, reach_px: int) -> np.ndarray:
    """The pixels of ``within`` with a pixel of ``mask`` of their own region within ``reach_px`` across or down.

    Measured around each 8-connected run of ``mask``, in its box widened by ``reach_px``.
    """
    out = np.zeros(ids.shape, dtype=bool)
    count, _labels, stats, _middles = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    h, w = ids.shape
    for left, top, width, height, _area in stats[1:count]:
        y0, x0 = max(0, top - reach_px), max(0, left - reach_px)
        y1, x1 = min(h, top + height + reach_px), min(w, left + width + reach_px)
        part_mask = np.pad(mask[y0:y1, x0:x1], reach_px)
        part_ids = np.pad(ids[y0:y1, x0:x1], reach_px, constant_values=-1)
        here = ids[y0:y1, x0:x1]
        found = np.zeros(here.shape, dtype=bool)
        for dy in range(2 * reach_px + 1):
            for dx in range(2 * reach_px + 1):
                rows, columns = slice(dy, dy + here.shape[0]), slice(dx, dx + here.shape[1])
                found |= part_mask[rows, columns] & (part_ids[rows, columns] == here)
        out[y0:y1, x0:x1] |= found
    return within & out


def _pieces_by_region(mask: np.ndarray, ids: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    """The 8-connected pieces of ``mask`` within each region of ``ids``, numbered from 0 (-1 off the mask); how many.

    Pieces of two regions that touch are told apart; a region's pieces joined
    only through another region's would be counted as one, which a region's
    pixels beyond its brush never are in practice.
    """
    _n, component = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    key = component.astype(np.int64) * count + ids
    pieces = np.full(ids.shape, -1, dtype=np.int64)
    unique, inverse = np.unique(key[mask], return_inverse=True)
    pieces[mask] = inverse
    return pieces, int(unique.size)


def _farthest_from_reach(ids: np.ndarray, pieces: np.ndarray, unreached: np.ndarray, piece: int, limit: float) -> float:
    """The straight distance from piece ``piece``'s farthest pixel to its region's pixels a brush reaches.

    Measured in the piece's box, widened by ``limit`` and a pixel: a piece
    whose region's reach lies further than that is farther than ``limit``.
    """
    ys, xs = np.nonzero(pieces == piece)
    pad = int(math.ceil(limit)) + 1
    h, w = ids.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    reached = (ids[y0:y1, x0:x1] == ids[ys[0], xs[0]]) & ~unreached[y0:y1, x0:x1]
    if not reached.any():
        return math.inf
    distance = cv2.distanceTransform((~reached).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(distance[ys - y0, xs - x0].max())


def _neighbor_pairs(ids: np.ndarray) -> tuple:
    """Neighboring pixels along rows, columns and both diagonals: (one side, the other side, spacing of those lines)."""
    return (
        (ids[:, :-1], ids[:, 1:], 1.0),
        (ids[:-1, :], ids[1:, :], 1.0),
        # Diagonal lines of pixel centers lie 1/√2 apart, rows and columns 1 apart.
        (ids[:-1, :-1], ids[1:, 1:], np.sqrt(0.5)),
        (ids[:-1, 1:], ids[1:, :-1], np.sqrt(0.5)),
    )


def _renumbered_regions(region_id_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the map with its regions renumbered 0..n-1 in id order and -1 outside any region, each region's area)."""
    ids = np.asarray(region_id_map, dtype=np.int64)
    inside = ids >= 0
    areas = np.bincount(ids[inside], minlength=1)
    present = np.flatnonzero(areas)
    new_id = np.full(areas.size, -1, dtype=np.int32)
    new_id[present] = np.arange(present.size, dtype=np.int32)
    return np.where(inside, new_id[np.where(inside, ids, 0)], -1).astype(np.int32), areas[present]


def _edge_adjacency_classes(ids: np.ndarray, count: int) -> np.ndarray:
    """A class per region such that regions sharing a pixel edge never share a class (greedy graph coloring)."""
    codes = []
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])):
        differ = (a != b) & (a >= 0) & (b >= 0)
        low = np.minimum(a[differ], b[differ]).astype(np.int64)
        high = np.maximum(a[differ], b[differ]).astype(np.int64)
        codes.append(low * count + high)
    neighbors: list[list[int]] = [[] for _ in range(count)]
    for low, high in zip(*(part.tolist() for part in np.divmod(np.unique(np.concatenate(codes)), count))):
        neighbors[low].append(high)
        neighbors[high].append(low)
    region_class = [-1] * count
    for region in sorted(range(count), key=lambda r: -len(neighbors[r])):  # most neighbors first
        taken = {region_class[other] for other in neighbors[region]}
        cls = 0
        while cls in taken:
            cls += 1
        region_class[region] = cls
    return np.asarray(region_class, dtype=np.int32)


def _out_of_region_distance_sq(ids: np.ndarray, count: int) -> np.ndarray:
    """Each pixel's squared distance to the nearest pixel outside its region, 0 for the pixels in no region.

    ``ids`` are numbered 0 to ``count`` - 1, as ``_renumbered_regions`` numbers
    them. Pixels in no region, and those just off the page, are outside every
    region. A distance transform measures the distance to one set of pixels, so
    regions are split into classes in which no two regions share an edge. The
    nearest pixel of another region always shares an edge with this region (one
    step from it towards the center lands inside), so it is in another class:
    one distance transform per class finds it for all of the class's regions at
    once.
    """
    region_class = _edge_adjacency_classes(ids, count)
    classes = np.where(ids >= 0, region_class[ids], -1)
    distance_sq = np.zeros(ids.shape)
    for cls in range(int(region_class.max()) + 1):
        in_class = classes == cls
        padded = np.pad(in_class, 1).astype(np.uint8)  # the padding is outside the page
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        distance_sq[in_class] = _squared_distance(distance[in_class])
    return distance_sq


def _squared_distance(distance: np.ndarray) -> np.ndarray:
    """Exact squared distances from ``cv2.distanceTransform`` output, which holds their float32 square roots."""
    return np.rint(distance.astype(np.float64) ** 2)


# -- The page's analysis, as the app shows it -------------------------------------------------------------------------
@dataclass(frozen=True)
class PageMeasures:
    """A page measured as the benchmark judges it, on paper of ``px_per_mm``: what the app's analysis shows.

    The counts are the benchmark's own (``sliver_share`` less the corners' tips, ``unlabeled_regions``,
    ``label_sizes``, the numbers on leaders); the region ids under each issue are there to find them by, largest first.
    """

    px_per_mm: float
    brush_mm: float
    min_area_mm2: float
    min_pt: float
    colors: int
    regions: int
    page_mm2: float
    smallest_mm2: float
    smallest_region: int | None
    narrowest_mm: float
    narrowest_region: int | None
    sliver_fraction: float  # of the page: what the brush can't reach, its corners' tips left out
    corner_tip_fraction: float
    leader_labels: int
    cramped_labels: int
    small_label_fraction: float | None  # of the numbers: printed smaller than min_pt
    min_label_pt: float | None
    unlabeled_regions: int
    unlabeled_area_fraction: float
    slivers: np.ndarray  # HxW bool
    small: np.ndarray  # region ids smaller than min_area_mm2
    narrow: np.ndarray  # region ids nowhere as wide as the brush
    unlabeled: np.ndarray  # region ids with no number
    cramped: np.ndarray  # region ids whose number is cramped
    small_numbers: np.ndarray  # region ids whose number prints smaller than min_pt
    on_leaders: np.ndarray  # region ids with a number written outside them, on a leader line
    leaders: list  # ((x, y), (x, y)) of each number's leader line, and its number's box: (leader, box)

    @property
    def regions_per_cm2(self) -> float:
        return self.regions / (self.page_mm2 / 100.0) if self.page_mm2 else 0.0


def widest_brushes_px(region_id_map: np.ndarray) -> np.ndarray:
    """Each region id's widest brush in pixels (as ``widest_brush_px`` measures one region), for every region at once:
    twice its farthest pixel from outside it, less a pixel; 0 for an id merged away."""
    ids, areas = _renumbered_regions(region_id_map)
    original = np.asarray(region_id_map)
    count = int(original.max()) + 1 if original.size and original.max() >= 0 else 0
    widest = np.zeros(count)
    if areas.size == 0:
        return widest
    inside = ids >= 0
    half_sq = np.zeros(areas.size)
    np.maximum.at(half_sq, ids[inside], _out_of_region_distance_sq(ids, areas.size)[inside])
    present = np.flatnonzero(np.bincount(original[original >= 0].ravel(), minlength=count))
    widest[present] = np.maximum(2.0 * np.sqrt(half_sq) - 1.0, 1.0)
    return widest


def _largest_first(ids: np.ndarray, areas: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return ids[np.lexsort((ids, -areas[ids]))] if ids.size else ids


def measure_page(region_id_map: np.ndarray, labels, colors: int, px_per_mm: float, brush_mm: float,
                 min_area_mm2: float, min_pt: float, px_per_pt: float) -> PageMeasures:
    """The page of ``region_id_map`` and its numbers (``labels``: each with ``region_id``, ``font_size`` in pixels,
    ``leader``, ``box``, ``cramped``), measured on paper: ``px_per_mm`` and ``px_per_pt`` its scale, ``brush_mm`` the
    brush it is painted with, ``min_area_mm2`` its smallest region and ``min_pt`` its smallest number."""
    ids = np.asarray(region_id_map)
    stats = region_stats(ids)
    areas = stats.areas
    present = np.flatnonzero(areas)
    brush_px = brush_mm * px_per_mm
    corners = corner_tips(ids, brush_px)
    slivers = sliver_mask(ids, brush_px) & ~corners
    widest = widest_brushes_px(ids)
    labels = list(labels)
    labeled = {label.region_id for label in labels}
    unlabeled = unlabeled_regions(ids, labeled)

    class _Scale:  # what ``label_sizes`` reads of a print scale
        pass

    scale = _Scale()
    scale.px_per_pt = px_per_pt
    sizes = label_sizes([label.font_size for label in labels], scale, min_pt)
    smallest = present[np.argmin(areas[present])] if present.size else None
    narrowest = present[np.argmin(widest[present])] if present.size else None
    mm2 = px_per_mm * px_per_mm
    unlabeled_ids = present[~np.isin(present, np.fromiter(labeled, dtype=np.int64, count=len(labeled)))]
    return PageMeasures(
        px_per_mm=px_per_mm, brush_mm=brush_mm, min_area_mm2=min_area_mm2, min_pt=min_pt, colors=int(colors),
        regions=int(present.size), page_mm2=ids.size / mm2,
        smallest_mm2=float(areas[smallest]) / mm2 if smallest is not None else 0.0,
        smallest_region=int(smallest) if smallest is not None else None,
        narrowest_mm=float(widest[narrowest]) / px_per_mm if narrowest is not None else 0.0,
        narrowest_region=int(narrowest) if narrowest is not None else None,
        sliver_fraction=float(slivers.mean()) if slivers.size else 0.0,
        corner_tip_fraction=float(corners.mean()) if corners.size else 0.0,
        leader_labels=sum(label.leader is not None for label in labels),
        cramped_labels=sum(bool(label.cramped) for label in labels),
        small_label_fraction=sizes["small_label_fraction"], min_label_pt=sizes["min_label_pt"],
        unlabeled_regions=unlabeled["unlabeled_regions"], unlabeled_area_fraction=unlabeled["unlabeled_area_fraction"],
        slivers=slivers,
        small=_largest_first(present[areas[present] < min_area_mm2 * mm2], areas),
        narrow=_largest_first(present[widest[present] < brush_px], areas),
        unlabeled=_largest_first(unlabeled_ids, areas),
        cramped=_largest_first(np.unique([label.region_id for label in labels if label.cramped]), areas),
        small_numbers=_largest_first(
            np.unique([label.region_id for label in labels if label.font_size / px_per_pt < min_pt]), areas),
        on_leaders=_largest_first(np.unique([label.region_id for label in labels if label.leader is not None]), areas),
        leaders=[(label.leader, label.box) for label in labels if label.leader is not None],
    )

"""Quality metrics for generated coloring pages.

Two kinds:

* **Absolute** metrics score one result on its own: how faithfully the
  finished painting (every region filled with its legend color) reproduces the
  source image, how paintable the page is at print size (slivers too thin for a
  brush, unlabeled regions, label size, region shape, leftover undersized
  regions, outline clutter), and how cleanly its lines are drawn (lines per
  boundary, boundaries between same-colored regions, jaggedness, lines on the
  source's edges).
* **Agreement** metrics score a result against a reference result (usually the
  previous version), to tell "identical output" apart from "different output".

Deliberately independent of the tessellatum package, so every version of the
pipeline is scored by the same yardstick. The one shared piece, the print-size
model, is loaded from this checkout's source file (``print_size``), never from
the version being measured.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

BOUNDARY_TOLERANCE_PX = 2
PRINT_SIZE_PATH = Path(__file__).resolve().parents[1] / "src" / "tessellatum" / "core" / "print_size.py"
# Line quality. Wiggles in a drawn line smaller than JAGGEDNESS_SMOOTHING_MM count as jaggedness. The
# source's edges are found after smoothing it by EDGE_SMOOTHING_MM, as color steps of at least
# EDGE_THRESHOLDS (Canny's hysteresis thresholds, in CIE Lab units), and a region boundary within
# EDGE_TOLERANCE_MM of an edge lies on it.
JAGGEDNESS_SMOOTHING_MM = 0.5
EDGE_SMOOTHING_MM = 0.5
EDGE_TOLERANCE_MM = 0.5
EDGE_THRESHOLDS = (5.0, 10.0)


def _load_print_size():
    """``tessellatum.core.print_size`` of this checkout, loaded by file path.

    A regular import would find the version being benchmarked first, which may
    judge pages differently or predate the model.
    """
    spec = importlib.util.spec_from_file_location("bench_print_size", PRINT_SIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look up their module while being created
    spec.loader.exec_module(module)
    return module


print_size = _load_print_size()


def reference_resize(image_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    """The source image at output size -- what every version is compared against."""
    h, w = image_bgr.shape[:2]
    if max(h, w) <= long_edge:
        return image_bgr
    scale = long_edge / max(h, w)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def fit_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize to ``size`` (w, h), only if it differs."""
    if image.shape[1::-1] == tuple(size):
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def paint(region_id_map: np.ndarray, region_color: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
    """Fill every region with its palette color: the "finished painting"."""
    lut = np.asarray(palette_bgr, dtype=np.uint8)[np.asarray(region_color, dtype=np.int64)]
    return lut[np.clip(region_id_map, 0, None)]


def bgr_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """uint8 sRGB (BGR order) -> float64 CIE Lab with L in 0..100."""
    lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab.astype(np.float64)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Per-element CIEDE2000 color difference (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c_bar7 = ((np.hypot(a1, b1) + np.hypot(a2, b2)) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma_zero = (c1p * c2p) == 0

    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(chroma_zero, 0.0, dh)
    d_big_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(np.abs(h1p - h2p) > 180, np.where(h_sum < 360, h_sum + 360, h_sum - 360), h_sum) / 2
    h_bar = np.where(chroma_zero, h_sum, h_bar)

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7))
    l50 = (l_bar - 50) ** 2
    s_l = 1 + 0.015 * l50 / np.sqrt(20 + l50)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c

    return np.sqrt(
        (dl / s_l) ** 2 + (dc / s_c) ** 2 + (d_big_h / s_h) ** 2 + r_t * (dc / s_c) * (d_big_h / s_h)
    )


def mean_de00(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    return float(ciede2000(bgr_to_lab(a_bgr), bgr_to_lab(b_bgr)).mean())


def ssim_gray(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004; 11x11 Gaussian window, sigma 1.5) on luma."""
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(ssim_map.mean())


def fidelity(source_bgr: np.ndarray, painted_bgr: np.ndarray) -> dict[str, float]:
    """How closely the finished painting matches the source image."""
    de = ciede2000(bgr_to_lab(source_bgr), bgr_to_lab(painted_bgr))
    return {
        "de00_mean": float(de.mean()),
        "de00_p95": float(np.percentile(de, 95)),
        "ssim": ssim_gray(source_bgr, painted_bgr),
    }


def count_undersized(region_id_map: np.ndarray, min_area_px: int) -> int:
    """Regions still smaller than the merge threshold (should be none)."""
    areas = np.bincount(region_id_map[region_id_map >= 0].ravel())
    areas = areas[areas > 0]
    return int((areas < min_area_px).sum())


def label_coverage(regions, labeled_region_ids, total_px: int) -> dict[str, float]:
    """Share of drawn regions (and of page area) that carry a number."""
    labeled = [r for r in regions if r.region_id in labeled_region_ids]
    return {
        "labeled_region_fraction": len(labeled) / len(regions) if regions else 0.0,
        "labeled_area_fraction": sum(r.area for r in labeled) / total_px if total_px else 0.0,
    }


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


def label_sizes(font_sizes_px, scale) -> dict[str, float | None]:
    """How large the numbers print: the smallest, in points, and the share below the minimum legible size.

    ``font_sizes_px`` are the numbers' em sizes on the page; ``scale`` is its
    ``print_size.PrintScale``. Both results are None on a page without numbers.
    """
    if len(font_sizes_px) == 0:
        return {"small_label_fraction": None, "min_label_pt": None}
    sizes_pt = np.asarray(font_sizes_px, dtype=np.float64) / scale.px_per_pt
    return {
        "small_label_fraction": float((sizes_pt < print_size.MIN_LABEL_SIZE_PT).mean()),
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

    # A brush fits where the nearest pixel of another region is farther than
    # its radius. A distance transform measures the distance to one set of
    # pixels, so regions are split into classes in which no two regions share
    # an edge. The nearest pixel of another region always shares an edge with
    # this region (one step from it towards the center lands inside), so it
    # is in another class: one distance transform per class finds it for all
    # of the class's regions at once.
    region_class = _edge_adjacency_classes(ids, areas.size)
    classes = np.where(ids >= 0, region_class[ids], -1)
    fits = np.zeros(ids.shape, dtype=bool)
    for cls in range(int(region_class.max()) + 1):
        in_class = classes == cls
        padded = np.pad(in_class, 1).astype(np.uint8)  # the padding is outside the page
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        fits |= in_class & (_squared_distance(distance) > radius_sq)

    if not fits.any():
        return ids >= 0
    to_brush = cv2.distanceTransform((~fits).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (ids >= 0) & (_squared_distance(to_brush) > radius_sq)


def sliver_share(region_id_map: np.ndarray, min_width_px: float) -> float:
    """Share of page area in slivers (see ``sliver_mask``)."""
    return float(sliver_mask(region_id_map, min_width_px).mean()) if region_id_map.size else 0.0


def compactness(region_id_map: np.ndarray) -> np.ndarray:
    """4πA/P² of every region in the map, in region-id order: 1 for a disk, lower the more stretched or ragged.

    A is the region's pixel count. P, its boundary length including holes and
    the page edge, is estimated with the Cauchy–Crofton formula from how often
    rows, columns and both diagonals of pixel centers cross the boundary.
    Counting pixel edges instead would make diagonal boundaries √2 times too
    long. Values are capped at 1, which the estimate can exceed for regions of
    a few pixels.
    """
    ids, areas = _renumbered_regions(region_id_map)
    crossings = np.zeros(areas.size)
    for a, b, line_spacing in _neighbor_pairs(np.pad(ids, 1, constant_values=-1)):
        differ = a != b
        for side in (a, b):
            crossings += line_spacing * np.bincount(side[differ & (side >= 0)], minlength=areas.size)
    perimeter = np.pi / 8 * crossings  # (1/2) · Σ over the 4 directions of crossings · line spacing · π/4
    return np.minimum(4 * np.pi * areas / np.maximum(perimeter, 1e-12) ** 2, 1.0)


def compactness_stats(region_id_map: np.ndarray) -> dict[str, float | None]:
    """Median and 10th percentile of region compactness (see ``compactness``)."""
    values = compactness(region_id_map)
    if values.size == 0:
        return {"compactness_median": None, "compactness_p10": None}
    return {"compactness_median": float(np.median(values)), "compactness_p10": float(np.percentile(values, 10))}


def boundary_lines(region_id_map: np.ndarray, strokes) -> dict[str, float | None]:
    """How many drawn lines run along the boundaries between regions: 1 means one line per boundary.

    ``strokes`` are the lines drawn on the page, each an (x, y) polyline with
    pixel centers at integer coordinates. Boundaries are measured per pixel
    edge between two regions; the page edge doesn't count. A line runs along a
    pixel edge if its rasterized centerline passes through or next to
    (8-neighborhood) either of the edge's two pixels, and counts once however
    often it passes. Returns the mean count, and the shares of boundary with two
    or more lines and with none (all None on a page without boundaries).
    """
    ids = np.asarray(region_id_map)
    h, w = ids.shape
    between_columns = (ids[:, :-1] != ids[:, 1:]) & (ids[:, :-1] >= 0) & (ids[:, 1:] >= 0)
    between_rows = (ids[:-1, :] != ids[1:, :]) & (ids[:-1, :] >= 0) & (ids[1:, :] >= 0)
    if not (between_columns.any() or between_rows.any()):
        return {"lines_per_boundary": None, "doubled_boundary_fraction": None, "undrawn_boundary_fraction": None}
    lines_between_columns = np.zeros(between_columns.shape, dtype=np.int32)
    lines_between_rows = np.zeros(between_rows.shape, dtype=np.int32)
    for (x0, y0), near in _line_neighborhoods(strokes, (w, h)):
        rows, cols = near.shape
        lines_between_columns[y0 : y0 + rows, x0 : x0 + cols - 1] += near[:, :-1] | near[:, 1:]
        lines_between_rows[y0 : y0 + rows - 1, x0 : x0 + cols] += near[:-1, :] | near[1:, :]
    counts = np.concatenate([lines_between_columns[between_columns], lines_between_rows[between_rows]])
    return {
        "lines_per_boundary": float(counts.mean()),
        "doubled_boundary_fraction": float((counts >= 2).mean()),
        "undrawn_boundary_fraction": float((counts == 0).mean()),
    }


def same_color_boundary_share(region_id_map: np.ndarray, region_color: np.ndarray) -> float | None:
    """Share of the boundary length between regions that separates two regions of the same color.

    Lengths are Cauchy–Crofton estimates, as in ``compactness``; the page edge
    doesn't count. None on a page without boundaries.
    """
    ids = np.asarray(region_id_map, dtype=np.int64)
    colors = np.asarray(region_color)
    total = same = 0.0
    for a, b, line_spacing in _neighbor_pairs(ids):
        differ = (a != b) & (a >= 0) & (b >= 0)
        total += line_spacing * np.count_nonzero(differ)
        same += line_spacing * np.count_nonzero(colors[a[differ]] == colors[b[differ]])
    return same / total if total else None


def jaggedness(strokes, region_id_map: np.ndarray, smoothing_px: float) -> float | None:
    """Length of the drawn lines over their length once wiggles smaller than ``smoothing_px`` are smoothed away.

    1 for straight lines and smooth curves; a staircase of 1 px steps scores
    about √2. Lines are cut ``smoothing_px`` short of junctions, where three
    regions, or two regions and the page edge, meet, and where they run along
    the page edge, so corners where lines meet don't count. Each piece is
    smoothed along its length by a Gaussian of standard deviation
    ``smoothing_px``, with its ends fixed; a closed line that meets no junction
    is smoothed all the way round. The result is the pieces' total length over
    their total smoothed length, so longer lines weigh more. Lines shorter than
    half a pixel are skipped; None if no line is long enough to measure.
    """
    ids = np.asarray(region_id_map)
    h, w = ids.shape
    to_junction = _distance_to_junctions(ids)
    total = smoothed_total = 0.0
    for stroke in strokes:
        points = _without_repeats(np.asarray(stroke, dtype=np.float64).reshape(-1, 2))
        closed = len(points) > 2 and np.array_equal(points[0], points[-1])
        samples, spacing = _resample(points, _RESAMPLE_PX)
        if samples is None:
            continue
        if closed:
            samples = samples[:-1]  # the same point as the first
        corner = np.rint(samples + 0.5).astype(np.int64)  # nearest pixel corner, as (column, row) of the corner grid
        cut = to_junction[np.clip(corner[:, 1], 0, h), np.clip(corner[:, 0], 0, w)] <= smoothing_px
        cut |= (samples < 0.5).any(axis=1) | (samples[:, 0] > w - 1.5) | (samples[:, 1] > h - 1.5)
        kernel = _gaussian_kernel(smoothing_px / spacing)
        if closed and not cut.any():
            length, smoothed = _smoothed_loop_lengths(samples, kernel)
        else:
            if closed:  # start at a cut, so that no piece wraps around the end of the array
                start = int(np.argmax(cut))
                samples, cut = np.roll(samples, -start, axis=0), np.roll(cut, -start)
            pieces = np.flatnonzero(np.diff(np.concatenate([[0], (~cut).astype(np.int8), [0]]))).reshape(-1, 2)
            pieces = pieces[pieces[:, 1] - pieces[:, 0] >= 2]
            length, smoothed = _smoothed_piece_lengths(samples, pieces[:, 0], pieces[:, 1] - pieces[:, 0], kernel)
        total, smoothed_total = total + length, smoothed_total + smoothed
    return total / smoothed_total if smoothed_total else None


def source_edges(image_bgr: np.ndarray, smoothing_px: float, thresholds: tuple[float, float] = EDGE_THRESHOLDS) -> np.ndarray:
    """The source image's edges: Canny on its CIE Lab colors after Gaussian smoothing by ``smoothing_px``.

    ``thresholds`` (low, high) are the heights of the clean, straight color step
    whose gradient would just reach Canny's hysteresis thresholds, in Lab units
    (L from 0 to 100). At each pixel the channel with the steepest gradient
    counts.
    """
    lab = bgr_to_lab(image_bgr).astype(np.float32)
    if smoothing_px > 0:
        lab = cv2.GaussianBlur(lab, (0, 0), smoothing_px)
    # The gradient a unit step reaches after the same smoothing and Sobel filter, so thresholds read as step heights.
    step = np.zeros((1, int(8 * smoothing_px) + 16), dtype=np.float32)
    step[:, step.shape[1] // 2 :] = 1
    if smoothing_px > 0:
        step = cv2.GaussianBlur(step, (0, 0), smoothing_px)
    scale = _EDGE_FIXED_POINT / float(np.abs(cv2.Sobel(step, cv2.CV_32F, 1, 0, ksize=3)).max())
    dx, dy = (
        np.clip(np.rint(cv2.Sobel(lab, cv2.CV_32F, *order, ksize=3) * scale), -32768, 32767).astype(np.int16)
        for order in ((1, 0), (0, 1))
    )
    low, high = thresholds
    return cv2.Canny(dx, dy, low * _EDGE_FIXED_POINT, high * _EDGE_FIXED_POINT, L2gradient=True) > 0


def edge_alignment(region_id_map: np.ndarray, edges: np.ndarray, tolerance_px: float) -> dict[str, float | None]:
    """Do the region boundaries lie on the source's edges?

    ``edge_precision`` is the share of boundary pixels (``boundary_map``) within
    ``tolerance_px`` of an edge pixel, ``edge_recall`` the share of edge pixels
    within it of a boundary pixel, and ``edge_f1`` their harmonic mean. A share
    of nothing is None; F1 is None only when there are neither boundaries nor
    edges.
    """
    boundary = boundary_map(region_id_map)
    precision = _share_near(boundary, edges, tolerance_px)
    recall = _share_near(edges, boundary, tolerance_px)
    if precision is None and recall is None:
        f1 = None
    elif not precision or not recall:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


_SUBPIXEL_BITS = 4  # cv2.polylines draws points given in 1/16 px
_NEIGHBORHOOD_3X3 = np.ones((3, 3), dtype=np.uint8)
_RESAMPLE_PX = 0.5  # jaggedness smooths lines resampled at most this far apart
_EDGE_FIXED_POINT = 16  # Canny takes 16-bit gradients: 1/16 of a Lab unit


def _line_neighborhoods(strokes, size: tuple[int, int]):
    """Each line's ((x0, y0), mask): the pixels in or next to its rasterized centerline, in a box at (x0, y0)."""
    w, h = size
    for stroke in strokes:
        points = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            continue
        x0, y0 = (int(v) for v in np.maximum(np.floor(points.min(axis=0)) - 2, 0))
        x1, y1 = (int(v) for v in np.minimum(np.ceil(points.max(axis=0)) + 3, (w, h)))
        if x1 <= x0 or y1 <= y0:
            continue
        canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        fixed = np.rint((points - (x0, y0)) * 2**_SUBPIXEL_BITS).astype(np.int32)
        if len(fixed) == 1:
            fixed = np.repeat(fixed, 2, axis=0)  # a dot
        cv2.polylines(canvas, [fixed], False, 1, thickness=1, lineType=cv2.LINE_8, shift=_SUBPIXEL_BITS)
        yield (x0, y0), cv2.dilate(canvas, _NEIGHBORHOOD_3X3).astype(bool)


def _distance_to_junctions(ids: np.ndarray) -> np.ndarray:
    """Distance from every pixel corner to the nearest junction, where three regions (or a crossing) meet.

    On the (h + 1) x (w + 1) grid of pixel corners: [i, j] is the corner at
    (x, y) = (j - 0.5, i - 0.5). Outside the page counts as a region of its own.
    """
    padded = np.pad(np.asarray(ids, dtype=np.int64), 1, constant_values=np.iinfo(np.int64).min)
    a, b, c, d = padded[:-1, :-1], padded[:-1, 1:], padded[1:, :-1], padded[1:, 1:]
    distinct = 1 + (b != a) + ((c != a) & (c != b)) + ((d != a) & (d != b) & (d != c))
    junction = (distinct >= 3) | ((a == d) & (b == c) & (a != b))
    if not junction.any():
        return np.full(junction.shape, np.inf)
    return cv2.distanceTransform((~junction).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _without_repeats(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.concatenate([[True], np.any(np.diff(points, axis=0) != 0, axis=1)])
    return points[keep]


def _resample(points: np.ndarray, max_spacing: float) -> tuple[np.ndarray | None, float]:
    """(points evenly spaced along the polyline, at most ``max_spacing`` apart and including both ends; their spacing).

    (None, 0) for a polyline shorter than ``max_spacing``. That keeps the
    spacing above half of ``max_spacing``, and the smoothing kernel it sets
    bounded, however short a line is.
    """
    if len(points) < 2:
        return None, 0.0
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    if along[-1] < max_spacing:
        return None, 0.0
    count = int(np.ceil(along[-1] / max_spacing)) + 1
    at = np.linspace(0.0, along[-1], count)
    return np.column_stack([np.interp(at, along, points[:, 0]), np.interp(at, along, points[:, 1])]), along[-1] / (count - 1)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    offsets = np.arange(-int(np.ceil(4 * sigma)), int(np.ceil(4 * sigma)) + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _smoothed_piece_lengths(
    samples: np.ndarray, starts: np.ndarray, counts: np.ndarray, kernel: np.ndarray
) -> tuple[float, float]:
    """(total length, total length after smoothing with ``kernel``) of the polylines ``samples[start : start + count]``.

    Each piece is extended past its ends by point reflection through them
    (2 end - point), again and again, which keeps its ends in place and a
    straight piece straight. Written as the piece's chord plus its offsets from
    the chord, that extension continues the chord and repeats the offsets
    back and forth, flipping their sign on the way back. All pieces are smoothed
    in one convolution, each padded far enough that the kernel doesn't reach
    the next.
    """
    if len(starts) == 0:
        return 0.0, 0.0
    pad = len(kernel) // 2
    padded_counts = counts + 2 * pad
    piece = np.repeat(np.arange(len(starts)), padded_counts)
    t = np.arange(padded_counts.sum()) - np.repeat(np.cumsum(padded_counts) - padded_counts, padded_counts) - pad
    last = (counts - 1)[piece]
    phase = np.mod(t, 2 * last)
    backwards = phase > last
    local = np.where(backwards, 2 * last - phase, phase)
    first = samples[starts][piece]
    chord_step = ((samples[starts + counts - 1] - samples[starts]) / (counts - 1)[:, None])[piece]
    offset = samples[starts[piece] + local] - (first + chord_step * local[:, None])
    extended = first + chord_step * t[:, None] + np.where(backwards[:, None], -offset, offset)
    smoothed = np.column_stack([np.convolve(extended[:, axis], kernel, mode="same") for axis in (0, 1)])
    inside = (t >= 0) & (t <= last)
    within_piece = piece[inside][1:] == piece[inside][:-1]
    length, smoothed_length = (float(np.hypot(*np.diff(line[inside], axis=0)[within_piece].T).sum()) for line in (extended, smoothed))
    return length, smoothed_length


def _smoothed_loop_lengths(samples: np.ndarray, kernel: np.ndarray) -> tuple[float, float]:
    """(length, length after smoothing with ``kernel``) of a closed polyline, smoothed all the way round."""
    pad = len(kernel) // 2
    extended = samples[np.arange(-pad, len(samples) + pad) % len(samples)]
    smoothed = np.column_stack([np.convolve(extended[:, axis], kernel, mode="valid") for axis in (0, 1)])
    return _loop_length(samples), _loop_length(smoothed)


def _loop_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points, axis=0, append=points[:1]).T).sum())


def _share_near(pixels: np.ndarray, targets: np.ndarray, tolerance_px: float) -> float | None:
    """Share of ``pixels`` within ``tolerance_px`` of a ``targets`` pixel; None without pixels."""
    if not pixels.any():
        return None
    if not targets.any():
        return 0.0
    distance = cv2.distanceTransform((~targets).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float((distance[pixels] <= tolerance_px).mean())


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


def _squared_distance(distance: np.ndarray) -> np.ndarray:
    """Exact squared distances from ``cv2.distanceTransform`` output, which holds their float32 square roots."""
    return np.rint(distance.astype(np.float64) ** 2)


def ink_fraction(page_rgb: np.ndarray) -> float:
    """Share of page pixels that are dark (outlines + numbers): visual clutter."""
    return float((page_rgb.mean(axis=2) < 128).mean())


def page_diff_fraction(page_a: np.ndarray, page_b: np.ndarray) -> float:
    if page_a.shape != page_b.shape:
        return 1.0
    return float(np.any(page_a != page_b, axis=2).mean())


def boundary_map(region_id_map: np.ndarray) -> np.ndarray:
    """1-px region boundaries: pixels whose left or upper neighbor is another region."""
    boundary = np.zeros(region_id_map.shape, dtype=bool)
    boundary[:, 1:] |= region_id_map[:, 1:] != region_id_map[:, :-1]
    boundary[1:, :] |= region_id_map[1:, :] != region_id_map[:-1, :]
    return boundary


def boundary_f1(map_a: np.ndarray, map_b: np.ndarray, tolerance: float = BOUNDARY_TOLERANCE_PX) -> float:
    """Boundary F-measure: do the two partitions draw their lines in the same places?"""
    if map_a.shape != map_b.shape:
        return 0.0
    ba, bb = boundary_map(map_a), boundary_map(map_b)
    if not ba.any() and not bb.any():
        return 1.0
    if not ba.any() or not bb.any():
        return 0.0
    dist_to_b = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_a = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    precision = float((dist_to_b[ba] <= tolerance).mean())
    recall = float((dist_to_a[bb] <= tolerance).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def partition_identical(map_a: np.ndarray, map_b: np.ndarray) -> bool:
    """True if both maps split the image into exactly the same regions (ids may differ)."""
    if map_a.shape != map_b.shape:
        return False
    a = map_a.ravel().astype(np.int64) + 1
    b = map_b.ravel().astype(np.int64) + 1
    pairs = np.unique(a * (int(b.max()) + 1) + b)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))

"""Quality metrics for generated coloring pages.

Two kinds:

* **Absolute** metrics score one result on its own: how faithfully the
  finished painting (every region filled with its legend color) reproduces the
  source image, and how paintable the page is at print size (slivers too thin
  for a brush, unlabeled regions, label size, region shape, leftover undersized
  regions, outline clutter).
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
    ``print_size.PrintScale``. Both are None on a page without numbers.
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
    padded = np.pad(ids, 1, constant_values=-1)
    neighbor_pairs = (
        (padded[:, :-1], padded[:, 1:], 1.0),
        (padded[:-1, :], padded[1:, :], 1.0),
        # Diagonal lines of pixel centers lie 1/√2 apart, rows and columns 1 apart.
        (padded[:-1, :-1], padded[1:, 1:], np.sqrt(0.5)),
        (padded[:-1, 1:], padded[1:, :-1], np.sqrt(0.5)),
    )
    crossings = np.zeros(areas.size)
    for a, b, line_spacing in neighbor_pairs:
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

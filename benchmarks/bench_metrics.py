"""Quality metrics for generated coloring pages.

Two kinds:

* **Absolute** metrics score one result on its own: how faithfully the
  finished painting (every region filled with its legend color) reproduces the
  source image, and how paintable the page is (labeled regions, leftover
  undersized regions, outline clutter).
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


def label_coverage(regions, label_radius: float, total_px: int) -> dict[str, float]:
    """Share of drawn regions (and of page area) that are big enough to carry a number."""
    labeled = [r for r in regions if r.interior_radius >= label_radius]
    return {
        "labeled_region_fraction": len(labeled) / len(regions) if regions else 0.0,
        "labeled_area_fraction": sum(r.area for r in labeled) / total_px if total_px else 0.0,
    }


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

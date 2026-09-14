"""Color quantization: reduce an image to a small palette via k-means in Lab space."""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels, parallel

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


def quantize(image_bgr: np.ndarray, num_colors: int, blur_sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``image_bgr`` to ``num_colors`` flat colors.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order (OpenCV convention).
        num_colors: number of clusters/colors in the output palette.
        blur_sigma: bilateral-filter smoothing strength (0 disables smoothing).
        seed: RNG seed so results are reproducible for the same inputs.

    Returns:
        (label_map, palette_bgr):
            label_map: HxW int32 array, each pixel's cluster index.
            palette_bgr: num_colors x 3 uint8 array of BGR cluster colors,
                ordered from darkest to lightest (by perceptual lightness)
                so numbering reads naturally.
    """
    smoothed = _smooth(image_bgr, blur_sigma)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITERATIONS, _KMEANS_EPSILON)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers_lab = cv2.kmeans(
        samples, num_colors, None, criteria, attempts=_KMEANS_ATTEMPTS, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(lab.shape[:2]).astype(np.int32)
    centers_lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    order = np.argsort(centers_lab[:, 0])  # sort by L (lightness), darkest first
    remap = np.empty(len(order), dtype=np.int32)
    remap[order] = np.arange(len(order), dtype=np.int32)
    labels = remap[labels].astype(np.int32)
    palette_bgr = centers_bgr[order]

    return labels, palette_bgr


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

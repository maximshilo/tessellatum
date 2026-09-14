"""Color quantization: reduce an image to a small palette via k-means in Lab space."""

from __future__ import annotations

import cv2
import numpy as np


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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers_lab = cv2.kmeans(
        samples, num_colors, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
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
    diameter = 0  # derive from sigma
    return cv2.bilateralFilter(image_bgr, diameter, sigmaColor=blur_sigma * 8, sigmaSpace=blur_sigma * 4)

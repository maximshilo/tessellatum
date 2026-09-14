import cv2
import numpy as np

from tessellatum.core.quantize import quantize


def test_quantize_returns_requested_color_count(sample_image_bgr):
    labels, palette = quantize(sample_image_bgr, num_colors=4, blur_sigma=5.0)

    assert labels.shape == sample_image_bgr.shape[:2]
    assert palette.shape == (4, 3)
    assert set(np.unique(labels)).issubset(set(range(4)))


def test_quantize_palette_sorted_by_lightness(sample_image_bgr):
    _, palette = quantize(sample_image_bgr, num_colors=4, blur_sigma=5.0)

    palette_lab = cv2.cvtColor(palette.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lightness = list(palette_lab[:, 0])
    assert lightness == sorted(lightness)


def test_quantize_handles_no_smoothing(sample_image_bgr):
    labels, palette = quantize(sample_image_bgr, num_colors=3, blur_sigma=0.0)

    assert labels.dtype == np.int32
    assert palette.shape == (3, 3)

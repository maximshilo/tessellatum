import cv2
import numpy as np

from tessellatum.core.quantize import _sparse_bilateral, bilateral_filter, quantize


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


def test_sparse_bilateral_sampling_every_pixel_matches_opencv(sample_image_bgr):
    expected = cv2.bilateralFilter(sample_image_bgr, 0, 40.0, 20.0)

    actual = _sparse_bilateral(sample_image_bgr, 40.0, 20.0, spacing=1)

    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 1  # float rounding only
    assert diff.mean() < 0.05


def test_bilateral_filter_stays_close_to_exact_filter(sample_image_bgr):
    # Easy-preset strength, where the sparse lattice is at its coarsest.
    expected = cv2.bilateralFilter(sample_image_bgr, 0, 72.0, 36.0)

    actual = bilateral_filter(sample_image_bgr, sigma_color=72.0, sigma_space=36.0)

    assert actual.shape == expected.shape and actual.dtype == np.uint8
    assert np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean() < 1.0

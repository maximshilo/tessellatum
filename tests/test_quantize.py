import cv2
import numpy as np
import pytest

from tessellatum.core.color import MIN_PALETTE_DE00, pairwise_de00
from tessellatum.core.quantize import (
    _lab_centers_to_bgr,
    _merge_close_colors,
    _sparse_bilateral,
    bilateral_filter,
    quantize,
)

# Two colors a painter cannot tell apart (2.1 ΔE00) and two that stand well clear of them and of each other.
NEAR_BLUE, NEARER_BLUE, GREEN, CYAN = (60, 80, 120), (66, 86, 126), (30, 180, 30), (200, 200, 30)


@pytest.fixture
def two_near_colors_bgr() -> np.ndarray:
    """Four flat blocks, two of which are almost the same color."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50, :50] = NEAR_BLUE
    image[:50, 50:] = NEARER_BLUE
    image[50:, :50] = GREEN
    image[50:, 50:] = CYAN
    return image


def test_quantize_keeps_every_cluster_whose_colors_stand_apart(sample_image_bgr):
    labels, palette = quantize(sample_image_bgr, num_colors=4, blur_sigma=5.0)

    assert labels.shape == sample_image_bgr.shape[:2]
    assert palette.shape == (4, 3)  # the fixture's four blocks are nowhere near each other
    assert set(np.unique(labels)).issubset(set(range(4)))


def test_quantize_merges_colors_closer_than_the_margin(two_near_colors_bgr):
    labels, palette = quantize(two_near_colors_bgr, num_colors=4, blur_sigma=0.0)

    assert len(palette) == 3  # the two blues became one
    assert pairwise_de00(palette).min() >= MIN_PALETTE_DE00
    assert set(np.unique(labels)) == set(range(3))
    # Both blue halves are painted in one color, the one the two average to.
    assert labels[10, 10] == labels[10, 90]
    merged = palette[labels[10, 10]]
    assert merged.tolist() == pytest.approx(np.mean([NEAR_BLUE, NEARER_BLUE], axis=0).tolist(), abs=2)


def test_quantize_without_a_margin_keeps_k_means_colors_as_they_are(two_near_colors_bgr):
    labels, palette = quantize(two_near_colors_bgr, num_colors=4, blur_sigma=0.0, min_de00=0.0)

    assert len(palette) == 4
    assert pairwise_de00(palette).min() < MIN_PALETTE_DE00
    assert set(np.unique(labels)) == set(range(4))


def test_quantize_labels_stay_inside_the_palette_it_returns(two_near_colors_bgr):
    labels, palette = quantize(two_near_colors_bgr, num_colors=12, blur_sigma=0.0)

    assert labels.min() >= 0
    assert labels.max() < len(palette)
    assert len(palette) <= 12


def test_merge_close_colors_joins_the_closest_pair_and_stops_at_the_margin():
    centers = cv2.cvtColor(
        np.array([NEAR_BLUE, NEARER_BLUE, GREEN, CYAN], dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float64)
    weights = np.array([300, 100, 50, 50])

    merged, group = _merge_close_colors(centers, weights, MIN_PALETTE_DE00)

    assert group.tolist() == [0, 0, 2, 3]  # the bigger of the two blues survives
    assert merged[0] == pytest.approx((centers[0] * 300 + centers[1] * 100) / 400)
    assert (merged[2:] == centers[2:]).all()  # colors that stand apart are left alone


def test_merge_close_colors_leaves_a_separated_palette_untouched():
    centers = cv2.cvtColor(
        np.array([NEAR_BLUE, GREEN, CYAN], dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float64)

    merged, group = _merge_close_colors(centers, np.array([10, 10, 10]), MIN_PALETTE_DE00)

    assert group.tolist() == [0, 1, 2]
    assert (merged == centers).all()


def test_merge_close_colors_can_collapse_a_whole_run_of_near_colors():
    ramp = np.linspace(100, 118, 10)  # ten grays a couple of ΔE00 apart
    centers = np.stack([ramp, np.full(10, 128.0), np.full(10, 128.0)], axis=1)

    merged, group = _merge_close_colors(centers, np.ones(10), MIN_PALETTE_DE00)

    kept = np.unique(group)
    assert len(kept) < 10
    assert pairwise_de00(_lab_centers_to_bgr(merged[kept])).min() >= MIN_PALETTE_DE00


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

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
    """Four flat blocks, two of which are almost the same color and cover nine tenths and one tenth of their half."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50, :90] = NEAR_BLUE
    image[:50, 90:] = NEARER_BLUE
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
    # Both blue areas are painted in one color, the one their pixels average to -- which is
    # nine tenths of the way to the bigger one, not the midpoint between the two.
    assert labels[10, 10] == labels[10, 95]
    merged = palette[labels[10, 10]].astype(float)
    by_area = 0.9 * np.array(NEAR_BLUE) + 0.1 * np.array(NEARER_BLUE)
    midpoint = np.mean([NEAR_BLUE, NEARER_BLUE], axis=0)
    assert np.abs(merged - by_area).max() < np.abs(merged - midpoint).max()
    assert merged.tolist() == pytest.approx(by_area.tolist(), abs=1.5)


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

    assert group.tolist() == [0, 0, 2, 3]  # the two blues share a slot; which of them keeps it does not matter
    assert merged[0] == pytest.approx((centers[0] * 300 + centers[1] * 100) / 400)
    assert (merged[2:] == centers[2:]).all()  # colors that stand apart are left alone


def test_merge_close_colors_leaves_a_separated_palette_untouched():
    centers = cv2.cvtColor(
        np.array([NEAR_BLUE, GREEN, CYAN], dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float64)

    merged, group = _merge_close_colors(centers, np.array([10, 10, 10]), MIN_PALETTE_DE00)

    assert group.tolist() == [0, 1, 2]
    assert (merged == centers).all()


def test_lab_centers_outside_the_8_bit_range_are_clipped_rather_than_wrapped():
    # k-means returns the mean of its cluster, and its float accumulation can land
    # just past 255: 255.34 on the reaper at Hard. Casting that to uint8 without
    # clipping first turns a value over 256 into black.
    inside, over = np.array([[255.0, 128.0, 128.0]]), np.array([[256.4, 128.0, 128.0]])

    assert _lab_centers_to_bgr(over).tolist() == _lab_centers_to_bgr(inside).tolist()
    assert _lab_centers_to_bgr(np.array([[-0.4, 128.0, 128.0]])).tolist() == _lab_centers_to_bgr(
        np.array([[0.0, 128.0, 128.0]])
    ).tolist()


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


def _two_fills_and_a_line() -> tuple[np.ndarray, np.ndarray]:
    """Two flat fills either side of a black line 4 px wide, whose edges are anti-aliased: a column half ink either side."""
    green, blue = np.array((60, 160, 60)), np.array((200, 120, 40))
    image = np.zeros((40, 100, 3), dtype=np.uint8)
    image[:, :47] = green
    image[:, 53:] = blue
    image[:, 47] = green // 2  # half ink, half fill
    image[:, 52] = blue // 2
    line = np.zeros((40, 100), dtype=bool)
    line[:, 48:52] = True
    return image, line


def test_the_ink_takes_no_color_and_its_edge_takes_the_color_of_the_fill_it_edges():
    image, line = _two_fills_and_a_line()

    labels, palette = quantize(image, 6, 0.0, ink=line, halo_px=1.0)

    assert (labels[line] == len(palette)).all() and (labels[~line] < len(palette)).all()
    # The fills' colors (to the rounding of 8-bit Lab), and nothing in between: the half-ink columns were never fitted.
    fills = np.array([(60, 160, 60), (200, 120, 40)])
    assert len(palette) == 2
    assert all(np.abs(fills - color).max(axis=1).min() <= 3 for color in palette.astype(int))
    left = labels[:, 0][0]
    right = labels[:, -1][0]
    assert left != right
    assert (labels[:, 47] == left).all() and (labels[:, 52] == right).all()
    # Given no ink, k-means spends colors on the line and its edges.
    _labels, all_colors = quantize(image, 6, 0.0)
    assert len(all_colors) > 2


def test_without_ink_the_colors_are_found_as_before(sample_image_bgr):
    plain = quantize(sample_image_bgr, 5, 1.0)
    no_ink = quantize(sample_image_bgr, 5, 1.0, ink=np.zeros(sample_image_bgr.shape[:2], dtype=bool), halo_px=2.0)

    assert (plain[0] == no_ink[0]).all() and (plain[1] == no_ink[1]).all()

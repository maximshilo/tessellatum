"""The optimized region and render stages must match the original implementation exactly."""

import dataclasses

import cv2
import numpy as np
import pytest

import reference_impl as ref
from tessellatum.core.regions import build_regions, extract_regions
from tessellatum.core.render import render_page


def _blobby_labels(seed: int, shape: tuple[int, int], num_colors: int, blur_sigma: float) -> np.ndarray:
    """Random label map; more blur = bigger blobs, no blur = per-pixel noise."""
    rng = np.random.default_rng(seed)
    field = rng.random(shape).astype(np.float32)
    if blur_sigma > 0:
        field = cv2.GaussianBlur(field, (0, 0), blur_sigma)
    edges = np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])
    return np.digitize(field, edges).astype(np.int32)


CASES = [
    # seed, (h, w), num_colors, blur_sigma, min_area_px, min_width_px
    pytest.param(0, (60, 80), 5, 2.0, 30, 5.0, id="blobs"),
    pytest.param(1, (97, 131), 12, 1.5, 60, 4.0, id="many-colors"),
    pytest.param(2, (40, 50), 3, 0.0, 8, 3.0, id="pixel-noise"),
    pytest.param(3, (73, 64), 8, 3.0, 4, 6.0, id="tiny-threshold"),
    pytest.param(4, (50, 50), 6, 1.0, 0, 0.0, id="no-merging"),
    pytest.param(5, (1, 90), 4, 0.0, 5, 3.0, id="single-row"),
    pytest.param(6, (240, 320), 6, 8.0, 400, 9.0, id="large-labeled-regions"),
]


@pytest.mark.parametrize("seed, shape, num_colors, blur_sigma, min_area_px, min_width_px", CASES)
def test_build_regions_matches_reference(seed, shape, num_colors, blur_sigma, min_area_px, min_width_px):
    labels = _blobby_labels(seed, shape, num_colors, blur_sigma)

    expected_map, expected_color = ref.build_regions(labels, num_colors, min_area_px, min_width_px)
    actual_map, actual_color = build_regions(labels, num_colors, min_area_px, min_width_px)

    np.testing.assert_array_equal(actual_map, expected_map)
    np.testing.assert_array_equal(actual_color, expected_color)


@pytest.mark.parametrize("seed, shape, num_colors, blur_sigma, min_area_px, min_width_px", CASES)
def test_the_brush_width_actually_changes_the_cases(seed, shape, num_colors, blur_sigma, min_area_px, min_width_px):
    # Guard for the comparison above: a width that happened to change nothing
    # would leave the new pass untested.
    labels = _blobby_labels(seed, shape, num_colors, blur_sigma)
    without = _painted(*build_regions(labels, num_colors, min_area_px, 0.0))

    with_brush = _painted(*build_regions(labels, num_colors, min_area_px, min_width_px))

    page_takes_a_brush = min(shape) > min_width_px
    assert (not np.array_equal(with_brush, without)) == (min_width_px > 0 and page_takes_a_brush)


def _painted(region_id_map: np.ndarray, region_color: np.ndarray) -> np.ndarray:
    """Each pixel's color index, -1 outside every region: the page as it will be painted."""
    inside = region_id_map >= 0
    return np.where(inside, region_color[np.where(inside, region_id_map, 0)], -1)


def test_build_regions_matches_reference_with_unlabeled_pixels():
    # Labels outside [0, num_colors) belong to no region: islands cut off by
    # them have no neighbor to merge into, and the brush never reaches them.
    labels = _blobby_labels(7, (45, 55), 4, 1.0)
    labels[10:30, 20:25] = -1
    labels[0, :] = 4

    expected_map, expected_color = ref.build_regions(labels, 4, 12, 5.0)
    actual_map, actual_color = build_regions(labels, 4, 12, 5.0)

    np.testing.assert_array_equal(actual_map, expected_map)
    np.testing.assert_array_equal(actual_color, expected_color)
    np.testing.assert_array_equal(actual_map[10:30, 20:25], -1)


@pytest.mark.parametrize("seed, shape, num_colors, blur_sigma, min_area_px, min_width_px", CASES)
def test_extract_regions_and_render_page_match_reference(
    seed, shape, num_colors, blur_sigma, min_area_px, min_width_px
):
    labels = _blobby_labels(seed, shape, num_colors, blur_sigma)
    region_id_map, region_color = ref.build_regions(labels, num_colors, min_area_px, min_width_px)

    expected = ref.extract_regions(region_id_map, region_color)
    actual = extract_regions(region_id_map, region_color)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert (got.region_id, got.color_index, got.area) == (want.region_id, want.color_index, want.area)
        assert got.interior_point == want.interior_point
        assert got.interior_radius == want.interior_radius
        np.testing.assert_array_equal(got.contour, want.contour)

    size = (shape[1], shape[0])
    rendered = render_page(size, actual)
    np.testing.assert_array_equal(np.asarray(rendered.image), np.asarray(ref.render_page(size, expected)))
    # The outline layer is the page as it would be drawn without numbers.
    without_numbers = [dataclasses.replace(r, interior_radius=0.0) for r in expected]
    np.testing.assert_array_equal(
        np.asarray(rendered.outlines.convert("RGB")), np.asarray(ref.render_page(size, without_numbers))
    )


def test_large_case_actually_draws_numbers():
    # Guard for the render comparison above: make sure some regions are big
    # enough to get a number, so text rendering is really being compared.
    labels = _blobby_labels(6, (240, 320), 6, 8.0)
    region_id_map, region_color = ref.build_regions(labels, 6, 400)

    regions = extract_regions(region_id_map, region_color)

    assert any(r.interior_radius >= 9.0 for r in regions)

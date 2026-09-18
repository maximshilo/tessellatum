"""The optimized region and render stages must match the original implementation exactly."""

import cv2
import numpy as np
import pytest

import reference_impl as ref
from tessellatum.core.labels import FONT_SIZE_RADIUS_RATIO, MAX_FONT_SIZE, _pixels_along, min_font_size, text_box_size
from tessellatum.core.regions import build_regions, extract_regions
from tessellatum.core.render import PageStyle, render_page


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
    rendered = render_page(size, actual, region_id_map)
    expected_lines = ref.trace_boundaries(region_id_map)
    assert len(rendered.strokes) == len(expected_lines)
    for got_line, want_line in zip(rendered.strokes, expected_lines):
        np.testing.assert_array_equal(got_line, want_line)
    np.testing.assert_array_equal(
        np.asarray(rendered.image), np.asarray(ref.render_page(size, expected, region_id_map))
    )
    np.testing.assert_array_equal(
        np.asarray(rendered.outlines), ref.line_layer(size, region_id_map, PageStyle())
    )


@pytest.mark.parametrize("width_px", [1.0, 1.3, 2.0, 2.84, 3.0, 4.75])
def test_the_line_layer_matches_the_reference_at_any_width(width_px):
    # The cases above all print at a few pixels per millimetre, where every line
    # is at its floor; these are the widths a page of ordinary size asks for,
    # including the ones that fall between two steps of the drawing grid.
    labels = _blobby_labels(0, (60, 80), 5, 2.0)
    region_id_map, _color = ref.build_regions(labels, 5, 30, 5.0)
    size = (80, 60)
    style = PageStyle(min_line_width_px=width_px)
    assert style.line_width_px(size) == width_px

    rendered = render_page(size, [], region_id_map, style)

    np.testing.assert_array_equal(np.asarray(rendered.outlines), ref.line_layer(size, region_id_map, style))


# Pages that print at a few pixels per millimetre, as a real page does, so that a number's smallest size and a
# leader's reach are what they are in the app. The pages above are so small that neither comes into play.
LABEL_CASES = [
    # seed, (h, w), num_colors, blur_sigma, min_area_px, min_width_px
    pytest.param(2, (180, 240), 12, 3.0, 40, 6.0, id="leaders"),
    pytest.param(6, (240, 320), 6, 8.0, 400, 9.0, id="roomy"),
]


@pytest.mark.parametrize("seed, shape, num_colors, blur_sigma, min_area_px, min_width_px", LABEL_CASES)
def test_numbers_are_placed_as_the_reference_places_them(seed, shape, num_colors, blur_sigma, min_area_px, min_width_px):
    labels = _blobby_labels(seed, shape, num_colors, blur_sigma)
    region_id_map, region_color = ref.build_regions(labels, num_colors, min_area_px, min_width_px)
    size = (shape[1], shape[0])

    rendered = render_page(size, extract_regions(region_id_map, region_color), region_id_map)

    expected = ref.render_page(size, ref.extract_regions(region_id_map, region_color), region_id_map)
    np.testing.assert_array_equal(np.asarray(rendered.image), np.asarray(expected))


def test_a_number_made_smaller_under_a_thick_line_is_placed_as_the_reference_places_it():
    # A line thick enough to eat the middle of a square that would take a 12 px number there.
    region_id_map = np.zeros((160, 160), dtype=np.int32)
    region_id_map[60:90, 60:90] = 1
    region_color = np.array([0, 1], dtype=np.int32)
    style = PageStyle(line_width_mm=0.0, min_line_width_px=24.0)
    regions = extract_regions(region_id_map, region_color)

    rendered = render_page((160, 160), regions, region_id_map, style)

    square = next(label for label in rendered.labels if label.region_id == 1)
    assert square.font_size < _preferred_size(regions[1], (160, 160)) and square.leader is None
    expected = ref.render_page((160, 160), ref.extract_regions(region_id_map, region_color), region_id_map, style)
    np.testing.assert_array_equal(np.asarray(rendered.image), np.asarray(expected))


def _preferred_size(region, size: tuple[int, int]) -> int:
    return max(min_font_size(size), int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))


def test_the_label_cases_reach_every_way_a_number_can_be_placed():
    # Guard for the comparison above: a way of placing a number that no case reaches would go untested.
    ways = set()
    for seed, shape, num_colors, blur_sigma, min_area_px, min_width_px in (case.values for case in LABEL_CASES):
        region_id_map, region_color = ref.build_regions(
            _blobby_labels(seed, shape, num_colors, blur_sigma), num_colors, min_area_px, min_width_px
        )
        size = (shape[1], shape[0])
        regions = {region.region_id: region for region in extract_regions(region_id_map, region_color)}
        rendered = render_page(size, list(regions.values()), region_id_map)
        ink = np.asarray(rendered.outlines) < 255
        for label in rendered.labels:
            region = regions[label.region_id]
            x0, y0, x1, y1 = (int(v) for v in label.box)
            if label.leader is not None:
                end, anchor = label.leader
                crossed = set(region_id_map[_pixels_along(anchor, end)].tolist()) - {label.region_id, region_id_map[y0, x0]}
                ways.add("a leader crossing another region" if crossed else "a leader")
            elif ink[y0:y1, x0:x1].any():
                ways.add("at its middle, with nowhere else to go")
            else:
                width, height = text_box_size(label.text, _preferred_size(region, size))
                x, y = region.interior_point
                centered = (min(max(x - width // 2, 0), size[0] - width), min(max(y - height // 2, 0), size[1] - height))
                ways.add("at its middle" if (x0, y0) == centered else "moved off its middle")

    assert ways == {
        "at its middle",
        "moved off its middle",
        "a leader",
        "a leader crossing another region",
        "at its middle, with nowhere else to go",
    }

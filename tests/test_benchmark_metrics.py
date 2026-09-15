"""Sanity checks for the benchmark harness's quality metrics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402
from tessellatum.core import difficulty, pipeline, render  # noqa: E402
from tessellatum.core.regions import extract_regions  # noqa: E402

# Reference pairs from Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference formula".
SHARMA_PAIRS = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]

# Pixels within 2.5 px of the center: a 5 x 5 square without its corner pixels.
BRUSH_5PX = 5.0


def test_ciede2000_matches_published_reference_values():
    lab1 = np.array([pair[0] for pair in SHARMA_PAIRS])
    lab2 = np.array([pair[1] for pair in SHARMA_PAIRS])
    expected = [pair[2] for pair in SHARMA_PAIRS]

    assert bm.ciede2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_ssim_is_one_for_identical_images_and_drops_with_noise():
    rng = np.random.default_rng(0)
    gradient = np.linspace(40, 200, 64, dtype=np.float64)
    image = np.repeat(np.tile(gradient, (64, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    noisy = np.clip(image.astype(np.int16) + rng.integers(-40, 40, size=image.shape), 0, 255).astype(np.uint8)

    assert bm.ssim_gray(image, image) == pytest.approx(1.0)
    assert bm.ssim_gray(image, noisy) < 0.9


def test_paint_fills_regions_with_their_palette_color():
    region_id_map = np.array([[0, 1], [1, 2]])
    region_color = np.array([2, 0, 1])
    palette_bgr = np.array([[0, 0, 0], [10, 20, 30], [255, 255, 255]], dtype=np.uint8)

    painted = bm.paint(region_id_map, region_color, palette_bgr)

    assert painted[0, 0].tolist() == [255, 255, 255]
    assert painted[0, 1].tolist() == [0, 0, 0]
    assert painted[1, 1].tolist() == [10, 20, 30]


def test_partition_identical_ignores_region_ids():
    a = np.array([[0, 0, 1], [2, 2, 1]])
    relabeled = np.array([[5, 5, 3], [4, 4, 3]])
    different = relabeled.copy()
    different[1, 0] = 5

    assert bm.partition_identical(a, relabeled)
    assert not bm.partition_identical(a, different)


def test_boundary_f1_tolerates_only_small_shifts():
    a = np.zeros((40, 40), dtype=np.int32)
    a[:, 20:] = 1
    shifted_1px = np.zeros_like(a)
    shifted_1px[:, 21:] = 1
    shifted_6px = np.zeros_like(a)
    shifted_6px[:, 26:] = 1

    assert bm.boundary_f1(a, a) == 1.0
    assert bm.boundary_f1(a, shifted_1px) == 1.0
    assert bm.boundary_f1(a, shifted_6px) == 0.0


def test_label_coverage_counts_the_regions_that_carry_a_number():
    regions = [
        SimpleNamespace(region_id=0, area=60),
        SimpleNamespace(region_id=3, area=30),
        SimpleNamespace(region_id=7, area=10),
    ]

    coverage = bm.label_coverage(regions, {3, 7}, total_px=200)

    assert coverage == pytest.approx({"labeled_region_fraction": 2 / 3, "labeled_area_fraction": 0.2})


def test_unlabeled_regions_are_counted_from_the_region_map():
    # Region 5 might be too small to get an outline, but it is on the page all the same.
    region_id_map = np.array([[0, 0, 2, 2], [5, 5, 2, 2]])

    assert bm.unlabeled_regions(region_id_map, {2, 9}) == {"unlabeled_regions": 2, "unlabeled_area_fraction": 0.5}


def test_label_sizes_are_judged_in_points_at_print_size():
    scale = bm.print_size.PrintScale(size_px=(800, 600), landscape=True, px_per_mm=2 * 72 / 25.4)  # 2 px per pt

    assert bm.label_sizes([26, 10, 13, 20], scale) == pytest.approx({"small_label_fraction": 0.25, "min_label_pt": 5.0})
    assert bm.label_sizes([], scale) == {"small_label_fraction": None, "min_label_pt": None}


def _bands(middle_rows: int) -> np.ndarray:
    """A page 30 px wide: an 8-row band, a band of ``middle_rows`` rows, and another 8-row band."""
    return np.repeat([10] * 8 + [20] * middle_rows + [30] * 8, 30).reshape(-1, 30)


def test_a_bar_as_wide_as_the_brush_loses_only_its_corners():
    wide, narrow = _bands(5), _bands(4)

    assert bm.sliver_mask(wide, BRUSH_5PX)[wide == 20].sum() == 4
    assert bm.sliver_mask(narrow, BRUSH_5PX)[narrow == 20].all()
    # 1 pixel at each corner of the outer bands, and all of the 4 x 30 bar.
    assert bm.sliver_share(narrow, BRUSH_5PX) == (8 + 120) / (20 * 30)


def test_a_ring_is_all_sliver_once_the_brush_is_wider_than_it():
    page = np.ones((20, 20), dtype=np.int32)
    page[5:15, 5:15] = 2  # a 10 x 10 square inside a square ring 5 px wide

    five, seven = bm.sliver_mask(page, BRUSH_5PX), bm.sliver_mask(page, 7.0)

    # A 5 px brush misses only the outer corner pixels of the ring and the corner pixels of the square.
    assert (five[page == 1].sum(), five[page == 2].sum()) == (4, 4)
    # A 7 px brush fits nowhere in the ring, and misses 3 pixels in each corner of the square.
    assert (seven[page == 1].sum(), seven[page == 2].sum()) == (400 - 100, 4 * 3)


def test_a_dumbbell_handle_is_sliver_except_where_the_brush_reaches_in_from_the_ends():
    page = np.zeros((15, 30), dtype=np.int32)
    page[3:12, 3:12] = 1  # two 9 x 9 squares...
    page[3:12, 18:27] = 1
    page[6:9, 12:18] = 1  # ...joined by a 3 px thick, 6 px long handle

    slivers = bm.sliver_mask(page, BRUSH_5PX)

    assert slivers[6:9, 13:17].all()
    assert not slivers[6:9, [12, 17]].any()  # a brush inside a square reaches 1 px into the handle
    assert slivers[page == 1].sum() == 2 * 4 + 3 * 4  # 4 corner pixels per square, and the handle's middle


def _brute_force_slivers(region_id_map: np.ndarray, width: float) -> np.ndarray:
    """``sliver_mask`` straight from its definition: try the brush at every pixel."""
    h, w = region_id_map.shape
    reach = int(width // 2)
    brush = [
        (dy, dx)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if dy * dy + dx * dx <= (width / 2) ** 2
    ]
    painted = np.zeros((h, w), dtype=bool)
    for y, x in np.argwhere(region_id_map >= 0):
        spots = [(y + dy, x + dx) for dy, dx in brush]
        if all(0 <= sy < h and 0 <= sx < w and region_id_map[sy, sx] == region_id_map[y, x] for sy, sx in spots):
            for sy, sx in spots:
                painted[sy, sx] = True
    return (region_id_map >= 0) & ~painted


@pytest.mark.parametrize("seed", range(3))
def test_sliver_mask_matches_trying_the_brush_at_every_pixel(seed):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random((30, 40)).astype(np.float32), (0, 0), 2.0)
    region_id_map = np.digitize(field, np.quantile(field, [0.25, 0.5, 0.75])).astype(np.int32) * 7
    region_id_map[rng.random(region_id_map.shape) < 0.02] = -1  # pixels in no region

    for width in (4.0, 5.0, rng.uniform(2.0, 10.0)):
        np.testing.assert_array_equal(bm.sliver_mask(region_id_map, width), _brute_force_slivers(region_id_map, width))


def test_compactness_of_rectangles_uses_the_crofton_perimeter():
    page = np.zeros((20, 30), dtype=np.int32)
    page[2:15, 2:15] = 1  # a 13 x 13 square
    page[17, 5:15] = 2  # a 1 x 10 bar

    def expected(width: int, height: int) -> float:
        # A rectangle crosses 2 lines per row and per column, and 2 (width + height - 1) along each diagonal.
        crossings = 2 * (width + height) + 2 * (2 * (width + height - 1)) * np.sqrt(0.5)
        return 4 * np.pi * width * height / (np.pi / 8 * crossings) ** 2

    assert bm.compactness(page)[1:] == pytest.approx([expected(13, 13), expected(10, 1)], rel=1e-12)


def test_compactness_is_near_one_for_a_disk_and_does_not_depend_on_orientation():
    yy, xx = np.mgrid[-45:46, -45:46]
    squared_radius = xx**2 + yy**2
    disk = (squared_radius <= 30**2).astype(np.int32)
    ring = ((squared_radius <= 40**2) & (squared_radius > 20**2)).astype(np.int32)
    square = ((np.abs(xx) <= 20) & (np.abs(yy) <= 20)).astype(np.int32)
    diamond = (np.abs(xx) + np.abs(yy) <= 29).astype(np.int32)  # about the same square, turned 45°

    assert bm.compactness(disk)[1] == pytest.approx(1.0, abs=0.03)
    assert bm.compactness(ring)[1] == pytest.approx((40 - 20) / (40 + 20), abs=0.01)  # 4πA/P² of an annulus
    assert bm.compactness(diamond)[1] == pytest.approx(bm.compactness(square)[1], rel=0.03)


def test_compactness_is_capped_at_one_for_tiny_regions():
    page = np.zeros((5, 5), dtype=np.int32)
    page[2, 2] = 1

    assert bm.compactness(page)[1] == 1.0


def test_paintability_metrics_of_a_page_without_regions():
    empty = np.full((4, 6), -1, dtype=np.int32)

    assert bm.sliver_share(empty, BRUSH_5PX) == 0.0
    assert bm.unlabeled_regions(empty, set()) == {"unlabeled_regions": 0, "unlabeled_area_fraction": 0.0}
    assert bm.compactness_stats(empty) == {"compactness_median": None, "compactness_p10": None}


def _drawn_outlines(region_id_map: np.ndarray) -> list[np.ndarray]:
    """The lines today's renderer draws for a region map: every region's contour, as a closed polyline."""
    regions = extract_regions(region_id_map, np.arange(int(region_id_map.max()) + 1, dtype=np.int32))
    contours = [c.reshape(-1, 2).astype(np.float64) for c in render.render_page(region_id_map.shape[::-1], regions).strokes]
    return [np.vstack([c, c[:1]]) for c in contours]


def test_todays_renderer_draws_two_lines_along_a_boundary_where_one_would_do():
    image = np.full((40, 60, 3), 230, dtype=np.uint8)
    image[:, 30:] = 20
    params = difficulty.DifficultyParams(num_colors=2, min_region_fraction=0.01, blur_sigma=0.0)
    analysis = pipeline.generate(image, params, long_edge=60, collect_analysis=True).analysis
    shared_line = np.array([[29.5, 0.0], [29.5, 39.0]])

    assert len(analysis.strokes) == 2  # each region's outline
    assert bm.boundary_lines(analysis.region_id_map, analysis.strokes) == {
        "lines_per_boundary": 2.0,
        "doubled_boundary_fraction": 1.0,
        "undrawn_boundary_fraction": 0.0,
    }
    assert bm.boundary_lines(analysis.region_id_map, [shared_line]) == {
        "lines_per_boundary": 1.0,
        "doubled_boundary_fraction": 0.0,
        "undrawn_boundary_fraction": 0.0,
    }
    assert bm.boundary_lines(analysis.region_id_map, []) == {
        "lines_per_boundary": 0.0,
        "doubled_boundary_fraction": 0.0,
        "undrawn_boundary_fraction": 1.0,
    }


def test_a_line_runs_along_a_boundary_within_a_pixel_of_it_and_counts_once():
    page = np.zeros((20, 20), dtype=np.int32)
    page[:, 10:] = 1  # the boundary lies between columns 9 and 10

    def lines_at(x: float) -> float:
        there_and_back = np.array([[x, 0.0], [x, 19.0], [x, 0.0]])
        return bm.boundary_lines(page, [there_and_back])["lines_per_boundary"]

    assert [lines_at(x) for x in (7, 8, 9, 10, 11, 12)] == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_the_page_edge_is_not_a_boundary():
    page = np.zeros((10, 10), dtype=np.int32)
    frame = np.array([[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0], [0.0, 0.0]])

    assert bm.boundary_lines(page, [frame]) == {
        "lines_per_boundary": None,
        "doubled_boundary_fraction": None,
        "undrawn_boundary_fraction": None,
    }
    assert bm.same_color_boundary_share(page, np.array([0])) is None


def test_same_color_boundary_share_is_the_boundary_between_regions_of_one_color():
    stripes = np.repeat([[0] * 10 + [1] * 10 + [2] * 10], 12, axis=0)  # two boundaries of the same length

    assert bm.same_color_boundary_share(stripes, np.array([4, 4, 7])) == 0.5
    assert bm.same_color_boundary_share(stripes, np.array([4, 5, 7])) == 0.0


def _staircase(step: int, steps: int) -> np.ndarray:
    """A polyline going ``step`` px right, then ``step`` px down, ``steps`` times."""
    return np.array([[10 + step * ((k + 1) // 2), 10 + step * (k // 2)] for k in range(2 * steps + 1)], dtype=np.float64)


def test_jaggedness_is_one_for_a_straight_line_and_about_root_two_for_a_pixel_staircase():
    page = np.zeros((300, 300), dtype=np.int32)
    tilted = np.array([[10.0, 10.0], [250.0, 90.0]])

    assert bm.jaggedness([tilted], page, 2.0) == pytest.approx(1.0, abs=1e-12)
    assert bm.jaggedness([_staircase(1, 200)], page, 2.0) == pytest.approx(np.sqrt(2), abs=0.005)
    assert bm.jaggedness([_staircase(20, 10)], page, 2.0) == pytest.approx(1.063, abs=0.001)


def test_jaggedness_ignores_corners_where_regions_meet():
    three = np.zeros((100, 100), dtype=np.int32)
    three[:50, 50:] = 1
    three[50:, 50:] = 2  # the three regions meet at (49.5, 49.5)
    corner = np.array([[20.0, 49.5], [49.5, 49.5], [49.5, 90.0]])
    yy, xx = np.mgrid[:80, :80]
    grid = (yy // 20 * 4 + xx // 20).astype(np.int32)

    assert bm.jaggedness([corner], three, 2.0) == pytest.approx(1.0, abs=1e-12)
    # The same corner inside one region gets rounded off.
    assert bm.jaggedness([corner], np.zeros_like(three), 2.0) == pytest.approx(1.018, abs=0.001)
    assert bm.jaggedness(_drawn_outlines(grid), grid, 2.0) == pytest.approx(1.0, abs=1e-12)


def test_jaggedness_smooths_a_closed_line_that_meets_no_junction_all_the_way_round():
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.column_stack([100 + 50 * np.cos(angles), 100 + 50 * np.sin(angles)])

    # Smoothing a circle by a Gaussian of standard deviation s shrinks it by a factor exp(-s² / 2r²).
    expected = np.exp(2.0**2 / (2 * 50**2))
    assert bm.jaggedness([np.vstack([circle, circle[:1]])], np.zeros((200, 200), dtype=np.int32), 2.0) == pytest.approx(
        expected, abs=1e-4
    )


def test_line_metrics_skip_empty_lines_and_lines_too_short_to_measure():
    page = np.zeros((40, 40), dtype=np.int32)
    page[:, 20:] = 1
    empty, dot, speck = np.zeros((0, 2)), np.array([[10.0, 10.0]]), np.array([[10.0, 10.0], [10.0001, 10.0]])
    straight = np.array([[5.0, 5.0], [30.0, 5.0]])

    assert bm.boundary_lines(page, [empty])["lines_per_boundary"] == 0.0
    # A line shorter than the 0.5 px resampling step is skipped rather than smoothed with a kernel sized to its length.
    assert bm.jaggedness([empty, dot, speck], page, 2.0) is None
    assert bm.jaggedness([empty, dot, speck, straight], page, 2.0) == pytest.approx(1.0, abs=1e-12)


def _two_halves(left_lab, right_lab) -> np.ndarray:
    """A 60 x 60 image in two halves of the given CIE Lab colors."""
    lab = np.empty((60, 60, 3), dtype=np.float32)
    lab[:, :30], lab[:, 30:] = left_lab, right_lab
    return np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255), 0, 255).astype(np.uint8)


def test_source_edges_are_color_steps_above_the_threshold_in_lab_units():
    def edge_pixels(right_lab) -> int:
        return int(bm.source_edges(_two_halves((50, 0, 0), right_lab), 2.2, thresholds=(5.0, 10.0)).sum())

    assert (edge_pixels((59, 0, 0)), edge_pixels((61, 0, 0))) == (0, 60)  # an edge is one pixel per row
    # A change of hue at the same lightness is an edge too.
    assert (edge_pixels((50, 8, 0)), edge_pixels((50, 12, 0))) == (0, 60)


def test_edge_alignment_scores_boundaries_on_and_off_the_source_edges():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    image[30:70, 30:70] = (128, 100, 160)
    square = np.zeros((100, 100), dtype=np.int32)
    square[30:70, 30:70] = 1
    one_region = np.zeros_like(square)
    edges, no_edges = bm.source_edges(image, 2.2), bm.source_edges(np.full_like(image, 128), 2.2)

    assert bm.edge_alignment(square, edges, 2.2) == {"edge_precision": 1.0, "edge_recall": 1.0, "edge_f1": 1.0}
    assert bm.edge_alignment(np.roll(square, 6, axis=(0, 1)), edges, 2.2)["edge_f1"] < 0.1
    assert bm.edge_alignment(one_region, edges, 2.2) == {"edge_precision": None, "edge_recall": 0.0, "edge_f1": 0.0}
    assert bm.edge_alignment(square, no_edges, 2.2) == {"edge_precision": 0.0, "edge_recall": None, "edge_f1": 0.0}
    assert bm.edge_alignment(one_region, no_edges, 2.2) == {"edge_precision": None, "edge_recall": None, "edge_f1": None}


def _gray_lightness(value: int) -> float:
    """CIE L* of the sRGB gray ``value``, from the sRGB and CIELAB definitions."""
    c = value / 255
    y = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 116 * np.cbrt(y) - 16 if y > (6 / 29) ** 3 else y * (29 / 3) ** 3


def _gray_de00(value1: int, value2: int) -> float:
    """CIEDE2000 between two sRGB grays: without chroma, only the lightness term is left."""
    l1, l2 = _gray_lightness(value1), _gray_lightness(value2)
    mid = ((l1 + l2) / 2 - 50) ** 2
    return abs(l1 - l2) / (1 + 0.015 * mid / np.sqrt(20 + mid))


def test_exact_lab_conversion_gives_published_values_and_grays_without_chroma():
    red_green_blue_white_black = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    published = [(53.24, 80.09, 67.20), (87.73, -86.18, 83.18), (32.30, 79.19, -107.86), (100, 0, 0), (0, 0, 0)]
    grays = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)

    # Published values use the sRGB matrix unrounded; the standard's 4-digit one moves red's a* and b* by 0.02.
    assert bm.bgr_to_lab_exact(red_green_blue_white_black) == pytest.approx(np.array(published, dtype=np.float64), abs=0.05)
    lab = bm.bgr_to_lab_exact(grays)
    assert lab[:, 0] == pytest.approx([_gray_lightness(v) for v in range(256)], abs=1e-9)
    assert np.abs(lab[:, 1:]).max() < 1e-9


def test_palette_separation_is_the_smallest_color_difference_and_the_pairs_closer_than_the_minimum():
    grays = [0, 255, 100, 118, 128, 160]
    palette = np.repeat(np.array(grays, dtype=np.uint8)[:, None], 3, axis=1)

    # Closer than 10 ΔE00: 100 and 118 (7.0), 118 and 128 (3.9). Not: 128 and 160 (10.8), 100 and 128 (11.1).
    assert bm.palette_separation(palette) == pytest.approx(
        {"palette_min_de00": _gray_de00(118, 128), "palette_close_pairs": 2}, abs=1e-9
    )
    assert bm.palette_separation(palette, min_de00=11.0)["palette_close_pairs"] == 3


def test_palette_separation_counts_every_close_pair_including_identical_colors_and_needs_two_colors():
    gray, black = [128, 128, 128], [0, 0, 0]
    five_near_grays = np.array([[100 + i] * 3 for i in range(5)], dtype=np.uint8)

    assert bm.palette_separation(np.array([gray, black, gray], dtype=np.uint8)) == {"palette_min_de00": 0.0, "palette_close_pairs": 1}
    assert bm.palette_separation(five_near_grays)["palette_close_pairs"] == 10
    assert bm.palette_separation(np.array([gray], dtype=np.uint8)) == {"palette_min_de00": None, "palette_close_pairs": 0}
    assert bm.palette_separation(np.zeros((0, 3), dtype=np.uint8)) == {"palette_min_de00": None, "palette_close_pairs": 0}


WHITE, FILL, BLACK = (255, 255, 255), (90, 150, 230), (0, 0, 0)  # BGR


def test_source_ink_is_the_pixels_in_an_ink_color_narrower_than_the_widest_line():
    image = np.full((60, 90, 3), WHITE, dtype=np.uint8)
    image[10:50, 10:50] = BLACK
    image[16:44, 16:44] = FILL  # a fill inside a black outline 6 px wide
    image[10:50, 60:84] = BLACK  # a black shape 24 px wide: a fill drawn in the ink color
    outline = np.zeros(image.shape[:2], dtype=bool)
    outline[10:50, 10:50] = True
    outline[16:44, 16:44] = False

    ink = bm.source_ink(image, np.array([WHITE, FILL], dtype=np.uint8), np.array([BLACK], dtype=np.uint8), 15.0)

    np.testing.assert_array_equal(ink[:, :55], outline[:, :55])
    assert not ink[18:42, 68:76].any()  # a 15 px disk fits into the wide shape...
    assert ink[10, 60] and ink[49, 83]  # ...but can't reach its corners
    assert not bm.source_ink(image, np.array([WHITE, FILL], dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8), 15.0).any()


def test_anti_aliasing_between_two_ink_colors_is_ink_even_where_a_flat_color_is_nearer():
    # The bold-line girl's colors: black and dark brown ink, and a dark brown fill close to both.
    flats = np.array([WHITE, (10, 20, 36)], dtype=np.uint8)  # #ffffff, #24140a
    inks = np.array([BLACK, (0, 17, 43)], dtype=np.uint8)  # #000000, #2b1100
    blend = np.array([0, 10, 25], dtype=np.uint8)  # #190a00, where the two inks meet
    image = np.full((20, 20, 3), WHITE, dtype=np.uint8)
    image[5:15, 5:8] = inks[0]
    image[5:15, 8] = blend
    image[5:15, 9:12] = inks[1]
    lab = bm.bgr_to_lab_exact(np.vstack([flats, inks, blend]))
    to_fill, to_black, to_brown = (float(bm.ciede2000(lab[4], lab[k])) for k in (1, 2, 3))

    assert to_fill < min(to_black, to_brown)  # 3.9 against 7.6 and 8.1
    assert bm.source_ink(image, flats, inks, 15.0)[5:15, 5:12].all()


def test_centerlines_thin_shapes_to_their_middle_one_pixel_wide():
    bar = np.zeros((25, 50), dtype=bool)
    bar[10:15, 10:40] = True  # 5 px wide
    blocks = np.zeros((20, 30), dtype=bool)
    blocks[3:5, 3:5] = True
    blocks[8:15, 15:22] = True
    yy, xx = np.mgrid[-30:31, -30:31]
    ring = (xx**2 + yy**2 > 14**2) & (xx**2 + yy**2 <= 20**2)

    # The middle row, shortened at each end as thinning eats into it.
    assert np.argwhere(bm.centerlines(bar)).tolist() == [[12, x] for x in range(12, 37)]
    assert np.argwhere(bm.centerlines(blocks)).tolist() == [[11, 18]]  # a 7 x 7 square thins to its center; 2 x 2 vanishes
    loop = bm.centerlines(ring)
    neighbors = cv2.filter2D(loop.astype(np.uint8), -1, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT) - loop
    assert (neighbors[loop] == 2).all()  # a closed loop
    assert np.abs(np.hypot(xx, yy)[loop] - 17).max() < 0.6  # halfway between the ring's radii
    assert not bm.centerlines(np.zeros((5, 5), dtype=bool)).any()


def test_ink_line_match_wants_lines_down_the_middle_of_the_ink_and_ignores_lines_away_from_it():
    ink = np.zeros((60, 80), dtype=bool)
    ink[20:31, 10:70] = True  # an ink line 11 px wide; its centerline is row 25, from x = 15 to 63
    middle, top_edge, bottom_edge, far_away = (np.array([[15.0, y], [63.0, y]]) for y in (25, 20, 30, 50))

    def match(strokes, ink=ink):
        return bm.ink_line_match(strokes, ink, 2.0)

    assert match([middle]) == {"ink_line_precision": 1.0, "ink_line_recall": 1.0, "ink_line_f1": 1.0}
    # A tube's outlines run along both edges of the line, 5 px from its middle.
    assert match([top_edge, bottom_edge]) == {"ink_line_precision": 0.0, "ink_line_recall": 0.0, "ink_line_f1": 0.0}
    # A line between two fills, 20 px from the ink, counts for nothing.
    assert match([middle, top_edge, bottom_edge, far_away]) == pytest.approx(
        {"ink_line_precision": 1 / 3, "ink_line_recall": 1.0, "ink_line_f1": 0.5}, abs=1e-12
    )
    assert match([far_away]) == {"ink_line_precision": None, "ink_line_recall": 0.0, "ink_line_f1": 0.0}
    assert match([middle], np.zeros_like(ink)) == {"ink_line_precision": None, "ink_line_recall": None, "ink_line_f1": None}


def test_tube_regions_are_ink_lines_turned_into_shapes_to_paint():
    ink = np.zeros((60, 80), dtype=bool)
    ink[20:31, 20:60] = True  # an ink line 11 px wide and 40 px long
    own = np.ones(ink.shape, dtype=np.int32)
    own[31:] = 3
    own[ink] = 2  # the line is a region of its own, between two fills
    half = own.copy()
    half[31:42, 20:60] = 2  # ...with as much fill again
    split = np.ones(ink.shape, dtype=np.int32)
    split[26:] = 3  # the fills meet down the middle of the line
    left_out = np.where(ink, -1, own)  # the page keeps the line out of every region
    merged = np.full(ink.shape, 2, dtype=np.int32)
    merged[:, :20] = 1
    merged[ink] = 1  # the line sticks out of a wide fill

    def tubes(ids, ink=ink):
        return bm.tube_regions(ids, ink, 15.0)

    assert tubes(own) == {"tube_regions": 1, "tube_ink_fraction": 1.0}
    assert tubes(half)["tube_regions"] == 1
    assert tubes(split) == tubes(left_out) == {"tube_regions": 0, "tube_ink_fraction": 0.0}
    # No region of its own, but paint all the same, except where a disk inside the fill reaches into it.
    assert tubes(merged)["tube_regions"] == 0 and 0.9 < tubes(merged)["tube_ink_fraction"] < 1
    assert tubes(own, np.zeros_like(ink)) == {"tube_regions": 0, "tube_ink_fraction": None}


def test_todays_renderer_turns_a_bold_ink_outline_into_a_tube():
    image = np.full((120, 160, 3), WHITE, dtype=np.uint8)
    image[20:100, 30:130] = BLACK
    image[28:92, 38:122] = FILL  # a flat fill inside a black outline 8 px wide
    params = difficulty.DifficultyParams(num_colors=3, min_region_fraction=0.001, blur_sigma=0.0)
    analysis = pipeline.generate(image, params, long_edge=160, collect_analysis=True).analysis
    ink = bm.source_ink(image, np.array([WHITE, FILL], dtype=np.uint8), np.array([BLACK], dtype=np.uint8), 15.0)
    one_line = np.array([[33.5, 23.5], [125.5, 23.5], [125.5, 95.5], [33.5, 95.5], [33.5, 23.5]])

    assert ink.sum() == 80 * 100 - 64 * 84
    # The outline becomes a region of its own, outlined along its edges, 3.5 px or more from its middle.
    assert bm.tube_regions(analysis.region_id_map, ink, 15.0) == {"tube_regions": 1, "tube_ink_fraction": 1.0}
    assert bm.ink_line_match(analysis.strokes, ink, 2.0) == {"ink_line_precision": 0.0, "ink_line_recall": 0.0, "ink_line_f1": 0.0}
    assert bm.ink_line_match([one_line], ink, 2.0) == {"ink_line_precision": 1.0, "ink_line_recall": 1.0, "ink_line_f1": 1.0}


def test_flat_color_match_is_the_difference_from_each_flat_color_to_the_nearest_legend_color():
    def grays(*values):
        return np.repeat(np.array(values, dtype=np.uint8)[:, None], 3, axis=1).reshape(-1, 3)

    assert bm.flat_color_match(grays(100, 200), grays(200, 110)) == pytest.approx(
        {"flat_color_de00_mean": _gray_de00(100, 110) / 2, "flat_color_de00_max": _gray_de00(100, 110)}, abs=1e-9
    )
    nothing = {"flat_color_de00_mean": None, "flat_color_de00_max": None}
    assert bm.flat_color_match(grays(), grays(100)) == bm.flat_color_match(grays(100), grays()) == nothing


def test_face_fidelity_scores_the_painting_inside_the_face_boxes_and_counts_overlaps_once():
    source = np.full((60, 90, 3), 100, dtype=np.uint8)
    inside = source.copy()
    inside[10:30, 10:20] = 110  # the left half of the 20 x 20 face box
    both = inside.copy()
    both[:, 60:] = 200  # outside the face, beyond the reach of SSIM's window
    face, overlapping = (10, 10, 20, 20), (15, 15, 10, 10)
    one_pixel = float(bm.ciede2000(bm.bgr_to_lab(source[:1, :1]), bm.bgr_to_lab(inside[10:11, 10:11]))[0, 0])

    scores = bm.face_fidelity(source, inside, [face])
    assert scores["face_de00_mean"] == pytest.approx(one_pixel / 2, rel=1e-12)
    assert scores["face_ssim"] < 1
    assert bm.face_fidelity(source, both, [face]) == scores
    assert bm.face_fidelity(source, inside, [face, overlapping]) == scores
    assert bm.face_fidelity(source, source, [face]) == {"face_de00_mean": 0.0, "face_ssim": 1.0}
    assert bm.face_fidelity(source, inside, []) == {"face_de00_mean": None, "face_ssim": None}


def test_ssim_gray_is_the_mean_of_the_ssim_map():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-30, 30, size=a.shape), 0, 255).astype(np.uint8)

    assert bm.ssim_map(a, b).shape == (30, 40)
    assert bm.ssim_gray(a, b) == float(bm.ssim_map(a, b).mean())


def test_a_feature_survives_as_lines_along_its_edges_or_as_a_drawn_region_of_its_own():
    background = np.zeros((40, 60), dtype=np.int32)
    edges = np.zeros(background.shape, dtype=bool)
    edges[10, 10:20] = True  # 10 edge pixels inside the feature box
    box = (5, 5, 20, 10)  # 200 px

    def score(ids=background, strokes=(), drawn=(0,), edges=edges):
        return bm.feature_survival([box], ids, set(drawn), list(strokes), edges, 0.5)[0]

    assert score() == {"edge_recall": 0.0, "region_share": 0.0, "survived": False}
    # A line along 3 of the 10 edge pixels is enough, along 2 it isn't.
    assert score(strokes=[np.array([[10.0, 10.0], [12.0, 10.0]])]) == {"edge_recall": 0.3, "region_share": 0.0, "survived": True}
    assert score(strokes=[np.array([[10.0, 10.0], [11.0, 10.0]])])["survived"] is False

    eye = background.copy()
    eye[8:13, 8:18] = 1  # 50 px inside the box: a quarter of it
    small_eye = background.copy()
    small_eye[8:12, 8:18] = 1  # 40 px
    assert score(eye, drawn=(0, 1)) == {"edge_recall": 0.0, "region_share": 0.25, "survived": True}
    assert score(small_eye, drawn=(0, 1)) == {"edge_recall": 0.0, "region_share": 0.2, "survived": False}
    assert score(eye, drawn=(0,))["survived"] is False  # a region without an outline isn't on the page

    half_inside = background.copy()
    half_inside[5:15, 20:30] = 1  # 100 px, 50 of them inside the box
    more_outside = half_inside.copy()
    more_outside[20, 20] = 1
    assert score(half_inside, drawn=(0, 1))["region_share"] == 0.25
    assert score(more_outside, drawn=(0, 1))["region_share"] == 0.0

    no_edges = np.zeros_like(edges)
    assert score(eye, drawn=(0, 1), edges=no_edges) == {"edge_recall": None, "region_share": 0.25, "survived": True}
    assert score(edges=no_edges) == {"edge_recall": None, "region_share": 0.0, "survived": False}


def test_lost_features_counts_the_features_that_did_not_survive_and_averages_their_edge_recall():
    scores = [
        {"edge_recall": 0.5, "region_share": 0.0, "survived": True},
        {"edge_recall": None, "region_share": 0.0, "survived": False},
        {"edge_recall": 0.1, "region_share": 0.1, "survived": False},
    ]

    assert bm.lost_features(scores) == pytest.approx({"features_lost": 2, "feature_edge_recall": 0.3}, abs=1e-12)
    assert bm.lost_features(scores[1:2]) == {"features_lost": 1, "feature_edge_recall": None}
    assert bm.lost_features([]) == {"features_lost": None, "feature_edge_recall": None}


def test_todays_pipeline_loses_small_face_features_at_a_coarse_setting():
    background, skin, dark, nose, mouth = (230, 200, 150), (150, 190, 235), (40, 40, 40), (90, 120, 170), (60, 60, 190)
    image = np.full((240, 240, 3), background, dtype=np.uint8)
    cv2.circle(image, (120, 125), 95, skin, -1)
    for center in ((85, 105), (155, 105)):
        cv2.ellipse(image, center, (14, 8), 0, 0, 360, dark, -1)  # eyes of about 350 px
    cv2.ellipse(image, (120, 135), (6, 5), 0, 0, 360, nose, -1)  # about 95 px
    cv2.ellipse(image, (120, 170), (28, 8), 0, 0, 360, mouth, -1)  # about 700 px
    face, features = (25, 30, 190, 190), [(65, 92, 40, 26), (135, 92, 40, 26), (110, 126, 20, 18), (88, 158, 64, 24)]
    edges = bm.source_edges(image, 2.0)

    def score(min_region_fraction):
        params = difficulty.DifficultyParams(num_colors=5, min_region_fraction=min_region_fraction, blur_sigma=0.0)
        analysis = pipeline.generate(image, params, long_edge=240, collect_analysis=True).analysis
        drawn = {region.region_id for region in analysis.regions}
        painted = bm.paint(analysis.region_id_map, analysis.region_color, analysis.palette_bgr)
        survival = bm.feature_survival(features, analysis.region_id_map, drawn, analysis.strokes, edges, 2.0)
        return [feature["survived"] for feature in survival], bm.face_fidelity(image, painted, [face])["face_de00_mean"]

    # Regions under 576 px merge into a neighbor: the eyes and the nose melt into the skin. Under 58 px, they keep their shapes.
    coarse, coarse_de00 = score(0.01)
    fine, fine_de00 = score(0.001)
    assert coarse == [False, False, False, True]
    assert fine == [True, True, True, True]
    assert coarse_de00 > fine_de00


def test_labels_on_boxes_counts_each_number_overlapping_a_box_once():
    eyes = [(10, 10, 20, 10), (40, 10, 20, 10)]
    labels = [
        (0.0, 0.0, 10.0, 10.0),  # touches the first eye's corner
        (29.5, 12.0, 40.5, 18.0),  # across both eyes
        (50.0, 19.0, 55.0, 25.0),  # 1 px into the second eye
        (60.0, 10.0, 70.0, 20.0),  # beside the second eye
    ]

    assert bm.labels_on_boxes(labels, eyes) == 2
    assert bm.labels_on_boxes(labels, []) == 0
    assert bm.labels_on_boxes([], eyes) == 0

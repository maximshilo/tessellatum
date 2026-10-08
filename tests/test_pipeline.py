import math
import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import boundaries, difficulty, faces, ink, labels, pipeline, print_size, render, subject, text
from tessellatum.core.legend import render_legend
from tessellatum.core.painting import Version, tint_rgb
from tessellatum.core.color import MIN_PALETTE_DE00, pairwise_de00
from tessellatum.core.pipeline import Handling, PipelineCancelled, generate


def test_generate_produces_page_and_legend(sample_image_bgr):
    params = difficulty.params_for_preset("Medium")
    result = generate(sample_image_bgr, params, long_edge=200)

    assert result.page.size == (200, 200)
    assert result.legend.width > 0 and result.legend.height > 0
    assert result.num_colors_used > 0
    assert result.num_regions > 0
    assert len(result.palette_rgb) == result.num_colors_used
    assert result.num_colors_used <= params.num_colors


def test_generate_respects_difficulty_color_count(sample_image_bgr):
    easy = generate(sample_image_bgr, difficulty.params_for_preset("Easy"), long_edge=200)
    hard = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200)

    assert easy.num_colors_used <= difficulty.params_for_preset("Easy").num_colors
    assert hard.num_colors_used <= difficulty.params_for_preset("Hard").num_colors


def test_generate_reports_monotonic_progress_to_100(sample_image_bgr):
    params = difficulty.params_for_preset("Medium")
    reported = []

    generate(sample_image_bgr, params, long_edge=200, progress_callback=reported.append)

    assert reported == sorted(reported)
    assert reported[0] > 0
    assert reported[-1] == 100


def test_generate_can_be_cancelled_mid_pipeline(sample_image_bgr):
    params = difficulty.params_for_preset("Medium")
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 2  # cancel on the second check, mid-pipeline

    with pytest.raises(PipelineCancelled):
        generate(sample_image_bgr, params, long_edge=200, should_cancel=should_cancel)


def test_generate_can_be_cancelled_while_the_numbers_are_placed(sample_image_bgr, monkeypatch):
    # Numbering a page of thousands of regions takes seconds, all in one stage: Abort is heard between two numbers.
    params = difficulty.params_for_preset("Medium")
    polled, started, finished = [], [], []
    real = render.place_labels

    def spy(*args, **kwargs):
        started.append(len(polled))
        result = real(*args, **kwargs)
        finished.append(result)
        return result

    monkeypatch.setattr(render, "place_labels", spy)

    def should_cancel():
        polled.append(True)
        return bool(started) and len(polled) == started[0] + 2  # before the second number

    pipeline.clear_cache()
    with pytest.raises(PipelineCancelled):
        generate(sample_image_bgr, params, long_edge=200, should_cancel=should_cancel)
    assert started and not finished
    page = generate(sample_image_bgr, params, long_edge=200, should_cancel=lambda: False)
    assert page.num_regions >= 2 and finished


def test_generate_reuses_quantization_across_region_size_changes(sample_image_bgr, monkeypatch):
    quantize_calls = []
    real_quantize = pipeline.quantize

    def counting_quantize(*args, **kwargs):
        quantize_calls.append(args)
        return real_quantize(*args, **kwargs)

    monkeypatch.setattr(pipeline, "quantize", counting_quantize)
    pipeline.clear_cache()
    medium = difficulty.params_for_preset("Medium")
    finer = difficulty.DifficultyParams(medium.num_colors, medium.min_region_area_mm2 / 4, medium.blur_sigma)

    first = generate(sample_image_bgr, medium, long_edge=200)
    generate(sample_image_bgr, finer, long_edge=200)
    again = generate(sample_image_bgr, medium, long_edge=200)
    assert len(quantize_calls) == 1
    assert np.array_equal(np.asarray(again.page), np.asarray(first.page))

    # A different image object is never served from the cache, even if equal.
    generate(sample_image_bgr.copy(), medium, long_edge=200)
    assert len(quantize_calls) == 2


def test_warm_up_runs_the_pipeline_without_error():
    pipeline.warm_up()


def test_analysis_is_collected_only_on_request_and_leaves_the_page_unchanged(sample_image_bgr):
    params = difficulty.params_for_preset("Hard")

    plain = generate(sample_image_bgr, params, long_edge=200)
    analyzed = generate(sample_image_bgr, params, long_edge=200, collect_analysis=True)

    assert plain.analysis is None
    assert analyzed.analysis is not None
    assert np.array_equal(np.asarray(analyzed.page), np.asarray(plain.page))


def test_analysis_regions_colors_and_labels_match_the_page(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True)
    analysis = result.analysis

    assert analysis.region_id_map.shape == analysis.outlines.shape == (200, 200)
    legend_bgr = analysis.palette_bgr[: analysis.legend_size]
    assert [tuple(int(c) for c in bgr[::-1]) for bgr in legend_bgr] == result.palette_rgb
    assert len(analysis.regions) == result.num_regions
    for region in analysis.regions:
        x, y = region.interior_point
        assert analysis.region_id_map[y, x] == region.region_id
        assert analysis.region_color[region.region_id] == region.color_index < analysis.legend_size

    # One line per boundary, not one outline per region, so the lines run along
    # pixel cracks -- smoothed off them by at most a pixel, and never off the page.
    assert len(analysis.strokes) > len(analysis.regions)
    for stroke in analysis.strokes:
        assert stroke.shape[1:] == (2,) and len(stroke) >= 2
        off_the_cracks = np.abs(stroke - 0.5 - np.rint(stroke - 0.5))
        assert np.hypot(*off_the_cracks.T).max() <= boundaries.MAX_SHIFT_PX + 1e-9
        assert stroke.min() >= -0.5 and stroke.max() <= 199.5

    # Every drawn region carries its number, on the page, at least as large as the paper needs.
    drawn = {r.region_id: r for r in analysis.regions}
    assert sorted(label.region_id for label in analysis.labels) == sorted(drawn)
    smallest = labels.min_font_size((200, 200))
    for label in analysis.labels:
        assert label.text == str(drawn[label.region_id].color_index + 1)
        assert smallest <= label.font_size <= max(smallest, labels.MAX_FONT_SIZE)
        x0, y0, x1, y1 = label.box
        assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 200
    assert analysis.leaders.shape == (200, 200) and analysis.leaders.dtype == np.uint8


def test_every_region_carries_a_number_clear_of_the_lines(sample_image_bgr):
    for preset in ("Easy", "Hard"):
        result = generate(sample_image_bgr, difficulty.params_for_preset(preset), long_edge=400, collect_analysis=True)
        analysis = result.analysis
        ink = (analysis.outlines != render.PAPER) | (analysis.leaders != render.PAPER)
        scale = print_size.print_scale((400, 400))

        regions = set(np.unique(analysis.region_id_map[analysis.region_id_map >= 0]).tolist())
        assert {label.region_id for label in analysis.labels} == regions
        for label in analysis.labels:
            x0, y0, x1, y1 = (int(v) for v in label.box)
            assert not ink[y0:y1, x0:x1].any()
            assert scale.px_to_pt(label.font_size) >= print_size.MIN_LABEL_SIZE_PT


def test_page_is_the_line_layer_inked_in_gray_plus_the_numbers(sample_image_bgr):
    style = render.PageStyle()
    result = generate(
        sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True, style=style
    )
    page = np.asarray(result.page)
    outlines = result.analysis.outlines

    near_numbers = np.zeros(outlines.shape, dtype=bool)
    for label in result.analysis.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        near_numbers[y0:y1, x0:x1] = True  # a number sits on whole pixels, and its ink stays inside its box

    # Away from the numbers the page is white paper with the line gray laid on
    # it as thickly as the line layer says, and the numbers' gray as thickly as
    # the leaders' layer does, the darker of the two where they meet.
    def inked(layer: np.ndarray, gray: int) -> np.ndarray:
        ink = render.PAPER - layer.astype(np.float64)
        return np.rint(render.PAPER - ink * ((render.PAPER - gray) / render.PAPER)).astype(np.uint8)

    inked = np.minimum(inked(outlines, style.line_gray), inked(result.analysis.leaders, style.label_gray))
    assert near_numbers.any()
    np.testing.assert_array_equal(page[~near_numbers], np.repeat(inked[~near_numbers][:, None], 3, axis=1))
    assert (page[near_numbers] != inked[near_numbers][:, None]).any()  # the numbers are really there


def test_the_line_layer_says_where_the_ink_is_whatever_tone_it_is_printed_in(sample_image_bgr):
    params = difficulty.params_for_preset("Hard")
    kwargs = dict(long_edge=200, collect_analysis=True)

    default = generate(sample_image_bgr, params, **kwargs)
    black = generate(sample_image_bgr, params, **kwargs, style=render.PageStyle(line_gray=0, label_gray=0))

    np.testing.assert_array_equal(default.analysis.outlines, black.analysis.outlines)
    assert not np.array_equal(np.asarray(default.page), np.asarray(black.page))


def test_analysis_lists_legend_colors_first(speckled_image_bgr):
    params = difficulty.DifficultyParams(num_colors=3, min_region_area_mm2=480.0, blur_sigma=0.0)

    result = generate(speckled_image_bgr, params, long_edge=80, collect_analysis=True)
    analysis = result.analysis

    # The specks' gray, the middle of the three quantized colors, was merged
    # away: it is left off the legend and moves to the end of the palette.
    assert (result.num_colors_used, analysis.legend_size, len(analysis.palette_bgr)) == (2, 2, 3)
    assert abs(int(analysis.palette_bgr[2, 0]) - 128) <= 3
    painted = analysis.palette_bgr[analysis.region_color[analysis.region_id_map]].astype(int)
    halves = np.where(np.arange(80) < 40, 20, 230)[None, :, None]
    assert np.abs(painted - halves).max() <= 3


def test_analysis_arrays_are_not_the_cached_ones(sample_image_bgr):
    params = difficulty.params_for_preset("Medium")
    pipeline.clear_cache()
    first = generate(sample_image_bgr, params, long_edge=200, collect_analysis=True)
    first.analysis.palette_bgr[:] = 0

    again = generate(sample_image_bgr, params, long_edge=200, collect_analysis=True)  # quantized colors from the cache

    assert again.palette_rgb == first.palette_rgb


def test_every_two_colors_on_the_legend_stand_clearly_apart():
    # A narrow band of browns cannot hold 20 colors a painter could tell
    # apart, so the page comes back with fewer of them rather than with a
    # legend of near-identical swatches.
    ramp = np.linspace(0, 1, 200)[None, :, None]
    gradient = (np.array([40, 60, 80]) + ramp * np.array([50, 50, 50])).repeat(200, axis=0).astype(np.uint8)
    params = difficulty.DifficultyParams(num_colors=20, min_region_area_mm2=36.0, blur_sigma=1.0)

    result = generate(gradient, params, long_edge=200, collect_analysis=True)

    legend = result.analysis.palette_bgr[: result.analysis.legend_size]
    assert result.num_colors_used < params.num_colors
    assert pairwise_de00(legend).min() >= MIN_PALETTE_DE00


def test_no_boundary_separates_two_regions_of_one_color(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True)
    ids = result.analysis.region_id_map.astype(np.int64)
    color = result.analysis.region_color

    # Every pair of 8-adjacent pixels: neighboring regions must differ in color.
    for a, b in (
        (ids[:, :-1], ids[:, 1:]),
        (ids[:-1, :], ids[1:, :]),
        (ids[:-1, :-1], ids[1:, 1:]),
        (ids[:-1, 1:], ids[1:, :-1]),
    ):
        differ = (a != b) & (a >= 0) & (b >= 0)
        assert differ.any()
        assert not (color[a[differ]] == color[b[differ]]).any()


def test_a_line_too_thin_to_paint_is_not_a_region_on_the_page():
    # Two blocks split by a line 2 px wide. The page prints at about 1 px per
    # mm, so the line is narrower than the 3 mm brush and cannot be painted.
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :100] = (200, 60, 60)
    image[:, 100:] = (60, 180, 60)
    image[:, 99:101] = (20, 20, 20)
    params = difficulty.DifficultyParams(num_colors=3, min_region_area_mm2=print_size.MIN_REGION_AREA_MM2, blur_sigma=0.0)

    result = generate(image, params, long_edge=200, collect_analysis=True)

    assert result.num_regions == result.num_colors_used == 2
    assert (20, 20, 20) not in result.palette_rgb
    ids = result.analysis.region_id_map
    assert (ids[:, :99] == ids[0, 0]).all() and (ids[:, 101:] == ids[0, -1]).all()


def test_the_region_limits_come_from_the_printed_page():
    params = difficulty.DifficultyParams(num_colors=4, min_region_area_mm2=40.0, blur_sigma=0.0, min_width_mm=4.0)
    scale = print_size.print_scale((200, 200))

    min_area_px, min_width_px = pipeline._paintable_limits(params, (200, 200))

    assert min_area_px == round(scale.mm2_to_px(40.0))
    assert min_width_px == scale.mm_to_px(4.0)

    # The brush is the one asked for, and no region is smaller than its footprint: at the default brush, the printed
    # page's own limit.
    default_brush = difficulty.DifficultyParams(num_colors=4, min_region_area_mm2=1.0, blur_sigma=0.0)
    min_area_px, min_width_px = pipeline._paintable_limits(default_brush, (1100, 778))
    scale = print_size.print_scale((1100, 778))
    assert min_width_px == scale.mm_to_px(print_size.MIN_PAINTABLE_WIDTH_MM)
    assert min_area_px == round(scale.mm2_to_px(print_size.MIN_REGION_AREA_MM2)) > 4

    fine = difficulty.DifficultyParams(num_colors=4, min_region_area_mm2=1.0, blur_sigma=0.0, min_width_mm=2.0)
    min_area_px, min_width_px = pipeline._paintable_limits(fine, (1100, 778))
    assert min_width_px == scale.mm_to_px(2.0)
    assert min_area_px == round(scale.mm2_to_px(math.pi)) > 4  # a 2 mm disk

    # No brush at all, and no region under 4 pixels.
    none = difficulty.DifficultyParams(num_colors=4, min_region_area_mm2=0.01, blur_sigma=0.0, min_width_mm=0.0)
    assert pipeline._paintable_limits(none, (1100, 778)) == (4, 0.0)


def test_a_preview_and_an_export_are_held_to_the_same_sizes_on_paper():
    params = difficulty.params_for_preset("Hard")
    preview_area, preview_width = pipeline._paintable_limits(params, (1100, 825))
    export_area, export_width = pipeline._paintable_limits(params, (2200, 1650))

    assert export_width == 2 * preview_width
    assert abs(export_area - 4 * preview_area) <= 4  # each rounded to a whole pixel

    # A panorama prints smaller than a picture of ordinary proportions, so its
    # smallest region is a larger share of it: it gets fewer regions, not
    # smaller ones.
    panorama_area, _ = pipeline._paintable_limits(params, (1100, 275))
    assert panorama_area / (1100 * 275) > 2 * preview_area / (1100 * 825)


def _line_drawing(width: int, height: int) -> np.ndarray:
    """Flat fills outlined in black, the outline about 1 mm wide on paper at this size."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    line = max(2, round(print_size.print_scale((width, height)).mm_to_px(1.0)))
    for i, color in enumerate([(90, 150, 230), (60, 160, 60), (200, 120, 40), (180, 180, 250)]):
        x0, y0 = (i % 2) * width // 2, (i // 2) * height // 2
        image[y0 : y0 + height // 2, x0 : x0 + width // 2] = 0
        image[y0 + line : y0 + height // 2 - line, x0 + line : x0 + width // 2 - line] = color
    return image


def test_analysis_says_whether_the_picture_is_line_art_and_where_its_ink_lines_are():
    drawing = _line_drawing(300, 220)

    result = generate(drawing, difficulty.params_for_preset("Hard"), long_edge=300, collect_analysis=True)

    analysis = result.analysis
    assert analysis.line_art == ink.line_art(drawing)
    assert analysis.line_art.is_line_art
    assert analysis.ink_lines.shape == (220, 300) and analysis.ink_lines.dtype == bool
    assert (analysis.ink_lines == (drawing == 0).all(axis=2)).all()  # every outline, and nothing else


def test_a_picture_that_is_not_line_art_has_no_ink_lines(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True)

    assert not result.analysis.line_art.is_line_art
    assert result.analysis.ink_lines.shape == (200, 200) and not result.analysis.ink_lines.any()


def test_line_art_is_decided_on_the_picture_at_preview_size(monkeypatch):
    monkeypatch.setattr(pipeline, "PREVIEW_LONG_EDGE", 300)
    drawing = _line_drawing(450, 330)
    params = difficulty.params_for_preset("Hard")

    preview = generate(drawing, params, long_edge=300, collect_analysis=True).analysis
    export = generate(drawing, params, long_edge=450, collect_analysis=True).analysis
    smaller = generate(drawing, params, long_edge=200, collect_analysis=True).analysis

    at_preview_size = ink.line_art(pipeline.resize_to_long_edge(drawing, 300))
    assert preview.line_art == export.line_art == smaller.line_art == at_preview_size
    assert export.ink_lines.shape == (330, 450) and smaller.ink_lines.shape == (147, 200)
    assert (export.ink_lines == ink.ink_lines(drawing)).all()


def _outlined_shapes() -> tuple[np.ndarray, dict]:
    """A 600 x 800 drawing (3.16 px/mm on paper): four flat fills outlined in black 4 px (1.3 mm) wide, and two small
    shapes the ink encloses on white: a "finger" 22 px (7 mm) across, and a speck 6 px (1.9 mm) across."""
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    for i, color in enumerate([(90, 150, 230), (60, 160, 60), (200, 120, 40), (180, 180, 250)]):
        x0, y0 = 40 + (i % 2) * 270, 40 + (i // 2) * 300
        image[y0 : y0 + 250, x0 : x0 + 250] = 0
        image[y0 + 4 : y0 + 246, x0 + 4 : x0 + 246] = color
    shapes = {"finger": (slice(660, 682), slice(100, 122)), "speck": (slice(660, 666), slice(300, 306))}
    for rows, columns in shapes.values():
        image[rows.start - 4 : rows.stop + 4, columns.start - 4 : columns.stop + 4] = 0
        image[rows, columns] = (90, 150, 230)
    return image, shapes


def test_line_art_prints_its_ink_and_paints_the_areas_it_encloses():
    drawing, shapes = _outlined_shapes()

    result = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True)

    analysis = result.analysis
    assert analysis.line_art.is_line_art
    printed = analysis.printed_ink
    # Every outline, as found: the found ink is all of it here (the speck's inner corners too, where 0.5 mm of gap
    # closing bridges the outline's diagonal).
    assert (printed == analysis.ink_lines).all() and printed[(drawing == 0).all(axis=2)].all()
    assert analysis.ink_gray == 0  # the drawing's own ink is black
    # The outlines are 1.3 mm wide, thinner than ink.THIN_INK_MM: the regions on either side reach to their middle,
    # where a seam a pixel wide keeps them from touching. Across the first square's left outline (columns 40-43):
    across = analysis.region_id_map[150, 38:46]
    background, fill = int(across[0]), int(across[-1])
    assert background >= 0 and fill >= 0 and background != fill
    assert list(across[:3]) == [background] * 3 and list(across[5:]) == [fill] * 3
    assert sorted([int(across[3]), int(across[4])]) in ([-1, background], [-1, fill]) and -1 in across
    ids = analysis.region_id_map  # and no two regions touch anywhere: every fill here is outlined
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :]), (ids[:-1, :-1], ids[1:, 1:]), (ids[:-1, 1:], ids[1:, :-1])):
        assert not ((a != b) & (a >= 0) & (b >= 0)).any()
    page = np.asarray(result.page.convert("L"))
    assert (page[printed] == 0).all()
    # Each fill is one region, and so is the finger, though it is far below Easy's 300 mm²: the ink encloses it alone.
    finger = analysis.region_id_map[shapes["finger"]]
    assert (finger >= 0).all() and len(np.unique(finger)) == 1
    labeled = {label.region_id for label in analysis.labels}
    assert int(finger[0, 0]) in labeled
    assert result.num_regions == 6  # the four fills, the finger, the white around them
    # No brush fits in the speck: it is left as bare paper, neither a region nor ink.
    speck = shapes["speck"]
    assert (analysis.region_id_map[speck] == -1).all()
    assert not printed[speck][1:-1, :].any() and not printed[speck][:, 1:-1].any()
    assert (page[speck][~printed[speck]] == 255).all()
    # No number is on the ink, and no line runs beside it: the ink is the line.
    for label in analysis.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert (analysis.outlines[y0:y1, x0:x1] == 255).all()
    assert (analysis.outlines[printed] == 0).all()


def test_a_picture_that_is_not_line_art_prints_no_ink(sample_image_bgr):
    analysis = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True).analysis

    assert not analysis.printed_ink.any() and analysis.ink_gray == 0
    assert (analysis.region_id_map >= 0).all()


def test_the_ink_is_found_once_per_picture_and_size(monkeypatch):
    drawing, _shapes = _outlined_shapes()
    calls = []
    find = ink.find_ink
    monkeypatch.setattr(ink, "find_ink", lambda *args, **kwargs: calls.append(1) or find(*args, **kwargs))

    for preset in ("Easy", "Medium", "Hard"):
        generate(drawing, difficulty.params_for_preset(preset), long_edge=800)

    assert len(calls) == 1


def test_the_ink_prints_in_the_drawing_s_own_tone():
    drawing, _shapes = _outlined_shapes()
    drawing[(drawing == 0).all(axis=2)] = 60  # outlines in dark gray instead of black

    result = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True)

    analysis = result.analysis
    assert analysis.ink_gray == 60
    assert (np.asarray(result.page.convert("L"))[analysis.printed_ink] == 60).all()


def test_the_ink_s_edge_is_left_out_by_at_least_a_pixel_and_the_enclosed_shapes_are_printed(monkeypatch):
    drawing, _shapes = _outlined_shapes()  # 3.16 px/mm on paper: 0.25 mm of edge is less than a pixel
    seen = {}
    real_quantize, real_join = pipeline.quantize, pipeline.join_ink

    def quantize_spy(*args, **kwargs):
        seen["halo_px"] = kwargs["halo_px"]
        return real_quantize(*args, **kwargs)

    def join_spy(labels, num_colors, image, ink_bgr, min_area_px, own=None):
        seen["own"] = own
        return real_join(labels, num_colors, image, ink_bgr, min_area_px, own)

    settled = np.zeros((800, 600), dtype=bool)
    settled[5:10, 5:10] = True  # what settle_enclosed says to print, made up here

    def settle_spy(ids, *args, **kwargs):
        return np.where(settled, -1, ids).astype(np.int32), settled

    monkeypatch.setattr(pipeline, "quantize", quantize_spy)
    monkeypatch.setattr(pipeline, "join_ink", join_spy)
    monkeypatch.setattr(pipeline, "settle_enclosed", settle_spy)
    result = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True)

    assert seen["halo_px"] == 1.0
    assert (seen["own"] == ~ink.near(result.analysis.ink_lines, 1.0)).all()  # colors judged off the ink's edge
    assert result.analysis.printed_ink[settled].all()
    assert (np.asarray(result.page.convert("L"))[settled] == result.analysis.ink_gray).all()


def test_a_line_art_legend_offers_the_artwork_s_own_colors():
    drawing, _shapes = _outlined_shapes()

    analysis = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True).analysis

    # Easy asks for 6 colors and the drawing has 5, so every one of them is on the legend, exactly as painted --
    # not the mean of a fill and the blends along its edges, which is what minimizing distance lands on.
    legend = analysis.palette_bgr[: analysis.legend_size]
    drawn = [(255, 255, 255), (90, 150, 230), (60, 160, 60), (200, 120, 40), (180, 180, 250)]
    assert len(legend) == len(drawn)
    for color in legend:
        assert np.abs(np.asarray(drawn, dtype=int) - color.astype(int)).max(axis=1).min() <= 3


def test_only_line_art_takes_the_flat_color_palette(sample_image_bgr, monkeypatch):
    calls = []
    original = pipeline.quantize

    def spy(*args, **kwargs):
        calls.append(kwargs.get("ink") is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "quantize", spy)
    pipeline.clear_cache()
    generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200)
    pipeline.clear_cache()
    generate(_outlined_shapes()[0], difficulty.params_for_preset("Easy"), long_edge=400)

    assert calls == [False, True]  # a photograph's colors still come from k-means; a drawing's from its fills


def _hatched_sheet() -> np.ndarray:
    """A 400 x 300 sheet (1.44 px/mm on paper) hatched from row 6 down with black strokes 2 px (1.4 mm) wide, 2 px apart:
    a band of paper 6 px tall along the top joins the gaps, too short for a number."""
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    for x in range(0, 400, 4):
        image[6:, x : x + 2] = 0
    return image


def test_paint_goes_over_hatching_and_a_number_on_it_clears_its_strokes():
    sheet = _hatched_sheet()

    result = generate(sheet, difficulty.params_for_preset("Easy"), long_edge=400, collect_analysis=True)

    analysis = result.analysis
    assert analysis.line_art.is_line_art
    ids = analysis.region_id_map
    assert result.num_regions == 1 and (ids == 0).all()  # the strokes are thin: the paint goes over them, gaps and all
    (label,) = analysis.labels
    assert label.clears and label.leader is None  # no room on the paper anywhere: the strokes under it are cleared
    x0, y0, x1, y1 = (int(v) for v in label.box)
    hatched = (sheet == 0).all(axis=2)
    gap = int(np.ceil(render.PageStyle().line_width_px((400, 300))))  # a line's width round the number
    cleared_area = (slice(max(0, y0 - gap), y1 + gap), slice(max(0, x0 - gap), x1 + gap))
    assert hatched[cleared_area].any()  # strokes ran under it
    assert (analysis.outlines[y0:y1, x0:x1] == 255).all()  # no line or ink in its box
    assert not analysis.printed_ink[cleared_area].any()  # nor in the line's width round it
    assert analysis.printed_ink[hatched].mean() > 0.99  # every other stroke is printed, as the page shows
    page = np.asarray(result.page.convert("L"))
    assert (page[analysis.printed_ink] == analysis.ink_gray).all()
    assert (page[hatched & ~analysis.printed_ink] > 0).all()


def test_the_paint_goes_over_ink_thinner_than_thin_ink_mm_and_the_numbers_keep_off_it(monkeypatch):
    drawing, _shapes = _outlined_shapes()
    seen = {}
    real_paint, real_detail, real_extract = pipeline.paint_over_thin_ink, pipeline.detail_ink, pipeline.extract_regions

    def paint_spy(ids, colors, printed, max_width_px, image, palette, own=None):
        seen["paint"] = (printed.copy(), max_width_px, own)
        return real_paint(ids, colors, printed, max_width_px, image, palette, own)

    def detail_spy(ids, printed, reach_px, apart_px):
        seen["reach_px"], seen["apart_px"] = reach_px, apart_px
        return real_detail(ids, printed, reach_px, apart_px)

    def extract_spy(ids, colors, printed=None):
        seen["extract_printed"] = printed
        return real_extract(ids, colors, printed=printed)

    monkeypatch.setattr(pipeline, "paint_over_thin_ink", paint_spy)
    monkeypatch.setattr(pipeline, "detail_ink", detail_spy)
    monkeypatch.setattr(pipeline, "extract_regions", extract_spy)
    analysis = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True).analysis

    thin_px = print_size.print_scale((600, 800)).mm_to_px(ink.THIN_INK_MM)
    printed, width, own = seen["paint"]
    assert width == thin_px and seen["reach_px"] == thin_px / 2  # detail ink: none of a line's halves
    assert seen["apart_px"] == render.PageStyle().line_width_px((600, 800))  # nor where two regions meet unlined
    assert (printed == analysis.printed_ink).all()  # every number fits here, so nothing is cleared
    assert (own == ~ink.near(analysis.ink_lines, 1.0)).all()  # paper's color judged off the ink's edge
    assert seen["extract_printed"] is not None and (seen["extract_printed"] == printed).all()  # numbers off the ink


# --- T3.4b: hatched patches, white areas, numbers with no room -------------------------------------------------------


SAMPLE_IMAGES = Path(__file__).resolve().parent / "sample_images"


def _white_areas_per_region(analysis) -> np.ndarray:
    """For each region, how many white areas a brush fits in it holds: 4-connected runs of its unprinted pixels."""
    ids, printed = analysis.region_id_map, analysis.printed_ink
    radius = analysis.min_paintable_width_px / 2
    counts = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    for rid in np.unique(ids[ids >= 0]).tolist():
        ys, xs = np.nonzero(ids == rid)
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        white = ((ids[box] == rid) & ~printed[box]).astype(np.uint8)
        _count, parts = cv2.connectedComponents(white, connectivity=4)
        distance = cv2.distanceTransform(np.pad(white, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        counts[rid] = len(np.unique(parts[(distance > radius) & (white > 0)]))
    return counts


@pytest.mark.parametrize("preset, long_edge", [("Max", 1100), ("Hard", 2400)])
def test_a_densely_hatched_scan_numbers_every_white_area_and_no_number_sits_on_a_line(preset, long_edge):
    image = pipeline.load_image_bgr(SAMPLE_IMAGES / "m-cartoon-complex.jpg")
    params = difficulty.finest_params() if preset == "Max" else difficulty.params_for_preset(preset)

    analysis = generate(image, params, long_edge=long_edge, collect_analysis=True).analysis

    assert analysis.line_art.is_line_art
    assert (_white_areas_per_region(analysis) <= 1).all()  # every white area a brush fits in has a number of its own
    numbered = {label.region_id for label in analysis.labels}
    assert numbered == set(np.unique(analysis.region_id_map[analysis.region_id_map >= 0]).tolist())
    for label in analysis.labels:  # none written where it found no room: every one clear of lines and ink
        x0, y0, x1, y1 = label.box
        box = (slice(max(0, int(np.floor(y0))), int(np.ceil(y1))), slice(max(0, int(np.floor(x0))), int(np.ceil(x1))))
        assert not label.cramped and (analysis.outlines[box] == 255).all()


def test_a_hatched_patch_is_painted_in_its_own_gaps_color_not_the_color_beyond_its_strokes():
    sky, orange = (230, 200, 160), (60, 140, 230)
    image = np.full((600, 800, 3), sky, dtype=np.uint8)
    image[150:450, 200:600] = orange
    for x in range(200, 600, 6):
        image[150:450, x : x + 2] = 0  # strokes 2 px wide, 4 px apart, open at both ends to the sky

    analysis = generate(image, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True).analysis

    assert analysis.line_art.is_line_art
    gaps = analysis.region_id_map[200:400, 203:596:6]  # the middle of every gap, well inside the patch
    assert len(np.unique(gaps)) == 1  # one area, strokes and all
    painted = analysis.palette_bgr[analysis.region_color[gaps[0, 0]]].astype(int)
    assert np.abs(painted - orange).sum() < np.abs(painted - sky).sum()


def test_a_one_pixel_diagonal_line_keeps_the_two_areas_it_divides_apart_and_is_printed_whole():
    drawing, _shapes = _outlined_shapes()
    y0, x0 = 44, 44  # inside the first square's outline
    steps = np.arange(242)
    drawing[y0 + steps, x0 + steps] = 0  # a line one pixel wide, corner to corner

    analysis = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True).analysis

    ids, printed = analysis.region_id_map, analysis.printed_ink
    below, above = ids[y0 + 180, x0 + 40], ids[y0 + 40, x0 + 180]
    assert below >= 0 and above >= 0 and below != above  # two areas, two regions ...
    assert {below, above} <= {label.region_id for label in analysis.labels}  # ... each with its number
    along = np.zeros(ids.shape, dtype=bool)
    middle = steps[20:-20]  # its ends make tips against the outline no brush reaches, left as paper (Q22)
    along[y0 + middle, x0 + middle] = True
    along = cv2.dilate(along.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert not (along & (ids < 0) & ~printed).any()  # what keeps them apart is printed, not left as a speck of paper


def test_the_pixels_splitting_takes_for_ink_are_printed(monkeypatch):
    drawing, _shapes = _outlined_shapes()
    real_split = pipeline.split_areas
    taken = {}

    def split_spy(ids, colors, printed, min_width_px, min_area_px, num_colors):
        split, split_colors, corners = real_split(ids, colors, printed, min_width_px, min_area_px, num_colors)
        spare = np.argwhere((split < 0) & ~printed & ~corners)  # a pixel of bare paper, to hand back as a corner
        taken["at"] = tuple(spare[0])
        corners = corners.copy()
        corners[taken["at"]] = True
        return split, split_colors, corners

    monkeypatch.setattr(pipeline, "split_areas", split_spy)
    analysis = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True).analysis

    assert analysis.printed_ink[taken["at"]]


def test_an_export_renders_at_300_dpi_on_a4_and_never_upscales():
    # D-052 (Q36): the long edge that prints at 300 dpi on A4, by the picture's shape, or the picture's own.
    big = np.zeros((3000, 4500, 3), dtype=np.uint8)
    assert pipeline.export_long_edge(big) == 3272
    page = pipeline.resize_to_long_edge(big, pipeline.export_long_edge(big))
    assert print_size.print_scale(page.shape[1::-1]).dpi == pytest.approx(300, abs=0.2)
    assert pipeline.export_long_edge(np.zeros((2400, 2400, 3), dtype=np.uint8)) == 2244  # 300 dpi is enough
    assert pipeline.export_long_edge(np.zeros((1000, 1500, 3), dtype=np.uint8)) == 1500  # never upscaled


def test_the_legend_is_drawn_at_the_page_s_print_scale(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Easy"), long_edge=200)

    palette_bgr = np.array([rgb[::-1] for rgb in result.palette_rgb], dtype=np.uint8)
    expected = render_legend(palette_bgr, result.page.width, print_size.print_scale(result.page.size).px_per_mm)
    assert result.legend.size == expected.size and result.legend.tobytes() == expected.tobytes()


def test_every_step_runs_by_default(sample_image_bgr):
    assert Handling() == Handling(line_art=True, detail=True, text=True)
    params = difficulty.params_for_preset("Medium")

    plain = generate(sample_image_bgr, params, long_edge=200)
    explicit = generate(sample_image_bgr, params, long_edge=200, handling=Handling(line_art=True, detail=True, text=True))
    assert plain.page.tobytes() == explicit.page.tobytes()


def test_with_line_art_off_a_drawing_is_drawn_from_its_colors_and_the_cache_keeps_the_two_apart():
    # D-052 (Q36). Its colors are quantized without holding out the ink, so the cache must not hand back the colors
    # taken around it: the page with line art off is the same after a page with it on as before one.
    drawing, _shapes = _outlined_shapes()
    params = difficulty.params_for_preset("Easy")
    pipeline.clear_cache()
    cold = generate(drawing, params, long_edge=800, collect_analysis=True, handling=Handling(line_art=False))
    on = generate(drawing, params, long_edge=800, collect_analysis=True).analysis
    off = generate(drawing, params, long_edge=800, collect_analysis=True, handling=Handling(line_art=False))
    pipeline.clear_cache()

    assert on.line_art.is_line_art and off.analysis.line_art.is_line_art  # still found, no longer used
    assert on.printed_ink.any() and (on.region_id_map < 0).any()
    assert not off.analysis.printed_ink.any() and (off.analysis.region_id_map >= 0).all()  # every pixel painted
    assert off.page.tobytes() == cold.page.tobytes()
    np.testing.assert_array_equal(off.analysis.region_id_map, cold.analysis.region_id_map)


def test_with_detail_off_no_face_or_subject_is_looked_for_and_the_page_is_drawn_as_if_none_were_there(
    sample_image_bgr, monkeypatch
):
    # D-052 (Q36).
    calls = []
    face = faces.Face(box=(50.0, 50.0, 100.0, 100.0), score=0.9, detector="yunet")
    monkeypatch.setattr(faces, "find_faces", lambda picture: calls.append("faces") or [face])
    monkeypatch.setattr(subject, "find_subject", lambda picture: calls.append("subject") or np.ones((40, 40), np.float32))
    params = difficulty.params_for_preset("Hard")
    pipeline.clear_cache()
    off = generate(sample_image_bgr, params, long_edge=200, handling=Handling(detail=False))
    assert calls == []
    on = generate(sample_image_bgr, params, long_edge=200, collect_analysis=True).analysis
    assert sorted(calls) == ["faces", "subject"] and on.detail.any()

    monkeypatch.setattr(faces, "find_faces", lambda picture: [])
    monkeypatch.setattr(subject, "find_subject", lambda picture: np.zeros((40, 40), np.float32))
    pipeline.clear_cache()
    nothing = generate(sample_image_bgr, params, long_edge=200)
    pipeline.clear_cache()
    assert off.page.tobytes() == nothing.page.tobytes()


def test_with_text_off_no_text_is_looked_for_and_no_lettering_is_printed(monkeypatch):
    # D-052 (Q36). A dark band of "lettering" the stubbed finder calls a line of text.
    picture = np.full((300, 400, 3), 235, dtype=np.uint8)
    picture[140:150, 60:340:6] = 30
    line = text.TextLine(quad=((50.0, 130.0), (350.0, 130.0), (350.0, 160.0), (50.0, 160.0)), score=0.9)
    calls = []
    monkeypatch.setattr(text, "find_text", lambda picture, source=None, max_height_mm=None: calls.append(1) or [line])
    params = difficulty.params_for_preset("Easy")
    pipeline.clear_cache()
    off = generate(picture, params, long_edge=400, collect_analysis=True, handling=Handling(text=False))
    assert calls == []
    assert off.analysis.text == [] and not off.analysis.lettering_area.any()
    on = generate(picture, params, long_edge=400, collect_analysis=True)
    assert calls == [1] and on.analysis.lettering_area.any()

    monkeypatch.setattr(text, "find_text", lambda picture, source=None, max_height_mm=None: [])
    pipeline.clear_cache()
    nothing = generate(picture, params, long_edge=400)
    pipeline.clear_cache()
    assert off.page.tobytes() == nothing.page.tobytes()


def test_the_page_comes_with_what_it_was_drawn_from_to_draw_it_again_off_the_pixel_grid(monkeypatch):
    # T7.3: the vector PDF draws the page again from these (see export.save_pdf).
    style = render.PageStyle.from_settings(0.45, "Dark")
    drawing, _ = _outlined_shapes()
    result = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True, style=style)
    drawn, analysis = result.drawing, result.analysis
    assert drawn.size == result.page.size and drawn.style == style
    assert drawn.strokes is analysis.strokes and drawn.labels is analysis.labels
    np.testing.assert_array_equal(drawn.ink, analysis.printed_ink)  # line art's ink, all of it printed solid
    assert drawn.ink_gray == analysis.ink_gray == 0 and drawn.lettering is None

    # On a picture with a line of text, its letters are printed there, in place of the printed ink lying in a region,
    # and drawn again from their outline.
    picture = np.full((300, 400, 3), 235, dtype=np.uint8)
    picture[140:150, 60:340:6] = 30
    line = text.TextLine(quad=((50.0, 130.0), (350.0, 130.0), (350.0, 160.0), (50.0, 160.0)), score=0.9)
    monkeypatch.setattr(text, "find_text", lambda picture, source=None, max_height_mm=None: [line])
    pipeline.clear_cache()
    result = generate(picture, difficulty.params_for_preset("Easy"), long_edge=400, collect_analysis=True)
    drawn, analysis = result.drawing, result.analysis
    letters = text.lettering(pipeline.resize_to_long_edge(picture, 400), [line])
    assert len(drawn.lettering) == len(letters.outline) == len(range(60, 340, 6))  # a ring round each stroke
    for got, want in zip(drawn.lettering, letters.outline):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(analysis.lettering, render.PAPER - letters.ink)  # the analysis has it as paper
    assert analysis.printed_ink.any() and analysis.lettering_area.any()
    in_region = (analysis.region_id_map >= 0) & analysis.lettering_area
    np.testing.assert_array_equal(drawn.ink, analysis.printed_ink & ~in_region)


def test_the_page_comes_with_what_it_is_painted_with_legend_colors_first(speckled_image_bgr):
    result = generate(speckled_image_bgr, difficulty.params_for_preset("Easy"), long_edge=80, collect_analysis=True)

    painting, analysis = result.painting, result.analysis
    assert painting.region_id_map is analysis.region_id_map
    np.testing.assert_array_equal(painting.region_color, analysis.region_color)
    assert painting.palette_rgb == [tuple(int(v) for v in bgr[::-1]) for bgr in analysis.palette_bgr]
    assert painting.palette_rgb[: len(result.palette_rgb)] == result.palette_rgb  # paint i is numbered i + 1
    for region in analysis.regions:
        assert painting.region_color[region.region_id] == region.color_index


def test_the_completed_version_is_the_painting_the_benchmarks_score():
    # Line art: its ink printed solid over the paint, the speck left as bare paper (see the line art test above).
    drawing, shapes = _outlined_shapes()
    result = generate(drawing, difficulty.params_for_preset("Easy"), long_edge=800, collect_analysis=True)

    analysis = result.analysis
    expected = np.array(analysis.palette_bgr[:, ::-1])[analysis.region_color][np.clip(analysis.region_id_map, 0, None)]
    expected[analysis.region_id_map < 0] = 255
    expected[analysis.printed_ink] = analysis.ink_gray
    completed = result.image(Version.COMPLETED)
    np.testing.assert_array_equal(np.asarray(completed), expected)
    assert (np.asarray(completed)[shapes["speck"]][~analysis.printed_ink[shapes["speck"]]] == 255).all()
    assert result.image(Version.COMPLETED) is completed  # drawn once, and kept
    assert result.image(Version.PAGE) is result.image() is result.page


def test_the_tinted_version_is_the_page_under_a_wash_of_its_paint(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Medium"), long_edge=200, collect_analysis=True)

    analysis, page = result.analysis, np.asarray(result.page).astype(int)
    tinted = np.asarray(result.image(Version.TINTED)).astype(int)
    wash = np.array([tint_rgb(tuple(bgr[::-1])) for bgr in analysis.palette_bgr])[analysis.region_color]
    wash = np.vstack([wash, [255, 255, 255]])[analysis.region_id_map]  # -1, in no region, takes the last: paper
    np.testing.assert_array_equal(tinted, np.rint(page * wash / 255))
    paper = (page == 255).all(axis=2)
    assert paper.any() and (tinted[paper] == wash[paper]).all()  # where the page is bare, the wash alone


def _two_close_grays_and_red() -> np.ndarray:
    picture = np.zeros((300, 400, 3), dtype=np.uint8)
    picture[:, :150] = 100
    picture[:, 150:300] = 115
    picture[:, 300:] = (40, 40, 200)
    return picture


def test_the_palette_margin_is_a_setting_and_the_colors_found_are_cached_by_it():
    picture = _two_close_grays_and_red()
    gap = float(pairwise_de00(np.array([(100,) * 3, (115,) * 3], dtype=np.uint8))[0, 1])
    assert 4.0 < gap < MIN_PALETTE_DE00  # the two grays are closer than the palette's margin
    easy = difficulty.custom_params(4, 30.0, 0.0)
    pipeline.clear_cache()
    merged = generate(picture, easy, long_edge=400)
    kept = generate(picture, dataclasses.replace(easy, palette_margin_de00=4.0), long_edge=400)
    again = generate(picture, easy, long_edge=400)  # the same picture: the margin is in the cache's key
    pipeline.clear_cache()
    assert (merged.num_colors_used, kept.num_colors_used, again.num_colors_used) == (2, 3, 2)
    assert merged.page.tobytes() == again.page.tobytes()


def test_the_brush_is_a_setting():
    # The line 2 px wide of test_a_line_too_thin_to_paint_is_not_a_region_on_the_page, 1.9 mm on paper: too thin for the
    # 3 mm brush, a region of its own for a 1 mm one.
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :100] = (200, 60, 60)
    image[:, 100:] = (60, 180, 60)
    image[:, 99:101] = (20, 20, 20)
    params = difficulty.custom_params(3, 2.0, 0.0)
    pipeline.clear_cache()
    thick = generate(image, params, long_edge=200, collect_analysis=True)
    fine = generate(image, dataclasses.replace(params, min_width_mm=1.0), long_edge=200, collect_analysis=True)
    pipeline.clear_cache()
    assert thick.num_regions == 2 and fine.num_regions == 3
    assert (20, 20, 20) in fine.palette_rgb
    assert fine.analysis.min_paintable_width_px == print_size.print_scale((200, 200)).mm_to_px(1.0)


def test_the_ink_and_the_text_found_are_cached_by_their_settings(monkeypatch):
    picture = np.full((300, 400, 3), 235, dtype=np.uint8)
    picture[140:150, 60:340:6] = 30
    line = text.TextLine(quad=((50.0, 130.0), (350.0, 130.0), (350.0, 160.0), (50.0, 160.0)), score=0.9)
    asked = []
    monkeypatch.setattr(
        text, "find_text", lambda picture, source=None, max_height_mm=None: asked.append(max_height_mm) or [line]
    )
    gaps = []
    real = ink.find_ink
    monkeypatch.setattr(
        ink, "find_ink", lambda *args, **kwargs: gaps.append(kwargs.get("gap_mm")) or real(*args, **kwargs)
    )
    params = difficulty.params_for_preset("Easy")
    pipeline.clear_cache()
    # Each found once for a setting, again for another.
    for handling in (Handling(), Handling(), Handling(text_max_height_mm=30.0), Handling(ink_gap_mm=1.0)):
        generate(picture, params, long_edge=400, handling=handling)
    pipeline.clear_cache()
    assert asked == [15.0, 30.0]
    assert gaps == [0.5, 1.0]


def test_a_face_s_tones_and_marks_and_the_vote_follow_their_settings(monkeypatch):
    from tessellatum.core import marks, texture, tones

    seen = {}
    real_tones, real_marks, real_smooth = tones.settle_tones, marks.detail_marks, pipeline.smooth_regions
    def spy(name, real, first):
        def call(*args):
            seen[name] = args[first:]  # the settings, after the arguments every call has
            return real(*args)

        return call

    monkeypatch.setattr(tones, "settle_tones", spy("tones", real_tones, 5))
    monkeypatch.setattr(marks, "detail_marks", spy("marks", real_marks, 6))
    monkeypatch.setattr(pipeline, "smooth_regions", spy("vote", real_smooth, 7))
    image = pipeline.load_image_bgr(Path(__file__).resolve().parent / "sample_images" / "l-photo-cats-face.jpg")
    params = difficulty.custom_params(
        12, 125.0, 5.0, min_width_mm=2.0, edge_settling=0.5, edge_color_step_de00=6.0, detail_weight=3,
        sharpest_corner_deg=30.0, corner_contrast_de00=15.0,
    )
    handling = Handling(mark_contrast=20.0, mark_length_mm=3.0, face_tone_step_de00=0.5)
    pipeline.clear_cache()
    generate(image, params, pipeline.PREVIEW_LONG_EDGE, handling=handling)
    pipeline.clear_cache()
    corners = seen["vote"][3]
    assert {**seen, "vote": seen["vote"][:3]} == {"vote": (0.5, 6.0, 3), "tones": (0.5,), "marks": (2.0, 20.0, 3.0)}
    assert (corners.sharpest_deg, corners.contrast_de00) == (30.0, 15.0)


def test_a_triangle_s_point_is_rounded_unless_corners_are_asked_to_keep_their_points():
    # A 30 degree triangle, dark blue on pale gray, its point at column 520 of a 600 px page (a brush of 6.7 px).
    picture = np.full((424, 600, 3), (215, 225, 230), dtype=np.uint8)
    half = math.radians(30) / 2
    point = np.array([520.0, 212.0])
    side = 380 * np.array([math.cos(half), math.sin(half)])
    corners = [point, point - side, point - side * (1, -1)]
    cv2.fillPoly(picture, [np.round(np.array(corners) * 16).astype(np.int32)], (120, 40, 30), cv2.LINE_AA, shift=4)
    medium = difficulty.params_for_preset("Medium")
    reach = {}
    for name, params in (("default", medium), ("20 degrees", dataclasses.replace(medium, sharpest_corner_deg=20.0))):
        pipeline.clear_cache()
        ids = generate(picture, params, 600, collect_analysis=True).analysis.region_id_map
        reach[name] = int(np.nonzero((ids == ids[212, 300]).any(axis=0))[0].max())
    pipeline.clear_cache()
    assert reach == {"default": 515, "20 degrees": 520}  # rounded to the brush 5 px short of its point, or to it

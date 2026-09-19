import numpy as np
import pytest

from tessellatum.core import boundaries, difficulty, ink, labels, pipeline, print_size, render
from tessellatum.core.color import MIN_PALETTE_DE00, pairwise_de00
from tessellatum.core.pipeline import PipelineCancelled, generate


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

    # Below the brush and its footprint, the printed page's own limits hold.
    finer = difficulty.DifficultyParams(num_colors=4, min_region_area_mm2=1.0, blur_sigma=0.0, min_width_mm=1.0)
    min_area_px, min_width_px = pipeline._paintable_limits(finer, (200, 200))

    assert min_width_px == scale.mm_to_px(print_size.MIN_PAINTABLE_WIDTH_MM)
    assert min_area_px == round(scale.mm2_to_px(print_size.MIN_REGION_AREA_MM2)) > 4


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
    assert (analysis.region_id_map[printed] == -1).all()  # the ink is in no region: it is printed, not painted
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
    monkeypatch.setattr(ink, "find_ink", lambda *args: calls.append(1) or find(*args))

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

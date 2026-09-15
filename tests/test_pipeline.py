import numpy as np
import pytest

from tessellatum.core import difficulty, pipeline, render
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
    finer = difficulty.DifficultyParams(medium.num_colors, medium.min_region_fraction / 4, medium.blur_sigma)

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

    labeled = {r.region_id: r for r in analysis.regions if r.interior_radius >= render.MIN_LABEL_RADIUS_PX}
    assert labeled
    assert [label.region_id for label in analysis.labels] == list(labeled)
    for label in analysis.labels:
        assert label.text == str(labeled[label.region_id].color_index + 1)
        assert render.MIN_FONT_SIZE <= label.font_size <= render.MAX_FONT_SIZE
        x0, y0, x1, y1 = label.box
        assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 200


def test_page_is_the_outline_layer_plus_the_numbers(sample_image_bgr):
    result = generate(sample_image_bgr, difficulty.params_for_preset("Hard"), long_edge=200, collect_analysis=True)
    page = np.asarray(result.page)
    outlines = result.analysis.outlines

    near_numbers = np.zeros(outlines.shape, dtype=bool)
    for label in result.analysis.labels:
        x0, y0, x1, y1 = label.box
        # A number drawn at a fractional position can shade the pixel just past its box.
        near_numbers[max(int(y0) - 1, 0) : int(y1) + 2, max(int(x0) - 1, 0) : int(x1) + 2] = True

    assert near_numbers.any()
    np.testing.assert_array_equal(page[~near_numbers], np.repeat(outlines[~near_numbers][:, None], 3, axis=1))
    assert (page[near_numbers] != outlines[near_numbers][:, None]).any()  # the numbers are really there


def test_analysis_lists_legend_colors_first(speckled_image_bgr):
    params = difficulty.DifficultyParams(num_colors=3, min_region_fraction=0.01, blur_sigma=0.0)

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

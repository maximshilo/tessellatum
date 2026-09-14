from tessellatum.core import difficulty
from tessellatum.core.pipeline import generate


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

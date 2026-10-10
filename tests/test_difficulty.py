"""Difficulty settings are sizes on the printed page."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tessellatum.core import difficulty, print_size

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_case  # noqa: E402


def test_presets_ask_for_more_smaller_regions_from_beginner_to_realistic():
    levels = [difficulty.params_for_preset(name) for name in ("Beginner", "Easy", "Medium", "Hard", "Realistic")]

    for coarser, finer in zip(levels, levels[1:]):
        assert coarser.num_colors < finer.num_colors
        assert coarser.min_region_area_mm2 > finer.min_region_area_mm2
        assert coarser.min_width_mm > finer.min_width_mm
        assert coarser.blur_sigma > finer.blur_sigma


def test_the_presets_are_the_ones_the_user_chose():
    # The user's five levels (0.1.52), chosen from their own pages: colors, smallest region, brush, smoothing.
    assert list(difficulty.PRESETS) == ["Beginner", "Easy", "Medium", "Hard", "Realistic"]
    assert {name: (p.num_colors, p.min_region_area_mm2, p.min_width_mm, p.blur_sigma) for name, p in difficulty.PRESETS.items()} == {
        "Beginner": (16, 10.0, 1.0, 1.2),
        "Easy": (20, 8.0, 0.8, 1.0),
        "Medium": (24, 6.0, 0.7, 0.9),
        "Hard": (28, 4.0, 0.6, 0.8),
        "Realistic": (32, 2.0, 0.5, 0.7),
    }
    assert difficulty.DEFAULT_PRESET == "Medium"
    # Every level keeps colors 4 ΔE00 apart, settles edges three times as far, held to the picture's within 1 ΔE00,
    # allows regions a quarter the size in faces and the subject, and rounds corners to the brush.
    for p in difficulty.PRESETS.values():
        assert (
            p.palette_margin_de00, p.edge_settling, p.edge_color_step_de00, p.detail_weight, p.sharpest_corner_deg,
            p.corner_contrast_de00,
        ) == (4.0, 3.0, 1.0, 4, 180.0, 0.0)
    # The pipeline's own defaults, which Max keeps, are where they were.
    assert difficulty.DifficultyParams(4, 100.0, 1.0).min_width_mm == print_size.MIN_PAINTABLE_WIDTH_MM == 3.0


def test_the_benchmark_s_max_is_pinned_where_the_sliders_used_to_stop():
    finest = difficulty.finest_params()

    assert (finest.num_colors, finest.min_region_area_mm2, finest.blur_sigma) == (40, 30.0, 0.0)
    assert finest == difficulty.DifficultyParams(40, 30.0, 0.0)  # every other setting at the pipeline's default
    assert finest == difficulty.DifficultyParams(40, 30.0, 0.0, 3.0, 10.0, 1.0, 10.0, 2, 180.0, 20.0)
    # The sliders reach past it now, and every preset lies within their range.
    lo, hi = difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE
    assert lo < finest.min_region_area_mm2 and difficulty.CUSTOM_COLORS_RANGE[1] > finest.num_colors
    for p in difficulty.PRESETS.values():
        for name, (low, high) in difficulty.CUSTOM_RANGES.items():
            assert low <= getattr(p, name) <= high, name


def test_custom_params_are_clamped_to_the_sliders():
    params = difficulty.custom_params(num_colors=1, min_region_area_mm2=10_000.0, blur_sigma=50.0)

    assert (params.num_colors, params.min_region_area_mm2, params.blur_sigma) == (
        difficulty.CUSTOM_COLORS_RANGE[0],
        difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE[1],
        difficulty.CUSTOM_BLUR_RANGE[1],
    )
    finer = difficulty.custom_params(
        64, 1.0, 0.0, min_width_mm=0.2, palette_margin_de00=-3.0, edge_settling=9.0, edge_color_step_de00=0.0,
        detail_weight=3.6, sharpest_corner_deg=1.0, corner_contrast_de00=-5.0,
    )
    assert finer == difficulty.DifficultyParams(64, 2.0, 0.0, 0.5, 0.0, 3.0, 1.0, 4, 5.0, 0.0)
    assert difficulty.custom_params(12, 125.0, 5.0, sharpest_corner_deg=500.0, corner_contrast_de00=99.0) == (
        difficulty.DifficultyParams(12, 125.0, 5.0, sharpest_corner_deg=180.0, corner_contrast_de00=60.0)
    )
    assert isinstance(finer.detail_weight, int) and isinstance(finer.num_colors, int)
    with pytest.raises(TypeError):
        difficulty.custom_params(12, 125.0, 5.0, brush=2.0)


def test_unknown_presets_are_refused():
    with pytest.raises(ValueError):
        difficulty.params_for_preset("Max")


def test_describe_speaks_in_print_units():
    assert difficulty.describe(difficulty.finest_params()) == (
        "Up to 40 colors, at least 10 ΔE00 apart. Regions of at least 30 mm² (about 5 × 5 mm; half that on a "
        "photograph's or painting's subject and faces) "
        "and 3 mm wide on the printed A4 page, corners rounded to the brush."
    )
    assert difficulty.describe(difficulty.params_for_preset("Easy")) == (
        "Up to 20 colors, at least 4 ΔE00 apart. Regions of at least 8 mm² (about 3 × 3 mm; 1/4 of that on a "
        "photograph's or painting's subject and faces) and 0.8 mm wide on the printed A4 page, corners rounded to the "
        "brush."
    )
    fine = difficulty.custom_params(
        30, 16.0, 0.0, min_width_mm=1.5, palette_margin_de00=6.0, detail_weight=3, sharpest_corner_deg=20.0
    )
    assert difficulty.describe(fine) == (
        "Up to 30 colors, at least 6 ΔE00 apart. Regions of at least 16 mm² (about 4 × 4 mm; 1/3 of that on a "
        "photograph's or painting's subject and faces) and 1.5 mm wide on the printed A4 page, corners down to 20° "
        "kept sharp."
    )


def test_the_benchmark_max_is_the_finest_setting_of_the_version_measured():
    assert bench_case.preset_params(difficulty, "Max") == difficulty.finest_params()
    assert bench_case.preset_params(difficulty, "Hard") == difficulty.params_for_preset("Hard")

    # A version from before 0.1.26 has no finest_params: its sliders stopped at 0.0002 of the image.
    def old_params(num_colors, min_region_fraction, blur_sigma):
        return SimpleNamespace(num_colors=num_colors, min_region_fraction=min_region_fraction, blur_sigma=blur_sigma)

    old = SimpleNamespace(DifficultyParams=old_params, params_for_preset=difficulty.params_for_preset)
    assert bench_case.preset_params(old, "Max") == old_params(40, 0.0002, 0.0)

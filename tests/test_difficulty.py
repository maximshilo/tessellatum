"""Difficulty settings are sizes on the printed page."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tessellatum.core import difficulty, print_size

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_case  # noqa: E402


def test_presets_ask_for_more_smaller_regions_from_easy_to_hard():
    easy, medium, hard = (difficulty.params_for_preset(name) for name in ("Easy", "Medium", "Hard"))

    assert easy.num_colors < medium.num_colors < hard.num_colors
    assert easy.min_region_area_mm2 > medium.min_region_area_mm2 > hard.min_region_area_mm2
    assert easy.blur_sigma > medium.blur_sigma > hard.blur_sigma


def test_the_presets_are_the_ones_agreed_in_print_units():
    # D-034: today's presets restated on paper, every one painted with the 3 mm brush.
    assert {name: (p.num_colors, p.min_region_area_mm2, p.blur_sigma, p.min_width_mm) for name, p in difficulty.PRESETS.items()} == {
        "Easy": (6, 300.0, 9.0, 3.0),
        "Medium": (12, 125.0, 5.0, 3.0),
        "Hard": (20, 40.0, 2.5, 3.0),
    }
    assert difficulty.DifficultyParams(4, 100.0, 1.0).min_width_mm == print_size.MIN_PAINTABLE_WIDTH_MM == 3.0


def test_the_finest_custom_setting_keeps_regions_of_30_square_millimeters():
    finest = difficulty.finest_params()

    assert (finest.num_colors, finest.min_region_area_mm2, finest.blur_sigma) == (40, 30.0, 0.0)
    assert difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE == (30.0, 500.0)
    # It is the sliders' own extreme, and every preset lies within the sliders' range.
    assert finest == difficulty.custom_params(1000, 0.0, -1.0)
    lo, hi = difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE
    assert print_size.MIN_REGION_AREA_MM2 < lo
    assert all(lo <= p.min_region_area_mm2 <= hi for p in difficulty.PRESETS.values())


def test_custom_params_are_clamped_to_the_sliders():
    params = difficulty.custom_params(num_colors=2, min_region_area_mm2=10_000.0, blur_sigma=50.0)

    assert (params.num_colors, params.min_region_area_mm2, params.blur_sigma) == (
        difficulty.CUSTOM_COLORS_RANGE[0],
        difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE[1],
        difficulty.CUSTOM_BLUR_RANGE[1],
    )


def test_unknown_presets_are_refused():
    with pytest.raises(ValueError):
        difficulty.params_for_preset("Max")


def test_describe_speaks_in_print_units():
    assert difficulty.describe(difficulty.params_for_preset("Easy")) == (
        "Up to 6 colors. Regions of at least 300 mm² (about 17 × 17 mm) and 3 mm wide on the printed A4 page."
    )


def test_the_benchmark_max_is_the_finest_setting_of_the_version_measured():
    assert bench_case.preset_params(difficulty, "Max") == difficulty.finest_params()
    assert bench_case.preset_params(difficulty, "Hard") == difficulty.params_for_preset("Hard")

    # A version from before 0.1.26 has no finest_params: its sliders stopped at 0.0002 of the image.
    def old_params(num_colors, min_region_fraction, blur_sigma):
        return SimpleNamespace(num_colors=num_colors, min_region_fraction=min_region_fraction, blur_sigma=blur_sigma)

    old = SimpleNamespace(DifficultyParams=old_params, params_for_preset=difficulty.params_for_preset)
    assert bench_case.preset_params(old, "Max") == old_params(40, 0.0002, 0.0)

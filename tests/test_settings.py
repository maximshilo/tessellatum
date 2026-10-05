"""The settings the app offers: one per field, each with a range around its default, applied within it."""

import dataclasses

import pytest

from tessellatum.core import difficulty, settings
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.pipeline import Handling
from tessellatum.core.render import PageStyle

OWNERS = {settings.DIFFICULTY: DifficultyParams, settings.STYLE: PageStyle, settings.HANDLING: Handling}


def test_every_setting_is_a_field_of_its_object_and_its_default_lies_in_its_range():
    names = [s.name for s in settings.SETTINGS]
    assert len(names) == len(set(names))
    assert len({s.label for s in settings.SETTINGS}) == len(names)
    for s in settings.SETTINGS:
        assert s.name in {f.name for f in dataclasses.fields(OWNERS[s.owner])}, s.name
        assert s.group in settings.GROUPS and s.label and s.tooltip
        assert s.minimum <= s.default <= s.maximum, s.name
        assert s.step > 0
        if not s.log:  # a slider steps from one end to the other
            steps = (s.maximum - s.minimum) / s.step
            assert steps == pytest.approx(round(steps)), s.name
        else:
            assert s.minimum > 0
    assert all(settings.in_group(group) for group in settings.GROUPS)


def test_the_defaults_are_the_pipeline_s_own():
    medium = difficulty.params_for_preset(difficulty.DEFAULT_PRESET)
    chosen = settings.values(medium, PageStyle(), Handling())
    assert {s.name: s.default for s in settings.SETTINGS} == chosen
    # Applying them changes nothing.
    assert settings.apply(chosen, medium) == (medium, PageStyle(), Handling())


def test_every_field_the_pipeline_takes_a_number_for_is_offered_or_set_otherwise():
    offered = {s.name for s in settings.SETTINGS}
    # The grays come with the tone, the thinnest line and the leader's dot are drawing details, the switches are boxes.
    otherwise = {"line_gray", "label_gray", "min_line_width_px", "leader_dot_ratio", "line_art", "detail", "text"}
    for owner in OWNERS.values():
        for field in dataclasses.fields(owner):
            assert field.name in offered | otherwise, field.name


def test_apply_holds_each_setting_in_its_range():
    params, style, handling = settings.apply(
        {"num_colors": 500, "min_width_mm": 0.0, "palette_margin_de00": 4.5, "line_width_mm": 9.0, "ink_gap_mm": 1.2,
         "detail_weight": 2.6},
        difficulty.params_for_preset("Hard"),
    )
    assert (params.num_colors, params.min_width_mm, params.palette_margin_de00, params.detail_weight) == (64, 0.5, 4.5, 3)
    assert params.min_region_area_mm2 == difficulty.params_for_preset("Hard").min_region_area_mm2  # the rest as given
    assert style == PageStyle(line_width_mm=1.0)
    assert handling == Handling(ink_gap_mm=1.2)
    with pytest.raises(KeyError):
        settings.apply({"brush": 2.0}, difficulty.params_for_preset("Hard"))


def test_a_value_is_shown_with_its_unit_to_the_setting_s_step():
    assert settings.setting("min_width_mm").text(1.5) == "1.5 mm"
    assert settings.setting("line_width_mm").text(0.3) == "0.30 mm"
    assert settings.setting("num_colors").text(12) == "12"
    assert settings.setting("palette_margin_de00").text(6) == "6.0 ΔE00"
    assert settings.setting("min_region_area_mm2").text(124.6) == "125 mm²"
    assert settings.setting("detail_weight").text(2) == "2 ×"

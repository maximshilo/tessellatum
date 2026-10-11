"""The controls offer every setting in the printed page's units and ask the pipeline for what they show."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty, settings  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.pipeline import Handling  # noqa: E402
from tessellatum.core.render import TONES, PageStyle  # noqa: E402
from tessellatum.gui import controls_panel  # noqa: E402
from tessellatum.gui.controls_panel import ControlsPanel  # noqa: E402


# One application for the whole module: letting it be collected between tests
# and making another is how Qt test suites come to crash at random.
_app = QApplication.instance() or QApplication([])


@pytest.fixture
def panel():
    widget = ControlsPanel()
    yield widget
    widget.deleteLater()
    _app.processEvents()


def _store(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def _move(panel: ControlsPanel, name: str, value: float) -> None:
    """Move a setting's slider to ``value``'s position, as the user would."""
    control = panel.controls[name]
    control.slider.setValue(controls_panel.slider_position(control.setting, value))


@pytest.mark.parametrize("setting", settings.SETTINGS, ids=lambda s: s.name)
def test_every_slider_spans_its_setting_s_range_and_finds_every_position_again(setting):
    steps = controls_panel.slider_steps(setting)
    assert controls_panel.slider_value(setting, 0) == setting.minimum
    assert controls_panel.slider_value(setting, steps) == pytest.approx(setting.maximum)
    for position in range(steps + 1):
        assert controls_panel.slider_position(setting, controls_panel.slider_value(setting, position)) == position


def test_the_region_slider_moves_in_equal_ratios():
    area = settings.setting("min_region_area_mm2")
    steps = controls_panel.slider_steps(area)
    assert controls_panel.slider_value(area, steps // 2) == pytest.approx((area.minimum * area.maximum) ** 0.5)


def test_every_setting_has_a_slider_in_its_section_with_its_tooltip_and_unit(panel):
    for s in settings.SETTINGS:
        control = panel.controls[s.name]
        assert control.setting is s and control.toolTip() == s.tooltip
        assert control.value_label.text() == s.text(control.value())
        assert panel.sections[s.group].body.isAncestorOf(control)
    # The tone sits with the lines; the faces', line art's and text's own settings start folded away.
    assert panel.sections[settings.LINES].body.isAncestorOf(panel.tone_combo)
    assert {group: section.is_expanded() for group, section in panel.sections.items()} == {
        settings.REGIONS: True, settings.FACES: False, settings.LINES: True, settings.LINE_ART: False,
        settings.TEXT: False,
    }
    panel.sections[settings.TEXT].toggle.click()
    assert panel.sections[settings.TEXT].is_expanded() and not panel.sections[settings.TEXT].body.isHidden()


def test_the_panel_starts_at_the_defaults(panel):
    assert panel.preset_combo.currentText() == difficulty.DEFAULT_PRESET
    assert panel.get_difficulty_params() == difficulty.params_for_preset(difficulty.DEFAULT_PRESET)
    assert panel.get_page_style() == PageStyle()
    assert panel.get_handling() == Handling()
    for s in settings.SETTINGS:
        assert panel.controls[s.name].value() == s.default


def test_a_preset_shows_its_values_and_is_passed_as_it_is(panel):
    levels = [panel.preset_combo.itemText(i) for i in range(panel.preset_combo.count())]
    assert levels == ["Beginner", "Easy", "Medium", "Hard", "Realistic", "Custom"]
    panel.preset_combo.setCurrentText("Hard")
    hard = difficulty.params_for_preset("Hard")
    assert panel.get_difficulty_params() == hard
    for name, control in panel.controls.items():
        if control.setting.owner == settings.DIFFICULTY:
            assert control.value() == getattr(hard, name)  # exactly, wherever the slider's steps fall
    tooltip = panel.preset_combo.itemData(panel.preset_combo.findText("Hard"), Qt.ToolTipRole)
    assert tooltip == difficulty.describe(hard) and "4 mm²" in tooltip
    assert panel.preset_combo.itemData(panel.preset_combo.findText("Custom"), Qt.ToolTipRole)


def test_moving_a_difficulty_setting_picks_custom_and_keeps_the_preset_s_others(panel):
    panel.preset_combo.setCurrentText("Hard")
    _move(panel, "min_width_mm", 1.5)
    _move(panel, "palette_margin_de00", 6.0)

    assert panel.preset_combo.currentText() == "Custom"
    params = panel.get_difficulty_params()
    hard = difficulty.params_for_preset("Hard")
    assert (params.min_width_mm, params.palette_margin_de00) == (1.5, 6.0)
    assert (params.num_colors, params.min_region_area_mm2, params.blur_sigma) == (
        hard.num_colors, hard.min_region_area_mm2, hard.blur_sigma
    )
    assert panel.controls["min_width_mm"].value_label.text() == "1.5 mm"
    # Picking a preset again shows the preset's values.
    panel.preset_combo.setCurrentText("Easy")
    assert panel.get_difficulty_params() == difficulty.params_for_preset("Easy")
    assert panel.controls["min_width_mm"].value() == 0.8


def test_the_sliders_reach_past_the_old_finest_page(panel):
    for name in ("num_colors", "palette_margin_de00", "min_region_area_mm2", "min_width_mm", "blur_sigma"):
        control = panel.controls[name]
        end = control.slider.maximum() if name == "num_colors" else control.slider.minimum()
        control.slider.setValue(end)
    params = panel.get_difficulty_params()
    assert (params.num_colors, params.palette_margin_de00, params.min_region_area_mm2, params.min_width_mm) == (
        64, 0.0, 2.0, 0.5
    )
    finest = difficulty.finest_params()
    assert params.num_colors > finest.num_colors and params.min_region_area_mm2 < finest.min_region_area_mm2


def test_the_style_and_the_handling_settings_leave_the_preset_alone(panel):
    panel.preset_combo.setCurrentText("Hard")
    _move(panel, "line_width_mm", 0.5)
    _move(panel, "line_smoothing_mm", 1.0)
    _move(panel, "min_label_pt", 8.0)
    _move(panel, "ink_gap_mm", 1.0)
    _move(panel, "text_max_height_mm", 25.0)
    assert panel.preset_combo.currentText() == "Hard"
    assert panel.get_page_style() == PageStyle(line_width_mm=0.5, line_smoothing_mm=1.0, min_label_pt=8.0)
    assert panel.get_handling() == Handling(ink_gap_mm=1.0, text_max_height_mm=25.0)


def test_line_widths_from_0_1_to_1_mm(panel):
    slider = panel.controls["line_width_mm"].slider
    widths = [controls_panel.slider_value(settings.setting("line_width_mm"), p) for p in range(slider.maximum() + 1)]
    assert widths[0] == 0.1 and widths[-1] == 1.0 and 0.2 in widths and len(widths) == 19
    assert panel.controls["line_width_mm"].value_label.text() == "0.20 mm"


def test_a_tone_sets_both_grays(panel):
    assert [panel.tone_combo.itemText(i) for i in range(panel.tone_combo.count())] == list(TONES)
    assert panel.tone_combo.currentText() == "Medium"
    for index, (name, (line_gray, label_gray)) in enumerate(TONES.items()):
        panel.tone_combo.setCurrentText(name)
        style = panel.get_page_style()
        assert (style.line_gray, style.label_gray) == (line_gray, label_gray)
        assert style.line_width_mm == PageStyle().line_width_mm
        assert panel.tone_combo.itemData(index, Qt.ToolTipRole)


def test_every_step_of_the_picture_handling_starts_on_and_each_box_turns_its_own_off(panel):
    assert panel.get_handling() == Handling()
    boxes = {"line_art": panel.line_art_check, "detail": panel.detail_check, "text": panel.text_check}
    for name, box in boxes.items():
        assert box.toolTip()
        box.setChecked(False)
        assert panel.get_handling() == Handling(**{name: False})
        box.setChecked(True)


def test_reset_puts_every_setting_back(panel):
    panel.preset_combo.setCurrentText("Hard")
    _move(panel, "min_width_mm", 1.0)
    _move(panel, "line_width_mm", 0.8)
    _move(panel, "mark_contrast", 30.0)
    panel.tone_combo.setCurrentText("Dark")
    panel.text_check.setChecked(False)
    panel.reset_button.click()
    assert panel.preset_combo.currentText() == difficulty.DEFAULT_PRESET
    assert panel.get_difficulty_params() == difficulty.params_for_preset(difficulty.DEFAULT_PRESET)
    assert panel.get_page_style() == PageStyle() and panel.get_handling() == Handling()
    for s in settings.SETTINGS:
        assert panel.controls[s.name].value() == s.default


def test_the_settings_are_kept_between_runs(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    first.preset_combo.setCurrentText("Hard")
    _move(first, "min_width_mm", 1.5)
    _move(first, "line_smoothing_mm", 0.25)
    _move(first, "face_tone_step_de00", 2.0)
    first.tone_combo.setCurrentText("Light")
    first.detail_check.setChecked(False)
    asked = (first.get_difficulty_params(), first.get_page_style(), first.get_handling())
    first.deleteLater()
    _app.processEvents()

    again = ControlsPanel(store=_store(tmp_path))
    assert again.preset_combo.currentText() == "Custom"
    assert (again.get_difficulty_params(), again.get_page_style(), again.get_handling()) == asked
    again.deleteLater()
    _app.processEvents()


def test_a_kept_preset_is_the_preset_and_a_store_that_can_t_be_read_is_ignored(tmp_path):
    store = _store(tmp_path)
    store.setValue("difficulty/preset", "Easy")
    store.setValue("settings/num_colors", 40)  # beside a preset, its difficulty is the preset's
    store.setValue("settings/line_width_mm", "thick")
    store.setValue("settings/text_gap_mm", 99)
    store.setValue("style/tone", "Neon")
    panel = ControlsPanel(store=store)
    assert panel.get_difficulty_params() == difficulty.params_for_preset("Easy")
    assert panel.get_page_style() == PageStyle(text_gap_mm=3.0)  # held to its range; the rest as by default
    panel.deleteLater()
    _app.processEvents()


def test_without_a_store_nothing_is_kept(panel, tmp_path):
    _move(panel, "min_width_mm", 1.0)
    other = ControlsPanel()
    assert other.get_difficulty_params() == difficulty.params_for_preset(difficulty.DEFAULT_PRESET)
    other.deleteLater()
    _app.processEvents()


def test_at_its_narrowest_every_slider_has_room_and_the_sliders_line_up(panel):
    for section in panel.sections.values():
        section.toggle.setChecked(True)
    panel.show()  # laid out only once shown
    panel.resize(panel.minimumSizeHint().width(), panel.sizeHint().height())
    _app.processEvents()
    for s in settings.SETTINGS:
        control = panel.controls[s.name]
        assert control.slider.width() >= controls_panel.MIN_SLIDER_WIDTH, s.name
        # A setting's label explains it too.
        assert panel.sections[s.group].form.labelForField(control).toolTip() == s.tooltip
    # Every section's labels are as wide as the widest, so the sliders start in one column.
    lefts = {panel.controls[s.name].slider.mapTo(panel, panel.controls[s.name].slider.rect().topLeft()).x()
             for s in settings.SETTINGS}
    assert len(lefts) == 1


def test_any_version_of_the_page_can_be_exported_and_the_page_itself_is_the_default(panel):
    combo = panel.version_combo
    assert [combo.itemData(i) for i in range(combo.count())] == list(Version)
    assert [combo.itemText(i) for i in range(combo.count())] == [version.value for version in Version]
    assert panel.get_export_version() is Version.PAGE
    for index, version in enumerate(Version):
        assert combo.itemData(index, Qt.ToolTipRole) == version.description
        combo.setCurrentIndex(index)
        assert panel.get_export_version() is version

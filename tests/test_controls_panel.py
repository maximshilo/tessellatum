"""The controls offer every setting in the printed page's units and ask the pipeline for what they show."""

import dataclasses
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
        # One that means nothing yet (the mixing settings, without My paints) is off, and says why first.
        reason = s.disabled_reason(panel.snapshot().chosen())
        assert control.setting is s and control.isEnabled() == (reason is None)
        assert control.toolTip() == (s.tooltip if reason is None else f"{reason} {s.tooltip}")
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


def test_the_pixels_across_the_brush_sit_by_the_brush_and_leave_the_preset_alone(tmp_path):
    # D-012 (T8): how finely a page is made is a setting of its own, kept between runs, whatever the preset.
    first = ControlsPanel(store=_store(tmp_path))
    rows = [s.name for s in settings.in_group(settings.REGIONS)]
    assert rows.index("brush_px") == rows.index("min_width_mm") + 1
    first.preset_combo.setCurrentText("Realistic")
    _move(first, "brush_px", 6.0)
    assert first.preset_combo.currentText() == "Realistic"
    assert first.get_difficulty_params() == difficulty.params_for_preset("Realistic")
    assert first.get_handling() == Handling(brush_px=6.0)
    assert first.controls["brush_px"].value_label.text() == "6.0 px"
    first.deleteLater()
    _app.processEvents()

    again = ControlsPanel(store=_store(tmp_path))
    assert again.preset_combo.currentText() == "Realistic" and again.get_handling() == Handling(brush_px=6.0)
    again.reset_button.click()
    assert again.get_handling().brush_px == 4.0
    again.deleteLater()
    _app.processEvents()


def test_line_widths_from_0_1_to_1_mm(panel):
    slider = panel.controls["line_width_mm"].slider
    widths = [controls_panel.slider_value(settings.setting("line_width_mm"), p) for p in range(slider.maximum() + 1)]
    assert widths[0] == 0.1 and widths[-1] == 1.0 and 0.2 in widths and len(widths) == 19
    assert panel.controls["line_width_mm"].value_label.text() == "0.20 mm"


def test_a_tone_sets_all_three_grays(panel):
    assert [panel.tone_combo.itemText(i) for i in range(panel.tone_combo.count())] == [*TONES, "Custom"]
    assert panel.tone_combo.currentText() == "Medium"
    for index, (name, grays) in enumerate(TONES.items()):
        panel.tone_combo.setCurrentText(name)
        style = panel.get_page_style()
        assert (style.line_gray, style.label_gray, style.leader_gray) == grays
        assert [panel.controls[field].value() for field in ("line_gray", "label_gray", "leader_gray")] == list(grays)
        assert style.line_width_mm == PageStyle().line_width_mm
        assert panel.tone_combo.itemData(index, Qt.ToolTipRole)
    assert panel.tone_combo.itemData(len(TONES), Qt.ToolTipRole)


def test_medium_shows_its_grays_as_34_34_and_70_fresh_or_kept_between_runs(panel, tmp_path):
    fields = ("line_gray", "label_gray", "leader_gray")
    assert [panel.controls[field].value_label.text() for field in fields] == ["34 L*", "34 L*", "70 L*"]
    store = _store(tmp_path)
    store.setValue("style/tone", "Medium")
    store.setValue("settings/label_gray", 0x81)  # kept by an older version, as Medium was: the tone's grays win
    kept = ControlsPanel(store=store)
    assert kept.tone_combo.currentText() == "Medium"
    assert [kept.controls[field].value_label.text() for field in fields] == ["34 L*", "34 L*", "70 L*"]


def test_moving_a_gray_picks_a_custom_tone_starting_from_the_tone(panel):
    panel.tone_combo.setCurrentText("Dark")
    _move(panel, "leader_gray", 0x70)
    assert panel.tone_combo.currentText() == "Custom"
    line_gray, label_gray, _ = TONES["Dark"]
    assert panel.get_page_style() == PageStyle(line_gray=line_gray, label_gray=label_gray, leader_gray=0x70)
    panel.tone_combo.setCurrentText("Light")  # a tone again sets all three
    assert panel.get_page_style() == PageStyle.from_settings(tone="Light")


def test_the_leader_width_is_a_setting_of_its_own(panel):
    _move(panel, "leader_width_mm", 0.3)
    assert panel.get_page_style() == PageStyle(leader_width_mm=0.3)
    assert panel.tone_combo.currentText() == "Medium"


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
    _move(panel, "label_gray", 0xA0)
    panel.text_check.setChecked(False)
    panel.reset_button.click()
    assert panel.tone_combo.currentText() == "Medium"
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


def test_a_custom_tone_s_grays_are_kept_and_a_tone_s_are_the_tone_s(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    first.tone_combo.setCurrentText("Light")
    _move(first, "leader_gray", 0x40)
    asked = first.get_page_style()
    first.deleteLater()
    _app.processEvents()
    again = ControlsPanel(store=_store(tmp_path))
    assert again.tone_combo.currentText() == "Custom" and again.get_page_style() == asked
    again.deleteLater()
    _app.processEvents()

    store = _store(tmp_path)
    store.setValue("style/tone", "Dark")
    store.setValue("settings/leader_gray", 0x40)  # beside a tone, its grays are the tone's
    panel = ControlsPanel(store=store)
    assert panel.get_page_style() == PageStyle.from_settings(tone="Dark")
    panel.deleteLater()
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


def test_the_paper_is_a_page_setting_kept_between_runs_and_reset_to_a4(tmp_path):
    # T3a: A4, A3 or A2, in the region settings, since it changes the page.
    first = ControlsPanel(store=_store(tmp_path))
    assert [first.paper_combo.itemText(i) for i in range(first.paper_combo.count())] == ["A4", "A3", "A2"]
    assert first.get_page_style().paper == "A4"
    first.paper_combo.setCurrentText("A2")
    assert first.get_page_style() == PageStyle(paper="A2")
    again = ControlsPanel(store=_store(tmp_path))
    assert again.get_page_style().paper == "A2"
    again.reset()
    assert again.get_page_style() == PageStyle()
    store = _store(tmp_path)
    store.setValue("style/paper", "Letter")  # not a sheet the app offers: the default
    assert ControlsPanel(store=store).get_page_style().paper == "A4"


def test_the_png_s_resolution_is_asked_for_in_dpi_or_pixels_and_shown_as_both():
    # T3c: 600 dpi by default; the other unit worked out from the paper and the picture's shape, shown live.
    panel = ControlsPanel()
    assert panel.get_export_resolution() == (controls_panel.DPI, 600) and panel.get_export_dpi() == 600
    assert panel.resolution_readout.text() == "6,543 px along the long edge at 600 dpi on A4."
    panel.set_picture_size((3000, 2000))  # 3:2, 277 mm long on A4
    assert panel.resolution_readout.text() == "6,543 × 4,362 px (29 Mpx) at 600 dpi on A4."

    panel.resolution_unit.setCurrentIndex(panel.resolution_unit.findData(controls_panel.PIXELS))
    assert panel.get_export_resolution() == (controls_panel.PIXELS, 6543)
    assert panel.get_export_dpi() == pytest.approx(600, abs=0.1)
    panel.resolution_spin.setValue(3272)
    assert panel.get_export_dpi() == pytest.approx(300, abs=0.1)
    assert panel.get_export_dpi(printed_long_mm=138.5) == pytest.approx(600, abs=0.1)
    assert "3,272 × 2,181 px" in panel.resolution_readout.text()
    panel.resolution_unit.setCurrentIndex(panel.resolution_unit.findData(controls_panel.DPI))
    assert panel.get_export_resolution() == (controls_panel.DPI, 300)

    panel.dpi_buttons[150].click()
    assert panel.get_export_resolution() == (controls_panel.DPI, 150)
    panel.pdf_radio.setChecked(True)  # a PDF is vector art: no resolution of its own
    assert not panel.resolution_spin.isEnabled() and not panel.dpi_buttons[600].isEnabled()
    assert "vector" in panel.resolution_readout.text()


def test_the_png_s_resolution_is_capped_at_what_a_png_takes_on_its_paper():
    panel = ControlsPanel()
    panel.set_picture_size((3000, 2000))
    panel.set_export_dpi(1200)
    assert panel.get_export_dpi() == 1200 and "The most" not in panel.resolution_readout.text()
    panel.paper_combo.setCurrentText("A2")  # 574 mm long: 16,384 px is 724 dpi
    assert panel.get_export_dpi() == 724
    assert panel.resolution_readout.text().endswith("The most a PNG takes: 16,384 px along a side.")
    panel.paper_combo.setCurrentText("A4")  # what was asked for, where the paper takes it
    assert panel.get_export_dpi() == 1200


def test_the_png_s_resolution_is_kept_between_runs_and_reset_to_600_dpi(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    first.resolution_unit.setCurrentIndex(first.resolution_unit.findData(controls_panel.PIXELS))
    first.resolution_spin.setValue(5000)
    again = ControlsPanel(store=_store(tmp_path))
    assert again.get_export_resolution() == (controls_panel.PIXELS, 5000)
    again.reset()
    assert again.get_export_resolution() == (controls_panel.DPI, 600)
    store = _store(tmp_path)
    store.setValue("export/unit", "furlongs")  # not a unit the app offers: the default
    assert ControlsPanel(store=store).get_export_resolution() == (controls_panel.DPI, 600)


def test_the_coloring_medium_drops_the_paper_black_and_caps_the_margin_on_a_screen_and_is_kept(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    assert first.medium_combo.currentText() == "Paper and paint"
    assert first.get_difficulty_params() == difficulty.params_for_preset("Medium")
    first.medium_combo.setCurrentText("Screen")
    screen = first.get_difficulty_params()
    assert (screen.paper_black_lstar, screen.palette_margin_de00) == (0.0, 2.5)
    assert screen.num_colors == difficulty.params_for_preset("Medium").num_colors
    assert not first.controls["paper_black_lstar"].isEnabled()
    # The margin's and the levels' tooltips say what the page keeps on a screen.
    assert "at most 2.5" in first.controls["palette_margin_de00"].toolTip()
    medium = first.preset_combo.findText("Medium")
    assert first.preset_combo.itemData(medium, Qt.ToolTipRole) == difficulty.describe(screen)
    again = ControlsPanel(store=_store(tmp_path))
    assert again.medium_combo.currentText() == "Screen"
    again.reset()
    assert again.medium_combo.currentText() == "Paper and paint"
    assert again.controls["paper_black_lstar"].isEnabled()
    assert "at most 2.5" not in again.controls["palette_margin_de00"].toolTip()
    store = _store(tmp_path)
    store.remove("difficulty/medium")  # kept by a version without the medium: paper and paint
    assert ControlsPanel(store=store).medium_combo.currentText() == "Paper and paint"


def test_a_screen_keeps_a_margin_already_under_its_own():
    custom = difficulty.custom_params(24, 6.0, 0.9, palette_margin_de00=1.0, paper_black_lstar=30.0)
    assert difficulty.for_medium(custom, "Screen") == dataclasses.replace(custom, paper_black_lstar=0.0)
    assert difficulty.for_medium(custom, "Paper and paint") is custom


def test_what_the_legend_writes_is_kept_and_says_so_when_changed(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    assert first.get_legend_detail() == "Name, hex and RGB" == first.get_page_style().legend_detail
    changed = []
    first.legend_detail_changed.connect(changed.append)
    first.legend_combo.setCurrentText("Number only")
    assert changed == ["Number only"] and first.get_page_style().legend_detail == "Number only"
    again = ControlsPanel(store=_store(tmp_path))
    assert again.get_legend_detail() == "Number only"
    again.reset()
    assert again.get_legend_detail() == "Name, hex and RGB"
    store = _store(tmp_path)
    store.remove("style/legend")  # kept by a version without it: the default
    assert ControlsPanel(store=store).get_legend_detail() == "Name, hex and RGB"
    store.setValue("style/legend", "Everything")  # not a choice the app offers
    assert ControlsPanel(store=store).get_legend_detail() == "Name, hex and RGB"


def test_the_palette_export_has_its_format_kept_and_is_suggested_coloring_on_a_screen(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    assert first.get_output_format() == "PNG" and not first.palette_combo.isEnabled()
    first.palette_radio.setChecked(True)
    assert first.get_output_format() == "Palette" and first.palette_combo.isEnabled()
    assert not first.version_combo.isEnabled() and not first.resolution_spin.isEnabled()
    assert first.get_palette_format() == ".swatches"  # Procreate first
    first.palette_combo.setCurrentIndex(first.palette_combo.findData(".gpl"))
    assert ControlsPanel(store=_store(tmp_path)).get_palette_format() == ".gpl"
    # Coloring on a screen, the panel says the palette is there to take.
    assert first.screen_hint.isHidden()
    first.medium_combo.setCurrentText("Screen")
    assert not first.screen_hint.isHidden()
    first.medium_combo.setCurrentText("Paper and paint")
    assert first.screen_hint.isHidden()
    again = ControlsPanel(store=_store(tmp_path))
    again.reset()
    assert again.get_palette_format() == ".swatches"


def test_the_paints_listed_are_kept_and_the_page_takes_its_colors_from_them_when_asked(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    assert first.colors_from_combo.currentText() == "The picture" and first.get_handling().paints == ()
    paints = (("Ultramarine", (32, 0, 177)), ("Sap green", (90, 107, 47)))
    first.set_paints(paints)  # listing paints asks for them
    assert first.colors_from_combo.currentText() == "My paints" and first.get_handling().paints == paints
    assert first.paints_button.text() == "Paints (2)…"
    again = ControlsPanel(store=_store(tmp_path))
    assert again.get_handling().paints == paints
    again.colors_from_combo.setCurrentText("The picture")
    assert again.get_handling().paints == () and again.paints_button.text() == "Paints (2)…"  # kept, not used
    again.reset()
    assert again.colors_from_combo.currentText() == "The picture"
    assert ControlsPanel(store=_store(tmp_path)).get_paints() == ()


def test_the_legend_offers_recipes_and_the_paints_listed_for_them_tell_the_window(tmp_path):
    panel = ControlsPanel(store=_store(tmp_path))
    assert [panel.legend_combo.itemText(i) for i in range(panel.legend_combo.count())][-1] == "Name and recipe"
    told = []
    panel.paints_changed.connect(lambda: told.append(panel.listed_paints()))
    panel.legend_combo.setCurrentText("Name and recipe")
    paints = (("Titanium white", (255, 255, 255)),)
    panel.set_paints(paints)
    assert told == [paints] and panel.colors_from_combo.currentText() == "The picture"
    assert panel.get_handling().paints == () and panel.listed_paints() == paints


def test_mixing_is_offered_with_my_paints_kept_between_runs_and_off_after_a_reset(tmp_path):
    first = ControlsPanel(store=_store(tmp_path))
    assert not first.mix_check.isChecked() and not first.mix_check.isEnabled()
    assert not first.controls["mix_gain_de00"].isEnabled() and not first.get_handling().mix_paints
    paints = (("Titanium white", (250, 250, 250)), ("Ivory black", (20, 20, 20)))
    first.set_paints(paints)
    assert first.mix_check.isEnabled() and not first.controls["mix_min_share_pct"].isEnabled()
    first.mix_check.setChecked(True)
    assert first.controls["mix_gain_de00"].isEnabled() and first.controls["mix_min_share_pct"].isEnabled()
    _move(first, "mix_gain_de00", 4.0)
    handling = first.get_handling()
    assert handling.mix_paints and handling.mix_gain_de00 == 4.0 and handling.mix_rule() is not None
    again = ControlsPanel(store=_store(tmp_path))
    assert again.mix_check.isChecked() and again.get_handling().mix_gain_de00 == 4.0
    again.colors_from_combo.setCurrentText("The picture")
    assert not again.mix_check.isEnabled() and not again.get_handling().mix_paints
    again.reset()
    assert not again.mix_check.isChecked()
    assert again.get_handling().mix_gain_de00 == settings.setting("mix_gain_de00").default == 3.0

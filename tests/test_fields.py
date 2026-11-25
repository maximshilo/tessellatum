"""W3: the fields settings are edited in. Exact values typed in the locale's way, a value out of range refused with
the range shown (and never reaching the document), lengths in millimetres or inches without the value drifting, the
changed-value marker and its reset, the reason a field is off, keyboard steps, and the segmented choices."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QLocale, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty, settings  # noqa: E402
from tessellatum.gui import fields  # noqa: E402
from tessellatum.gui.controls_panel import ControlsPanel  # noqa: E402
from tessellatum.gui.fields import INCH, MM, Display, SegmentedControl, SettingField, parse_number  # noqa: E402

_app = QApplication.instance() or QApplication([])

US = QLocale(QLocale.English, QLocale.UnitedStates)
GERMAN = QLocale(QLocale.German, QLocale.Germany)
FRENCH = QLocale(QLocale.French, QLocale.France)


# -- Numbers in three locales ------------------------------------------------------------------
@pytest.mark.parametrize("locale, text, expected", [
    (US, "0.7", 0.7), (US, "12", 12.0), (US, "0.7 mm", 0.7), (US, "1,234.5", 1234.5), (US, " 3.0 ΔE00 ", 3.0),
    (US, "−2", -2.0), (US, "125 mm²", 125.0),
    (GERMAN, "0,7", 0.7), (GERMAN, "0,7 mm", 0.7), (GERMAN, "1.234,5", 1234.5), (GERMAN, "0.7", 0.7),
    (FRENCH, "0,7", 0.7), (FRENCH, "1 234,5", 1234.5), (FRENCH, "1 234,5", 1234.5), (FRENCH, "0.7 mm", 0.7),
])
def test_a_number_is_read_the_locale_s_way_or_the_plain_way(locale, text, expected):
    assert parse_number(text, locale) == pytest.approx(expected)


@pytest.mark.parametrize("locale", [US, GERMAN, FRENCH])
@pytest.mark.parametrize("text", ["", "mm", "abc", "1,2,3.4.5", "--1"])
def test_what_isn_t_a_number_isn_t_read_as_one(locale, text):
    assert parse_number(text, locale) is None


def test_numbers_are_shown_the_locale_s_way():
    setting = settings.setting("min_width_mm")
    display = Display(setting)
    assert display.text(0.7, US) == "0.7 mm"
    assert display.text(0.7, GERMAN) == "0,7 mm"
    assert display.text(0.7, FRENCH) == "0,7 mm"
    assert display.range_text(GERMAN) == "0,5 to 6,0 mm"


def test_a_german_value_typed_in_a_german_window(monkeypatch):
    QLocale.setDefault(GERMAN)
    try:
        field = SettingField(settings.setting("min_width_mm"))
        assert field.text() == "0,7 mm"
        assert field.type_text("1,5") and field.value() == pytest.approx(1.5) and field.text() == "1,5 mm"
    finally:
        QLocale.setDefault(US)


# -- Units -----------------------------------------------------------------------------
def test_millimetres_to_inches_and_back_never_change_the_value():
    panel = ControlsPanel()
    lengths = [c for c in panel.controls.values() if c.setting.unit_kind in (settings.LENGTH, settings.AREA)]
    before = {c.setting.name: c.value() for c in lengths}
    for _ in range(5):
        panel.set_length_units(INCH)
        panel.set_length_units(MM)
    assert {c.setting.name: c.value() for c in lengths} == before
    panel.set_length_units(INCH)
    brush = panel.controls["min_width_mm"]
    assert brush.text() == "0.028 in" and brush.value() == pytest.approx(0.7)
    assert panel.controls["min_region_area_mm2"].text().endswith(" in²")
    assert panel.controls["min_label_pt"].text() == "3.0 pt"  # points stay points
    assert panel.controls["line_gray"].text().endswith("L*")
    # A length typed in inches is kept in millimetres.
    assert brush.type_text("0.04 in") and brush.value() == pytest.approx(1.016)
    panel.set_length_units(MM)
    assert brush.text() == "1.0 mm" and brush.value() == pytest.approx(1.016)
    with pytest.raises(ValueError):
        brush.set_units("cubits")


# -- Typing and its errors -------------------------------------------------------------------
def test_a_value_out_of_range_is_refused_with_the_range_and_never_reaches_the_document():
    panel = ControlsPanel()
    announced = []
    panel.settings_changed.connect(announced.append)
    brush = panel.controls["min_width_mm"]
    before = panel.snapshot()

    assert not brush.type_text("9 mm")
    assert brush.value() == pytest.approx(0.7) and announced == [] and panel.snapshot() == before
    assert brush.error() == "Out of range: from 0.5 to 6.0 mm." and brush.message.text() == brush.error()
    assert not brush.message.isHidden() and brush.entry.property("invalid") is True
    assert brush.entry.accessibleDescription() == brush.error()

    assert not brush.type_text("wide")
    assert brush.error() == "Type a number from 0.5 to 6.0 mm." and announced == []

    # Esc puts the box back as it was.
    brush.entry.setFocus()
    QTest.keyClick(brush.entry, Qt.Key_Escape)
    assert brush.error() is None and brush.message.isHidden() and brush.entry.text() == "0.7 mm"

    # A good value is taken, and announced once.
    assert brush.type_text("1.2")
    assert brush.value() == pytest.approx(1.2) and len(announced) == 1
    assert announced[-1].values["min_width_mm"] == pytest.approx(1.2)
    # The same value again is no edit.
    assert brush.type_text("1.2 mm") and len(announced) == 1


def test_a_gray_is_typed_as_its_lightness():
    field = SettingField(settings.setting("line_gray"))
    assert field.type_text("50")
    assert field.text() == "50 L*" and 0 <= field.value() <= 254 and field.value() == int(field.value())
    assert not field.type_text("120") and "0 to 100 L*" in field.error()


def test_a_count_takes_whole_numbers():
    field = SettingField(settings.setting("num_colors"))
    assert field.type_text("30.4") and field.value() == 30 and field.text() == "30"
    assert not field.type_text("65") and field.error() == "Out of range: from 2 to 64."


# -- The keyboard --------------------------------------------------------------------------
def test_up_down_and_the_page_keys_step_it_in_the_box_and_on_the_slider():
    field = SettingField(settings.setting("min_width_mm"))
    edits = []
    field.edited.connect(lambda: edits.append(field.value()))
    field.entry.setFocus()
    QTest.keyClick(field.entry, Qt.Key_Up)
    assert field.value() == pytest.approx(0.8)
    QTest.keyClick(field.entry, Qt.Key_PageUp)
    assert field.value() == pytest.approx(1.8)
    QTest.keyClick(field.entry, Qt.Key_Down)
    assert field.value() == pytest.approx(1.7)
    field.slider.setFocus()
    QTest.keyClick(field.slider, Qt.Key_Right)
    assert field.value() == pytest.approx(1.8) and field.entry.text() == "1.8 mm"
    QTest.keyClick(field.slider, Qt.Key_PageDown)
    assert field.value() == pytest.approx(0.8)
    assert len(edits) == 5
    # At an end, a step further is no edit.
    field.set_value(6.0)
    QTest.keyClick(field.entry, Qt.Key_Up)
    assert field.value() == 6.0 and len(edits) == 5


def test_a_drag_is_announced_as_it_starts_and_ends():
    field = SettingField(settings.setting("min_width_mm"))
    happened = []
    field.edit_started.connect(lambda: happened.append("start"))
    field.edited.connect(lambda: happened.append("edit"))
    field.edit_finished.connect(lambda: happened.append("end"))
    field.slider.setSliderDown(True)
    field.slider.setValue(field.slider.value() + 3)
    field.slider.setValue(field.slider.value() + 2)
    field.slider.setSliderDown(False)
    assert happened == ["start", "edit", "edit", "end"]


def test_names_values_and_descriptions_for_a_screen_reader():
    setting = settings.setting("min_width_mm")
    field = SettingField(setting)
    assert field.slider.accessibleName() == "Brush width"
    assert field.slider.accessibleDescription().startswith("0.7 mm. ")
    assert field.entry.accessibleName() == "Brush width, exact value" and field.entry.text() == "0.7 mm"
    field.set_value(2.0)
    assert field.slider.accessibleDescription().startswith("2.0 mm. ")


# -- The marker, reset and the reason it is off -----------------------------------------------
def test_a_changed_field_is_marked_and_its_marker_puts_it_back():
    panel = ControlsPanel()
    brush = panel.controls["min_width_mm"]
    medium = difficulty.params_for_preset("Medium").min_width_mm
    assert not brush.is_changed() and brush.marker.isHidden() and panel.changed_settings() == []

    assert brush.type_text("1.5")
    assert panel.preset_combo.currentText() == "Custom"
    assert brush.is_changed() and not brush.marker.isHidden()
    assert brush.marker.toolTip().startswith(f"Changed from Medium: {medium:.1f} mm")
    assert brush.marker.text() == fields.MARKER  # a shape, not a color only
    assert panel.changed_settings() == ["min_width_mm"]

    brush.marker.click()
    assert brush.value() == pytest.approx(medium) and brush.marker.isHidden()
    assert panel.preset_combo.currentText() == "Medium"  # every setting the preset's again: the preset again

    # Off another preset, it is compared with that one.
    panel.preset_combo.setCurrentText("Beginner")
    assert panel.changed_settings() == []
    brush.step(3)
    assert brush.marker.toolTip().startswith("Changed from Beginner:")
    # The grays against their tone, the rest against their defaults.
    panel.controls["line_gray"].step(5)
    assert panel.tone_combo.currentText() == "Custom"
    assert "the Medium tone" in panel.controls["line_gray"].marker.toolTip()
    panel.controls["line_width_mm"].step(2)
    assert "the default" in panel.controls["line_width_mm"].marker.toolTip()


def test_the_context_menu_resets_to_the_preset_or_the_factory_default():
    panel = ControlsPanel()
    panel.preset_combo.setCurrentText("Beginner")
    brush = panel.controls["min_width_mm"]
    brush.step(2)
    actions = [a for a in brush.menu().actions()]
    beginner = difficulty.params_for_preset("Beginner").min_width_mm
    assert actions[0].text() == f"Reset to Beginner ({beginner:.1f} mm)" and actions[0].isEnabled()
    assert actions[1].text() == f"Reset to the factory default ({brush.setting.default:.1f} mm)"
    actions[1].trigger()
    assert brush.value() == pytest.approx(brush.setting.default) and panel.preset_combo.currentText() == "Custom"
    brush.menu().actions()[0].trigger()
    assert brush.value() == pytest.approx(beginner) and panel.preset_combo.currentText() == "Beginner"


def test_the_marker_s_preset_is_kept_between_runs(tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    first = ControlsPanel(store=store)
    first.preset_combo.setCurrentText("Hard")
    first.controls["min_width_mm"].step(4)
    again = ControlsPanel(store=store)
    assert again.preset_combo.currentText() == "Custom"
    assert again.changed_settings() == ["min_width_mm"]
    assert again.controls["min_width_mm"].marker.toolTip().startswith("Changed from Hard:")


def test_a_field_that_means_nothing_says_why_under_it():
    panel = ControlsPanel()
    mix = panel.controls["mix_gain_de00"]
    assert not mix.isEnabled() and mix.disabled_reason() == "Only with Colors from: My paints."
    assert mix.message.text() == "Only with Colors from: My paints." and not mix.message.isHidden()
    assert mix.message.property("role") == "reason"
    panel.detail_check.setChecked(False)
    assert panel.controls["detail_weight"].message.text() == settings.ONLY_WITH_DETAIL
    # Said once in its section, and once in the Advanced group: the fields under the first have it in their tooltips.
    marks = panel.controls["mark_contrast"]
    assert not marks.isEnabled() and marks.message.text() == settings.ONLY_WITH_DETAIL and not marks.message.isHidden()
    length = panel.controls["mark_length_mm"]
    assert not length.isEnabled() and length.message.isHidden()
    assert length.toolTip().startswith(settings.ONLY_WITH_DETAIL)
    assert panel.controls["mix_min_share_pct"].message.isHidden()
    panel.detail_check.setChecked(True)
    detail = panel.controls["detail_weight"]
    assert detail.isEnabled() and detail.message.isHidden() and detail.disabled_reason() is None


# -- Segmented choices ------------------------------------------------------------------------
def test_a_segmented_control_answers_as_a_combo_and_moves_by_arrow_keys():
    control = SegmentedControl()
    added = []
    control.currentTextChanged.connect(added.append)
    control.addItems(["A4", "A3", "A2"])
    assert added == ["A4"]  # the first choice added is chosen, and said, as a combo's is
    changes = []
    control.currentTextChanged.connect(changes.append)
    control.setItemData(1, "Twice the A4", Qt.ToolTipRole)
    assert control.count() == 3 and control.currentText() == "A4" and control.currentIndex() == 0
    assert control.itemText(2) == "A2" and control.itemData(1, Qt.ToolTipRole) == "Twice the A4"
    control.setCurrentText("A2")
    assert control.currentText() == "A2" and changes == ["A2"]
    control.setCurrentText("Letter")  # not one of its choices: nothing changes, as with a combo
    assert control.currentText() == "A2"
    buttons = control.buttons()
    QTest.keyClick(buttons[2], Qt.Key_Left)
    assert control.currentText() == "A3"
    QTest.keyClick(buttons[1], Qt.Key_Right)
    QTest.keyClick(buttons[2], Qt.Key_Right)  # at the end it stays
    assert control.currentText() == "A2" and changes == ["A2", "A3", "A2"]
    # One stop in the Tab order: the choice made.
    assert [b.focusPolicy() == Qt.StrongFocus for b in buttons] == [False, False, True]
    control.blockSignals(True)
    control.setCurrentText("A4")
    control.blockSignals(False)
    assert changes == ["A2", "A3", "A2"]


def test_the_panel_s_few_choices_are_segmented():
    panel = ControlsPanel()
    for combo in (panel.paper_combo, panel.medium_combo, panel.colors_from_combo, panel.tone_combo):
        assert isinstance(combo, SegmentedControl) and 2 <= combo.count() <= 4
    panel.paper_combo.buttons()[1].click()
    assert panel.snapshot().paper == "A3"

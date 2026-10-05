"""The controls speak in the printed page's units and ask the pipeline for what they show."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty  # noqa: E402
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


def test_the_region_slider_spans_the_custom_range_in_equal_ratios():
    lo, hi = difficulty.CUSTOM_MIN_REGION_AREA_MM2_RANGE
    steps = controls_panel.REGION_SLIDER_STEPS

    assert controls_panel.region_slider_area_mm2(0) == lo
    assert controls_panel.region_slider_area_mm2(steps) == pytest.approx(hi)
    assert controls_panel.region_slider_area_mm2(steps // 2) == pytest.approx((lo * hi) ** 0.5)
    for position in range(steps + 1):
        assert controls_panel.region_slider_position(controls_panel.region_slider_area_mm2(position)) == position


def test_a_preset_is_passed_as_it_is_and_described_in_print_units(panel):
    panel.preset_combo.setCurrentText("Hard")

    assert panel.get_difficulty_params() == difficulty.params_for_preset("Hard")
    tooltip = panel.preset_combo.itemData(panel.preset_combo.findText("Hard"), Qt.ToolTipRole)
    assert tooltip == difficulty.describe(difficulty.params_for_preset("Hard"))
    assert "40 mm²" in tooltip


def test_the_custom_sliders_start_at_medium_and_show_their_units(panel):
    panel.preset_combo.setCurrentText("Custom")
    medium = difficulty.params_for_preset("Medium")
    params = panel.get_difficulty_params()

    assert (params.num_colors, params.blur_sigma) == (medium.num_colors, medium.blur_sigma)
    assert params.min_region_area_mm2 == pytest.approx(medium.min_region_area_mm2, rel=0.02)
    labels = [label.text() for label in panel.custom_group.findChildren(controls_panel.QLabel)]
    assert f"up to {medium.num_colors}" in labels
    assert f"{params.min_region_area_mm2:.0f} mm²" in labels
    # The subject and the faces found on a photograph or a painting may have regions half as large (D-043, D-048).
    assert "half that on a photograph's or painting's subject and faces" in panel.min_region_slider.toolTip()
    assert f"{medium.blur_sigma:.1f}" in labels


def test_the_custom_sliders_finest_setting_is_the_finest_page(panel):
    panel.preset_combo.setCurrentText("Custom")
    panel.colors_slider.setValue(panel.colors_slider.maximum())
    panel.min_region_slider.setValue(panel.min_region_slider.minimum())
    panel.blur_slider.setValue(panel.blur_slider.minimum())

    assert panel.get_difficulty_params() == difficulty.finest_params()


def test_the_style_controls_start_at_the_default_style_and_offer_widths_from_0_2_to_0_5_mm(panel):
    # D-052 (Q36).
    assert panel.get_page_style() == PageStyle()
    slider = panel.line_width_slider
    widths = [controls_panel.line_width_mm(position) for position in range(slider.minimum(), slider.maximum() + 1)]
    assert widths == [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    labels = [label.text() for label in panel.findChildren(controls_panel.QLabel)]
    assert "0.30 mm" in labels
    slider.setValue(slider.maximum())
    assert panel.get_page_style().line_width_mm == 0.5
    assert "0.50 mm" in [label.text() for label in panel.findChildren(controls_panel.QLabel)]


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


def test_any_version_of_the_page_can_be_exported_and_the_page_itself_is_the_default(panel):
    combo = panel.version_combo
    assert [combo.itemData(i) for i in range(combo.count())] == list(Version)
    assert [combo.itemText(i) for i in range(combo.count())] == [version.value for version in Version]
    assert panel.get_export_version() is Version.PAGE
    for index, version in enumerate(Version):
        assert combo.itemData(index, Qt.ToolTipRole) == version.description
        combo.setCurrentIndex(index)
        assert panel.get_export_version() is version


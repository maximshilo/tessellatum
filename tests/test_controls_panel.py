"""The difficulty controls speak in the printed page's units and ask the pipeline for what they show."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty  # noqa: E402
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
    assert f"{medium.blur_sigma:.1f}" in labels


def test_the_custom_sliders_finest_setting_is_the_finest_page(panel):
    panel.preset_combo.setCurrentText("Custom")
    panel.colors_slider.setValue(panel.colors_slider.maximum())
    panel.min_region_slider.setValue(panel.min_region_slider.minimum())
    panel.blur_slider.setValue(panel.blur_slider.minimum())

    assert panel.get_difficulty_params() == difficulty.finest_params()

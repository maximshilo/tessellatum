"""W8: Preferences. Each applies at once and is kept between runs; none is part of a document or its undo; units never
change a document's values; the interface's scale is apart from the canvas's zoom."""

import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QKeySequence, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import main_window, preferences, theme  # noqa: E402
from tessellatum.gui.fields import INCH, MM  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.preferences import (  # noqa: E402
    CACHE_KEY,
    CANVAS_KEY,
    SCALE_KEY,
    UNITS_KEY,
    Preferences,
    PreferencesDialog,
)

_app = QApplication.instance() or QApplication([])


def _store(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow(store=_store(tmp_path))
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.set_cache_pages(pipeline.DEFAULT_CACHE_PAGES)


def test_each_preference_is_kept_between_runs(tmp_path):
    first = Preferences(_store(tmp_path))
    first.set_value(UNITS_KEY, "Inches")
    first.set_value(CANVAS_KEY, "Neutral gray")
    first.set_value(SCALE_KEY, "125%")
    first.set_value(CACHE_KEY, "4 pages")
    again = Preferences(_store(tmp_path))
    assert again.units() == INCH and again.canvas_color() == "#808080" and again.cache_pages() == 4
    assert preferences.scale_at_start(_store(tmp_path)) == 1.25
    with pytest.raises(ValueError):
        again.set_value(UNITS_KEY, "furlongs")


def test_a_value_kept_that_isnt_a_choice_is_left_at_its_default(tmp_path):
    store = _store(tmp_path)
    store.setValue(UNITS_KEY, "furlongs")
    store.setValue(SCALE_KEY, "300%")
    prefs = Preferences(store)
    assert prefs.units() == MM and preferences.scale_at_start(store) is None


def test_the_scale_is_set_before_the_app_starts_unless_the_environment_set_it(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.setValue(SCALE_KEY, "150%")
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    preferences.apply_scale_at_start(store)
    assert os.environ["QT_SCALE_FACTOR"] == "1.5"
    monkeypatch.setenv("QT_SCALE_FACTOR", "2")
    preferences.apply_scale_at_start(store)
    assert os.environ["QT_SCALE_FACTOR"] == "2"


def test_units_change_how_lengths_read_and_never_a_value(window):
    controls = window.controls
    before = controls.snapshot()
    history = window.history.count()
    window.preferences.set_value(UNITS_KEY, "Inches")
    _app.processEvents()
    assert controls.controls["line_width_mm"].text().endswith("in")
    assert controls.snapshot() == before and window.history.count() == history  # no value, no undo step
    window.preferences.set_value(UNITS_KEY, "Millimeters")
    assert controls.controls["line_width_mm"].text().endswith("mm") and controls.snapshot() == before


def test_the_canvas_background(window):
    view = window.preview
    window.preferences.set_value(CANVAS_KEY, "Black")
    assert "#000000" in view.styleSheet()
    assert view.palette().color(QPalette.WindowText).lightnessF() > 0.5  # the placeholder readable on it
    window.preferences.set_value(CANVAS_KEY, "White")
    assert view.palette().color(QPalette.WindowText).lightnessF() < 0.5
    window.preferences.set_value(CANVAS_KEY, "Theme")
    assert view.styleSheet() == ""


def test_the_results_kept(window):
    window.preferences.set_value(CACHE_KEY, "8 pages")
    assert pipeline.cache_pages() == 8
    window.preferences.set_value(CACHE_KEY, "Automatic")
    assert pipeline.cache_pages() == pipeline.DEFAULT_CACHE_PAGES


def test_the_dialog_applies_each_change_at_once(window):
    window.commands["preferences"].trigger()
    dialog = window.preferences_dialog
    assert dialog.isVisible() and dialog.stack.count() == len(PreferencesDialog.PAGES)
    assert window.commands["preferences"].menuRole() == window.commands["preferences"].MenuRole.PreferencesRole
    dialog.units_combo.setCurrentText("Inches")
    assert window.controls.controls["min_width_mm"].text().endswith("in")
    dialog.canvas_combo.setCurrentText("Neutral gray")
    assert "#808080" in window.preview.styleSheet()
    assert dialog.restart_label.isHidden()
    dialog.scale_combo.setCurrentText("150%")
    assert not dialog.restart_label.isHidden()  # applies when started again
    dialog.scale_combo.setCurrentText("Automatic")
    assert dialog.restart_label.isHidden()
    dialog.cache_combo.setCurrentText("1 page")
    assert pipeline.cache_pages() == 1
    dialog.clear_cache_button.click()
    assert dialog.cache_label.text() == "Cleared."
    # The workspace's reset.
    window.set_inspector_visible(False)
    dialog.reset_layout_button.click()
    assert window.inspector_visible()
    # The shortcuts listed: every command with keys.
    table = dialog.shortcuts
    rows = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
    # In the platform's own words: Ctrl+F on Windows and Linux, ⌘F on macOS.
    assert QKeySequence("Ctrl+F").toString(QKeySequence.NativeText) in rows["Find Setting"] and "Preferences" in rows
    dialog.close()


def test_the_theme_and_motion_in_the_dialog_are_the_managers():
    manager = theme.install(_app)
    try:
        prefs = Preferences()
        dialog = PreferencesDialog(prefs)
        dialog.theme_combo.setCurrentText(theme.DARK_MODE)
        assert manager.mode() == theme.DARK_MODE
        manager.set_mode(theme.LIGHT_MODE)
        assert dialog.theme_combo.currentText() == theme.LIGHT_MODE
        dialog.motion_combo.setCurrentText("Reduced")
        assert manager.reduced_motion() and manager.duration_ms() == 0
        dialog.deleteLater()
    finally:
        manager.set_mode(theme.SYSTEM)
        manager.set_motion_mode(theme.SYSTEM)
        theme.uninstall(_app)


_SCALE_PROBE = r"""
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from PIL import Image
app = QApplication([])
from tessellatum.gui.preview_widget import PreviewWidget
view = PreviewWidget()
view.resize(800, 600)
view.show()
from tessellatum.gui.preview_widget import Version
view.show_page({Version.PAGE: Image.new("RGB", (1000, 700), "white")}, Image.new("RGB", (1000, 100), "white"))
view.zoom_to(1.0)
app.processEvents()
print(json.dumps({"ratio": view.devicePixelRatioF(), "zoom": view.zoom(), "m11": view.transform().m11()}))
"""


@pytest.mark.parametrize("scale", ["1", "1.5"])
def test_the_interface_scale_leaves_the_canvas_zoom_exact(scale):
    env = {**os.environ, "QT_SCALE_FACTOR": scale, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run([sys.executable, "-c", _SCALE_PROBE], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    assert probe["ratio"] == pytest.approx(float(scale))
    # 100% is a pixel of the page to a pixel of the screen, however large the interface is drawn.
    assert probe["zoom"] == pytest.approx(1.0) and probe["m11"] == pytest.approx(1 / float(scale))

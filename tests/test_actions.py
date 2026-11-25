"""The window's commands: one action each, in the menu bar and on its shortcut, behind its button and enabled with it;
and the document the window keeps, which the panel writes its settings into."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import SIGNAL, QEvent, QSettings  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from tessellatum import __version__  # noqa: E402
from tessellatum.app.snapshot import PDF, PIXELS, DocumentSettings  # noqa: E402
from tessellatum.core.legend import NAME  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui import actions, main_window, theme  # noqa: E402
from tessellatum.gui.controls_panel import ControlsPanel  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])

PAINTS = (("Ultramarine", (32, 0, 177)), ("Cadmium yellow", (255, 200, 0)), ("Titanium white", (250, 250, 250)))


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    widget = MainWindow()
    yield widget
    widget.close()
    widget.deleteLater()
    _app.processEvents()


def _menu_actions(menu: QMenu) -> list:
    found = []
    for action in menu.actions():
        if action.menu() is not None:
            found += _menu_actions(action.menu())
        elif not action.isSeparator():
            found.append(action)
    return found


def _wait(window: MainWindow, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while not window.jobs.is_idle():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)
    _app.processEvents()


def _blocks() -> np.ndarray:
    picture = np.zeros((200, 300, 3), np.uint8)
    picture[:100, :150] = (200, 40, 40)
    picture[100:, 150:] = (40, 170, 40)
    return picture


# -- The commands --------------------------------------------------------------------------------
def test_every_command_is_in_the_menu_bar_once(window):
    in_menus = [a for menu in window.menuBar().findChildren(QMenu) if menu.parent() is window.menuBar()
                for a in _menu_actions(menu)]
    assert sorted(a.objectName() for a in in_menus) == sorted(actions.COMMAND_IDS)


def test_no_two_commands_share_a_key(window):
    keys = [key.toString() for action in window.commands for key in action.shortcuts()]
    assert len(keys) == len(set(keys))
    assert not window.findChildren(QShortcut)  # the preview's zoom keys are the View menu's now


def test_no_command_has_a_key_twice(window):
    """A platform's standard key may also be named (Redo: Ctrl+Shift+Z); bound twice, Qt would call it ambiguous."""
    for action in window.commands:
        keys = [sequence.toString() for sequence in action.shortcuts()]
        assert len(keys) == len(set(keys)), action.objectName()
    assert window.commands["undo"].shortcuts() and len(window.commands["redo"].shortcuts()) >= 1


def test_tooltips_name_the_shortcut(window):
    tip = window.commands["generate"].toolTip()
    assert window.commands["generate"].shortcut().toString(QKeySequence.NativeText) in tip
    assert window.controls.generate_button.toolTip() == tip


def test_a_button_triggers_its_command(window, monkeypatch):
    asked = []
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: asked.append(1) or ("", "")))
    triggered = []
    window.commands["open"].triggered.connect(lambda: triggered.append(1))
    window.controls.open_button.click()
    assert triggered == [1] and asked == [1]


def test_commands_are_enabled_with_their_buttons(window):
    commands = window.commands
    assert commands["generate"].isEnabled() and not commands["export"].isEnabled()
    assert not commands["cancel"].isEnabled() and not commands["fit"].isEnabled()
    assert not commands[actions.VERSION_ACTIONS[Version.TINTED]].isEnabled()
    window.controls.set_busy(True)
    assert not commands["generate"].isEnabled() and not commands["open"].isEnabled() and commands["cancel"].isEnabled()
    window.controls.set_busy(True, cancellable=False)
    assert not commands["cancel"].isEnabled()
    window.controls.set_busy(False)
    assert commands["generate"].isEnabled() and not commands["cancel"].isEnabled()


def test_generate_then_view_and_export_by_command(window):
    window.current_image_bgr = _blocks()
    window.commands["generate"].trigger()
    _wait(window)
    assert window.current_page is not None
    commands = window.commands
    assert commands["export"].isEnabled() and commands["fit"].isEnabled()
    tinted = commands[actions.VERSION_ACTIONS[Version.TINTED]]
    assert tinted.isEnabled()
    tinted.trigger()
    assert window.version_bar.version() is Version.TINTED and tinted.isChecked()
    window.version_bar.buttons[Version.COMPLETED].click()
    assert commands[actions.VERSION_ACTIONS[Version.COMPLETED]].isChecked() and not tinted.isChecked()
    before = window.preview.zoom()
    commands["zoom_in"].trigger()
    assert window.preview.zoom() > before
    commands["fit"].trigger()
    assert window.preview.zoom() == pytest.approx(before)


def test_the_theme_commands_choose_the_appearance(tmp_path):
    store = QSettings(str(tmp_path / "theme.ini"), QSettings.IniFormat)
    manager = theme.install(_app, store)
    signals = ("changed(QString)", "mode_changed(QString)")
    listening = [manager.receivers(SIGNAL(signal)) for signal in signals]
    try:
        window = MainWindow()
        dark = window.commands[actions.THEME_ACTIONS[theme.DARK_MODE]]
        assert window.commands[actions.THEME_ACTIONS[manager.mode()]].isChecked()
        dark.trigger()
        assert manager.mode() == theme.DARK_MODE and dark.isChecked()
        window.commands[actions.THEME_ACTIONS[theme.LIGHT_MODE]].trigger()
        assert manager.mode() == theme.LIGHT_MODE and not dark.isChecked()
        manager.set_mode(theme.SYSTEM)  # chosen elsewhere (Preferences, later): the menu follows
        assert window.commands[actions.THEME_ACTIONS[theme.SYSTEM]].isChecked()
        # The empty canvas's text is drawn in the scene, not a widget: it takes the theme's text color too.
        for mode, scheme in ((theme.DARK_MODE, theme.tokens.DARK), (theme.LIGHT_MODE, theme.tokens.LIGHT)):
            manager.set_mode(mode)
            _app.processEvents()
            placeholder = window.preview._placeholder.defaultTextColor().name().upper()
            assert placeholder == theme.tokens.COLORS[scheme].text.upper()
        window.close()
        window.deleteLater()
        _app.sendPostedEvents(None, QEvent.DeferredDelete)
        # The manager outlives the window: nothing of the window is left listening to it.
        assert [manager.receivers(SIGNAL(signal)) for signal in signals] == listening
        manager.set_mode(theme.DARK_MODE)
    finally:
        theme.uninstall(_app)


def test_without_a_theme_its_commands_stand_aside(window):
    assert theme.current() is None
    assert not any(window.commands[i].isEnabled() for i in actions.THEME_ACTIONS.values())


def test_about_names_the_version_and_licenses():
    text = main_window.about_text()
    for part in (__version__, "Apache License 2.0", "Open Font License", "ISC", "LGPL"):
        assert part in text


# -- The document --------------------------------------------------------------------------------
def test_the_panel_writes_its_edits_into_the_document(window):
    doc = window.document
    assert doc.settings() == window.controls.snapshot()
    revisions = []
    doc.settings_changed.connect(revisions.append)
    window.controls.paper_combo.setCurrentText("A3")
    assert doc.settings().paper == "A3" and len(revisions) == 1
    window.controls.reset()
    assert doc.settings().paper == "A4" and len(revisions) == 2  # one change for the whole reset


def test_the_page_made_keeps_the_settings_it_was_made_with(window):
    window.current_image_bgr = _blocks()
    window.commands["generate"].trigger()
    _wait(window)
    doc = window.document
    assert doc.result() is window.current_page and doc.result_is_current()
    window.controls.legend_combo.setCurrentText(NAME)
    assert doc.result_is_current()  # the legend is drawn again, not the page
    window.controls.paper_combo.setCurrentText("A3")
    assert not doc.result_is_current() and doc.result_settings().paper == "A4"
    window.current_image_bgr = _blocks()
    assert window.current_page is None  # another picture: the page was of the last


def test_a_snapshot_shown_reads_back_the_same():
    edited = ControlsPanel()
    edited.set_paints(PAINTS)
    edited.mix_check.setChecked(True)
    edited.preset_combo.setCurrentText("Hard")
    edited.tone_combo.setCurrentText("Dark")
    edited.paper_combo.setCurrentText("A3")
    edited.legend_combo.setCurrentText(NAME)
    edited.text_check.setChecked(False)
    edited.pdf_radio.setChecked(True)
    edited.version_combo.setCurrentIndex(edited.version_combo.findData(Version.TINTED))
    edited.palette_combo.setCurrentIndex(1)
    edited.resolution_unit.setCurrentIndex(edited.resolution_unit.findData(PIXELS))
    wanted = edited.snapshot()
    assert wanted != DocumentSettings() and wanted.export.format == PDF and wanted.handling().mix_paints

    shown = ControlsPanel()
    announced = []
    shown.settings_changed.connect(announced.append)
    shown.show_settings(wanted)
    assert shown.snapshot() == wanted and announced == [wanted]
    assert shown.get_difficulty_params() == edited.get_difficulty_params()
    assert shown.get_handling() == edited.get_handling() and shown.get_page_style() == edited.get_page_style()
    shown.show_settings(wanted)
    assert announced == [wanted]  # nothing changed: nothing announced
    for panel in (edited, shown):
        panel.deleteLater()
    _app.processEvents()

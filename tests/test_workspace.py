"""The workspace (W1): the toolbar, the activity bar, the sidebar's pages, the inspector, and their layout kept
between runs, focus mode, Reset Layout, and the panels folding away on a narrow window."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

from tessellatum.core import settings  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.actions import ACTIVITY_ACTIONS  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_warm_up(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)


def _store(tmp_path) -> QSettings:
    """A settings file of the test's own: never the user's."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def _window(store=None, size=(1280, 840)) -> MainWindow:
    window = MainWindow(store=store)
    window.resize(*size)
    window.show()
    _app.processEvents()
    return window


def _close(window: MainWindow) -> None:
    window.close()
    window.deleteLater()
    _app.processEvents()


def _wait(window: MainWindow) -> None:
    deadline = time.monotonic() + 120
    while not window.jobs.is_idle():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)
    _app.processEvents()


def _blocks() -> np.ndarray:
    picture = np.zeros((200, 300, 3), np.uint8)
    picture[:, 150:] = (40, 170, 40)
    picture[100:, :150] = (40, 40, 200)
    return picture


def test_the_workspace_starts_with_every_panel_and_the_generation_page():
    window = _window()
    try:
        assert window.sidebar_visible() and window.inspector_visible() and not window.toolbar.isHidden()
        assert window.controls.category() == settings.GENERATION
        assert window.activity_bar.current() == settings.GENERATION
        assert window.commands[ACTIVITY_ACTIONS[settings.GENERATION]].isChecked()
        assert window.commands["toggle_sidebar"].isChecked() and window.commands["toggle_inspector"].isChecked()
        # Each activity names itself to a screen reader and in its tooltip.
        for category, button in window.activity_bar.buttons.items():
            assert button.accessibleName() == category and category in button.toolTip()
        assert window.inspector.color_count() == 0 and "Generate" in window.inspector.summary.text()
        assert window.inspector.title.geometry().top() < window.inspector.height() // 4  # at the top, not centered
    finally:
        _close(window)


def test_every_panel_toggles():
    window = _window()
    try:
        window.commands["toggle_sidebar"].trigger()
        assert not window.sidebar_visible() and not window.commands["toggle_sidebar"].isChecked()
        window.commands["toggle_sidebar"].trigger()
        assert window.sidebar_visible()
        window.commands["toggle_inspector"].trigger()
        assert not window.inspector_visible() and not window.commands["toggle_inspector"].isChecked()
        window.commands["toggle_inspector"].trigger()
        assert window.inspector_visible()

        # An activity shows its page; the page already shown, clicked, folds the sidebar; clicked again, unfolds it.
        window.activity_bar.buttons[settings.COLORS].click()
        assert window.controls.category() == settings.COLORS and window.sidebar_visible()
        window.activity_bar.buttons[settings.COLORS].click()
        assert not window.sidebar_visible()
        window.activity_bar.buttons[settings.COLORS].click()
        assert window.sidebar_visible() and window.controls.category() == settings.COLORS
        # Its command too, which unfolds the sidebar if it was folded.
        window.set_sidebar_visible(False)
        window.commands[ACTIVITY_ACTIONS[settings.SOURCE]].trigger()
        assert window.sidebar_visible() and window.controls.category() == settings.SOURCE
        assert window.activity_bar.current() == settings.SOURCE
    finally:
        _close(window)


def test_the_layout_is_kept_between_runs(tmp_path):
    store = _store(tmp_path)
    window = _window(store, size=(1280, 560))
    window.show_activity(settings.DOCUMENT)
    window.set_inspector_visible(False)
    window.controls.sections["Document · Advanced"].toggle.setChecked(True)
    window.controls.sections["Document · Advanced · Numbers"].toggle.setChecked(True)  # a subsection, kept too
    window.controls.sections["Tone"].toggle.setChecked(False)
    window.splitter.setSizes([420, 860, 0])
    sizes = window.splitter.sizes()
    _app.processEvents()
    scroll = window.controls.scrolls[settings.DOCUMENT].verticalScrollBar()
    scroll.setValue(scroll.maximum() // 2)
    kept_scroll = scroll.value()
    _close(window)

    again = _window(store, size=(1280, 560))
    try:
        assert again.controls.category() == settings.DOCUMENT and again.activity_bar.current() == settings.DOCUMENT
        assert again.sidebar_visible() and not again.inspector_visible()
        assert again.controls.sections["Document · Advanced"].is_expanded()
        assert again.controls.sections["Document · Advanced · Numbers"].is_expanded()
        assert not again.controls.sections["Document · Advanced · Lines"].is_expanded()
        assert not again.controls.sections["Tone"].is_expanded()
        assert again.splitter.sizes()[0] == sizes[0]
        _app.processEvents()  # the scroll positions come back once the pages are laid out
        assert kept_scroll > 0
        assert again.controls.scrolls[settings.DOCUMENT].verticalScrollBar().value() == kept_scroll
    finally:
        _close(again)


def test_focus_mode_hides_every_panel_and_brings_them_back_exactly():
    window = _window()
    try:
        window.set_inspector_visible(False)
        window.splitter.setSizes([380, 900, 0])
        before = window.splitter.sizes()
        window.commands["focus_mode"].trigger()
        assert window.in_focus_mode() and window.commands["focus_mode"].isChecked()
        for panel in (window.toolbar, window.activity_bar, window.controls, window.inspector, window.statusBar()):
            assert panel.isHidden()
        assert not window.canvas_stack.isHidden()  # the canvas: the page, or the welcome page with no picture
        # The way back is said, and stays on the canvas's own bar while the toolbar is hidden.
        assert window.notification.isVisible() and "Esc" in window.notification.text.text()
        assert window.exit_focus_action.isVisible() and not window.canvas_bar.isHidden()
        window.exit_focus_action.trigger()
        assert not window.in_focus_mode() and not window.exit_focus_action.isVisible()
        assert not window.notification.isVisible()
        window.commands["focus_mode"].trigger()

        QTest.keyClick(window, Qt.Key_Escape)  # Esc brings them back; the same command does too
        assert not window.in_focus_mode() and not window.commands["focus_mode"].isChecked()
        assert window.sidebar_visible() and not window.inspector_visible()
        assert not window.toolbar.isHidden() and not window.statusBar().isHidden()
        assert window.splitter.sizes() == before

        window.commands["focus_mode"].trigger()
        window.commands["focus_mode"].trigger()
        assert not window.in_focus_mode() and window.splitter.sizes() == before
    finally:
        _close(window)


def test_reset_layout_puts_the_defaults_back(tmp_path):
    store = _store(tmp_path)
    window = _window(store)
    try:
        expanded = {name: section.is_expanded() for name, section in window.controls.sections.items()}
        window.show_activity(settings.SOURCE)
        window.set_inspector_visible(False)
        window.set_sidebar_visible(False)
        window.controls.sections["Detail"].toggle.setChecked(False)
        window.save_layout()
        window.commands["reset_layout"].trigger()

        assert window.sidebar_visible() and window.inspector_visible()
        assert window.controls.category() == settings.GENERATION
        assert {name: section.is_expanded() for name, section in window.controls.sections.items()} == expanded
        assert not [key for key in store.allKeys() if key.startswith("workspace/")]  # nothing kept to override it
    finally:
        _close(window)


def test_a_narrow_window_folds_the_inspector_then_the_sidebar_and_a_wider_one_brings_them_back():
    window = _window(size=(1280, 840))
    try:
        window.resize(1000, 700)  # 1280 x 720 at 125%
        _app.processEvents()
        assert not window.inspector_visible() and window.sidebar_visible()
        window.resize(740, 600)
        _app.processEvents()
        assert not window.inspector_visible() and not window.sidebar_visible()
        assert not window.activity_bar.isHidden()  # the way back to the settings stays
        window.resize(1280, 840)
        _app.processEvents()
        assert window.inspector_visible() and window.sidebar_visible()

        # Folded by the user, it stays folded however wide the window grows.
        window.set_inspector_visible(False)
        window.resize(1000, 700)
        window.resize(1400, 840)
        _app.processEvents()
        assert not window.inspector_visible()
    finally:
        _close(window)


def test_at_the_smallest_size_nothing_clips():
    window = _window(size=(main_window.MIN_WINDOW.width(), main_window.MIN_WINDOW.height()))
    try:
        window.resize(main_window.MIN_WINDOW)
        _app.processEvents()
        assert window.width() == main_window.MIN_WINDOW.width()
        # Every toolbar button shows whole, or has gone into the toolbar's overflow menu.
        for bar in window.findChildren(QToolBar):
            if bar.isHidden():
                continue
            for action in bar.actions():
                widget = bar.widgetForAction(action)
                if widget is not None and widget.isVisible():
                    assert bar.rect().contains(widget.geometry()), action.text() or widget.objectName()
        # Asked for at the smallest size, the sidebar gets the room its widest page needs.
        window.set_sidebar_visible(True)
        _app.processEvents()
        page = window.controls.pages[window.controls.category()]
        assert window.controls.scrolls[window.controls.category()].viewport().width() >= page.minimumSizeHint().width()
    finally:
        _close(window)


def test_the_inspector_lists_the_page_s_colors_and_the_toolbar_runs_or_cancels():
    window = _window()
    try:
        assert window.run_button.defaultAction() is window.commands["generate"]
        window.current_image_bgr = _blocks()
        window.commands["generate"].trigger()
        assert window.run_button.defaultAction() is window.commands["cancel"]  # one button: stop while it runs
        _wait(window)
        assert window.run_button.defaultAction() is window.commands["generate"]
        page = window.current_page
        assert window.inspector.color_count() == len(page.palette_rgb) > 0
        first = window.inspector.list.item(0)
        r, g, b = (int(v) for v in page.palette_rgb[0])
        assert first.text().startswith("1") and f"#{r:02X}{g:02X}{b:02X}" in first.toolTip()
        assert f"{len(page.palette_rgb)} colors" in window.inspector.summary.text()
        window.current_image_bgr = _blocks()  # a new picture: its page isn't made yet
        assert window.inspector.color_count() == 0
    finally:
        _close(window)

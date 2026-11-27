"""X1: keyboard and accessibility. Every control a screen reader reaches has a name; Tab goes round the workspace's
areas in order (toolbar, activity bar, sidebar, canvas, inspector, status bar) without a trap, Shift+Tab the other way;
a control reached inside a scrolled page is scrolled into view; the focus comes back after a dialog; the novice's and
the expert's paths work from the keyboard alone; nothing is said by color alone; the canvas can be shown as seen with
another color vision, on the screen only."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QAccessible, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox, QHeaderView,
    QAbstractScrollArea, QLineEdit, QPlainTextEdit, QScrollArea, QScrollBar, QTabBar, QTextEdit, QWidget,
)

from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import accessibility, freshness, main_window, vision  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])

INTERACTIVE = (QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QComboBox, QLineEdit, QTabBar,
               QTextEdit, QPlainTextEdit)
# The workspace's areas, in the order Tab goes round them.
AREAS = ("mainToolbar", "activityBar", "sidebar", "canvas", "inspector", "statusBar")


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    window.activateWindow()
    _app.processEvents()
    yield window
    window.export_dialog.reject()
    if window.preferences_dialog is not None:
        window.preferences_dialog.reject()
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _blocks(size=(400, 300)) -> np.ndarray:
    width, height = size
    picture = np.zeros((height, width, 3), np.uint8)
    picture[:, width // 2 :] = (40, 170, 40)
    picture[height // 2 :, : width // 2] = (40, 40, 200)
    picture[: height // 4, : width // 4] = (220, 220, 230)
    return picture


def _wait(window: MainWindow) -> None:
    deadline = time.monotonic() + 120
    while True:
        _app.processEvents()
        if window.jobs.is_idle():
            _app.processEvents()
            if window.jobs.is_idle():
                return
        assert time.monotonic() < deadline
        time.sleep(0.005)


def _previewed(window: MainWindow) -> None:
    window.current_image_bgr = _blocks()
    window.generate_preview()
    _wait(window)


def _name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    return interface.text(QAccessible.Text.Name).strip() if interface is not None else ""


def _qt_internal(widget: QWidget, popups: set) -> bool:
    """A part Qt makes and names by its role (a scroll bar, a list's header, a combo box's pop-up list, in
    ``popups``), or the line edit inside a spin box, which a screen reader reads as the spin box. (An editable combo
    box's line edit is an element of its own in Windows' UI Automation tree, so it needs a name.)"""
    if isinstance(widget, (QScrollBar, QHeaderView)) or widget in popups \
            or widget.objectName() == "qt_tableview_cornerbutton":  # a table's select-all corner, no Tab stop
        return True
    return isinstance(widget, QLineEdit) and isinstance(widget.parentWidget(), QAbstractSpinBox)


def _unnamed(root: QWidget) -> list[str]:
    found = []
    popups = {combo.view() for combo in root.findChildren(QComboBox)}
    for widget in root.findChildren(QWidget):
        focusable = widget.focusPolicy() & Qt.TabFocus
        ours = type(widget).__module__.startswith("tessellatum")
        if not (isinstance(widget, INTERACTIVE) or (focusable and ours)) or _qt_internal(widget, popups):
            continue
        if not _name(widget):
            found.append(f"{type(widget).__name__} {widget.objectName()!r} in {widget.parentWidget().objectName()!r}")
    # A graphics view's viewport is an element of its own in the UI Automation tree (an item view's or a text box's
    # is the view itself, and a scroll area's holds its controls).
    plain = (QAbstractItemView, QScrollArea, QTextEdit, QPlainTextEdit)
    for view in root.findChildren(QAbstractScrollArea):
        if view.focusPolicy() & Qt.TabFocus and not isinstance(view, plain) and not _name(view.viewport()):
            found.append(f"the viewport of {type(view).__name__} {view.objectName()!r}")
    return found


def _area(widget: QWidget, window: MainWindow) -> str:
    while widget is not None and widget is not window:
        if widget.objectName() in AREAS:
            return widget.objectName()
        if widget.objectName() in ("canvasToolbar", "canvasFooter", "welcome"):
            return "canvas"
        widget = widget.parentWidget()
    return "?"


def _tab_round(window: MainWindow, forward: bool = True) -> list[QWidget]:
    """The widgets Tab (or Shift+Tab) stops at, from the one with the focus round to it again."""
    first = _app.focusWidget()
    stops = []
    for _ in range(200):
        window.focusNextPrevChild(forward)
        _app.processEvents()
        widget = _app.focusWidget()
        if widget is first:
            return stops
        stops.append(widget)
    raise AssertionError("Tab never came back to where it started: a trap, or a cycle that skips it")


def _runs(areas: list[str]) -> list[str]:
    """The areas in the order a round visits them, each once per visit; a round started inside an area visits it
    at both ends, which is one visit."""
    runs = [area for i, area in enumerate(areas) if i == 0 or areas[i - 1] != area]
    if len(runs) > 1 and runs[0] == runs[-1]:
        runs.pop()
    return runs


def _rotation_of(sequence: list[str], wanted: tuple[str, ...]) -> bool:
    if sorted(sequence) != sorted(set(sequence)):
        return False  # an area visited twice in one round
    order = [area for area in wanted if area in sequence]
    start = order.index(sequence[0])
    return sequence == order[start:] + order[:start]


# -- Names --------------------------------------------------------------------------------------
def test_every_control_has_a_name_on_the_welcome_and_with_a_page(window):
    assert _unnamed(window) == []
    _previewed(window)
    window.set_vision(vision.PROTANOPIA)  # its badge shown
    assert _unnamed(window) == []


def test_every_control_in_the_dialogs_has_a_name(window):
    _previewed(window)
    window.commands["export"].trigger()
    assert _unnamed(window.export_dialog) == []
    window.export_dialog.reject()
    window.open_preferences()
    assert _unnamed(window.preferences_dialog) == []


def test_the_parts_qt_makes_get_names(window):
    for bar in ("mainToolbar", "canvasToolbar", "canvasFooter"):
        toolbar = window.findChild(QWidget, bar)
        overflow = toolbar.findChild(QAbstractButton, "qt_toolbar_ext_button")
        assert _name(overflow) == accessibility.OVERFLOW_NAME
    clear = [button for button in window.inspector.search.findChildren(QAbstractButton)]
    assert clear and all(_name(button) == accessibility.CLEAR_NAME for button in clear)


def test_the_palette_s_colors_are_read_out_with_what_they_are(window):
    _previewed(window)
    interface = QAccessible.queryAccessibleInterface(window.inspector.list)
    items = [interface.child(i).text(QAccessible.Text.Name) for i in range(interface.childCount())]
    assert len(items) == window.inspector.list.count() >= 3
    for number, text in enumerate(items, start=1):
        assert text.startswith(f"Color {number}, ") and "#" in text and "% of the page" in text


# -- The Tab path -------------------------------------------------------------------------------
def test_tab_goes_round_the_areas_in_order_and_shift_tab_the_other_way(window):
    _previewed(window)
    window.preview.setFocus()
    _app.processEvents()
    forward = _tab_round(window)
    areas = [_area(widget, window) for widget in forward]
    assert "?" not in areas
    assert _rotation_of(_runs(areas), AREAS), _runs(areas)
    assert all(_name(widget) for widget in forward), [type(w).__name__ for w in forward if not _name(w)]
    # Shift+Tab stops at the same widgets in the opposite order.
    backward = _tab_round(window, forward=False)
    assert backward == list(reversed(forward))


def test_the_welcome_s_scroll_areas_are_no_stop_of_their_own(window):
    stops = _tab_round(window)
    assert window.welcome not in stops
    assert window.welcome.open_picture_button in stops
    assert not any(type(widget).__name__ == "QScrollArea" for widget in stops)


def test_a_control_reached_by_keyboard_is_scrolled_into_view(window):
    window.resize(1280, 480)  # the smallest window: a sidebar's page taller than its pane
    _app.processEvents()
    tallest, shown = None, None
    for category in main_window.registry.CATEGORIES:
        window.show_activity(category)
        _app.processEvents()
        scroll = window.controls.pages[category].parentWidget().parentWidget()
        if tallest is None or scroll.verticalScrollBar().maximum() > tallest.verticalScrollBar().maximum():
            tallest, shown = scroll, category
    window.show_activity(shown)
    _app.processEvents()
    scroll = tallest
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(0)
    # Shift+Tab from the canvas's first stop into the sidebar lands on its last control, at the page's foot.
    window.controls.generate_button.setFocus()  # a stop just past the sidebar's scrolled page
    _app.processEvents()
    for _ in range(3):
        window.focusNextPrevChild(False)
        _app.processEvents()
        widget = _app.focusWidget()
        if scroll.widget().isAncestorOf(widget):
            break
    assert scroll.widget().isAncestorOf(widget)
    top_left = widget.mapTo(scroll.viewport(), widget.rect().topLeft())
    assert 0 <= top_left.y() and top_left.y() + widget.height() <= scroll.viewport().height()


def test_the_focus_comes_back_after_a_dialog(window):
    _previewed(window)
    window.preview.setFocus()
    _app.processEvents()
    window.commands["export"].trigger()
    _app.processEvents()
    assert window.export_dialog.isVisible() and _app.focusWidget() is window.export_dialog.export_button
    QTest.keyClick(window.export_dialog, Qt.Key_Escape)
    _app.processEvents()
    assert not window.export_dialog.isVisible()
    window.activateWindow()
    _app.processEvents()
    assert _app.focusWidget() is window.preview


def test_a_one_down_row_is_one_stop_its_arrows_move_along_it(window):
    _previewed(window)
    window.resize(1920, 1080)  # room for the view buttons, with the offscreen platform's wide fonts
    _app.processEvents()
    bar = window.version_bar
    assert not bar.is_compact()
    page = bar.buttons[main_window.Version.PAGE]
    page.setFocus()
    _app.processEvents()
    QTest.keyClick(page, Qt.Key_Right)
    _app.processEvents()
    assert _app.focusWidget() is bar.buttons[main_window.Version.COMPLETED]
    stops = _tab_round(window)
    assert sum(1 for button in bar.buttons.values() if button in stops) <= 1


def test_where_the_view_buttons_give_way_the_list_in_their_place_is_one_stop_and_its_arrows_choose(window):
    _previewed(window)
    window.resize(900, 700)
    _app.processEvents()
    bar = window.version_bar
    assert bar.is_compact()
    bar.compact.setFocus()
    _app.processEvents()
    QTest.keyClick(bar.compact, Qt.Key_Down)
    _app.processEvents()
    assert bar.version() == main_window.Version.COMPLETED and window.preview.view() == main_window.Version.COMPLETED
    assert _name(bar.compact)
    stops = _tab_round(window)
    assert not any(button in stops for button in bar.buttons.values())  # the round came back to the list


# -- The novice's and the expert's paths, keys only ---------------------------------------------
def test_the_novice_path_from_the_keyboard(window, tmp_path, monkeypatch):
    from PIL import Image

    Image.fromarray(_blocks()[:, :, ::-1]).save(tmp_path / "blocks.png")
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "blocks.png"), "")))
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "page.png"), "")))
    window.controls.set_export_dpi(150)
    target = window.welcome.open_picture_button
    target.setFocus()
    QTest.keyClick(target, Qt.Key_Space)  # Open Picture…, from the welcome
    _wait(window)
    assert window.current_image_bgr is not None
    QTest.keyClick(_app.focusWidget() or window, Qt.Key_Return, Qt.ControlModifier)  # Generate Preview
    _wait(window)
    assert window.current_page is not None
    QTest.keyClick(_app.focusWidget() or window, Qt.Key_E, Qt.ControlModifier)  # Export…
    _app.processEvents()
    assert window.export_dialog.isVisible()
    QTest.keyClick(_app.focusWidget(), Qt.Key_Return)  # Export, the dialog's default
    _wait(window)
    assert (tmp_path / "page.png").exists()


def test_the_expert_path_from_the_keyboard(window):
    _previewed(window)
    QTest.keyClick(_app.focusWidget() or window, Qt.Key_F, Qt.ControlModifier)  # Find Setting…
    _app.processEvents()
    search = _app.focusWidget()
    assert isinstance(search, QLineEdit)
    QTest.keyClicks(search, "brush width")
    QTest.keyClick(search, Qt.Key_Return)  # to the setting found
    _app.processEvents()
    focused = _app.focusWidget()
    assert "Brush width" in _name(focused)
    before = window.controls.snapshot()
    QTest.keyClick(focused, Qt.Key_PageUp)  # a page's step along the slider
    _app.processEvents()
    assert window.controls.snapshot() != before
    QTest.keyClick(focused, Qt.Key_Return, Qt.ControlModifier)  # Generate Preview with it
    _wait(window)
    assert window.current_page is not None and window.freshness.state() == freshness.CURRENT


# -- Not by color alone -------------------------------------------------------------------------
def test_each_freshness_state_says_it_in_words():
    badge = freshness.FreshnessBadge()
    for state in freshness.STATES:
        badge.set_state(state)
        assert badge.text().strip() and _name(badge)


def test_a_changed_setting_is_marked_by_a_sign_and_named(window):
    from tessellatum.gui.fields import SettingField

    field = window.controls.findChildren(SettingField)[0]
    field.slider.setValue(field.slider.maximum() if field.slider.value() != field.slider.maximum()
                          else field.slider.minimum())
    _app.processEvents()
    assert not field.marker.isHidden()
    assert field.marker.text().strip() and _name(field.marker).startswith("Reset ")


# -- Color-vision simulation --------------------------------------------------------------------
def test_the_simulations_keep_black_white_and_grays():
    grays = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], np.uint8)
    for mode in vision.MODES:
        seen = vision.simulate_rgb(grays, mode).astype(int)
        assert np.abs(seen - grays.astype(int)).max() <= 2, mode


def test_red_and_green_look_alike_to_the_red_and_green_blind_not_to_the_blue_blind():
    red_green = np.array([[[200, 60, 50], [70, 140, 50]]], np.uint8)  # a red and a green of about the same value

    def apart(mode):
        seen = vision.simulate_rgb(red_green, mode).astype(float)
        return float(np.linalg.norm(seen[0, 0] - seen[0, 1]))

    normal = apart(vision.NORMAL)
    assert apart(vision.PROTANOPIA) < normal / 2 and apart(vision.DEUTERANOPIA) < normal / 4
    assert apart(vision.TRITANOPIA) > normal / 2
    gray = vision.simulate_rgb(red_green, vision.GRAYSCALE)
    assert (gray[..., 0] == gray[..., 1]).all() and (gray[..., 1] == gray[..., 2]).all()


def test_simulate_keeps_the_image_s_size_and_transparency():
    image = QImage(5, 3, QImage.Format_ARGB32)
    image.fill(0x80FF0000)
    seen = vision.simulate(image, vision.DEUTERANOPIA)
    assert (seen.width(), seen.height()) == (5, 3) and seen.hasAlphaChannel()
    assert seen.pixelColor(2, 1).alpha() == image.pixelColor(2, 1).alpha()
    assert vision.simulate(image, vision.NORMAL) is image


def test_the_canvas_is_shown_as_seen_and_says_so_on_the_screen_only(window):
    _previewed(window)
    window.version_bar.buttons[main_window.Version.COMPLETED].click()  # colored: the outlined page is black on white
    _app.processEvents()
    page_before = window.preview.shown_image().copy()
    shown_before = window.preview._page_item.pixmap().toImage()
    window.commands[main_window.VISION_ACTIONS[vision.DEUTERANOPIA]].trigger()
    _app.processEvents()
    assert window.preview.vision() == window.compare_view.vision() == vision.DEUTERANOPIA
    assert window.preview._page_item.pixmap().toImage() != shown_before  # what the screen shows changed
    assert window.preview.shown_image() == page_before  # the page itself didn't
    # Said under the canvas, and shown there on the default window.
    assert window.vision_badge.isVisible() and "Deuteranopia" in window.vision_badge.text()
    window.vision_badge.click()  # back to the page's own colors
    _app.processEvents()
    assert window.preview.vision() == vision.NORMAL and not window._vision_badge_action.isVisible()
    assert window.commands[main_window.VISION_ACTIONS[vision.NORMAL]].isChecked()
    assert window.preview._page_item.pixmap().toImage() == shown_before


# -- Reduced motion -----------------------------------------------------------------------------
def test_nothing_moves_unless_it_asks_the_theme_how_long(window):
    # The interface's own widgets don't animate, and Qt's layout animations (a toolbar's, a splitter's) are instant. An
    # animation added takes its duration from ``theme.current().duration_ms``, 0 with reduced motion; this catches one
    # added without it (the tests run with no theme, so every duration is the reduced one).
    from PySide6.QtCore import QAbstractAnimation

    _previewed(window)
    durations = [animation.duration() for animation in window.findChildren(QAbstractAnimation)]
    assert all(duration == 0 for duration in durations), durations

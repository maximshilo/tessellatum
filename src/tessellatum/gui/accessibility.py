"""Keyboard and screen-reader help Qt doesn't give on its own.

- A widget reached with the keyboard inside a scrolled page is scrolled into view, however the focus got there
  (Qt's scroll area does it only for Tab pressed inside it, not for focus arriving from outside or Shift+Tab into it).
- The parts Qt makes inside its own widgets get names a screen reader can say: a toolbar's or the menu bar's overflow
  button ("»") and a search field's clear button.
- The scroll areas around a page of controls are no stop of their own on the Tab path: the controls in them are, and
  the area scrolls to each.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QMenuBar, QScrollArea, QToolBar, QToolButton, QWidget

OVERFLOW_NAME = "More"
CLEAR_NAME = "Clear"

_installed: set[int] = set()  # the applications the focus is followed in, by id


def follow_focus(app: QApplication | None = None) -> None:
    """Scroll each widget the focus moves to into view in the scroll area it is in, from now on (once per
    application)."""
    app = app or QApplication.instance()
    if app is None or id(app) in _installed:
        return
    _installed.add(id(app))
    app.focusChanged.connect(_scroll_to_focus)


def _scroll_to_focus(_old: QWidget | None, new: QWidget | None) -> None:
    if new is None:
        return
    widget, parent = new, new.parentWidget()
    while parent is not None:
        area = parent.parentWidget()
        if isinstance(area, QScrollArea) and area.widget() is not None and area.widget().isAncestorOf(new):
            area.ensureWidgetVisible(new, 8, 8)
            return
        widget, parent = parent, parent.parentWidget()


def no_tab_stop(area: QScrollArea) -> QScrollArea:
    """``area`` scrolls to the controls in it as each takes the focus, rather than being a stop of its own."""
    area.setFocusPolicy(Qt.NoFocus)
    return area


def name_parts(root: QWidget) -> None:
    """Name the parts Qt makes inside ``root``'s toolbars, menu bar and search fields (see the module's doc)."""
    for bar in (*root.findChildren(QToolBar), *root.findChildren(QMenuBar)):
        for button in bar.findChildren(QToolButton, options=Qt.FindDirectChildrenOnly):
            if button.objectName() in ("qt_toolbar_ext_button", "qt_menubar_ext_button") \
                    and not button.accessibleName():
                button.setAccessibleName(OVERFLOW_NAME)
                button.setToolTip(f"{OVERFLOW_NAME}: what doesn't fit")
    for field in root.findChildren(QLineEdit):
        if field.isClearButtonEnabled():
            for button in field.findChildren(QToolButton):
                if not button.accessibleName():
                    button.setAccessibleName(CLEAR_NAME)

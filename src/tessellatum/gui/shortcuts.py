"""The keyboard reference: every command's keys from the window's actions, and the keys that work in one place (the
canvas, the palette, a field), searchable, in the platform's notation (⌘ on macOS).

Shown by Help > Keyboard Shortcuts and on the Preferences' Shortcuts page. Built from the actions themselves, so it
lists what the keys are, not what a document says they were.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHeaderView, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from tessellatum.gui.i18n import translate

MACOS = sys.platform == "darwin"
COMMAND, KEYS, WHERE = range(3)
PORTABLE_ROLE = Qt.UserRole  # the keys in Qt's portable words (Ctrl+F), found when typed so on macOS too


def native(keys: str) -> str:
    """``keys`` in Qt's portable words ("Ctrl+F", "Esc") as the platform writes them ("⌘F" on macOS)."""
    return QKeySequence(keys).toString(QKeySequence.NativeText)


def _modifier(name: str) -> str:
    """A modifier alone, as the platform writes it before something that isn't a key (a wheel, a drag)."""
    written = native(f"{name}+A")
    return written[:-1] if written.endswith("A") else name + "+"


def context_keys() -> tuple[tuple[str, str, str], ...]:
    """(what it does, the keys, where they work): the keys no command has, as this platform writes them."""
    ctrl, copy = _modifier(translate("shortcuts", "Ctrl")), QKeySequence(QKeySequence.Copy).toString(
        QKeySequence.NativeText
    )
    menu_bar = (
        translate("shortcuts", "Ctrl+F2 (with keyboard access on)") if MACOS else translate("shortcuts", "Alt, or F10")
    )
    return (
        (
            translate("shortcuts", "Next control, previous control"),
            translate("shortcuts", "Tab, Shift+Tab"),
            translate("shortcuts", "Window"),
        ),
        (
            translate("shortcuts", "Along a row of views or activities"),
            translate("shortcuts", "Arrow keys"),
            translate("shortcuts", "Window"),
        ),
        (translate("shortcuts", "Open the menu bar"), menu_bar, translate("shortcuts", "Window")),
        (translate("shortcuts", "Leave focus mode"), translate("shortcuts", "Esc"), translate("shortcuts", "Window")),
        (
            translate("shortcuts", "Move the page"),
            translate("shortcuts", "Space+drag, middle-button drag, or drag"),
            translate("shortcuts", "Canvas"),
        ),
        (
            translate("shortcuts", "Zoom about the pointer"),
            translate("shortcuts", "Wheel, {ctrl}wheel, or pinch").format(ctrl=ctrl),
            translate("shortcuts", "Canvas"),
        ),
        (
            translate("shortcuts", "Scroll the page"),
            translate("shortcuts", "Arrow keys"),
            translate("shortcuts", "Canvas"),
        ),
        (
            translate("shortcuts", "Move between colors"),
            translate("shortcuts", "Arrow keys"),
            translate("shortcuts", "Palette"),
        ),
        (
            translate("shortcuts", "List the color's regions"),
            translate("shortcuts", "Enter"),
            translate("shortcuts", "Palette"),
        ),
        (translate("shortcuts", "Copy the color's hex value"), copy, translate("shortcuts", "Palette")),
        (
            translate("shortcuts", "Let go of the color"),
            translate("shortcuts", "Esc"),
            translate("shortcuts", "Palette"),
        ),
        (
            translate("shortcuts", "Next region, previous region"),
            translate("shortcuts", "Down, Up"),
            translate("shortcuts", "Regions"),
        ),
        (
            translate("shortcuts", "Back to the palette"),
            translate("shortcuts", "Esc"),
            translate("shortcuts", "Regions"),
        ),
        (
            translate("shortcuts", "Step a value"),
            translate("shortcuts", "Up, Down; Page Up, Page Down for ten steps"),
            translate("shortcuts", "Settings"),
        ),
        (
            translate("shortcuts", "Put back the value being typed"),
            translate("shortcuts", "Esc"),
            translate("shortcuts", "Settings"),
        ),
        (
            translate("shortcuts", "Through the settings found; go to one"),
            translate("shortcuts", "Down, Up; Enter"),
            translate("shortcuts", "Find Setting"),
        ),
        (
            translate("shortcuts", "Clear the search"),
            translate("shortcuts", "Esc"),
            translate("shortcuts", "Find Setting"),
        ),
    )


class ShortcutReference(QWidget):
    """A search field over a table: each command with keys (its menu in the third column), then the keys that work in
    one place. ``commands`` is the window's ``Actions`` (or any iterable of QActions)."""

    def __init__(self, commands=(), parent: QWidget | None = None):
        super().__init__(parent)
        self.search = QLineEdit()
        self.search.setObjectName("shortcutSearch")
        self.search.setPlaceholderText(translate("shortcuts", "Find a command or a key"))
        self.search.setAccessibleName(translate("shortcuts", "Find a shortcut"))
        self.search.setClearButtonEnabled(True)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("shortcutTable")
        self.table.setAccessibleName(translate("shortcuts", "Keyboard shortcuts"))
        self.table.setHorizontalHeaderLabels(
            [translate("shortcuts", "Command"), translate("shortcuts", "Keys"), translate("shortcuts", "Where")])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COMMAND, QHeaderView.Stretch)
        header.setSectionResizeMode(KEYS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(WHERE, QHeaderView.ResizeToContents)
        for action in commands:
            sequences = [key for key in action.shortcuts() if not key.isEmpty()]
            if sequences:
                where = action.property("menu") or ""
                self._add(action.text().replace("&", "").rstrip("…"),
                          ", ".join(key.toString(QKeySequence.NativeText) for key in sequences), where,
                          ", ".join(key.toString(QKeySequence.PortableText) for key in sequences))
        for what, keys, where in context_keys():
            self._add(what, keys, where, keys)
        self.empty = QLabel(translate("shortcuts", "No shortcut matches."))
        self.empty.setProperty("role", "hint")
        self.empty.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty)
        self.search.textChanged.connect(self.filter)

    def _add(self, command: str, keys: str, where: str, portable: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, text in ((COMMAND, command), (KEYS, keys), (WHERE, where)):
            item = QTableWidgetItem(text)
            if column == KEYS:
                item.setData(PORTABLE_ROLE, portable)
            self.table.setItem(row, column, item)

    def rows(self, shown_only: bool = True) -> list[tuple[str, str, str]]:
        """The rows (command, keys, where), those the search leaves shown by default."""
        return [tuple(self.table.item(row, column).text() for column in (COMMAND, KEYS, WHERE))
                for row in range(self.table.rowCount()) if not (shown_only and self.table.isRowHidden(row))]

    def filter(self, text: str) -> None:
        """Show the rows whose command, keys (as shown, or in portable words) or place contain every word of
        ``text``, ignoring case."""
        words = text.casefold().split()
        shown = 0
        for row in range(self.table.rowCount()):
            keys = self.table.item(row, KEYS)
            haystack = " ".join((*(self.table.item(row, c).text() for c in (COMMAND, KEYS, WHERE)),
                                 str(keys.data(PORTABLE_ROLE) or ""))).casefold()
            hidden = not all(word in haystack for word in words)
            self.table.setRowHidden(row, hidden)
            shown += not hidden
        self.empty.setVisible(shown == 0)


class ShortcutsDialog(QDialog):
    """Help > Keyboard Shortcuts: the reference in a window of its own, beside the main one (not modal)."""

    def __init__(self, commands=(), parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(translate("shortcuts", "Keyboard Shortcuts"))
        self.reference = ShortcutReference(commands)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.reference, 1)
        layout.addWidget(buttons)
        self.resize(640, 560)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reference.search.setFocus()
        self.reference.search.selectAll()

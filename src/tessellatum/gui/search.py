"""Settings search: a field at the top of the sidebar that finds any setting by its name, the words it may be called
by, what it does, or where it is, and takes you to it.

The matches are ranked: a label that is the text, or starts with it, before one that only holds it; then the words it
may be searched by (the registry's ``aliases``: "outline" for Line width); then its tooltip, and its category or
section. Each result says where the setting is and what it is set to; choosing one opens its page, unfolds its section
(the Advanced group too), scrolls to the control itself, gives it the focus and outlines it for a moment. The results
are only a list: the setting is always edited in its one place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QT_TRANSLATE_NOOP, QEvent, Qt, Signal
from PySide6.QtWidgets import QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from tessellatum.core import settings
from tessellatum.gui.i18n import term, translate

MAX_RESULTS = 12
PLACEHOLDER = QT_TRANSLATE_NOOP("search", "Search settings")


@dataclass(frozen=True)
class Match:
    entry: settings.Setting | settings.Choice
    score: int


def _words(text: str) -> list[str]:
    return re.findall(r"[\w²°]+", text.lower())


def _score(entry, needle: str) -> int:
    """How well ``entry`` matches ``needle`` (lower case, stripped), or its singular ("numbers"): 0 not at all."""
    singular = needle[:-1] if len(needle) > 3 and needle.endswith("s") and not needle.endswith("ss") else None
    return max(_score_one(entry, needle), _score_one(entry, singular) if singular else 0)


def _score_one(entry, needle: str) -> int:
    return max(_score_words(entry, needle, entry.label, entry.tooltip),
               _score_words(entry, needle, term(entry.label), term(entry.tooltip)))  # English, or as shown


def _score_words(entry, needle: str, label: str, tooltip: str) -> int:
    label = label.lower()
    if label == needle:
        return 100
    if label.startswith(needle):
        return 90
    if any(word.startswith(needle) for word in _words(label)):
        return 80
    aliases = [alias.lower() for alias in entry.aliases]
    if needle in aliases:
        return 70
    if any(alias.startswith(needle) or any(w.startswith(needle) for w in _words(alias)) for alias in aliases):
        return 60
    if needle in label or entry.name.lower().startswith(needle):
        return 50
    if any(needle in alias for alias in aliases):
        return 40
    if needle in tooltip.lower():
        return 20
    if needle in entry.section.lower().replace("advanced · ", "") or needle in entry.category.lower():
        return 10
    return 0


def search(text: str, entries: Sequence | None = None) -> list[Match]:
    """The entries matching ``text``, best first (registry order among equals); every word of ``text`` must match."""
    needles = [word for word in text.lower().split() if word]
    if not needles:
        return []
    found = []
    for index, entry in enumerate(entries if entries is not None else settings.entries()):
        scores = [_score(entry, needle) for needle in needles]
        if all(scores):
            found.append((-sum(scores), index, Match(entry, sum(scores))))
    return [match for *_key, match in sorted(found, key=lambda item: item[:2])]


def where(entry) -> str:
    """Where ``entry`` is: its page, and its section, or the page's Advanced group and its part of it."""
    if entry.level == settings.ADVANCED:
        section = term(entry.section.replace("Advanced · ", ""))
        return f"{term(entry.category)} · {translate('search', 'Advanced')} · {section}"
    return f"{term(entry.category)} · {term(entry.section)}"


class SettingsSearch(QWidget):
    """The search field and its results. ``chosen(name)`` asks for the setting to be shown; ``value_of(entry)`` says
    what an entry is set to, for its result."""

    chosen = Signal(str)

    def __init__(self, value_of: Callable[[object], str], parent: QWidget | None = None):
        super().__init__(parent)
        self._value_of = value_of
        self.field = QLineEdit()
        self.field.setObjectName("settingsSearch")
        self.field.setPlaceholderText(translate("search", PLACEHOLDER))
        self.field.setClearButtonEnabled(True)
        self.field.setAccessibleName(translate("search", "Search settings"))
        self.results = QListWidget()
        self.results.setObjectName("searchResults")
        self.results.setAccessibleName(translate("search", "Settings found"))
        self.results.setFocusPolicy(Qt.NoFocus)  # the field keeps the focus: Up and Down move in the list from it
        self.results.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 0)
        layout.setSpacing(2)
        layout.addWidget(self.field)
        layout.addWidget(self.results)
        self.field.textChanged.connect(self._show_results)
        self.field.returnPressed.connect(self.choose_current)
        self.results.itemClicked.connect(lambda item: self._choose(item))
        self.field.installEventFilter(self)

    def _show_results(self, text: str) -> None:
        self.results.clear()
        matches = search(text)[:MAX_RESULTS]
        for match in matches:
            entry = match.entry
            value = self._value_of(entry)
            item = QListWidgetItem(f"{term(entry.label)}    {value}\n{where(entry)}")
            item.setData(Qt.UserRole, entry.name)
            item.setToolTip(term(entry.tooltip))
            self.results.addItem(item)
        if matches:
            self.results.setCurrentRow(0)
        elif text.strip():
            item = QListWidgetItem(translate("search", "No setting matches."))
            item.setFlags(Qt.NoItemFlags)
            self.results.addItem(item)
        self.results.setVisible(bool(text.strip()))
        rows = max(1, self.results.count())
        self.results.setFixedHeight(min(rows, 6) * self.results.sizeHintForRow(0) + 2 * self.results.frameWidth())

    def names(self) -> list[str]:
        """The settings found, best first, by name."""
        return [self.results.item(i).data(Qt.UserRole) for i in range(self.results.count())
                if self.results.item(i).data(Qt.UserRole)]

    def choose_current(self) -> None:
        item = self.results.currentItem()
        if item is not None and item.data(Qt.UserRole):
            self._choose(item)

    def _choose(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.UserRole)
        if not name:
            return
        self.clear()
        self.chosen.emit(name)

    def clear(self) -> None:
        self.field.clear()

    def start(self) -> None:
        """The focus in the field, its text selected to type over."""
        self.field.setFocus(Qt.ShortcutFocusReason)
        self.field.selectAll()

    def eventFilter(self, watched, event) -> bool:
        # The results only while the field has the focus: they give the sidebar back when you move on, and come back
        # as they were when you return (a click on one doesn't take the focus from the field).
        if watched is self.field and event.type() == QEvent.FocusOut:
            self.results.hide()
        elif watched is self.field and event.type() == QEvent.FocusIn:
            self.results.setVisible(bool(self.field.text().strip()))
        if watched is self.field and event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape \
                and self.field.text():
            event.accept()  # Esc clears the search, before the window's Cancel can take it
            return True
        if watched is self.field and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up) and self.results.isVisible() and self.names():
                row = self.results.currentRow() + (1 if key == Qt.Key_Down else -1)
                self.results.setCurrentRow(max(0, min(len(self.names()) - 1, row)))
                return True
            if key == Qt.Key_Escape and self.field.text():
                self.clear()
                return True
        return super().eventFilter(watched, event)

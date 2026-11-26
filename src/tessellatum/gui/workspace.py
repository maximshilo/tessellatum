"""The workspace around the canvas: the activity bar, the palette inspector, and the layout's persistence.

- The **activity bar** picks the settings sidebar's page (see ``ControlsPanel.show_category``): one checkable icon
  button per category, its name in its tooltip and accessible name. Clicking the page already shown folds the sidebar
  away, and clicking any page shows it again.
- The **inspector**, on the right, is the palette inspector (see ``gui.palette_inspector``).
- ``LayoutStore`` keeps what the user arranged in the preferences store between runs (``workspace/`` keys): the
  window's geometry, the splitter's sizes, which panels show, the page shown, and each page's scroll position and
  folded sections.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QToolButton, QVBoxLayout, QWidget

from tessellatum.core import settings
from tessellatum.gui import theme
from tessellatum.gui.actions import ACTIVITIES, ANALYSIS
from tessellatum.gui.palette_inspector import PaletteInspector  # noqa: F401  (the inspector, where it was)
from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import COLORS, LIGHT

ACTIVITY_ICONS = {
    settings.SOURCE: "source", settings.GENERATION: "generation", settings.COLORS: "colors",
    settings.DOCUMENT: "document", ANALYSIS: "analysis",
}
ACTIVITY_TIPS = {
    settings.SOURCE: "Source: the picture, and how it is read",
    settings.GENERATION: "Generation: the difficulty, the regions and their detail",
    settings.COLORS: "Colors: what the page is colored with, and its palette",
    settings.DOCUMENT: "Document: the paper, the lines and numbers, the legend, and the export",
    ANALYSIS: "Analysis: the page measured on paper, its issues, and overlays to see them",
}


def _colors():
    manager = theme.current()
    return manager.colors() if manager is not None else COLORS[LIGHT]


class ActivityBar(QWidget):
    """A column of checkable buttons, one per category; one is down, the page the sidebar shows."""

    activity_chosen = Signal(str)  # a button was clicked: its category, whether it was down already or not

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("activityBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.buttons: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        for category in ACTIVITIES:
            button = QToolButton()
            button.setObjectName("activityButton")
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setIconSize(QSize(22, 22))
            button.setToolTip(ACTIVITY_TIPS[category])
            button.setAccessibleName(category)
            button.setAccessibleDescription(ACTIVITY_TIPS[category])
            button.setFocusPolicy(Qt.TabFocus)
            self._group.addButton(button)
            self.buttons[category] = button
            layout.addWidget(button)
            button.clicked.connect(lambda _checked=False, category=category: self.activity_chosen.emit(category))
        layout.addStretch(1)
        self.refresh_icons()
        manager = theme.current()
        if manager is not None:
            manager.changed.connect(self._on_scheme_changed)

    def _on_scheme_changed(self, _scheme: str) -> None:
        self.refresh_icons()

    def refresh_icons(self) -> None:
        colors = _colors()
        for category, button in self.buttons.items():
            button.setIcon(icons.icon(ACTIVITY_ICONS[category], colors))

    def set_current(self, category: str) -> None:
        self.buttons[category].setChecked(True)

    def current(self) -> str:
        return next(category for category, button in self.buttons.items() if button.isChecked())


class LayoutStore:
    """The workspace's arrangement in the preferences store (``workspace/`` keys), or nowhere if there is no store."""

    PREFIX = "workspace/"

    def __init__(self, store: QSettings | None):
        self._store = store

    def get(self, key: str, default=None):
        if self._store is None:
            return default
        value = self._store.value(self.PREFIX + key)
        return default if value is None else value

    def get_bool(self, key: str, default: bool) -> bool:
        value = self.get(key)
        if value is None:
            return default
        return value in (True, "true", "True", 1, "1")

    def get_int(self, key: str, default: int) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bytes(self, key: str) -> QByteArray | None:
        value = self.get(key)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None

    def set(self, key: str, value) -> None:
        if self._store is not None:
            self._store.setValue(self.PREFIX + key, value)

    def clear(self) -> None:
        """Forget every arrangement kept: the next start lays the workspace out as new."""
        if self._store is None:
            return
        for key in self._store.allKeys():
            if key.startswith(self.PREFIX):
                self._store.remove(key)

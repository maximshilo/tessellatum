"""Whether the preview shown is the page the settings make: its state, on the canvas's bar and in the status bar.

A preview is in one of these states:

- **none**: no page made yet (or the picture was replaced);
- **current**: the page shown is what the settings as they are make;
- **modified**: the settings changed since the page shown was made, in a way that changes the page (the legend's
  detail and the export's options don't count); the page shown stays, labeled out of date;
- **processing**: a page is being made;
- **cancelled** / **failed**: the last page asked for was cancelled or failed; the page shown, if any, is the one
  before it, kept and labeled as such, until the settings change or a page is asked for again.

How soon a change is applied follows the registry's policy (``core.settings.update_policy``): the legend at once, the
lines, numbers and tone a moment after the last change (the window redraws the page by itself), everything else with
Generate Preview, or a second after the last change if the preferences have it made by itself (Generate automatically:
always, or while the picture's last page was quick). A preview's state is apart from whether the document has unsaved changes (``app.history``).
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from tessellatum.gui import theme
from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import COLORS, LIGHT
from tessellatum.gui.i18n import translate

NO_PREVIEW, CURRENT, MODIFIED, PROCESSING, CANCELLED, FAILED = (
    "none", "current", "modified", "processing", "cancelled", "failed"
)
STATES = (NO_PREVIEW, CURRENT, MODIFIED, PROCESSING, CANCELLED, FAILED)

# Each state's icon, the color role it is drawn in, and its words on the canvas's bar.
_LOOK = {
    NO_PREVIEW: ("info", "text_secondary", QT_TRANSLATE_NOOP("freshness", "No preview")),
    CURRENT: ("success", "success", QT_TRANSLATE_NOOP("freshness", "Up to date")),
    MODIFIED: ("warning", "warning", QT_TRANSLATE_NOOP("freshness", "Out of date")),
    PROCESSING: ("busy", "accent", QT_TRANSLATE_NOOP("freshness", "Generating…")),
    CANCELLED: ("info", "text_secondary", QT_TRANSLATE_NOOP("freshness", "Cancelled")),
    FAILED: ("error", "error", QT_TRANSLATE_NOOP("freshness", "Failed")),
}


def preview_state(has_page: bool, making: bool, changed: tuple, outcome: str | None) -> str:
    """The preview's state: ``has_page`` whether a page is shown, ``making`` whether one is being made, ``changed``
    the settings changed since the page shown was made, and ``outcome`` how the last page asked for ended, if it was
    cancelled or failed and nothing changed since."""
    if making:
        return PROCESSING
    if outcome in (CANCELLED, FAILED):
        return outcome
    if not has_page:
        return NO_PREVIEW
    return MODIFIED if changed else CURRENT


def state_text(state: str, has_page: bool = True) -> str:
    """The words for ``state`` on the canvas's bar: a cancelled or failed page says that the one shown is from
    before."""
    text = translate("freshness", _LOOK[state][2])
    if state in (CANCELLED, FAILED) and has_page:
        text += translate("freshness", " · earlier preview")
    return text


def _colors():
    manager = theme.current()
    return manager.colors() if manager is not None else COLORS[LIGHT]


class FreshnessBadge(QWidget):
    """The preview's state on the canvas's bar: an icon and its words, with the detail in its tooltip."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewState")
        self.icon = QLabel()
        self.label = QLabel()
        self.label.setObjectName("previewStateText")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)
        layout.addWidget(self.icon)
        layout.addWidget(self.label)
        self.setAccessibleName(translate("freshness", "Preview"))
        self.setFocusPolicy(Qt.NoFocus)
        self._state = NO_PREVIEW
        self.set_state(NO_PREVIEW)
        manager = theme.current()
        if manager is not None:
            manager.changed.connect(self._show_icon)

    def state(self) -> str:
        return self._state

    def text(self) -> str:
        return self.label.text()

    def set_state(self, state: str, detail: str = "", has_page: bool = True) -> None:
        """Show ``state`` (one of ``STATES``), ``detail`` saying more in the tooltip."""
        if state not in STATES:
            raise ValueError(f"no state {state!r}")
        self._state = state
        text = state_text(state, has_page)
        self.label.setText(text)
        self.setToolTip(detail or text)
        self.setAccessibleDescription(f"{text}. {detail}" if detail else text)
        self._show_icon()

    def _show_icon(self, _scheme: str = "") -> None:
        name, role, _text = _LOOK[self._state]
        self.icon.setPixmap(icons.pixmap(name, getattr(_colors(), role), 16, self.devicePixelRatioF()))

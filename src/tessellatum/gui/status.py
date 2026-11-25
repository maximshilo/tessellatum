"""The status bar: what the app is doing, how far along, what the page came to, and the zoom.

From the left: the state (an icon and its words: "Generating preview…", "Preview ready.", "Cancelled.") with its
progress, measured or, for a step that can't say, indeterminate; then, on the right, the page's colors and regions,
its size in pixels and on paper, the zoom, and a button that counts the diagnostics' warnings and errors and opens
them. The state is a widget of its own, not a status message, so a menu's tip passing over it doesn't wipe it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QProgressBar, QStatusBar, QToolButton, QWidget

from tessellatum.gui import theme
from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import COLORS, LIGHT

IDLE, BUSY, DONE, CANCELLED, FAILED = "idle", "busy", "done", "cancelled", "failed"
STATES = (IDLE, BUSY, DONE, CANCELLED, FAILED)
_ICONS = {IDLE: ("info", "text_secondary"), BUSY: ("busy", "accent"), DONE: ("success", "success"),
          CANCELLED: ("info", "text_secondary"), FAILED: ("error", "error")}


def _colors():
    manager = theme.current()
    return manager.colors() if manager is not None else COLORS[LIGHT]


class StatusBar(QStatusBar):
    diagnostics_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("statusBar")
        self.state_icon = QLabel()
        self.state_label = QLabel()
        self.state_label.setObjectName("statusText")
        self.state_label.setAccessibleName("Status")
        self.state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.progress = QProgressBar()
        self.progress.setObjectName("statusProgress")
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(120)
        self.progress.setAccessibleName("Progress")
        self.progress.hide()
        self.page_label = QLabel()
        self.page_label.setAccessibleName("The page")
        self.size_label = QLabel()
        self.size_label.setAccessibleName("The page's size")
        self.zoom_label = QLabel()
        self.zoom_label.setAccessibleName("Zoom")
        self.zoom_label.setToolTip("How far in: 100% shows a pixel of the page as it prints on a pixel of the screen")
        self.diagnostics_button = QToolButton()
        self.diagnostics_button.setObjectName("diagnosticsButton")
        self.diagnostics_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.diagnostics_button.setAutoRaise(True)
        self.diagnostics_button.setAccessibleName("Diagnostics")
        self.diagnostics_button.clicked.connect(self.diagnostics_requested)
        self.addWidget(self.state_icon)
        self.addWidget(self.state_label, 1)
        self.addWidget(self.progress)
        for widget in (self.page_label, self.size_label, self.zoom_label, self.diagnostics_button):
            self.addPermanentWidget(widget)
        self._state = IDLE
        self.set_counts(0, 0)
        self.set_state("", IDLE)
        manager = theme.current()
        if manager is not None:
            manager.changed.connect(self._on_scheme_changed)

    def _on_scheme_changed(self, _scheme: str) -> None:
        self._show_icon()
        self.set_counts(*self._counts)

    # -- The state ------------------------------------------------------------------
    def set_state(self, text: str, state: str = IDLE) -> None:
        """Say ``text``, in ``state`` (one of ``STATES``); a busy state shows the progress bar."""
        if state not in STATES:
            raise ValueError(f"no state {state!r}")
        self._state = state
        self.state_label.setText(text)
        self.state_label.setAccessibleDescription(text)
        self._show_icon()
        self.progress.setVisible(state == BUSY)
        if state == BUSY:
            self.set_progress(None)

    def text(self) -> str:
        return self.state_label.text()

    def state(self) -> str:
        return self._state

    def set_progress(self, percent: int | None) -> None:
        """How far along, 0-100; None for a step that can't say (an indeterminate bar)."""
        if percent is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, percent)))

    def _show_icon(self) -> None:
        name, role = _ICONS[self._state]
        self.state_icon.setPixmap(icons.pixmap(name, getattr(_colors(), role), 16, self.devicePixelRatioF()))

    # -- The page and the view ----------------------------------------------------------
    def set_page(self, colors: int | None, regions: int | None) -> None:
        self.page_label.setText("" if colors is None else f"{colors} colors · {regions:,} regions")

    def set_size(self, pixels: tuple[int, int] | None, printed_mm: tuple[float, float] | None, paper: str = "") -> None:
        if pixels is None:
            self.size_label.setText("")
            return
        text = f"{pixels[0]:,} × {pixels[1]:,} px"
        if printed_mm is not None:
            text += f" · {printed_mm[0]:.0f} × {printed_mm[1]:.0f} mm on {paper}"
        self.size_label.setText(text)

    def set_zoom(self, printed_zoom: float | None) -> None:
        self.zoom_label.setText("" if printed_zoom is None else f"{printed_zoom * 100:.0f}%")

    def set_counts(self, warnings: int, errors: int) -> None:
        """What the diagnostics hold: their button shows the errors, else the warnings, else just its name."""
        self._counts = (warnings, errors)
        colors = _colors()
        button = self.diagnostics_button
        if errors:
            button.setIcon(icons.icon("error", colors, role="error"))
            button.setText(str(errors))
        elif warnings:
            button.setIcon(icons.icon("warning", colors, role="warning"))
            button.setText(str(warnings))
        else:
            button.setIcon(icons.icon("info", colors, role="text_secondary"))
            button.setText("")
        what = f"{errors} errors, {warnings} warnings" if errors or warnings else "no warnings or errors"
        button.setToolTip(f"Diagnostics: {what}. Click to show the log (Ctrl+Shift+D).")
        button.setAccessibleDescription(what)

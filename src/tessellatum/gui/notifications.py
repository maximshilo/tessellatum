"""Non-modal feedback: the notification banner over the canvas, and the diagnostics panel under it.

- A **notification** says what just happened -- "Exported to page.png", "Generation failed: …" -- with what can be
  done next ("Show in Folder", "Details"), in a banner at the top of the canvas. It never takes the keyboard focus
  from where it is. A success or an info goes by itself after a few seconds; a warning or an error stays until it is
  closed or replaced. Modal dialogs are kept for decisions with consequences (unsaved changes, overwriting a file,
  recovery).
- The **diagnostics panel** is a log: each job, what it did, how long it took and how it ended, and every warning
  and error, with its details and a Copy button. It starts closed and never opens by itself: the status bar counts
  the warnings and errors it holds, and View > Diagnostics opens it.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.app import jobs
from tessellatum.gui import theme
from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import COLORS, LIGHT

INFO, SUCCESS, WARNING, ERROR = "info", "success", "warning", "error"
LEVELS = (INFO, SUCCESS, WARNING, ERROR)
SUCCESS_MS = 6000  # how long a notification that needs nothing of the user stays: time to read it twice
_ICONS = {INFO: "info", SUCCESS: "success", WARNING: "warning", ERROR: "error"}


def _colors():
    manager = theme.current()
    return manager.colors() if manager is not None else COLORS[LIGHT]


def show_in_folder(path: Path) -> None:
    """Show ``path`` in the system's file manager: selected in it on Windows and macOS, its folder elsewhere."""
    path = Path(path)
    if sys.platform == "win32":
        QProcess.startDetached("explorer", [f"/select,{path}"])
    elif sys.platform == "darwin":
        QProcess.startDetached("open", ["-R", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class NotificationBar(QFrame):
    """One notification at a time: its icon, its text, its actions, and a close button."""

    notified = Signal(str, str)  # its level and text, as each is shown
    closed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("notification")
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.icon = QLabel()
        self.text = QLabel()
        self.text.setObjectName("notificationText")
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._buttons = QHBoxLayout()
        self._buttons.setContentsMargins(0, 0, 0, 0)
        self.close_button = QToolButton()
        self.close_button.setObjectName("notificationClose")
        self.close_button.setToolTip("Close")
        self.close_button.setAccessibleName("Close the notification")
        self.close_button.setFocusPolicy(Qt.TabFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.addWidget(self.icon, 0, Qt.AlignTop)
        layout.addWidget(self.text, 1)
        layout.addLayout(self._buttons)
        layout.addWidget(self.close_button, 0, Qt.AlignTop)
        self.close_button.clicked.connect(self.dismiss)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        self.level = INFO
        self.hide()

    def notify(self, level: str, text: str, actions: tuple[tuple[str, Callable[[], None]], ...] = ()) -> None:
        """Show ``text`` at ``level``, with ``actions`` (a label and what it does) as buttons, in place of what
        showed. A success or an info goes by itself after ``SUCCESS_MS``. Never takes the keyboard focus."""
        if level not in LEVELS:
            raise ValueError(f"no level {level!r}")
        self.level = level
        self.setProperty("level", level)
        self.style().unpolish(self)
        self.style().polish(self)
        colors = _colors()
        self.icon.setPixmap(icons.pixmap(_ICONS[level], getattr(colors, level), 16, self.devicePixelRatioF()))
        self.close_button.setIcon(icons.icon("close", colors))
        self.text.setText(text)
        self.setAccessibleName(f"{level.capitalize()}: {text}")
        while self._buttons.count():
            widget = self._buttons.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for label, action in actions:
            button = QPushButton(label)
            button.setFocusPolicy(Qt.TabFocus)  # reached by Tab; never focused by appearing
            button.clicked.connect(action)
            button.clicked.connect(self.dismiss)
            self._buttons.addWidget(button)
        self.show()
        if level in (INFO, SUCCESS):
            self._timer.start(SUCCESS_MS)
        else:
            self._timer.stop()
        self.notified.emit(level, text)

    def dismiss(self) -> None:
        self._timer.stop()
        if not self.isHidden():
            self.hide()
            self.closed.emit()

    def action_labels(self) -> list[str]:
        return [self._buttons.itemAt(i).widget().text() for i in range(self._buttons.count())]

    def trigger(self, label: str) -> None:
        """Press the action ``label``."""
        for i in range(self._buttons.count()):
            button = self._buttons.itemAt(i).widget()
            if button.text() == label:
                button.click()
                return
        raise KeyError(label)


@dataclass
class Entry:
    """One line of the diagnostics log."""

    level: str
    text: str
    details: str = ""
    when: float = 0.0

    def copy_text(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.when))
        return f"{stamp} {self.level.upper()} {self.text}" + (f"\n{self.details}" if self.details else "")


class DiagnosticsPanel(QWidget):
    """The log of jobs, warnings and errors, newest last, with the selected entry's details and a Copy button."""

    counts_changed = Signal(int, int)  # warnings, errors
    close_requested = Signal()

    MAX_ENTRIES = 500

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("diagnostics")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.entries: list[Entry] = []
        self._icons: dict[str, object] = {}  # by level, in the scheme now in use
        title = QLabel("Diagnostics")
        title.setObjectName("panelTitle")
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy the selected entry, with its details (all entries if none is selected)")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Empty the log")
        self.close_button = QToolButton()
        self.close_button.setToolTip("Close the diagnostics (Ctrl+Shift+D)")
        self.close_button.setAccessibleName("Close the diagnostics")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 6, 0)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.copy_button)
        header.addWidget(self.clear_button)
        header.addWidget(self.close_button)
        self.list = QListWidget()
        self.list.setObjectName("diagnosticsList")
        self.list.setAccessibleName("Diagnostics log")
        self.details = QPlainTextEdit()
        self.details.setObjectName("diagnosticsDetails")
        self.details.setReadOnly(True)
        self.details.setAccessibleName("Details of the selected entry")
        self.details.setPlaceholderText("Select an entry to see its details.")
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.details)
        split.setSizes([500, 400])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(split, 1)
        self.list.currentRowChanged.connect(self._show_details)
        self.copy_button.clicked.connect(self.copy)
        self.clear_button.clicked.connect(self.clear)
        self.close_button.clicked.connect(self.close_requested)
        self.refresh_icons()
        manager = theme.current()
        if manager is not None:
            manager.changed.connect(self._on_scheme_changed)

    def _on_scheme_changed(self, _scheme: str) -> None:
        self.refresh_icons()

    def refresh_icons(self) -> None:
        self._icons.clear()
        self.close_button.setIcon(icons.icon("close", _colors()))
        for row, entry in enumerate(self.entries):
            self.list.item(row).setIcon(self._icon(entry.level))

    def _icon(self, level: str):
        if level not in self._icons:
            self._icons[level] = icons.icon(_ICONS[level], _colors(), role=level)
        return self._icons[level]

    def add(self, level: str, text: str, details: str = "") -> Entry:
        """Log ``text`` at ``level``, with ``details`` (a traceback, say) shown when it is selected."""
        entry = Entry(level, text, details, time.time())
        self.entries.append(entry)
        item = QListWidgetItem(self._icon(level), entry.copy_text().split("\n")[0])
        item.setData(Qt.AccessibleTextRole, f"{level}: {text}")
        self.list.addItem(item)
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries.pop(0)
            self.list.takeItem(0)
        self.list.scrollToBottom()
        self.counts_changed.emit(*self.counts())
        return entry

    def add_record(self, level: str, text: str, details: str) -> None:
        self.add(level, text, details)

    def counts(self) -> tuple[int, int]:
        """How many warnings and errors the log holds."""
        return (sum(e.level == WARNING for e in self.entries), sum(e.level == ERROR for e in self.entries))

    def select(self, entry: Entry) -> None:
        if entry in self.entries:
            self.list.setCurrentRow(self.entries.index(entry))

    def _show_details(self, row: int) -> None:
        self.details.setPlainText(self.entries[row].copy_text() if 0 <= row < len(self.entries) else "")

    def copy(self) -> None:
        row = self.list.currentRow()
        chosen = [self.entries[row]] if 0 <= row < len(self.entries) else self.entries
        QGuiApplication.clipboard().setText("\n".join(entry.copy_text() for entry in chosen))

    def clear(self) -> None:
        self.entries.clear()
        self.list.clear()
        self.details.clear()
        self.counts_changed.emit(0, 0)

    # -- What it logs ---------------------------------------------------------------
    def watch(self, manager: jobs.JobManager) -> None:
        """Log every job of ``manager`` as it ends: what it did, how long it took, and how it ended."""
        manager.finished.connect(self._on_job_finished)

    def _on_job_finished(self, job: jobs.Job) -> None:
        seconds = job.seconds()
        took = f" in {seconds:.2f} s" if seconds is not None else ""
        if job.state == jobs.FAILED:
            self.add(ERROR, f"{job.label}: failed{took}: {job.error}", job.details)
        elif job.state == jobs.SUCCEEDED:
            self.add(INFO, f"{job.label}: done{took}")
        elif job.state == jobs.CANCELLED:
            self.add(INFO, f"{job.label}: cancelled{took}")
        else:
            self.add(INFO, f"{job.label}: dropped{took} (superseded, or of a picture since replaced)")


class _Relay(QObject):
    record = Signal(str, str, str)


class LogHandler(logging.Handler):
    """Brings the app's own warnings and errors (the ``tessellatum`` loggers) into ``panel``, from any thread: a
    record logged on a worker thread reaches the panel through a queued signal. Jobs are left to ``watch``, which
    logs them with their timing. ``close()`` it with the panel."""

    def __init__(self, panel: DiagnosticsPanel):
        super().__init__(logging.WARNING)
        self._relay = _Relay(panel)
        self._relay.record.connect(panel.add_record)  # a bound slot of the panel: delivered on its thread
        self.addFilter(lambda record: not record.name.startswith(jobs.__name__))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            details = logging.Formatter().formatException(record.exc_info) if record.exc_info else ""
            self._relay.record.emit(ERROR if record.levelno >= logging.ERROR else WARNING, record.getMessage(), details)
        except RuntimeError:
            pass  # the panel is gone: the app is closing


def install_log_handler(panel: DiagnosticsPanel, logger_name: str = "tessellatum") -> LogHandler:
    handler = LogHandler(panel)
    logging.getLogger(logger_name).addHandler(handler)
    return handler


def remove_log_handler(handler: LogHandler, logger_name: str = "tessellatum") -> None:
    logging.getLogger(logger_name).removeHandler(handler)
    handler.close()

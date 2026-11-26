"""Preferences: how the app looks and works on this computer, kept apart from the page's settings.

A preference is never part of a document, is never undone with Edit > Undo, and is kept globally between runs (in the
same ``QSettings`` as the window, under ``preferences/``; the theme and motion under the theme's own keys). They are:

- **Appearance:** the theme (System, Light, Dark) and the motion (System, Reduced, Full), both the theme manager's;
  the interface's scale, applied as the app starts (``QT_SCALE_FACTOR``, read by ``main`` before the application
  exists), so changing it says a restart is needed; the canvas's background behind the page.
- **Units:** lengths and areas shown in millimeters or inches. Only how they are shown and typed: a document's values
  are kept in millimeters whatever is picked.
- **Workspace:** the panels put back where they start.
- **Performance:** how much memory the pages made may take, kept so that undo, redo and a setting put back show them
  at once (``app.results``; Automatic: a quarter of the memory, at most 2 GB); how many pages' intermediate results the
  generator keeps to make the next preview faster; clearing both; and how soon a change to the lines, the numbers or the tone draws the preview again by itself, or
  whether it waits for Generate Preview as every other change does.
- **Shortcuts:** every command's keys.

The dialog applies each change at once; Close closes it.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from tessellatum.app import results
from tessellatum.core import pipeline
from tessellatum.gui import theme
from tessellatum.gui.fields import INCH, MM

SCALE_KEY = "preferences/ui_scale"
CANVAS_KEY = "preferences/canvas"
UNITS_KEY = "preferences/units"
CACHE_KEY = "preferences/cache_pages"
REDRAW_KEY = "preferences/redraw"
RESULTS_KEY = "preferences/result_memory"

AUTOMATIC = "Automatic"
SCALES = {AUTOMATIC: None, "90%": 0.9, "100%": 1.0, "110%": 1.1, "125%": 1.25, "150%": 1.5, "175%": 1.75, "200%": 2.0}
THEME_CANVAS = "Theme"
CANVASES = {THEME_CANVAS: None, "Neutral gray": "#808080", "White": "#FFFFFF", "Black": "#000000"}
UNITS = {"Millimeters": MM, "Inches": INCH}
CACHE_PAGES = {AUTOMATIC: pipeline.DEFAULT_CACHE_PAGES, "1 page": 1, "4 pages": 4, "8 pages": 8}
# The memory the pages made may take, in bytes; None: automatic (``results.automatic_budget``).
MB = 1024 ** 2
RESULT_MEMORY = {AUTOMATIC: None, "256 MB": 256 * MB, "512 MB": 512 * MB, "1 GB": 1024 * MB, "2 GB": 2048 * MB,
                 "4 GB": 4096 * MB}
# How long after the last change to the lines, the numbers or the tone the preview is drawn again by itself, in ms;
# None: only with Generate Preview. Long enough that a slider dragged draws once, when it stops.
REDRAW_AFTER_MS = 400
REDRAW = {"After 0.4 s": REDRAW_AFTER_MS, "After 0.2 s": 200, "After 1 s": 1000, "Only with Generate": None}
DEFAULT_REDRAW = "After 0.4 s"
RESTART_NOTE = "The new scale applies when Tessellatum is started again."


def scale_at_start(store: QSettings) -> float | None:
    """The interface's scale picked, to apply as the app starts; None for the system's (Automatic)."""
    return SCALES.get(store.value(SCALE_KEY), None)


def apply_scale_at_start(store: QSettings) -> None:
    """Set ``QT_SCALE_FACTOR`` from the scale picked, before the application exists, unless the environment set it."""
    scale = scale_at_start(store)
    if scale is not None and "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = f"{scale:g}"


class Preferences(QObject):
    """The preferences, read from ``store`` (kept in memory only without one) and written to it as they change. The
    theme and motion are the theme manager's, which keeps them itself."""

    changed = Signal(str)  # the key of the preference changed

    def __init__(self, store: QSettings | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._store = store
        self._values = {SCALE_KEY: AUTOMATIC, CANVAS_KEY: THEME_CANVAS, UNITS_KEY: "Millimeters", CACHE_KEY: AUTOMATIC,
                        REDRAW_KEY: DEFAULT_REDRAW, RESULTS_KEY: AUTOMATIC}
        if store is not None:
            for key, choices in _CHOICES.items():
                kept = store.value(key)
                if kept in choices:
                    self._values[key] = kept
        self.started_scale = self._values[SCALE_KEY]  # the scale this run started with

    def value(self, key: str) -> str:
        return self._values[key]

    def set_value(self, key: str, value: str) -> None:
        choices = _CHOICES[key]
        if value not in choices:
            raise ValueError(f"{value!r} is not one of {', '.join(choices)}")
        if value == self._values[key]:
            return
        self._values[key] = value
        if self._store is not None:
            self._store.setValue(key, value)
        self.changed.emit(key)

    def units(self) -> str:
        return UNITS[self._values[UNITS_KEY]]

    def canvas_color(self) -> str | None:
        return CANVASES[self._values[CANVAS_KEY]]

    def cache_pages(self) -> int:
        return CACHE_PAGES[self._values[CACHE_KEY]]

    def result_memory(self) -> int:
        """The bytes the pages made may take (see ``app.results``)."""
        chosen = RESULT_MEMORY[self._values[RESULTS_KEY]]
        return results.automatic_budget() if chosen is None else chosen

    def redraw_after_ms(self) -> int | None:
        """How long after the last change to the lines, numbers or tone the preview is drawn again; None: only with
        Generate Preview."""
        return REDRAW[self._values[REDRAW_KEY]]

    def restart_needed(self) -> bool:
        return self._values[SCALE_KEY] != self.started_scale


_CHOICES = {SCALE_KEY: SCALES, CANVAS_KEY: CANVASES, UNITS_KEY: UNITS, CACHE_KEY: CACHE_PAGES, REDRAW_KEY: REDRAW,
            RESULTS_KEY: RESULT_MEMORY}


def apply_canvas(view: QWidget, color: str | None) -> None:
    """The canvas ``view`` behind the page in ``color`` (None: the theme's), its placeholder text readable on it."""
    if color is None:
        view.setStyleSheet("")
        view.setPalette(QPalette())  # the application's again
        return
    view.setStyleSheet(f"QGraphicsView#canvas {{ background: {color}; border: none; }}")
    palette = view.palette()
    light = QColor(color).lightnessF() > 0.45
    palette.setColor(QPalette.WindowText, QColor("#1C1D20" if light else "#E8E9EC"))
    view.setPalette(palette)


class PreferencesDialog(QDialog):
    """The preferences, one page each in a list on the left; each change applies at once."""

    PAGES = ("Appearance", "Units", "Workspace", "Performance", "Shortcuts")
    reset_layout_requested = Signal()
    clear_requested = Signal()  # Clear Now: the window forgets the pages it keeps

    def __init__(self, preferences: Preferences, commands=(), parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setObjectName("preferences")
        self.preferences = preferences
        self.pages = QListWidget()
        self.pages.setObjectName("preferencePages")
        self.pages.addItems(self.PAGES)
        self.pages.setFixedWidth(140)
        self.pages.setAccessibleName("Preference pages")
        self.stack = QStackedWidget()
        manager = theme.current()

        # Appearance
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(theme.MODES)
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(theme.MOTION_MODES)
        self.motion_combo.setToolTip("Reduced turns off the interface's animations; System follows the system's setting")
        if manager is not None:
            self.theme_combo.setCurrentText(manager.mode())
            self.motion_combo.setCurrentText(manager.motion_mode())
            self.theme_combo.currentTextChanged.connect(manager.set_mode)
            self.motion_combo.currentTextChanged.connect(manager.set_motion_mode)
            manager.mode_changed.connect(self._show_theme_mode)
        else:
            self.theme_combo.setEnabled(False)
            self.motion_combo.setEnabled(False)
        self.scale_combo = self._choice(SCALE_KEY, SCALES)
        self.scale_combo.setToolTip("The size of the whole interface; the page's zoom is apart from it")
        self.restart_label = QLabel(RESTART_NOTE)
        self.restart_label.setProperty("role", "hint")
        self.restart_label.setWordWrap(True)
        self.canvas_combo = self._choice(CANVAS_KEY, CANVASES)
        self.canvas_combo.setToolTip("Behind the page: the theme's, or a neutral gray to judge its colors against")
        appearance = QWidget()
        form = QFormLayout(appearance)
        form.addRow("Theme", self.theme_combo)
        form.addRow("Motion", self.motion_combo)
        form.addRow("Interface scale", self.scale_combo)
        form.addRow("", self.restart_label)
        form.addRow("Canvas background", self.canvas_combo)

        # Units
        self.units_combo = self._choice(UNITS_KEY, UNITS)
        units_note = QLabel("How lengths and areas are shown and typed. A page's settings are kept in millimeters "
                            "whatever is picked, so changing this never changes them.")
        units_note.setProperty("role", "hint")
        units_note.setWordWrap(True)
        units = QWidget()
        form = QFormLayout(units)
        form.addRow("Lengths in", self.units_combo)
        form.addRow("", units_note)

        # Workspace
        self.reset_layout_button = QPushButton("Reset Layout")
        self.reset_layout_button.setToolTip("Put the panels back where they start")
        self.reset_layout_button.clicked.connect(self.reset_layout_requested)
        workspace = QWidget()
        form = QFormLayout(workspace)
        form.addRow("Panels", self.reset_layout_button)

        # Performance
        self.results_combo = self._choice(RESULTS_KEY, RESULT_MEMORY)
        self.results_combo.setToolTip(
            "How much memory the pages made may take, kept so that Undo, Redo and a setting put back show them at once")
        automatic = results.automatic_budget()
        results_note = QLabel(
            "Kept so that Undo, Redo and a setting put back show their page at once. Automatic: a quarter of the "
            f"memory, at most 2 GB (here {automatic / 1024 ** 3:.1f} GB).")
        results_note.setProperty("role", "hint")
        results_note.setWordWrap(True)
        self.cache_combo = self._choice(CACHE_KEY, CACHE_PAGES)
        self.cache_combo.setToolTip(
            "How many pages' intermediate results are kept, so that trying one setting after another makes the next "
            "preview faster; more takes more memory (a few tens of MB a page)")
        self.clear_cache_button = QPushButton("Clear Now")
        self.clear_cache_button.setToolTip("Forget the pages and intermediate results kept, to free their memory")
        self.clear_cache_button.clicked.connect(self._clear_cache)
        self.cache_label = QLabel()
        self.cache_label.setProperty("role", "hint")
        cache_note = QLabel("Kept so that trying one setting after another makes the next preview faster. Automatic "
                            "keeps two pages' worth.")
        cache_note.setProperty("role", "hint")
        cache_note.setWordWrap(True)
        performance = QWidget()
        form = QFormLayout(performance)
        form.addRow("Memory for pages made", self.results_combo)
        form.addRow("", results_note)
        form.addRow("Results kept", self.cache_combo)
        form.addRow("", cache_note)
        row = QHBoxLayout()
        row.addWidget(self.clear_cache_button)
        row.addWidget(self.cache_label, 1)
        form.addRow("", row)
        self.redraw_combo = self._choice(REDRAW_KEY, REDRAW)
        self.redraw_combo.setToolTip(
            "How soon the preview is drawn again by itself after the lines, the numbers or the tone change")
        redraw_note = QLabel("A change to the lines, the numbers or the tone only draws the page again, so the preview "
                             "does it by itself this long after the last change.")
        redraw_note.setProperty("role", "hint")
        redraw_note.setWordWrap(True)
        form.addRow("Redraw lines and numbers", self.redraw_combo)
        form.addRow("", redraw_note)

        # Shortcuts
        self.shortcuts = QTableWidget(0, 2)
        self.shortcuts.setObjectName("shortcutTable")
        self.shortcuts.setHorizontalHeaderLabels(["Command", "Keys"])
        self.shortcuts.verticalHeader().hide()
        self.shortcuts.setEditTriggers(QTableWidget.NoEditTriggers)
        self.shortcuts.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.shortcuts.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for action in commands:  # the window's Actions: its QActions
            keys = [key.toString(QKeySequence.NativeText) for key in action.shortcuts() if not key.isEmpty()]
            if keys:
                row = self.shortcuts.rowCount()
                self.shortcuts.insertRow(row)
                self.shortcuts.setItem(row, 0, QTableWidgetItem(action.text().replace("&", "").rstrip("…")))
                self.shortcuts.setItem(row, 1, QTableWidgetItem(", ".join(keys)))
        shortcuts = QWidget()
        layout = QVBoxLayout(shortcuts)
        layout.addWidget(self.shortcuts)

        for page in (appearance, units, workspace, performance):
            # Each scrolls if it is taller than the dialog (a large interface scale), rather than being cut off.
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            self.stack.addWidget(scroll)
        self.stack.addWidget(shortcuts)  # its table scrolls itself
        self.pages.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.pages.setCurrentRow(0)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        body = QHBoxLayout()
        body.addWidget(self.pages)
        body.addWidget(self.stack, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        self.resize(600, 460)
        preferences.changed.connect(self._show_restart)
        self._show_restart()

    def _choice(self, key: str, choices: dict) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(choices))
        combo.setCurrentText(self.preferences.value(key))
        combo.currentTextChanged.connect(lambda value, key=key: self.preferences.set_value(key, value))
        return combo

    def _show_theme_mode(self, mode: str) -> None:
        if self.theme_combo.currentText() != mode:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(mode)
            self.theme_combo.blockSignals(False)

    def _show_restart(self, _key: str = "") -> None:
        self.restart_label.setVisible(self.preferences.restart_needed())

    def _clear_cache(self) -> None:
        pipeline.clear_cache()
        self.clear_requested.emit()
        self.cache_label.setText("Cleared.")

    def show_page(self, name: str) -> None:
        self.pages.setCurrentRow(self.PAGES.index(name))

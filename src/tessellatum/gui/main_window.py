"""Top-level application window: the workspace around the canvas.

From the top: the menu bar; the global toolbar (Open, Generate/Cancel, Export, the panel toggles); then the activity
bar, the settings sidebar (``ControlsPanel``), the canvas with its own toolbar (the version shown, the zoom), and the
inspector (the page's colors), side by side in a splitter; the status bar. The sidebar and the inspector can be
resized, folded away (View menu, or the toolbar), or all hidden at once in focus mode; on a narrow window the inspector
and then the sidebar fold away by themselves. The arrangement is kept between runs (see ``workspace.LayoutStore``).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum import __version__
from tessellatum.app.document import Document
from tessellatum.app.history import History
from tessellatum.app.jobs import EXPORT, PREVIEW, PRINT_LAYOUT, RECIPES, SAVE, Job, JobManager, page_task
from tessellatum.app.snapshot import DocumentSettings
from tessellatum.core import export, mixing, pipeline, swatches
from tessellatum.core.difficulty import SCREEN_MEDIUM
from tessellatum.core.legend import NAME_RECIPE, render_legend
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import GeneratedPage
from tessellatum.core.print_size import MM_PER_INCH, print_scale
from tessellatum.gui import preferences, theme
from tessellatum.core import settings as registry
from tessellatum.gui.actions import ACTIVITY_ACTIONS, THEME_ACTIONS, VERSION_ACTIONS, ZOOM_ACTIONS, Actions
from tessellatum.app.presets import PresetStore
from tessellatum.gui.controls_panel import DPI, PALETTE, ControlsPanel
from tessellatum.gui.export_dialog import ExportDialog, ExportState, palette_pixmap, pixmap_of, warnings_for
from tessellatum.gui.notifications import (
    ERROR, INFO, SUCCESS, DiagnosticsPanel, NotificationBar, install_log_handler, remove_log_handler, show_in_folder,
)
from tessellatum.gui.preview_widget import PreviewWidget, VersionBar, ZoomBar
from tessellatum.gui.status import BUSY, CANCELLED, DONE, FAILED, IDLE, StatusBar
from tessellatum.gui.workspace import ActivityBar, LayoutStore, PaletteInspector

OPEN_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)"
# The preview's legend is drawn at least this many pixels a millimeter on paper (300 dpi), finer than a preview's
# page, so that its names and codes stay sharp zoomed in.
PREVIEW_LEGEND_PX_PER_MM = 300 / MM_PER_INCH

# The workspace's layout. Below these window widths (logical pixels) the inspector, then the sidebar, fold away by
# themselves, and come back as the window widens again: 1280 x 720 at 125% scaling is 1024 px wide, where the canvas
# keeps the room the inspector would take.
INSPECTOR_MIN_WINDOW = 1100
SIDEBAR_MIN_WINDOW = 760
MIN_WINDOW = QSize(720, 480)  # nothing clips at this size: the canvas's toolbar folds what doesn't fit into a menu
SCROLL_RESTORE_MS = 1000  # how long a kept scroll position waits for its page's layout
DEFAULT_SPLIT = (340, 700, 240)  # the sidebar, the canvas, the inspector
DEFAULT_ACTIVITY = registry.GENERATION
_NOT_MIXED = object()  # recipes asked for but not mixed yet (see MainWindow._recipes)

log = logging.getLogger(__name__)


def mix_recipes(palette_rgb, paints) -> list[list[str]]:
    """How to mix each of the colors ``palette_rgb`` from ``paints`` ((name, RGB) each), as the legend writes it: up to
    a second with many paints, so run off the GUI thread."""
    found = mixing.recipes(palette_rgb, [rgb for _name, rgb in paints])
    return [mixing.recipe_lines(recipe, paints) for recipe in found]

FOCUS_HINT = "Focus mode: the panels are hidden. Press Esc, or Exit Focus Mode above the page, to bring them back."
STALE_NAMED = 3  # how many changed settings the out-of-date status names


class MainWindow(QMainWindow):
    def __init__(self, store: QSettings | None = None, presets: PresetStore | None = None):
        """``store``, if given, is where the panel keeps its settings between runs (see ``ControlsPanel``)."""
        super().__init__()
        self.setWindowTitle("Tessellatum")

        self._layout = LayoutStore(store)
        self._store = store  # the preferences read it too (see ``_build_commands``)
        self.controls = ControlsPanel(store=store, presets=presets)  # the settings sidebar: its pages scroll on a short window
        # The export's options, kept by the panel, shown in the export dialog beside what they will write.
        self.export_dialog = ExportDialog(self.controls.export_options, self)
        self.controls.resolution_readout.hide()  # the dialog's summary says it, and more
        self.export_dialog.export_requested.connect(self.export_page)
        self.controls.export_options_changed.connect(
            lambda: self.export_dialog.isVisible() and self._show_export_state()  # only while it is shown
        )
        self.preview = PreviewWidget()
        self.preview.setObjectName("canvas")  # the theme's canvas surface; the page keeps its own pixels
        # Every version of a page is drawn with it, so the bar switches between them at once.
        self.version_bar = VersionBar()
        self.version_bar.setEnabled(False)
        self.zoom_bar = ZoomBar(self.preview)
        # The canvas's own toolbar: a toolbar, so that on a narrow window what doesn't fit goes into its overflow menu
        # rather than being cut off.
        self.canvas_bar = QToolBar("Canvas")
        self.canvas_bar.setObjectName("canvasToolbar")
        self.canvas_bar.setMovable(False)
        self.canvas_bar.addWidget(self.version_bar)
        gap = QWidget()
        gap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.canvas_bar.addWidget(gap)
        self.canvas_bar.addWidget(self.zoom_bar)
        # In focus mode the toolbar and its Focus Mode button are hidden: this is the way back, on the canvas's own bar.
        self.exit_focus_action = self.canvas_bar.addAction("Exit Focus Mode")
        self.exit_focus_action.setToolTip("Show the panels again (Esc)")
        self.exit_focus_action.setVisible(False)
        self.exit_focus_action.triggered.connect(lambda: self.set_focus_mode(False))
        canvas_pane = QWidget()
        canvas_layout = QVBoxLayout(canvas_pane)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self.canvas_bar)
        # What just happened, over the page: never a dialog in the way (see ``gui.notifications``).
        self.notification = NotificationBar()
        canvas_layout.addWidget(self.notification)
        canvas_layout.addWidget(self.preview, 1)

        self.inspector = PaletteInspector()
        self.activity_bar = ActivityBar()
        self.splitter = QSplitter()
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.setChildrenCollapsible(False)  # folded away with the View menu, never squeezed to nothing
        for widget in (self.controls, canvas_pane, self.inspector):
            self.splitter.addWidget(widget)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes(list(DEFAULT_SPLIT))
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.activity_bar)
        central_layout.addWidget(self.splitter, 1)
        # The diagnostics under it all: closed until asked for, and never opened by itself.
        self.diagnostics = DiagnosticsPanel()
        self.diagnostics.hide()
        self.bottom_splitter = QSplitter(Qt.Vertical)
        self.bottom_splitter.setObjectName("diagnosticsSplitter")
        self.bottom_splitter.setChildrenCollapsible(False)
        self.bottom_splitter.addWidget(central)
        self.bottom_splitter.addWidget(self.diagnostics)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 0)
        self.bottom_splitter.setSizes([640, 180])
        self.setCentralWidget(self.bottom_splitter)
        self.setMinimumSize(MIN_WINDOW)
        self._focus: dict | None = None  # in focus mode: what showed before it, to put back exactly
        self._auto_hidden: set[str] = set()  # the panels folded away by a narrow window, not by the user
        self._scroll_restored = False
        self._scroll_pending: dict[str, int] = {}  # pages whose kept scroll position waits for their layout

        self.status = StatusBar()
        self.setStatusBar(self.status)
        self._say("Open an image to get started.")

        self._previewed_image: np.ndarray | None = None  # the picture current_page is of
        self._pending_export_path: Path | None = None
        self._pending_export_format: str | None = None
        self._pending_export_version: Version | None = None
        self._pending_export_resolution: tuple[str, int] | None = None  # the PNG's, as the panel asked for it
        self._pending_export_drawn = False  # whether the export waits for the preview's page drawn at print size
        self._last_export: tuple | None = None  # the last export's file and options, for Retry
        self._ready_message = ""  # the status of the preview shown
        self._stale = False  # whether the status bar says the preview is out of date
        # The preview's page drawn at print size: what the zoom shows past the preview's own pixels, and what the
        # export saves. Drawn in the background when either first asks for it, and dropped with the preview.
        self._print_page: GeneratedPage | None = None

        self.controls.legend_detail_changed.connect(self._show_legend)
        self.controls.paints_changed.connect(self._show_legend)
        self._recipes_for: tuple | None = None  # what the recipes kept were mixed for, and them (see _recipes)
        self.version_bar.version_changed.connect(self.preview.show_version)
        self.preview.detail_wanted.connect(self._draw_print_page)

        # The document: the picture, its settings and the page made of them. The panel writes its edits into it.
        self.document = Document(self.controls.snapshot(), self)
        self.controls.settings_changed.connect(self.document.set_settings)
        # Undo and redo (see ``app.history``): each change to the settings or the picture, its states kept whole. The
        # edits made in one go (a preset, a reset) are one change; a slider dragged is one from press to release.
        self.history = History(self._restore_settings, self._restore_source, self)
        self._recorded = self.controls.snapshot()  # the settings as the history last saw them
        self._record_scheduled = False
        self._restoring = False
        self.controls.settings_changed.connect(self._on_settings_edited)
        self.controls.settings_changed.connect(lambda _settings: self._show_freshness())
        self.controls.edit_started.connect(self.history.begin_merge)
        # After the drag's last edit is recorded, which waits for the event loop too.
        self.controls.edit_finished.connect(lambda: QTimer.singleShot(0, self.history.end_merge))

        # Everything slow runs as a job (see ``app.jobs``): one of each kind at a time, its result delivered only if
        # still wanted.
        self.jobs = JobManager(self.document, self)
        self.jobs.succeeded.connect(self._on_job_succeeded)
        self.jobs.failed.connect(self._on_job_failed)
        self.jobs.cancelled.connect(self._on_job_cancelled)
        self.jobs.dropped.connect(self._on_job_dropped)
        self.jobs.progress.connect(self._on_job_progress)
        # Each job, and the app's own warnings and errors, in the diagnostics; the status bar counts them.
        self.diagnostics.watch(self.jobs)
        self._log_handler = install_log_handler(self.diagnostics)
        self.diagnostics.counts_changed.connect(self.status.set_counts)
        self.diagnostics.close_requested.connect(lambda: self.set_diagnostics_visible(False))
        self.status.diagnostics_requested.connect(lambda: self.set_diagnostics_visible(True))
        self.preview.zoom_changed.connect(self.status.set_zoom)

        self._build_commands()

        self.resize(1280, 840)
        self.document.result_changed.connect(lambda: self.inspector.show_page(self.current_page))
        self.document.result_changed.connect(self._show_page_status)
        self._restore_layout()

        # Get one-time costs (loading/compiling the compiled kernels, first
        # OpenCV/Pillow calls) out of the way while the user picks an image,
        # so the first preview is as fast as later ones.
        threading.Thread(target=_warm_up_pipeline, name="pipeline-warm-up", daemon=True).start()

    # -- The document's picture and page ---------------------------------------
    @property
    def current_image_bgr(self) -> np.ndarray | None:
        """The picture (BGR) the document holds, or None."""
        source = self.document.source()
        return None if source is None else source.image

    @current_image_bgr.setter
    def current_image_bgr(self, image: np.ndarray | None) -> None:
        self.document.set_source(image)

    @property
    def current_page(self) -> GeneratedPage | None:
        """The page last made of the picture (the preview), or None."""
        return self.document.result()

    @current_page.setter
    def current_page(self, page: GeneratedPage | None) -> None:
        self.document.set_result(page)

    # -- Commands: the menu bar, the keyboard and the buttons -------------------
    def _build_commands(self) -> None:
        """Every command as one action (see ``gui.actions``): in the menu bar, on its shortcut, and behind its button,
        enabled as its button is."""
        self.commands = Actions(self)
        self.commands.build_menus(self.menuBar())
        c, commands = self.controls, self.commands
        for command_id, method, button in (
            ("open", self.open_image, c.open_button),
            ("generate", self.generate_preview, c.generate_button),
            ("export", self.open_export_dialog, c.export_button),
        ):
            commands[command_id].triggered.connect(method)
            button.setToolTip(commands[command_id].toolTip())
        # The panel's buttons ask for the commands, so that each one works one way, from wherever it is asked.
        c.open_image_requested.connect(commands["open"].trigger)
        c.generate_requested.connect(commands["generate"].trigger)
        c.export_requested.connect(commands["export"].trigger)
        c.abort_requested.connect(commands["cancel"].trigger)
        commands["cancel"].triggered.connect(self.abort_generation)
        c.abort_button.setToolTip(commands["cancel"].toolTip())
        commands["reset"].triggered.connect(c.request_reset)
        c.confirm_reset = self._confirm_reset
        commands["undo"].triggered.connect(self.undo)
        commands["find"].triggered.connect(self.find_setting)
        commands["redo"].triggered.connect(self.redo)
        self.history.changed.connect(self._show_history)
        self._show_history()
        commands["quit"].triggered.connect(self.close)
        commands["about"].triggered.connect(self.show_about)
        for version, command_id in VERSION_ACTIONS.items():
            commands[command_id].triggered.connect(lambda _checked=False, v=version: self.version_bar.buttons[v].click())
        self.version_bar.version_changed.connect(lambda v: commands[VERSION_ACTIONS[v]].setChecked(True))
        for command_id, method in zip(ZOOM_ACTIONS, (
            self.preview.zoom_in, self.preview.zoom_out, self.preview.fit_to_window,
            self.preview.zoom_to_printed_pixels, self.preview.zoom_to_print_size,
        )):
            commands[command_id].triggered.connect(method)
        manager = theme.current()
        for mode, command_id in THEME_ACTIONS.items():
            action = commands[command_id]
            action.setEnabled(manager is not None)
            if manager is not None:
                action.triggered.connect(lambda _checked=False, m=mode: self._set_theme_mode(m))
        if manager is not None:
            commands[THEME_ACTIONS[manager.mode()]].setChecked(True)
            manager.mode_changed.connect(self._show_theme_mode)  # a method: it stops with the window
        self._build_toolbar()
        self.activity_bar.activity_chosen.connect(self._on_activity_chosen)
        for category, command_id in ACTIVITY_ACTIONS.items():
            commands[command_id].triggered.connect(lambda _checked=False, c=category: self.show_activity(c))
        commands["toggle_sidebar"].triggered.connect(self.set_sidebar_visible)
        commands["toggle_inspector"].triggered.connect(self.set_inspector_visible)
        commands["focus_mode"].triggered.connect(self.set_focus_mode)
        commands["reset_layout"].triggered.connect(self.reset_layout)
        commands["preferences"].triggered.connect(self.open_preferences)
        self.preferences = preferences.Preferences(self._store, self)
        self.preferences_dialog: preferences.PreferencesDialog | None = None
        self.preferences.changed.connect(self._apply_preference)
        self._apply_preference()
        commands["toggle_diagnostics"].triggered.connect(self.set_diagnostics_visible)
        # Enabled as the buttons are: whatever turns one on or off turns its command with it.
        self._enablement = _Watcher(self._sync_commands, self)
        for widget in (c.open_button, c.generate_button, c.export_button, c.abort_button, c.progress_row,
                       self.version_bar, self.zoom_bar):
            widget.installEventFilter(self._enablement)
        self._sync_commands()

    def _sync_commands(self) -> None:
        c, commands = self.controls, self.commands
        commands["open"].setEnabled(c.open_button.isEnabled())
        commands["generate"].setEnabled(c.generate_button.isEnabled())
        commands["export"].setEnabled(c.export_button.isEnabled())
        commands["cancel"].setEnabled(not c.progress_row.isHidden() and c.abort_button.isEnabled())
        for command_id in VERSION_ACTIONS.values():
            commands[command_id].setEnabled(self.version_bar.isEnabled())
        for command_id in ZOOM_ACTIONS:
            commands[command_id].setEnabled(self.zoom_bar.isEnabled())
        # One button starts the page or stops it, whichever can be done.
        if hasattr(self, "run_button"):
            busy = not c.progress_row.isHidden()
            wanted = commands["cancel"] if busy else commands["generate"]
            if self.run_button.defaultAction() is not wanted:
                self.run_button.setDefaultAction(wanted)

    def _build_toolbar(self) -> None:
        """The global toolbar: Open, Generate (Cancel while a page is made), Export; the panel toggles at its end."""
        commands = self.commands
        bar = QToolBar("Main")
        bar.setObjectName("mainToolbar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, bar)
        bar.addAction(commands["open"])
        self.run_button = QToolButton()
        self.run_button.setObjectName("runButton")
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.run_button.setDefaultAction(commands["generate"])
        bar.addWidget(self.run_button)
        bar.addAction(commands["export"])
        gap = QWidget()
        gap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(gap)
        for command_id in ("toggle_sidebar", "toggle_inspector", "focus_mode"):
            bar.addAction(commands[command_id])
            button = bar.widgetForAction(commands[command_id])
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar = bar

    # -- The workspace's layout ---------------------------------------------------
    def show_activity(self, category: str) -> None:
        """Show ``category``'s settings in the sidebar, unfolding it if it was folded away."""
        self.controls.show_category(category)
        self.activity_bar.set_current(category)
        self.commands[ACTIVITY_ACTIONS[category]].setChecked(True)
        self.set_sidebar_visible(True)

    def _on_activity_chosen(self, category: str) -> None:
        # The page already shown, clicked: the sidebar folds away; clicked again, it comes back.
        if category == self.controls.category() and self.sidebar_visible():
            self.set_sidebar_visible(False)
        else:
            self.show_activity(category)

    def sidebar_visible(self) -> bool:
        return not self.controls.isHidden()

    def inspector_visible(self) -> bool:
        return not self.inspector.isHidden()

    def set_sidebar_visible(self, visible: bool) -> None:
        self._set_panel_visible("sidebar", self.controls, visible)

    def set_inspector_visible(self, visible: bool) -> None:
        self._set_panel_visible("inspector", self.inspector, visible)

    def _set_panel_visible(self, name: str, widget: QWidget, visible: bool) -> None:
        if self._focus is not None:
            self.set_focus_mode(False)  # asking for a panel ends focus mode
        self._auto_hidden.discard(name)  # the user's choice now, whatever the window's width
        widget.setVisible(visible)
        self._show_panel_state()

    def _show_panel_state(self) -> None:
        self.commands["toggle_sidebar"].setChecked(self.sidebar_visible())
        self.commands["toggle_inspector"].setChecked(self.inspector_visible())
        self.commands["focus_mode"].setChecked(self._focus is not None)
        self.commands["toggle_diagnostics"].setChecked(not self.diagnostics.isHidden())

    def in_focus_mode(self) -> bool:
        return self._focus is not None

    def set_focus_mode(self, on: bool) -> None:
        """Hide every panel around the page (the toolbar, the activity bar, the sidebar, the inspector, the
        diagnostics and the status bar), or put them back exactly as they were."""
        if on == (self._focus is not None):
            self._show_panel_state()
            return
        panels = (self.toolbar, self.activity_bar, self.controls, self.inspector, self.diagnostics, self.statusBar())
        if on:
            self._focus = {"shown": [not panel.isHidden() for panel in panels], "sizes": self.splitter.sizes()}
            for panel in panels:
                panel.hide()
            self.notify(INFO, FOCUS_HINT)
        else:
            focus, self._focus = self._focus, None
            for panel, shown in zip(panels, focus["shown"]):
                panel.setVisible(shown)
            self.splitter.setSizes(focus["sizes"])
            self._reflow()
            if self.notification.isVisible() and self.notification.text.text() == FOCUS_HINT:
                self.notification.dismiss()
        self.exit_focus_action.setVisible(on)
        self._show_panel_state()

    def keyPressEvent(self, event) -> None:
        # Esc leaves focus mode (when it isn't cancelling a page: Cancel's shortcut takes it first).
        if event.key() == Qt.Key_Escape and self._focus is not None:
            self.set_focus_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        """On a narrow window fold the inspector away, then the sidebar; bring back what was folded so as the window
        widens. What the user folded stays folded."""
        if not hasattr(self, "inspector"):
            return  # still being built
        if self._focus is None:
            width = self.width()
            for name, threshold, widget in (
                ("inspector", INSPECTOR_MIN_WINDOW, self.inspector), ("sidebar", SIDEBAR_MIN_WINDOW, self.controls),
            ):
                if width < threshold and not widget.isHidden():
                    widget.hide()
                    self._auto_hidden.add(name)
                elif width >= threshold and name in self._auto_hidden:
                    widget.show()
                    self._auto_hidden.discard(name)
        if hasattr(self, "commands"):
            self._show_panel_state()

    def _wanted(self, name: str, widget: QWidget) -> bool:
        """Whether the user wants the panel ``name`` shown: shown, or folded away only by the window's width or by
        focus mode."""
        if self._focus is not None:
            index = {"sidebar": 2, "inspector": 3}[name]
            return self._focus["shown"][index] or name in self._auto_hidden
        return not widget.isHidden() or name in self._auto_hidden

    def _restore_layout(self) -> None:
        """The arrangement kept from the last run, where there is one."""
        self._default_expanded = {name: section.is_expanded() for name, section in self.controls.sections.items()}
        layout = self._layout
        geometry = layout.get_bytes("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        split = layout.get_bytes("splitter")
        if split is not None:
            self.splitter.restoreState(split)
        activity = layout.get("activity", DEFAULT_ACTIVITY)
        self.controls.show_category(activity if activity in ACTIVITY_ACTIONS else DEFAULT_ACTIVITY)
        self.activity_bar.set_current(self.controls.category())
        self.commands[ACTIVITY_ACTIONS[self.controls.category()]].setChecked(True)
        for name, section in self.controls.sections.items():
            section.toggle.setChecked(layout.get_bool(f"expanded/{name}", self._default_expanded[name]))
        self.controls.setVisible(layout.get_bool("sidebar", True))
        self.inspector.setVisible(layout.get_bool("inspector", True))
        self._reflow()
        self._show_panel_state()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._scroll_restored:
            self._scroll_restored = True
            # Once the pages are laid out, so that their scroll ranges reach the positions kept.
            QTimer.singleShot(0, self._restore_scroll)

    def _restore_scroll(self) -> None:
        """Each page's scroll position as kept. A page whose layout hasn't reached its full height yet (its rows'
        heights depend on its width, settled over more than one pass) gets its position as its range grows to it, for
        up to ``SCROLL_RESTORE_MS``, so that a section unfolded later never moves it."""
        self._scroll_pending = {}
        for category, scroll in self.controls.scrolls.items():
            bar = scroll.verticalScrollBar()
            wanted = self._layout.get_int(f"scroll/{category}", 0)
            bar.setValue(wanted)
            if bar.value() != wanted:
                self._scroll_pending[category] = wanted
                bar.rangeChanged.connect(self._on_scroll_range)
        if self._scroll_pending:
            QTimer.singleShot(SCROLL_RESTORE_MS, self._end_scroll_restore)

    def _on_scroll_range(self, _minimum: int, _maximum: int) -> None:
        for category, wanted in list(self._scroll_pending.items()):
            bar = self.controls.scrolls[category].verticalScrollBar()
            bar.setValue(wanted)
            if bar.value() == wanted:
                del self._scroll_pending[category]
                bar.rangeChanged.disconnect(self._on_scroll_range)

    def _end_scroll_restore(self) -> None:
        for category in self._scroll_pending:
            self.controls.scrolls[category].verticalScrollBar().rangeChanged.disconnect(self._on_scroll_range)
        self._scroll_pending = {}

    def save_layout(self) -> None:
        """Keep the arrangement for the next run."""
        layout = self._layout
        layout.set("geometry", self.saveGeometry())
        layout.set("splitter", self.splitter.saveState() if self._focus is None else layout.get_bytes("splitter"))
        layout.set("activity", self.controls.category())
        layout.set("sidebar", self._wanted("sidebar", self.controls))
        layout.set("inspector", self._wanted("inspector", self.inspector))
        for category, scroll in self.controls.scrolls.items():
            layout.set(f"scroll/{category}", scroll.verticalScrollBar().value())
        for name, section in self.controls.sections.items():
            layout.set(f"expanded/{name}", section.is_expanded())

    def reset_layout(self) -> None:
        """Put the panels back where they start: both shown, at their first sizes, the Generation page at its top
        with its sections folded as they start. The window's own size stays."""
        if self._focus is not None:
            self.set_focus_mode(False)
        self._layout.clear()
        self._auto_hidden.clear()
        self._end_scroll_restore()
        self.controls.show()
        self.inspector.show()
        self.splitter.setSizes(list(DEFAULT_SPLIT))
        for name, section in self.controls.sections.items():
            section.toggle.setChecked(self._default_expanded[name])
        for scroll in self.controls.scrolls.values():
            scroll.verticalScrollBar().setValue(0)
        self.show_activity(DEFAULT_ACTIVITY)
        self._reflow()

    # -- Preferences: how the app looks and works, apart from the page's settings and their undo -------------
    def _apply_preference(self, key: str | None = None) -> None:
        prefs = self.preferences
        if key in (None, preferences.UNITS_KEY):
            self.controls.set_length_units(prefs.units())
        if key in (None, preferences.CANVAS_KEY):
            preferences.apply_canvas(self.preview, prefs.canvas_color())
        if key in (None, preferences.CACHE_KEY):
            pipeline.set_cache_pages(prefs.cache_pages())

    def open_preferences(self) -> None:
        """The Preferences dialog, made the first time."""
        if self.preferences_dialog is None:
            self.preferences_dialog = preferences.PreferencesDialog(self.preferences, self.commands, self)
            self.preferences_dialog.reset_layout_requested.connect(self.reset_layout)
        self.preferences_dialog.show()
        self.preferences_dialog.raise_()
        self.preferences_dialog.activateWindow()

    def _show_theme_mode(self, mode: str) -> None:
        self.commands[THEME_ACTIONS[mode]].setChecked(True)

    def _set_theme_mode(self, mode: str) -> None:
        manager = theme.current()
        if manager is not None:
            manager.set_mode(mode)  # its mode_changed checks the mode's command

    # -- Feedback: the status bar, notifications and the diagnostics ------------------
    def _say(self, text: str, state: str = IDLE) -> None:
        """Say ``text`` in the status bar, in ``state`` (see ``gui.status``)."""
        self.status.set_state(text, state)

    def notify(self, level: str, text: str, actions=()) -> None:
        """Tell ``text`` in the banner over the page (see ``NotificationBar.notify``)."""
        self.notification.notify(level, text, tuple(actions))

    def _details(self) -> tuple[str, object]:
        """A notification's action that shows the last error in the diagnostics, with its details."""
        return ("Details", self.show_last_error)

    def show_last_error(self) -> None:
        self.set_diagnostics_visible(True)
        errors = [entry for entry in self.diagnostics.entries if entry.level == ERROR]
        if errors:
            self.diagnostics.select(errors[-1])

    def set_diagnostics_visible(self, on: bool) -> None:
        self.diagnostics.setVisible(bool(on))
        self._show_panel_state()

    def _show_page_status(self) -> None:
        """The page's colors, regions and size in the status bar."""
        page = self.current_page
        if page is None or page.drawing is None:
            self.status.set_page(None, None)
            self.status.set_size(None, None)
            return
        paper = page.drawing.style.paper_format
        self.status.set_page(page.num_colors_used, page.num_regions)
        self.status.set_size(page.page.size, print_scale(page.page.size, paper).printed_size_mm, paper.name)

    def show_about(self) -> None:
        QMessageBox.about(self, "About Tessellatum", about_text())

    # -- Open -----------------------------------------------------------
    def open_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Image", "", OPEN_IMAGE_FILTER)
        if not path_str:
            return
        path = Path(path_str)
        try:
            image = pipeline.load_image_bgr(path)
        except Exception as exc:  # noqa: BLE001
            text = f"Could not open {path.name}: {exc}"
            self.diagnostics.add(ERROR, text)
            self._say(text, FAILED)
            self.notify(ERROR, text)
            return
        old = self.document.source()
        self._show_source(image, path)
        if old is not None:  # a picture replaced in an open document: undoable
            self.history.record_source((old.image, old.path), (image, path), f"Open {path.name}")

    def _show_source(self, image: np.ndarray | None, path: Path | None) -> None:
        """Work on ``image``, read from ``path``, or on no picture: the last picture's page dropped."""
        self.document.set_source(image, path)  # drops the last picture's page
        self._drop_print_page()
        self.controls.set_export_enabled(False)
        self.version_bar.setEnabled(False)  # the preview left showing is the last picture's, until the next one
        if image is None:
            self.controls.set_picture_size(None)
            self._say("No picture open.")
            return
        self.controls.set_thumbnail(_bgr_to_pixmap(image))
        height, width = image.shape[:2]
        name = path.name if path is not None else "the picture"
        self.controls.set_picture_size((width, height), path.name if path is not None else "")
        self._say(f"Loaded {name}. Click \"Generate Preview\".")

    def find_setting(self) -> None:
        """The focus in the settings search, the sidebar shown for it (out of focus mode, if in it)."""
        if self._focus is not None:
            self.set_focus_mode(False)
        if not self.sidebar_visible():
            self.set_sidebar_visible(True)
        self.controls.search.start()

    # -- Undo and redo ------------------------------------------------------
    def _on_settings_edited(self, _settings: DocumentSettings) -> None:
        # Recorded once the edits made in one go are all in: a preset sets many settings, each announced.
        if self._restoring or self._record_scheduled:
            return
        self._record_scheduled = True
        QTimer.singleShot(0, self._record_settings)

    def _record_settings(self) -> None:
        self._record_scheduled = False
        label, self.controls.edit_label = self.controls.edit_label, None
        if self._restoring:
            return
        before, after = self._recorded, self.controls.snapshot()
        if after == before:
            return
        self._recorded = after
        self.history.record_settings(before, after, label or after.change_text(before))

    def undo(self) -> None:
        if self._record_scheduled:
            self._record_settings()  # an edit not recorded yet is the one to undo
        self.history.undo()

    def redo(self) -> None:
        if self._record_scheduled:
            self._record_settings()
        self.history.redo()

    def _restore_settings(self, settings: DocumentSettings) -> None:
        self._restoring = True
        try:
            self.controls.show_settings(settings)
        finally:
            self._restoring = False
        self.controls.edit_label = None
        self._recorded = self.controls.snapshot()
        self._show_freshness()

    def _show_freshness(self) -> None:
        """Say in the status bar when the preview shown isn't made with the settings as they are, naming what
        changed, and that it is ready again once they are back. Not while a page is being made, which says so."""
        made_with = self.document.result_settings()
        if self._making() or self.current_page is None or made_with is None:
            return
        changed = self.controls.snapshot().changes_from(made_with)
        if changed:
            names = ", ".join(changed[:STALE_NAMED])
            if len(changed) > STALE_NAMED:
                names += f" and {len(changed) - STALE_NAMED} more"
            self._stale = True
            self._say(f"The preview is out of date ({names} changed): Generate Preview to see the page with "
                      f"{'them' if len(changed) > 1 else 'it'}.")
        elif self._stale:
            self._stale = False
            self._say(self._ready_message, DONE)

    def _restore_source(self, source: tuple | None) -> None:
        image, path = source if source is not None else (None, None)
        self._show_source(image, path)

    def _show_history(self) -> None:
        """Edit > Undo and Redo, named for what they would do ("Undo Brush width"), on when there is something."""
        history, commands = self.history, self.commands
        commands["undo"].setEnabled(history.can_undo())
        commands["redo"].setEnabled(history.can_redo())
        commands["undo"].setText(f"&Undo {history.undo_text()}" if history.can_undo() else "&Undo")
        commands["redo"].setText(f"&Redo {history.redo_text()}" if history.can_redo() else "&Redo")

    # -- Preview ----------------------------------------------------------
    def _making(self) -> bool:
        """Whether a page is being made: a preview, or a page to export made afresh."""
        return self.jobs.is_running(PREVIEW) or self.jobs.is_running(EXPORT)

    def _exporting(self) -> bool:
        """Whether an export is under way: waiting for its page, or writing its file."""
        return self._pending_export_path is not None or self.jobs.is_running(SAVE)

    def _no_picture(self) -> None:
        self.notify(INFO, "Open a picture first.", [("Open Image…", self.commands["open"].trigger)])

    def generate_preview(self) -> None:
        if self.current_image_bgr is None:
            self._no_picture()
            return
        if self._making() or self._exporting():
            return

        self._drop_print_page()  # the zoom asks for the new page's
        self.document.set_settings(self.controls.snapshot())  # as the panel shows them, to the last edit
        settings = self.document.settings()
        self.controls.set_busy(True)
        self._say("Generating preview…", BUSY)

        # The page at its working size, where its level's brush is a few pixels across: what an export draws larger.
        image = self.current_image_bgr
        params, style, handling = settings.difficulty_params(), settings.page_style(), settings.handling()
        long_edge = pipeline.working_long_edge(image, params, handling, style.paper_format)
        self.jobs.start(
            PREVIEW, page_task(image, params, long_edge, style, handling, versions=tuple(Version)),
            label=f"Preview at {long_edge} px",
        )

    def abort_generation(self) -> None:
        # An export of the preview's page waits for it to be drawn at print size.
        kind = PRINT_LAYOUT if self._pending_export_drawn else EXPORT if self.jobs.is_running(EXPORT) else PREVIEW
        job = self.jobs.current(kind)
        if job is not None and job.cancellable and self.jobs.cancel(kind):
            # It stops at the pipeline's next stage: the button stays off meanwhile, against a second press.
            self.controls.abort_button.setEnabled(False)
            self._say("Cancelling…", BUSY)

    # -- What the jobs come to ------------------------------------------------
    def _on_job_succeeded(self, job: Job, result) -> None:
        if job.kind == PREVIEW:
            self._on_preview_ready(result, job.settings)
        elif job.kind == EXPORT:
            self._on_export_ready(result)
        elif job.kind == PRINT_LAYOUT:
            self._on_print_page_ready(result)
        elif job.kind == SAVE:
            job.context(result)  # what the file written reports
        elif job.kind == RECIPES:
            self._recipes_for = (job.context, result)
            self._show_legend()

    def _on_job_failed(self, job: Job, message: str) -> None:
        if job.kind in (PREVIEW, EXPORT):
            self._on_worker_failed(message)
        elif job.kind == PRINT_LAYOUT:
            self._on_print_page_failed(message)
        elif job.kind == SAVE:
            self._on_export_failed(message)
        # Recipes that couldn't be mixed: the legend stays without them (the job's details are logged).

    def _on_job_cancelled(self, job: Job) -> None:
        if job.kind in (PREVIEW, EXPORT):
            self._on_worker_cancelled()
        elif job.kind == PRINT_LAYOUT:
            self._on_print_page_cancelled()

    def _on_job_dropped(self, job: Job) -> None:
        # Superseded, or made of a picture since replaced: whatever waited for it alone is over.
        if job.kind == EXPORT:
            self._clear_pending_export()
        if job.kind in (PREVIEW, EXPORT) and not self._making() and not self._exporting():
            self.controls.set_busy(False)

    def _on_job_progress(self, job: Job, percent: int) -> None:
        # The zoom's drawing tells nothing, unless an export waits for it.
        if job.kind in (PREVIEW, EXPORT) or (job.kind == PRINT_LAYOUT and self._pending_export_drawn):
            self.controls.set_progress(percent)
            self.status.set_progress(percent)

    def _on_worker_cancelled(self) -> None:
        self.controls.set_busy(False)
        self._clear_pending_export()
        self._say("Cancelled.", CANCELLED)

    def _on_preview_ready(self, page: GeneratedPage, made_with: DocumentSettings) -> None:
        # A new page of the picture shown is shown as far in and in the same place, to compare the two.
        same_picture = self.current_page is not None and self._previewed_image is self.current_image_bgr
        self._drop_print_page()  # any drawn of the last page, for a zoom while this one was made
        self.document.set_result(page, made_with)
        self._previewed_image = self.current_image_bgr
        self.controls.set_busy(False)
        self.controls.set_export_enabled(True)
        if page.model is not None:
            self.controls.set_picture_reading(page.model.line_art.is_line_art)
        # Its colors, regions and size are in the status bar's own fields (see _show_page_status).
        self._ready_message = "Preview ready."
        self._stale = False
        self._say(self._ready_message, DONE)
        versions = {version: page.image(version) for version in Version}
        long_edge = self._print_long_edge()
        print_scale = 1.0 if long_edge is None else long_edge / max(page.page.size)
        legend, legend_scale = self._preview_legend(page)
        self.preview.show_page(
            versions, legend, self.version_bar.version(), print_scale, keep_view=same_picture,
            paper=page.drawing.style.paper_format, legend_scale=legend_scale,
        )
        self.version_bar.setEnabled(True)

    def _preview_legend(self, page: GeneratedPage):
        """The legend of ``page`` as the panel asks for it now, drawn finer than the page, and how many of its pixels go
        to one of the page's (see ``PreviewWidget.show_page``). Recipes not mixed yet are mixed in the background, and
        the legend drawn again with them."""
        paper = page.drawing.style.paper_format
        page_px_per_mm = print_scale(page.page.size, paper).px_per_mm
        legend_scale = max(1.0, PREVIEW_LEGEND_PX_PER_MM / page_px_per_mm)
        palette_bgr = np.asarray(page.palette_rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
        key, recipes = self._recipes(page)
        if recipes is _NOT_MIXED:
            recipes = None
            self._mix_recipes(key)
        legend = render_legend(
            palette_bgr, round(page.page.width * legend_scale), page_px_per_mm * legend_scale,
            self.controls.get_legend_detail(), names=page.color_names or None, recipes=recipes,
            mixes=page.mixes or None,
        )
        return legend, legend_scale

    def _recipes(self, page: GeneratedPage):
        """What ``page``'s legend recipes are mixed for, and them: (None, None) if the legend shows none; the mixed
        ones kept, as the legend is drawn again and again; or ``_NOT_MIXED`` if they aren't yet."""
        paints = self.controls.listed_paints()
        if self.controls.get_legend_detail() != NAME_RECIPE or not paints:
            return None, None
        key = (tuple(page.palette_rgb), paints)
        if self._recipes_for is not None and self._recipes_for[0] == key:
            return key, self._recipes_for[1]
        return key, _NOT_MIXED

    def _mix_recipes(self, key: tuple) -> None:
        """Mix the recipes for ``key`` (the palette and the paints) in the background: up to a second with many
        paints."""
        job = self.jobs.current(RECIPES)
        if job is not None and job.context == key:
            return
        palette, paints = key
        self.jobs.start(RECIPES, lambda _progress, _should_cancel: mix_recipes(palette, paints), context=key,
                        label=f"Recipes from {len(paints)} paints")

    def _show_legend(self, _detail: str = "") -> None:
        """Show the page's legend as the panel now asks for it: the page itself is the same."""
        if self.current_page is not None and self.current_page.drawing is not None:
            self.preview.set_legend(*self._preview_legend(self.current_page))

    def _on_worker_failed(self, message: str) -> None:
        self.controls.set_busy(False)
        self._clear_pending_export()
        self._say("Generation failed.", FAILED)
        self.notify(ERROR, f"Generation failed: {message}", [self._details()])

    # -- Export -----------------------------------------------------------
    def open_export_dialog(self) -> None:
        """The export's options beside what they will write (see ``export_dialog``); Export there asks for the file."""
        if self.current_image_bgr is None:
            self._no_picture()
            return
        if self._making() or self.jobs.is_running(SAVE):
            return
        self._show_export_state()
        self.export_dialog.open_for_export()

    def _show_export_state(self) -> None:
        if hasattr(self, "export_dialog"):
            self.export_dialog.show_state(self.export_state())

    def export_state(self) -> ExportState:
        """What the export dialog shows for the options as they are: what will be written, its warnings, whether the
        preview is out of date, and its thumbnail."""
        page = self.current_page
        if page is None or self.current_image_bgr is None:
            return ExportState("Generate a preview first: the export writes the page it shows.",
                               blocked="Generate a preview first.")
        now = self.controls.snapshot()
        fmt = now.export.format
        version = now.export.version
        made_with = self.document.result_settings()
        changed = () if made_with is None else now.changes_from(made_with)
        height, width = self.current_image_bgr.shape[:2]
        paper = self.controls.get_page_style().paper_format
        width_mm, height_mm = print_scale((width, height), paper).printed_size_mm
        on_paper = f"{width_mm:.0f} × {height_mm:.0f} mm on {paper.name}"
        dpi = None
        if fmt == PALETTE:
            suffix = now.export.palette_format
            title = swatches.FORMATS[suffix].split(" (")[0]
            summary = f"The legend's {len(page.palette_rgb)} colors as a {title} palette, named as on the legend."
            thumbnail = palette_pixmap(page.palette_rgb)
        else:
            what = version.value.lower() if version is not Version.PAGE else "page"
            if fmt == "PNG":
                dpi = self.controls.get_export_dpi(max(width_mm, height_mm))
                pixels = tuple(max(1, round(mm * dpi / MM_PER_INCH)) for mm in (width_mm, height_mm))
                summary = (f"A PNG of the {what}: {pixels[0]:,} × {pixels[1]:,} px at {dpi:.0f} dpi, printing "
                           f"{on_paper}, with its legend under it.")
                if self.controls.resolution_capped():
                    summary += f" The most a PNG takes is {export.MAX_RASTER_LONG_EDGE:,} px along a side."

            else:
                summary = (f"A PDF of the {what}: vector art printing {on_paper}, sharp at any size, with its legend "
                           f"on a sheet of its own.")
            thumbnail = pixmap_of(page.image(version))
        warnings = warnings_for(
            fmt, dpi, now.export.palette_format, len(page.palette_rgb), now.values["min_label_pt"],
            now.medium == SCREEN_MEDIUM,
        )
        return ExportState(summary, warnings, changed, thumbnail)

    def export_page(self, previewed: bool = False) -> None:
        """Ask for the file, and write the page to it with the export's options: the page as previewed if
        ``previewed``, else made with the settings as they are now (the preview's own page when they are the same)."""
        if self.current_image_bgr is None:
            self._no_picture()
            return
        if self._making() or self.jobs.is_running(SAVE):
            return

        fmt = self.controls.get_output_format()
        version = self.controls.get_export_version()
        source = self.document.source()
        stem = source.path.stem if source is not None and source.path is not None else "coloring_page"
        if fmt == PALETTE:
            suffix = self.controls.get_palette_format()
            file_filter = swatches.FORMATS[suffix]
            name = f"{stem}_palette"
        else:
            suffix = ".png" if fmt == "PNG" else ".pdf"
            file_filter = "PNG image (*.png)" if fmt == "PNG" else "PDF document (*.pdf)"
            name = f"{stem}_page" if version is Version.PAGE else f"{stem}_{version.value.lower()}"
        start = f"{name}{suffix}"
        if source is not None and source.path is not None:
            start = str(source.path.with_name(start))  # beside the picture
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Coloring Page", start, file_filter)
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)
        # A palette written as several files: the save dialog asked about the name chosen, not about them.
        page = self.current_page
        if fmt == PALETTE and page is not None:
            others = [p for p in swatches.palette_files(path, len(page.palette_rgb)) if p != path and p.exists()]
            if others and not self._confirm_replace(others):
                return
        self._start_export(path, fmt, version, self.controls.get_export_resolution(), previewed)

    def _confirm_reset(self) -> bool:
        answer = QMessageBox.question(
            self, "Reset All Settings",
            "Put every setting back to its default, and the difficulty to Medium? The paints listed are kept, and "
            "Edit > Undo takes it back.",
            QMessageBox.Reset | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        return answer == QMessageBox.Reset

    def _confirm_replace(self, paths: list[Path]) -> bool:
        names = "\n".join(path.name for path in paths)
        answer = QMessageBox.question(
            self, "Replace Files",
            f"The palette is written as {len(paths) if len(paths) > 1 else 'a'} file{'s' if len(paths) > 1 else ''} "
            f"that already exist{'' if len(paths) > 1 else 's'}:\n\n{names}\n\nReplace them?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def retry_export(self) -> None:
        """The last export again, to the same file with the same options."""
        if self._last_export is not None and not (self._making() or self.jobs.is_running(SAVE)):
            self._start_export(*self._last_export)

    def _start_export(self, path: Path, fmt: str, version: Version, resolution: tuple[str, int],
                      previewed: bool) -> None:
        self._last_export = (path, fmt, version, resolution, previewed)
        self._pending_export_path = path
        self._pending_export_format = fmt
        self._pending_export_version = version
        self._pending_export_resolution = resolution
        self.controls.set_busy(True)
        self._say(f"Rendering the {fmt} at print resolution…", BUSY)

        # Every export is drawn from the page's print layout: the preview's page laid out at 300 dpi on its paper, its
        # numbers placed once, as small as the style asks (see ``pipeline.draw_page``), drawn once for the zoom and
        # the export alike; made first if the settings changed since, unless the page as previewed was asked for.
        # The PDF draws it as vectors, the PNG draws the same vectors in pixels.
        if previewed and self.current_page is not None and self.current_page.model is not None:
            preview = self.current_page
        else:
            long_edge = pipeline.export_long_edge(self.current_image_bgr, self.controls.get_page_style().paper_format)
            preview = self._preview_to_draw(long_edge)
        if preview is not None and fmt == PALETTE:
            self._on_export_ready(preview)  # the preview's legend: nothing to draw
            return
        if preview is not None:
            if self._print_page is not None:
                self._on_export_ready(self._print_page)
            else:
                self._pending_export_drawn = True
                self._draw_print_page()  # saved once drawn (see _on_print_page_ready)
            return
        settings = self.controls.snapshot()
        long_edge = pipeline.export_long_edge(self.current_image_bgr, settings.page_style().paper_format)
        self.jobs.start(
            EXPORT, page_task(
                self.current_image_bgr, settings.difficulty_params(), long_edge, settings.page_style(),
                settings.handling(), vector=True,
            ),
            label=f"Page to export at {long_edge} px",
        )

    def _on_export_ready(self, page: GeneratedPage) -> None:
        path = self._pending_export_path
        fmt = self._pending_export_format
        version = self._pending_export_version
        resolution = self._pending_export_resolution
        self._clear_pending_export()
        if path is None or fmt is None or version is None:
            self.controls.set_busy(False)
            return
        # The legend as the panel asks for it now: changing it doesn't make the page again (see _preview_to_draw).
        drawing = page.drawing
        style = dataclasses.replace(drawing.style, legend_detail=self.controls.get_legend_detail())
        page = dataclasses.replace(page, drawing=dataclasses.replace(drawing, style=style))
        # Its recipes, as kept, or mixed as the file is written, off the GUI thread.
        key, recipes = self._recipes(page)

        def legend_recipes():
            return mix_recipes(*key) if recipes is _NOT_MIXED else recipes

        # Written in the background, and not stopped once started: at 600 dpi on A2 the PNG takes seconds.
        self.controls.set_busy(True, cancellable=False)
        if fmt == PALETTE:
            names = _palette_names(page)
            self._say("Writing the palette…", BUSY)
            self.jobs.start(
                SAVE, lambda _p, _c: swatches.save_palette(path, page.palette_rgb, names=names),
                cancellable=False, context=self._on_palette_saved, label=f"Palette {path.name}",
            )
            return
        if fmt == "PDF":
            self._say("Writing the PDF…", BUSY)

            def save_pdf(_progress, _should_cancel):
                export.save_pdf(
                    page.drawing, page.palette_rgb, path, version=version, painting=page.painting,
                    names=page.color_names or None, recipes=legend_recipes(), mixes=page.mixes or None,
                )
                return page.page.size

            self.jobs.start(
                SAVE, save_pdf, cancellable=False, label=f"PDF {path.name}",
                context=lambda size: self._on_export_saved(page, path, size),
            )
            return

        # The vector page drawn in pixels, as the PDF prints it (see ``export.rasterize``), at the resolution asked
        # for.
        unit, value = resolution
        printed_long_mm = max(print_scale(page.page.size, page.drawing.style.paper_format).printed_size_mm)
        dpi = value if unit == DPI else value * MM_PER_INCH / printed_long_mm
        self._say(f"Drawing the PNG at {dpi:.0f} dpi…", BUSY)
        self.jobs.start(
            SAVE,
            lambda _p, _c: export.export_png(
                page.drawing, page.palette_rgb, path, dpi, version, page.painting, names=page.color_names or None,
                recipes=legend_recipes(), mixes=page.mixes or None,
            ),
            cancellable=False, label=f"PNG {path.name} at {dpi:.0f} dpi",
            context=lambda size: self._on_export_saved(page, path, size),
        )

    def _on_export_saved(self, page: GeneratedPage, path: Path, size: tuple[int, int]) -> None:
        self.controls.set_busy(False)
        if self.preview.wants_detail():
            self._draw_print_page()  # asked for by a zoom while the export was made
        paper = page.drawing.style.paper_format
        width, height = size
        dpi = print_scale(size, paper).dpi
        printed = f"{width:,} × {height:,} px, {dpi:.0f} dpi on {paper.name}"
        self._say(f"Exported to {path} ({printed})", DONE)
        self.notify(SUCCESS, f"Exported {path.name}: {printed}.", [("Show in Folder", lambda: show_in_folder(path))])

    def _on_palette_saved(self, written: list[Path]) -> None:
        self.controls.set_busy(False)
        if self.preview.wants_detail():
            self._draw_print_page()  # asked for by a zoom while the export was made
        note = f"{len(written)} palettes of at most {swatches.PROCREATE_MAX_SWATCHES} colors" if len(written) > 1 else ""
        self._say(f"Exported the palette to {written[0]}" + (f" ({note})" if note else ""), DONE)
        names = ", ".join(path.name for path in written)
        self.notify(SUCCESS, f"Exported {names}" + (f": {note}." if note else "."),
                    [("Show in Folder", lambda: show_in_folder(written[0]))])

    def _on_export_failed(self, message: str) -> None:
        self.controls.set_busy(False)
        self._say("Export failed.", FAILED)
        self.notify(ERROR, f"Export failed: {message}", [("Retry", self.retry_export), self._details()])

    def _clear_pending_export(self) -> None:
        self._pending_export_path = None
        self._pending_export_format = None
        self._pending_export_version = None
        self._pending_export_resolution = None
        self._pending_export_drawn = False

    def _preview_to_draw(self, long_edge: int) -> GeneratedPage | None:
        """The page previewed, if an export at ``long_edge`` is drawn from it, or is it: it is no larger, and made with
        the panel's settings as they are now. None if not, and the export is made of the picture itself."""
        page = self.current_page
        if page is None or page.model is None or long_edge < max(page.page.size):
            return None
        model = page.model
        # What the legend writes doesn't change the page: the export writes the panel's (see _on_export_ready).
        style = dataclasses.replace(self.controls.get_page_style(), legend_detail=model.style.legend_detail)
        current = (self.controls.get_difficulty_params(), style, self.controls.get_handling())
        return page if (model.params, model.style, model.handling) == current else None

    # -- The preview's page at print size -------------------------------------
    def _print_long_edge(self) -> int | None:
        """The long edge the preview's page is laid out at to print (see ``pipeline.export_long_edge``); None if there
        is no page."""
        page = self.current_page
        if page is None or page.model is None or self._previewed_image is None:
            return None
        return max(pipeline.export_long_edge(self._previewed_image, page.drawing.style.paper_format), *page.page.size)

    def _draw_print_page(self) -> None:
        """Lay the preview's page out to print, in the background, unless it is drawn or being drawn: for the zoom,
        which shows it past the preview's own pixels, and for the export, which draws it. Not while a page is being
        made: a preview, which will replace this one, or an export with other settings."""
        long_edge = self._print_long_edge()
        if long_edge is None or self._print_page is not None or self.jobs.is_running(PRINT_LAYOUT) or self._making():
            return
        page = self.current_page
        model = page.model
        self.jobs.start(
            PRINT_LAYOUT,
            page_task(self._previewed_image, model.params, long_edge, model.style, model.handling,
                      versions=tuple(Version), drawn_from=page, vector=True),
            accept=lambda _job, page=page: self.current_page is page,  # not of a preview since replaced
            label=f"Print layout at {long_edge} px",
        )
        if self._pending_export_path is None:
            self._say("Drawing the page as it prints, for the zoom…", BUSY)

    def _drop_print_page(self) -> None:
        """Forget the preview's page at print size, and stop drawing it: the preview is about to change."""
        self._print_page = None
        self.jobs.drop(PRINT_LAYOUT)

    # Each of these is the export's, if it waits for the page; else the zoom's alone, telling nothing over an export
    # with other settings that may be under way.
    def _on_print_page_ready(self, drawn: GeneratedPage) -> None:
        self._print_page = drawn
        self.preview.show_detail({version: drawn.image(version) for version in Version})
        if self._pending_export_drawn:
            self._on_export_ready(drawn)
        elif self._pending_export_path is None:
            self._say(self._ready_message, DONE)

    def _on_print_page_failed(self, message: str) -> None:
        if self._pending_export_drawn:
            self._on_worker_failed(message)
        elif self._pending_export_path is None:
            self._say(f"The page could not be drawn as it prints: {message}", FAILED)

    def _on_print_page_cancelled(self) -> None:
        if self._pending_export_drawn:
            self._on_worker_cancelled()
        elif self._pending_export_path is None:
            self._say(self._ready_message, DONE)

    def closeEvent(self, event) -> None:
        self.save_layout()
        # Stop what is still running, without waiting for it: the threads outlive the window, and the application
        # waits for them, bounded, when it quits (see ``app.jobs.wait_for_all``).
        self.jobs.shutdown()
        remove_log_handler(self._log_handler)
        super().closeEvent(event)


class _Watcher(QObject):
    """Calls ``changed`` whenever a widget it watches is turned on or off, shown or hidden."""

    EVENTS = (QEvent.EnabledChange, QEvent.ShowToParent, QEvent.HideToParent)

    def __init__(self, changed, parent: QObject | None = None):
        super().__init__(parent)
        self._changed = changed

    def eventFilter(self, watched, event) -> bool:
        if event.type() in self.EVENTS:
            self._changed()
        return False


def about_text() -> str:
    """The About box: the version, what the app is, and the licenses of what it bundles."""
    return (
        f"<h3>Tessellatum {__version__}</h3>"
        "<p>Turns a picture into a paint-by-numbers page: regions to paint, numbered, with a legend of their "
        "colors.</p>"
        "<p>Apache License 2.0.</p>"
        "<p>It bundles the Inter typeface (SIL Open Font License 1.1), the Lucide icons (ISC license), Qt through "
        "PySide6 (LGPL 3), and its detection models (MIT or Apache 2.0); their license texts are in its "
        "<i>resources</i> folder.</p>"
    )


def _warm_up_pipeline() -> None:
    try:
        pipeline.warm_up()
    except RuntimeError as exc:
        if "after shutdown" not in str(exc):  # the app quit while it ran: nothing to tell
            log.warning("warming up the pipeline failed", exc_info=True)
    except Exception:  # noqa: BLE001 - best effort; a real failure resurfaces on Generate
        log.warning("warming up the pipeline failed", exc_info=True)


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = image_bgr[:, :, ::-1].copy()
    h, w, _ = rgb.shape
    qimage = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


def _palette_names(page: GeneratedPage) -> list[str] | None:
    """The names a palette file gives ``page``'s colors: the legend's, a mixed color's with its recipe after it,
    "gray (1 Ivory black + 1 Titanium white)"."""
    if not page.color_names:
        return None
    if not page.mixes:
        return list(page.color_names)
    return [
        name if lines is None else f"{name} ({' + '.join(lines)})" for name, lines in zip(page.color_names, page.mixes)
    ]

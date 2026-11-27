"""The app's commands, each one ``QAction``: its text, icon, tooltip, shortcut and menu, in one place.

The menu bar, the keyboard and the buttons all trigger the same action, so a command can't work from one and not
another, and its enablement shows everywhere at once. Shortcuts use the platform's standard keys where Qt has one
(Open, Quit, Zoom In/Out), so they read Cmd on macOS. Later tasks add their commands here (undo, projects, recent
files, paste, focus mode, preferences); this holds the ones the app has.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QT_TRANSLATE_NOOP, QObject
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QMenuBar, QWidget

from tessellatum.core import settings
from tessellatum.core.painting import Version
from tessellatum.gui import theme, vision
from tessellatum.gui.theme import icons
from tessellatum.gui.preview_widget import (
    COMPARE_OFF,
    ORIGINAL,
    ORIGINAL_DESCRIPTION,
    PREVIOUS,
    SIDE_BY_SIDE,
    SPLIT_HORIZONTAL,
    SPLIT_VERTICAL,
    view_description,
    view_name,
)
from tessellatum.gui.theme.tokens import COLORS, LIGHT
from tessellatum.gui.i18n import translate


@dataclass(frozen=True)
class Command:
    id: str
    text: str  # with its mnemonic (&)
    menu: str  # the menu it sits in
    tip: str = ""
    icon: str | None = None  # a key of ``theme.icons.NAMES``
    keys: tuple = ()  # QKeySequence.StandardKey members or strings, the first shown in the menu
    checkable: bool = False
    role: QAction.MenuRole = QAction.NoRole
    submenu: str = ""  # a submenu of ``menu`` it sits in instead, by title


FILE, EDIT, VIEW, GENERATE, HELP = "&File", "&Edit", "&View", "&Generate", "&Help"
MENUS = (FILE, EDIT, VIEW, GENERATE, HELP)
COMPARE_MENU = "Co&mpare"
VISION_MENU = QT_TRANSLATE_NOOP("actions", "Color &Vision")
# The views a comparison can be with, by the key of their command.
COMPARE_WITH = {**{v.name.lower(): v for v in Version}, "original": ORIGINAL, "previous": PREVIOUS}

COMMANDS = (
    Command("open", "&Open…", FILE, "Open a picture to make a page of, or a project saved before", "open",
            (QKeySequence.Open,)),
    Command("save", "&Save Project", FILE, "Save the picture and its settings as a project, to open again later", "save",
            (QKeySequence.Save,)),
    Command("save_as", "Save Project &As…", FILE, "Save the picture and its settings as a new project",
            keys=(QKeySequence.SaveAs, "Ctrl+Shift+S")),
    Command("export", "&Export…", FILE, "Save the page as a PNG, a PDF or a palette file", "export", ("Ctrl+E",)),
    Command("close", "&Close", FILE, "Close the picture and its project", keys=("Ctrl+W", QKeySequence.Close)),
    Command("quit", "&Quit", FILE, "Close Tessellatum", "quit", (QKeySequence.Quit,), role=QAction.QuitRole),
    Command("undo", "&Undo", EDIT, "Undo the last change to the settings or the picture", "undo", (QKeySequence.Undo,)),
    Command("redo", "&Redo", EDIT, "Redo what was undone", "redo", (QKeySequence.Redo, "Ctrl+Shift+Z")),
    Command("paste", "&Paste Picture", EDIT, "Open the picture copied, or the picture or project whose file was copied",
            "paste", (QKeySequence.Paste,)),
    Command("find", "&Find Setting…", EDIT, "Search the settings by name or by what they do", "search", ("Ctrl+F",)),
    Command("preferences", "&Preferences…", EDIT, "How the app looks and works: theme, scale, units, the results kept",
            "settings", (QKeySequence.Preferences, "Ctrl+,"), role=QAction.PreferencesRole),
    Command("reset", "&Reset All Settings", EDIT, "Put every setting back to its default, and the difficulty to Medium",
            "reset"),
    Command("generate", "&Generate Preview", GENERATE, "Make the page with the settings as they are", "generate",
            ("Ctrl+Return", "F5")),
    Command("cancel", "&Cancel", GENERATE, "Stop making the page", "cancel", ("Esc",)),
    *(Command(f"show_{v.name.lower()}", f"&{v.value}", VIEW, v.description, None, (f"Ctrl+{i}",), checkable=True)
      for i, v in enumerate(Version, start=1)),
    Command("show_original", "O&riginal", VIEW, ORIGINAL_DESCRIPTION, None, ("Ctrl+4",), checkable=True),
    Command("zoom_in", "Zoom &In", VIEW, "Zoom in; the mouse wheel zooms too", "zoom_in", (QKeySequence.ZoomIn, "Ctrl+=")),
    Command("zoom_out", "Zoom &Out", VIEW, "Zoom out; the mouse wheel zooms too", "zoom_out", (QKeySequence.ZoomOut,)),
    Command("fit", "&Fit Page", VIEW, "The whole page in the window", "fit", ("Ctrl+0",)),
    Command("actual_pixels", "&100%", VIEW, "A pixel of the page as it prints on a pixel of the screen",
            "actual_size", ("Ctrl+Alt+0",)),
    Command("print_size", "&Print Size", VIEW,
            "The page as large as on paper: approximate, by the size the screen reports"),
    Command("rulers", "R&ulers", VIEW, "Rulers along the canvas, in millimeters or inches on paper", None,
            ("Ctrl+R",), checkable=True),
    Command("grid", "&Grid", VIEW, "A grid over the page, every centimeter (or inch) on paper", None, ("Ctrl+'",),
            checkable=True),
    Command("page_bounds", "Page &Bounds", VIEW, "The sheet the page prints on, and its margins", checkable=True),
    Command("compare_off", "&No Comparison", VIEW, "One view of the page", checkable=True, submenu=COMPARE_MENU),
    Command("compare_side", "&Side by Side", VIEW, "Two views of the page side by side, moving together",
            checkable=True, submenu=COMPARE_MENU),
    Command("compare_split", "Split &Left and Right", VIEW,
            "The view compared with past a divider up the page; drag the divider to move it", checkable=True,
            submenu=COMPARE_MENU),
    Command("compare_split_h", "Split &Top and Bottom", VIEW,
            "The view compared with past a divider across the page; drag the divider to move it", checkable=True,
            submenu=COMPARE_MENU),
    *(Command(f"compare_with_{key}", f"With {view_name(view)}", VIEW, view_description(view), checkable=True,
              submenu=COMPARE_MENU) for key, view in COMPARE_WITH.items()),
    # The canvas as seen with another color vision: on the screen only, never in a file (see ``gui.vision``).
    *(Command(f"vision_{mode}", f"&{vision.NAMES[mode]}", VIEW, vision.DESCRIPTIONS[mode], checkable=True,
              submenu=VISION_MENU) for mode in vision.MODES),
    *(Command(f"activity_{category.lower()}", f"&{category} Settings", VIEW, f"Show the {category} settings",
              category.lower(), (f"Ctrl+Shift+{i}",), checkable=True)
      for i, category in enumerate(settings.CATEGORIES, start=1)),
    Command("activity_analysis", "&Analysis", VIEW, "Show the page's analysis: its measures, issues and overlays",
            "analysis", (f"Ctrl+Shift+{len(settings.CATEGORIES) + 1}",), checkable=True),
    Command("toggle_sidebar", "&Sidebar", VIEW, "Show or hide the settings sidebar", "left_panel", ("Ctrl+B",),
            checkable=True),
    Command("toggle_inspector", "&Inspector", VIEW, "Show or hide the inspector: the page's colors", "right_panel",
            ("Ctrl+Alt+B",), checkable=True),
    Command("toggle_diagnostics", "&Diagnostics", VIEW,
            "Show or hide the log: what the app did, how long it took, and its warnings and errors", None,
            ("Ctrl+Shift+D",), checkable=True),
    Command("focus_mode", "F&ocus Mode", VIEW, "Hide every panel around the page; Esc brings them back", "focus",
            ("Ctrl+Shift+F",), checkable=True),
    Command("reset_layout", "Reset &Layout", VIEW, "Put the panels back where they start"),
    *(Command(f"theme_{mode.lower()}", f"&{mode}", VIEW, f"The {mode.lower()} appearance" if mode != theme.SYSTEM
              else "The system's appearance, following it as it changes", checkable=True) for mode in theme.MODES),
    Command("shortcuts", "&Keyboard Shortcuts", HELP, "Every command's keys, and the keys that work in one place",
            "shortcuts", ("Ctrl+/",)),
    Command("about", "&About Tessellatum", HELP, "Its version and licenses", "info", role=QAction.AboutRole),
)
COMMAND_IDS = tuple(command.id for command in COMMANDS)
VERSION_ACTIONS = {**{v: f"show_{v.name.lower()}" for v in Version}, ORIGINAL: "show_original"}
GUIDE_ACTIONS = ("rulers", "grid", "page_bounds")
COMPARE_ACTIONS = {COMPARE_OFF: "compare_off", SIDE_BY_SIDE: "compare_side", SPLIT_VERTICAL: "compare_split",
                   SPLIT_HORIZONTAL: "compare_split_h"}
COMPARE_WITH_ACTIONS = {view: f"compare_with_{key}" for key, view in COMPARE_WITH.items()}
VISION_ACTIONS = {mode: f"vision_{mode}" for mode in vision.MODES}
THEME_ACTIONS = {mode: f"theme_{mode.lower()}" for mode in theme.MODES}
ZOOM_ACTIONS = ("zoom_in", "zoom_out", "fit", "actual_pixels", "print_size")
ANALYSIS = QT_TRANSLATE_NOOP(
    "settings", "Analysis"
)  # the sidebar's page that isn't settings: the page measured (see ``gui.analysis``)
ACTIVITIES = (*settings.CATEGORIES, ANALYSIS)
ACTIVITY_ACTIONS = {category: f"activity_{category.lower()}" for category in ACTIVITIES}


class Actions(QObject):
    """Every command's ``QAction``, by id, owned by ``parent`` (the window, so their shortcuts work in it). A QObject
    child of it, so that the theme's signals stop reaching it with the window."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._by_id: dict[str, QAction] = {}
        for command in COMMANDS:
            # The commands are data, translated as they are shown (``packaging/update_translations.py`` collects them).
            text, command_tip = translate("actions", command.text), (
                translate("actions", command.tip) if command.tip else ""
            )
            action = QAction(text, parent)
            action.setObjectName(command.id)
            tip = command_tip or text.replace("&", "").rstrip("…")
            action.setStatusTip(command_tip)
            if command.keys:
                # Each once: a platform's standard key may be one also named (Redo is Ctrl+Shift+Z on Linux and macOS),
                # or none (Quit and Preferences have none on Windows).
                sequences = []
                for key in command.keys:
                    sequence = QKeySequence(key)
                    if not sequence.isEmpty() and sequence not in sequences:
                        sequences.append(sequence)
                action.setShortcuts(sequences)
                if sequences:
                    tip += f" ({action.shortcut().toString(QKeySequence.NativeText)})"
            action.setToolTip(tip)
            action.setCheckable(command.checkable)
            action.setMenuRole(command.role)
            action.setProperty("menu", command.menu.replace("&", ""))  # for the keyboard reference
            parent.addAction(action)  # its shortcut works anywhere in the window, menu bar or not
            self._by_id[command.id] = action
        self.versions = QActionGroup(parent)
        for action_id in VERSION_ACTIONS.values():
            self.versions.addAction(self._by_id[action_id])
        self._by_id[VERSION_ACTIONS[Version.PAGE]].setChecked(True)
        self.themes = QActionGroup(parent)
        for action_id in THEME_ACTIONS.values():
            self.themes.addAction(self._by_id[action_id])
        self.activities = QActionGroup(parent)
        for action_id in ACTIVITY_ACTIONS.values():
            self.activities.addAction(self._by_id[action_id])
        self.compares = QActionGroup(parent)
        for action_id in COMPARE_ACTIONS.values():
            self.compares.addAction(self._by_id[action_id])
        self._by_id[COMPARE_ACTIONS[COMPARE_OFF]].setChecked(True)
        self.compare_withs = QActionGroup(parent)
        for action_id in COMPARE_WITH_ACTIONS.values():
            self.compare_withs.addAction(self._by_id[action_id])
        self._by_id[COMPARE_WITH_ACTIONS[ORIGINAL]].setChecked(True)
        self.visions = QActionGroup(parent)
        for action_id in VISION_ACTIONS.values():
            self.visions.addAction(self._by_id[action_id])
        self._by_id[VISION_ACTIONS[vision.NORMAL]].setChecked(True)
        self.submenus: dict[str, object] = {}  # by title, once the menus are built
        self.refresh_icons()
        manager = theme.current()
        if manager is not None:
            manager.changed.connect(self._on_scheme_changed)

    def __getitem__(self, command_id: str) -> QAction:
        return self._by_id[command_id]

    def __iter__(self):
        return iter(self._by_id.values())

    def _on_scheme_changed(self, _scheme: str) -> None:
        self.refresh_icons()

    def refresh_icons(self) -> None:
        """Draw the icons in the theme's colors now in use."""
        manager = theme.current()
        colors = manager.colors() if manager is not None else COLORS[LIGHT]
        for command in COMMANDS:
            if command.icon is not None:
                self._by_id[command.id].setIcon(icons.icon(command.icon, colors))

    def build_menus(self, bar: QMenuBar) -> dict[str, object]:
        """The menu bar's menus, every command in its own; the menus by title."""
        menus = {title: bar.addMenu(translate("actions", title)) for title in MENUS}
        by_menu: dict[str, list[Command]] = {title: [] for title in MENUS}
        for command in COMMANDS:
            by_menu[command.menu].append(command)
        for title, commands in by_menu.items():
            menu = menus[title]
            appearance = None
            for command in commands:
                action = self._by_id[command.id]
                if command.submenu:
                    if command.submenu not in self.submenus:
                        menu.addSeparator()
                        self.submenus[command.submenu] = menu.addMenu(translate("actions", command.submenu))
                    submenu = self.submenus[command.submenu]
                    if command.id == COMPARE_WITH_ACTIONS[Version.PAGE]:
                        submenu.addSeparator()
                    submenu.addAction(action)
                    if command.id == VISION_ACTIONS[vision.NORMAL]:
                        submenu.addSeparator()
                    continue
                if command.id.startswith("theme_"):
                    if appearance is None:
                        menu.addSeparator()
                        appearance = menu.addMenu(translate("actions", "&Appearance"))
                    appearance.addAction(action)
                    continue
                if command.id in ("save", "export", "quit", "find", "preferences", "zoom_in", "cancel", "rulers",
                                  ACTIVITY_ACTIONS[settings.SOURCE], "toggle_sidebar") \
                        and not menu.isEmpty():
                    menu.addSeparator()
                menu.addAction(action)
        return menus

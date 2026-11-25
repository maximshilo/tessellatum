"""The app's commands, each one ``QAction``: its text, icon, tooltip, shortcut and menu, in one place.

The menu bar, the keyboard and the buttons all trigger the same action, so a command can't work from one and not
another, and its enablement shows everywhere at once. Shortcuts use the platform's standard keys where Qt has one
(Open, Quit, Zoom In/Out), so they read Cmd on macOS. Later tasks add their commands here (undo, projects, recent
files, paste, focus mode, preferences); this holds the ones the app has.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QMenuBar, QWidget

from tessellatum.core import settings
from tessellatum.core.painting import Version
from tessellatum.gui import theme
from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import COLORS, LIGHT


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


FILE, EDIT, VIEW, GENERATE, HELP = "&File", "&Edit", "&View", "&Generate", "&Help"
MENUS = (FILE, EDIT, VIEW, GENERATE, HELP)

COMMANDS = (
    Command("open", "&Open Image…", FILE, "Open a picture to make a page of", "open", (QKeySequence.Open,)),
    Command("export", "&Export…", FILE, "Save the page as a PNG, a PDF or a palette file", "export", ("Ctrl+E",)),
    Command("quit", "&Quit", FILE, "Close Tessellatum", "quit", (QKeySequence.Quit,), role=QAction.QuitRole),
    Command("undo", "&Undo", EDIT, "Undo the last change to the settings or the picture", "undo", (QKeySequence.Undo,)),
    Command("redo", "&Redo", EDIT, "Redo what was undone", "redo", (QKeySequence.Redo, "Ctrl+Shift+Z")),
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
    Command("zoom_in", "Zoom &In", VIEW, "Zoom in; the mouse wheel zooms too", "zoom_in", (QKeySequence.ZoomIn, "Ctrl+=")),
    Command("zoom_out", "Zoom &Out", VIEW, "Zoom out; the mouse wheel zooms too", "zoom_out", (QKeySequence.ZoomOut,)),
    Command("fit", "&Fit Page", VIEW, "The whole page in the window", "fit", ("Ctrl+0",)),
    Command("actual_pixels", "&100%", VIEW, "A pixel of the page as it prints on a pixel of the screen",
            "actual_size", ("Ctrl+Alt+0",)),
    Command("print_size", "&Print Size", VIEW, "The page as large as on paper, by the size the screen reports"),
    *(Command(f"activity_{category.lower()}", f"&{category} Settings", VIEW, f"Show the {category} settings",
              category.lower(), (f"Ctrl+Shift+{i}",), checkable=True)
      for i, category in enumerate(settings.CATEGORIES, start=1)),
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
    Command("about", "&About Tessellatum", HELP, "Its version and licenses", "info", role=QAction.AboutRole),
)
COMMAND_IDS = tuple(command.id for command in COMMANDS)
VERSION_ACTIONS = {v: f"show_{v.name.lower()}" for v in Version}
THEME_ACTIONS = {mode: f"theme_{mode.lower()}" for mode in theme.MODES}
ZOOM_ACTIONS = ("zoom_in", "zoom_out", "fit", "actual_pixels", "print_size")
ACTIVITY_ACTIONS = {category: f"activity_{category.lower()}" for category in settings.CATEGORIES}


class Actions(QObject):
    """Every command's ``QAction``, by id, owned by ``parent`` (the window, so their shortcuts work in it). A QObject
    child of it, so that the theme's signals stop reaching it with the window."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._by_id: dict[str, QAction] = {}
        for command in COMMANDS:
            action = QAction(command.text, parent)
            action.setObjectName(command.id)
            tip = command.tip or command.text.replace("&", "").rstrip("…")
            action.setStatusTip(command.tip)
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
        menus = {title: bar.addMenu(title) for title in MENUS}
        by_menu: dict[str, list[Command]] = {title: [] for title in MENUS}
        for command in COMMANDS:
            by_menu[command.menu].append(command)
        for title, commands in by_menu.items():
            menu = menus[title]
            appearance = None
            for command in commands:
                action = self._by_id[command.id]
                if command.id.startswith("theme_"):
                    if appearance is None:
                        menu.addSeparator()
                        appearance = menu.addMenu("&Appearance")
                    appearance.addAction(action)
                    continue
                if command.id in ("quit", "find", "preferences", "zoom_in", "cancel", ACTIVITY_ACTIONS[settings.SOURCE],
                                  "toggle_sidebar") \
                        and not menu.isEmpty():
                    menu.addSeparator()
                menu.addAction(action)
        return menus

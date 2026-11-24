"""The interface's look: design tokens, light and dark themes, the Inter font and themed icons.

``install`` sets it up once, on the application, before any window: the Fusion style, Inter, and the palette and
stylesheet of the scheme in use. The scheme follows the system's (``QStyleHints.colorScheme``), live, unless the user
picked Light or Dark, which is kept between runs. Widgets name tokens, never colors, so a switch repaints everything
without a restart; ``ThemeManager.changed`` tells what draws icons of its own to draw them again.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from tessellatum.gui.theme import icons, motion
from tessellatum.gui.theme.style import build_palette, build_stylesheet
from tessellatum.gui.theme.tokens import COLORS, DARK, FONT_FAMILY, FONT_FILES, LIGHT, MOTION_MS, Colors

FONTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "fonts"

SYSTEM, LIGHT_MODE, DARK_MODE = "System", "Light", "Dark"
MODES = (SYSTEM, LIGHT_MODE, DARK_MODE)
MOTION_MODES = (SYSTEM, "Reduced", "Full")
THEME_KEY = "appearance/theme"
MOTION_KEY = "appearance/motion"


def load_fonts() -> str | None:
    """Register Inter's three weights with Qt; its family name, or None if they couldn't be (the platform's UI font is
    used then)."""
    loaded = [QFontDatabase.addApplicationFont(str(FONTS_DIR / name)) for name in FONT_FILES]
    if any(font_id < 0 for font_id in loaded):
        return None
    families = QFontDatabase.applicationFontFamilies(loaded[0])
    return families[0] if families else None


class ThemeManager(QObject):
    """The scheme in use and the user's choice of it; applies it to the application, and again whenever it changes."""

    changed = Signal(str)  # the scheme now in use, LIGHT or DARK

    def __init__(self, app: QApplication, store: QSettings | None = None):
        super().__init__(app)
        self._app = app
        self._store = store
        self._mode = SYSTEM
        self._motion_mode = SYSTEM
        self._system_motion = motion.system_reduces_motion()
        self._scheme: str | None = None
        if store is not None:
            if store.value(THEME_KEY) in MODES:
                self._mode = store.value(THEME_KEY)
            if store.value(MOTION_KEY) in MOTION_MODES:
                self._motion_mode = store.value(MOTION_KEY)
        app.styleHints().colorSchemeChanged.connect(self._on_system_scheme_changed)

    # -- The choice -----------------------------------------------------------
    def mode(self) -> str:
        """System, Light or Dark."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown theme {mode!r}: one of {', '.join(MODES)}")
        was = self._mode
        self._mode = mode
        if self._store is not None:
            self._store.setValue(THEME_KEY, mode)
        if mode == SYSTEM and was != SYSTEM:
            self._app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)  # back to the system's own
        self.apply()

    def scheme(self) -> str:
        """The scheme in use: the one picked, or the system's (light where it says neither)."""
        if self._mode == LIGHT_MODE:
            return LIGHT
        if self._mode == DARK_MODE:
            return DARK
        return DARK if self._app.styleHints().colorScheme() == Qt.ColorScheme.Dark else LIGHT

    def colors(self) -> Colors:
        return COLORS[self.scheme()]

    # -- Motion ---------------------------------------------------------------
    def motion_mode(self) -> str:
        return self._motion_mode

    def set_motion_mode(self, mode: str) -> None:
        if mode not in MOTION_MODES:
            raise ValueError(f"unknown motion setting {mode!r}: one of {', '.join(MOTION_MODES)}")
        self._motion_mode = mode
        if self._store is not None:
            self._store.setValue(MOTION_KEY, mode)

    def reduced_motion(self) -> bool:
        """Whether animations are skipped: as the user picked, else as the system asks (no, if it can't be told)."""
        if self._motion_mode != SYSTEM:
            return self._motion_mode == "Reduced"
        return bool(self._system_motion)

    def duration_ms(self, speed: str = "normal") -> int:
        """How long an animation of ``speed`` (a key of ``tokens.MOTION_MS``) runs: 0 with reduced motion."""
        return 0 if self.reduced_motion() else MOTION_MS[speed]

    # -- Applying -------------------------------------------------------------
    def apply(self) -> None:
        scheme = self.scheme()
        if self._mode != SYSTEM:
            # Tell the platform too, so the window frames and native dialogs match.
            wanted = Qt.ColorScheme.Dark if scheme == DARK else Qt.ColorScheme.Light
            if self._app.styleHints().colorScheme() != wanted:
                self._app.styleHints().setColorScheme(wanted)
        colors = COLORS[scheme]
        self._app.setPalette(build_palette(colors))
        self._app.setStyleSheet(build_stylesheet(colors))
        changed = scheme != self._scheme
        self._scheme = scheme
        if changed:
            self.changed.emit(scheme)

    def _on_system_scheme_changed(self, _scheme) -> None:
        if self._mode == SYSTEM:
            self.apply()

    def icon(self, name: str, role: str = "text"):
        """Icon ``name`` in the scheme in use (see ``icons.icon``)."""
        return icons.icon(name, self.colors(), role)


_installed: ThemeManager | None = None
_replaced: tuple | None = None  # the style, font and palette install replaced


def install(app: QApplication, store: QSettings | None = None) -> ThemeManager:
    """Give ``app`` the Fusion style, Inter and the theme, and keep it following the theme; the manager."""
    global _installed, _replaced
    if _replaced is None:
        _replaced = (app.style().name(), QFont(app.font()), QPalette(app.palette()))
    app.setStyle("Fusion")
    family = load_fonts()
    if family is not None:
        font = QFont(app.font())
        font.setFamily(family)
        app.setFont(font)
    _installed = ThemeManager(app, store)
    _installed.apply()
    return _installed


def current() -> ThemeManager | None:
    """The manager ``install`` set up, or None (in tests that build widgets without it)."""
    return _installed


def uninstall(app: QApplication) -> None:
    """Undo ``install``'s palette and stylesheet (for tests that share one application)."""
    global _installed, _replaced
    if _installed is not None:
        app.styleHints().colorSchemeChanged.disconnect(_installed._on_system_scheme_changed)
        _installed.setParent(None)
        app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    _installed = None
    app.setStyleSheet("")
    if _replaced is not None:
        style, font, palette = _replaced
        if style:  # empty when a stylesheet hid it
            app.setStyle(style)
        app.setFont(font)
        app.setPalette(palette)
    _replaced = None


__all__ = [
    "DARK", "DARK_MODE", "LIGHT", "LIGHT_MODE", "MODES", "MOTION_MODES", "SYSTEM", "ThemeManager", "current",
    "install", "load_fonts", "uninstall",
]

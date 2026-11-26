"""The palette and stylesheet built from a scheme's tokens, applied to the whole app on Qt's Fusion style.

Fusion draws the same on Windows, macOS and Linux and follows the palette everywhere, so the tokens decide every
color; the native file and color dialogs, menus and window frames stay the system's. The stylesheet is the app's
own, set once on the application and set again on every theme change, never on a widget: a widget's own stylesheet
freezes the colors it was given, and kept white text on a light panel after a switch from dark (BASELINE B1).

Stylesheet rules write a sub-control before its state (``QSlider::handle:horizontal:focus``): the other order styles
the whole widget.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

from tessellatum.gui.theme import icons
from tessellatum.gui.theme.tokens import RADIUS, Colors


def build_palette(c: Colors) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.Window: c.window, QPalette.WindowText: c.text, QPalette.Base: c.field, QPalette.AlternateBase: c.sidebar,
        QPalette.Text: c.text, QPalette.Button: c.raised, QPalette.ButtonText: c.text, QPalette.BrightText: c.error,
        QPalette.Highlight: c.accent, QPalette.HighlightedText: c.on_accent, QPalette.ToolTipBase: c.tooltip,
        QPalette.ToolTipText: c.tooltip_text, QPalette.PlaceholderText: c.text_hint, QPalette.Link: c.accent,
        QPalette.LinkVisited: c.accent_pressed, QPalette.Light: c.raised, QPalette.Midlight: c.hover,
        QPalette.Mid: c.border_strong, QPalette.Dark: c.border_strong, QPalette.Shadow: "#000000", QPalette.Accent: c.accent,
    }
    for role, color in roles.items():
        palette.setColor(QPalette.All, role, QColor(color))
    for role, color in {
        QPalette.WindowText: c.text_disabled, QPalette.Text: c.text_disabled, QPalette.ButtonText: c.text_disabled,
        QPalette.Base: c.window, QPalette.Button: c.window, QPalette.Highlight: c.pressed,
        QPalette.HighlightedText: c.text_disabled, QPalette.PlaceholderText: c.text_disabled,
    }.items():
        palette.setColor(QPalette.Disabled, role, QColor(color))
    return palette


def build_stylesheet(c: Colors) -> str:
    r, rs = RADIUS["m"], RADIUS["s"]
    chevron = icons.svg_file("down", c.text_secondary)
    chevron_off = icons.svg_file("down", c.text_disabled)
    up = icons.svg_file("up", c.text_secondary)
    up_off = icons.svg_file("up", c.text_disabled)
    fields = "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox"

    def each(selector_list: str, state: str) -> str:
        return ", ".join(f"{s.strip()}{state}" for s in selector_list.split(","))

    return f"""
QToolTip {{ color: {c.tooltip_text}; background: {c.tooltip}; border: none; padding: 4px 6px; }}
QStatusBar {{ background: {c.chrome}; color: {c.text_secondary}; border-top: 1px solid {c.border}; }}
QStatusBar QLabel {{ color: {c.text_secondary}; }}
QMenuBar {{ background: {c.chrome}; color: {c.text}; }}
QMenuBar::item {{ background: transparent; padding: 4px 8px; border-radius: {rs}px; }}
QMenuBar::item:selected {{ background: {c.hover}; }}
QMenuBar::item:pressed {{ background: {c.pressed}; }}
QMenu {{ background: {c.raised}; color: {c.text}; border: 1px solid {c.border}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 8px; border-radius: {rs}px; }}
QMenu::item:selected {{ background: {c.selected}; }}
QMenu::item:disabled {{ color: {c.text_disabled}; }}
QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 6px; }}
QSplitter::handle {{ background: {c.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QWidget#sidebar, QScrollArea#sidebarScroll > QWidget#qt_scrollarea_viewport {{ background: {c.sidebar}; }}
QToolBar#mainToolbar {{ background: {c.chrome}; border: none; border-bottom: 1px solid {c.border}; padding: 2px 6px;
    spacing: 4px; }}
QToolBar#canvasToolbar {{ background: {c.window}; border: none; border-bottom: 1px solid {c.border}; padding: 2px 6px;
    spacing: 6px; }}
QToolButton#runButton {{ background: {c.accent}; color: {c.on_accent}; font-weight: 600; padding: 3px 10px; }}
QToolButton#runButton:hover {{ background: {c.accent_hover}; }}
QToolButton#runButton:pressed {{ background: {c.accent_pressed}; }}
QToolButton#runButton:focus {{ border: 2px solid {c.text}; }}
QToolButton#runButton:disabled {{ background: {c.pressed}; color: {c.text_disabled}; }}
QToolButton#runButton::menu-indicator {{ image: none; width: 0px; }}
QToolButton#presetMenu {{ border: 1px solid {c.border_strong}; border-radius: {r}px; background: {c.field}; padding: 3px 18px 3px 8px; }}
QToolButton#presetMenu:hover {{ background: {c.hover}; }}
QToolButton#presetMenu:focus {{ border: 1px solid {c.accent}; }}
QToolButton#presetMenu::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: right center; right: 5px; }}
QWidget#activityBar {{ background: {c.chrome}; border-right: 1px solid {c.border}; }}
QToolButton#activityButton {{ padding: 6px; border-radius: {r}px; border-left: 2px solid transparent; }}
QToolButton#activityButton:checked {{ background: {c.selected}; border-left: 2px solid {c.accent}; }}
QToolButton#activityButton:focus {{ border: 1px solid {c.accent}; border-left: 2px solid {c.accent}; }}
QLabel#sidebarTitle, QLabel#panelTitle {{ background: {c.sidebar}; color: {c.text}; font-weight: 600;
    padding: 8px 12px 4px; }}
QWidget#sidebarFooter {{ background: {c.sidebar}; border-top: 1px solid {c.border}; }}
QWidget#inspector {{ background: {c.sidebar}; border-left: 1px solid {c.border}; }}
QWidget#inspector > QLabel[role="secondary"] {{ background: {c.sidebar}; padding: 0px 12px 6px; }}
QListWidget#paletteList {{ background: {c.sidebar}; color: {c.text}; border: none; outline: 0; padding: 0px 6px; }}
QListWidget#paletteList::item {{ padding: 3px 4px; border-radius: {rs}px; }}
QListWidget#paletteList::item:hover {{ background: {c.hover}; }}
QListWidget#paletteList::item:selected {{ background: {c.selected}; color: {c.text}; border: 2px solid {c.accent}; }}
QLabel#detailTitle {{ font-weight: 600; }}
QWidget#paletteDetail {{ background: {c.sidebar}; border-top: 1px solid {c.border}; }}
QFrame#notification {{ background: {c.raised}; border: 1px solid {c.border}; border-left: 4px solid {c.info};
    border-radius: {rs}px; margin: 6px 8px 0px; }}
QFrame#notification[level="success"] {{ border-left-color: {c.success}; }}
QFrame#notification[level="warning"] {{ border-left-color: {c.warning}; }}
QFrame#notification[level="error"] {{ border-left-color: {c.error}; }}
QLabel#notificationText {{ color: {c.text}; }}
QLineEdit#fieldEntry[invalid="true"] {{ border: 2px solid {c.error}; padding: 1px 5px; }}
QLabel#fieldMessage[role="error"] {{ color: {c.error}; }}
QLabel#fieldMessage[role="reason"], QLabel#fieldMessage[role="reason"]:disabled {{ color: {c.text_hint}; }}
QToolButton#changedMarker {{ color: {c.accent}; border: 1px solid transparent; padding: 0px 2px; font-size: 9px; }}
QToolButton#changedMarker:hover {{ background: {c.hover}; }}
QPushButton#segment {{ border-radius: 0px; padding: 3px 8px; margin-left: -1px; }}
QPushButton#segment[position="first"] {{ border-top-left-radius: {r}px; border-bottom-left-radius: {r}px;
    margin-left: 0px; }}
QPushButton#segment[position="last"] {{ border-top-right-radius: {r}px; border-bottom-right-radius: {r}px; }}
QPushButton#segment[position="only"] {{ border-radius: {r}px; margin-left: 0px; }}
QPushButton#segment:checked {{ background: {c.selected}; border-color: {c.accent}; }}
QPushButton#segment:focus {{ border: 2px solid {c.accent}; padding: 2px 7px; }}
QWidget#diagnostics {{ background: {c.sidebar}; border-top: 1px solid {c.border}; }}
QListWidget#diagnosticsList, QPlainTextEdit#diagnosticsDetails {{ background: {c.field}; color: {c.text};
    border: none; border-top: 1px solid {c.border}; }}
QListWidget#diagnosticsList::item {{ padding: 2px 4px; }}
QListWidget#diagnosticsList::item:selected {{ background: {c.selected}; color: {c.text}; }}
QLabel#statusText {{ color: {c.text}; }}
QLabel#previewStateText {{ color: {c.text_secondary}; }}
QProgressBar#statusProgress {{ background: {c.field}; border: 1px solid {c.border}; border-radius: {rs}px;
    max-height: 6px; }}
QProgressBar#statusProgress::chunk {{ background: {c.accent}; border-radius: {rs}px; }}
QGraphicsView#canvas {{ background: {c.canvas}; border: none; }}
QLabel#canvasCaption {{ background: {c.tooltip}; color: {c.tooltip_text}; border-radius: {rs}px; padding: 2px 6px; }}
QScrollArea#welcome, QWidget#welcomeBody {{ background: {c.canvas}; border: none; }}
QLabel#thumbnail {{ border: 1px solid {c.border_strong}; color: {c.text_hint}; background: {c.field};
    border-radius: {rs}px; }}
QLabel[role="hint"] {{ color: {c.text_hint}; }}
QLabel[role="secondary"] {{ color: {c.text_secondary}; }}
QLabel[role="error"] {{ color: {c.error}; }}

QPushButton {{ background: {c.raised}; color: {c.text}; border: 1px solid {c.border_strong}; border-radius: {r}px;
    padding: 4px 12px; }}
QPushButton:hover {{ background: {c.hover}; }}
QPushButton:pressed {{ background: {c.pressed}; }}
QPushButton:checked {{ background: {c.selected}; border-color: {c.accent}; }}
QPushButton:focus {{ border: 2px solid {c.accent}; padding: 3px 11px; }}
QPushButton:disabled {{ color: {c.text_disabled}; background: {c.window}; border-color: {c.border}; }}
QPushButton#abortButton, QPushButton#abortButton:focus {{ padding: 0px; }}
QPushButton[primary="true"] {{ background: {c.accent}; color: {c.on_accent}; border-color: {c.accent};
    font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {c.accent_hover}; border-color: {c.accent_hover}; }}
QPushButton[primary="true"]:pressed {{ background: {c.accent_pressed}; border-color: {c.accent_pressed}; }}
QPushButton[primary="true"]:focus {{ border: 2px solid {c.text}; }}
QPushButton[primary="true"]:disabled {{ background: {c.pressed}; color: {c.text_disabled}; border-color: {c.pressed}; }}

QToolButton {{ background: transparent; color: {c.text}; border: 1px solid transparent; border-radius: {r}px;
    padding: 3px; }}
QToolButton:hover {{ background: {c.hover}; }}
QToolButton:pressed {{ background: {c.pressed}; }}
QToolButton:checked {{ background: {c.selected}; }}
QToolButton:focus {{ border: 1px solid {c.accent}; }}
QToolButton:disabled {{ color: {c.text_disabled}; }}
QToolButton#sectionHeader {{ background: transparent; border: 1px solid transparent; color: {c.text};
    font-weight: 600; padding: 3px 2px; }}
QToolButton#sectionHeader:hover {{ background: {c.hover}; }}
QToolButton#sectionHeader:focus {{ border: 1px solid {c.accent}; }}
QToolButton#sectionReset {{ color: {c.accent}; padding: 1px 6px; }}
QToolButton#subsectionHeader {{ background: transparent; border: 1px solid transparent; color: {c.text_secondary};
    font-weight: 600; padding: 2px 2px; }}
QToolButton#subsectionHeader:hover {{ background: {c.hover}; color: {c.text}; }}
QToolButton#subsectionHeader:focus {{ border: 1px solid {c.accent}; }}
QRadioButton::indicator {{
    width: 12px; height: 12px; border-radius: 7px; border: 1px solid {c.border_strong}; background: {c.field};
}}
QRadioButton::indicator:checked {{
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, stop: 0 {c.accent}, stop: 0.5 {c.accent},
                                stop: 0.6 {c.field}, stop: 1 {c.field});
    border-color: {c.accent};
}}
QRadioButton::indicator:disabled {{ border-color: {c.text_disabled}; }}
QRadioButton::indicator:checked:disabled {{
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, stop: 0 {c.text_disabled},
                                stop: 0.5 {c.text_disabled}, stop: 0.6 {c.field}, stop: 1 {c.field});
}}
QLabel#exportWarnings {{ color: {c.text}; }}
QLineEdit#settingsSearch {{ padding: 4px 6px; }}
QListWidget#searchResults {{ background: {c.field}; border: 1px solid {c.border}; }}
QListWidget#searchResults::item {{ padding: 3px 4px; }}
QListWidget#preferencePages {{ background: {c.field}; border: 1px solid {c.border}; }}
QListWidget#preferencePages::item {{ padding: 6px 8px; }}
QListWidget#preferencePages::item:selected {{ background: {c.selected}; color: {c.text}; }}
QListWidget#searchResults::item:selected {{ background: {c.selected}; color: {c.text}; }}
*[found="true"] {{ border: 2px solid {c.accent}; border-radius: 4px; }}

{fields} {{ background: {c.field}; color: {c.text}; border: 1px solid {c.border_strong}; border-radius: {r}px;
    padding: 2px 6px; selection-background-color: {c.accent}; selection-color: {c.on_accent}; }}
{each(fields, ':hover')} {{ border-color: {c.text_secondary}; }}
{each(fields, ':focus')} {{ border: 2px solid {c.accent}; padding: 1px 5px; }}
{each(fields, ':disabled')} {{ color: {c.text_disabled}; background: {c.window}; border-color: {c.border}; }}
QComboBox {{ padding-right: 22px; }}
QComboBox:focus {{ padding-right: 21px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 20px; border: none; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: url({chevron_off}); }}
QComboBox QAbstractItemView {{ background: {c.raised}; color: {c.text}; border: 1px solid {c.border};
    selection-background-color: {c.selected}; selection-color: {c.text}; outline: 0; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 18px; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ padding-right: 17px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; border: none; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
    width: 18px; border: none; background: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {{ background: {c.hover}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({chevron}); width: 10px; height: 10px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({up_off}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({chevron_off}); }}

QSlider {{ background: transparent; min-height: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c.border}; border: none; border-radius: 2px; margin: 0; }}
QSlider::add-page:horizontal {{ background: {c.border}; border: none; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c.accent}; border: none; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: {c.text_disabled}; }}
QSlider::handle:horizontal {{ background: {c.raised}; border: 1px solid {c.border_strong}; width: 12px;
    height: 12px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ border-color: {c.accent}; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {c.accent}; }}
QSlider::handle:horizontal:disabled {{ background: {c.window}; border-color: {c.border}; }}

QGroupBox {{ border: 1px solid {c.border}; border-radius: {RADIUS['l']}px; margin-top: 1.1em; padding: 6px 4px 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {c.text}; font-weight: 600; }}
QProgressBar {{ background: {c.border}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 3px; }}
"""

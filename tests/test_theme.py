"""The theme: tokens that keep their contrast, the font and icons it ships, and a switch that repaints everything."""

import dataclasses
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QToolButton,
    QWidget,
)

from tessellatum.gui import theme  # noqa: E402
from tessellatum.gui.theme import icons, tokens  # noqa: E402
from tessellatum.gui.theme.tokens import COLORS, DARK, LIGHT, contrast  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


@pytest.fixture
def installed(store):
    manager = theme.install(_app, store)
    yield manager
    theme.uninstall(_app)


# -- Tokens -------------------------------------------------------------------
@pytest.mark.parametrize("scheme", tokens.SCHEMES)
def test_every_token_pair_keeps_its_contrast(scheme):
    colors = dataclasses.asdict(COLORS[scheme])
    short = [
        f"{fg} on {bg}: {contrast(colors[fg], colors[bg]):.2f} < {minimum}"
        for fg, bg, minimum in tokens.CONTRAST_PAIRS
        if contrast(colors[fg], colors[bg]) < minimum
    ]
    assert not short, short


def test_the_schemes_share_one_structure_and_differ():
    assert set(dataclasses.asdict(COLORS[LIGHT])) == set(dataclasses.asdict(COLORS[DARK]))
    assert COLORS[LIGHT].window != COLORS[DARK].window
    assert all(re.fullmatch(r"#[0-9A-F]{6}", v) for c in COLORS.values() for v in dataclasses.asdict(c).values())


def test_contrast_is_wcags():
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21)
    assert contrast("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)
    assert contrast("#123456", "#123456") == 1


# -- Font and icons -------------------------------------------------------------
def test_inter_is_bundled_with_its_license():
    assert theme.load_fonts() == tokens.FONT_FAMILY
    assert "SIL Open Font License" in (theme.FONTS_DIR / "OFL.txt").read_text(encoding="utf-8")


def test_every_named_icon_is_bundled_with_its_license():
    assert icons.missing() == []
    assert "ISC License" in (icons.ICONS_DIR / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(icons.NAMES))
def test_each_icon_draws_in_the_color_asked(name):
    image = icons.pixmap(name, "#0D7468", 96).toImage()  # large enough that a stroke covers whole pixels
    drawn = [
        image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    ]
    assert drawn, f"{name} draws nothing"
    assert {color.name() for color in drawn} == {"#0d7468"}


def test_an_icon_has_its_states_and_sizes():
    icon = icons.icon("generate", COLORS[LIGHT])
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    assert {(16, 16), (20, 20), (24, 24)} <= sizes
    disabled = icon.pixmap(24, QIcon_mode("Disabled")).toImage()
    opaque = {disabled.pixelColor(x, y).name() for x in range(24) for y in range(24) if disabled.pixelColor(x, y).alpha() == 255}
    assert opaque == {COLORS[LIGHT].text_disabled.lower()}


def QIcon_mode(name):
    from PySide6.QtGui import QIcon

    return getattr(QIcon, name)


def test_a_stylesheet_icon_is_a_file_in_the_color():
    path = icons.svg_file("down", "#123456")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "#123456" in text and "currentColor" not in text


# -- The manager ------------------------------------------------------------------
def test_install_gives_inter_the_palette_and_the_stylesheet(installed):
    assert theme.current() is installed and "QToolButton#sectionHeader" in _app.styleSheet()
    assert _app.font().family() == tokens.FONT_FAMILY
    colors = installed.colors()
    assert _app.palette().color(QPalette.Window).name().upper() == colors.window
    assert _app.palette().color(QPalette.Highlight).name().upper() == colors.accent


def test_uninstall_puts_back_what_install_replaced(store):
    style, font = _app.style().name(), _app.font().family()
    theme.install(_app, store)
    theme.uninstall(_app)
    assert _app.style().name() == style and _app.font().family() == font and _app.styleSheet() == ""


def test_the_choice_is_kept(installed, store):
    seen = []
    installed.changed.connect(seen.append)
    installed.set_mode(theme.LIGHT_MODE)
    installed.set_mode(theme.DARK_MODE)
    assert installed.scheme() == DARK and seen[-1] == DARK
    installed.set_mode(theme.DARK_MODE)  # no change, no signal
    installed.set_mode(theme.LIGHT_MODE)
    assert installed.scheme() == LIGHT and seen[-2:] == [DARK, LIGHT]
    assert store.value(theme.THEME_KEY) == theme.LIGHT_MODE
    again = theme.ThemeManager(_app, store)
    assert again.mode() == theme.LIGHT_MODE and again.scheme() == LIGHT
    with pytest.raises(ValueError):
        installed.set_mode("Sepia")


def test_system_mode_follows_the_system_live(installed):
    hints = _app.styleHints()
    hints.setColorScheme(Qt.ColorScheme.Dark)
    if hints.colorScheme() != Qt.ColorScheme.Dark:
        pytest.skip("this platform plugin doesn't take a color scheme")
    try:
        assert installed.mode() == theme.SYSTEM and installed.scheme() == DARK
        assert _app.palette().color(QPalette.Window).name().upper() == COLORS[DARK].window
        hints.setColorScheme(Qt.ColorScheme.Light)
        assert installed.scheme() == LIGHT
        assert _app.palette().color(QPalette.Window).name().upper() == COLORS[LIGHT].window
    finally:
        hints.setColorScheme(Qt.ColorScheme.Unknown)


def test_reduced_motion_zeroes_the_durations(installed, store):
    installed.set_motion_mode("Reduced")
    assert installed.reduced_motion() and installed.duration_ms("slow") == 0
    installed.set_motion_mode("Full")
    assert not installed.reduced_motion() and installed.duration_ms("fast") == tokens.MOTION_MS["fast"]
    assert store.value(theme.MOTION_KEY) == "Full"


# -- The whole window, switched both ways (BASELINE B1) -------------------------------------------------
def _text_problems(root: QWidget, colors) -> list[str]:
    """Every visible, enabled label and button under ``root`` whose text color fails 4.5:1 on its surface."""
    problems = []
    for widget in [root, *root.findChildren(QWidget)]:
        if not widget.isVisibleTo(root) or not widget.isEnabled():
            continue
        if isinstance(widget, QPushButton):
            primary = widget.property("primary") is True
            surface = colors.accent if primary else (colors.selected if widget.isChecked() else colors.raised)
            text = widget.palette().color(QPalette.ButtonText).name()
        elif isinstance(widget, (QToolButton, QCheckBox, QRadioButton)):
            surface = colors.sidebar
            text = widget.palette().color(widget.foregroundRole()).name()
        elif isinstance(widget, QLabel) and widget.text() and widget.pixmap().isNull():
            surface = colors.sidebar
            text = widget.palette().color(widget.foregroundRole()).name()
        else:
            continue
        ratio = contrast(text.upper(), surface)
        if ratio < tokens.TEXT:
            label = widget.text() if isinstance(widget, (QLabel, QAbstractButton)) else type(widget).__name__
            problems.append(f"{type(widget).__name__} {label!r}: {text} on {surface} = {ratio:.2f}")
    return problems


def test_switching_repaints_every_label_and_button(installed):
    from tessellatum.gui.main_window import MainWindow

    window = MainWindow()
    window.resize(1280, 720)
    for section in window.controls.sections.values():
        section.toggle.setChecked(True)  # every group's labels on show
    try:
        for mode, scheme in ((theme.DARK_MODE, DARK), (theme.LIGHT_MODE, LIGHT), (theme.DARK_MODE, DARK)):
            installed.set_mode(mode)
            _app.processEvents()
            assert _text_problems(window.controls, COLORS[scheme]) == [], scheme
            # The Advanced groups' subsections a step quieter than the sections.
            for s in window.controls.sections.values():
                want = COLORS[scheme].text_secondary if s.nested else COLORS[scheme].text
                assert s.toggle.palette().color(QPalette.ButtonText).name().upper() == want.upper(), s.title
    finally:
        window.close()


def test_no_widget_keeps_a_color_of_its_own():
    """A widget's own stylesheet freezes the colors it names: every color comes from the theme's (BASELINE B1)."""
    from tessellatum.gui.main_window import MainWindow
    from tessellatum.gui.paints_dialog import PaintsDialog

    window = MainWindow()
    dialog = PaintsDialog((), window)
    try:
        for root in (window, dialog):
            for widget in [root, *root.findChildren(QWidget)]:
                sheet = widget.styleSheet()
                assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|\bcolor\s*:", sheet), (type(widget).__name__, sheet)
    finally:
        dialog.close()
        window.close()

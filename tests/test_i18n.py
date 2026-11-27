"""X3: the interface ready to be translated. Every string it shows goes through a translation, so a pseudo-locale
shows no English left; every string asked for is in the catalog ``packaging/update_translations.py`` writes, and a
compiled translation is shown; a choice is known by its value whatever its words; numbers are written and typed the
locale's way; files with any characters in their paths open, save, export and recover; and right to left, the
artwork's coordinates are the same."""

import importlib.util
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QLocale, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QAction, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel, QLineEdit, QListWidget, QMenu, QTableWidget, QWidget,
)

from tessellatum.app import project as projects  # noqa: E402
from tessellatum.app.recovery import RecoveryStore  # noqa: E402
from tessellatum.app.snapshot import DocumentSettings  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import i18n, main_window  # noqa: E402
from tessellatum.gui.fields import ChoiceCombo, SegmentedControl  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]
US = QLocale(QLocale.English, QLocale.UnitedStates)
GERMAN = QLocale(QLocale.German, QLocale.Germany)


def _tool():
    spec = importlib.util.spec_from_file_location("update_translations", ROOT / "packaging" / "update_translations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Recorder(i18n.PseudoTranslator):
    """The pseudo-locale, noting every string it is asked for."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.asked: set[tuple[str, str]] = set()

    def translate(self, context, source_text, disambiguation=None, n=-1):
        if source_text:
            self.asked.add((context, source_text))
        return super().translate(context, source_text, disambiguation, n)


@pytest.fixture
def translated(monkeypatch):
    """The pseudo-locale installed while the test runs."""
    recorder = Recorder(_app)
    _app.installTranslator(recorder)
    yield recorder
    _app.removeTranslator(recorder)


def _window(monkeypatch) -> MainWindow:
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 720)
    window.show()
    _app.processEvents()
    return window


def _close(window: MainWindow) -> None:
    window.export_dialog.reject()
    for dialog in (window.preferences_dialog, window.shortcuts_dialog):
        if dialog is not None:
            dialog.close()
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _wait(window: MainWindow) -> None:
    deadline = time.monotonic() + 120
    while True:
        _app.processEvents()
        if window.jobs.is_idle():
            _app.processEvents()
            if window.jobs.is_idle():
                return
        assert time.monotonic() < deadline
        time.sleep(0.005)


def _blocks(size=(400, 300)) -> np.ndarray:
    width, height = size
    picture = np.zeros((height, width, 3), np.uint8)
    picture[:, width // 2 :] = (40, 170, 40)
    picture[height // 2 :, : width // 2] = (40, 40, 200)
    picture[: height // 4, : width // 4] = (220, 220, 230)
    return picture


def _previewed(window: MainWindow) -> None:
    window.current_image_bgr = _blocks()
    window.generate_preview()
    _wait(window)


# -- The pseudo-locale --------------------------------------------------------------------------
def test_a_pseudo_translation_is_longer_and_keeps_what_a_translation_keeps():
    text = "Open {name} with &Ctrl+O at %n <b>dpi</b> &amp; more"
    shown = i18n.pseudo(text)
    assert i18n.is_pseudo(shown) and len(shown) > len(text) + 8
    prose = "The page shown is the one the settings make."
    assert len(i18n.pseudo(prose)) > len(prose) * 1.3
    for kept in ("{name}", "%n", "<b>", "</b>", "&amp;"):
        assert kept in shown
    assert shown.count("&") == text.count("&")  # the mnemonic kept, on its letter
    assert i18n.pseudo("   ") == "   " and not i18n.is_pseudo("Open")


# What a widget may show untranslated: numbers with their units and signs, file names, and the symbols the
# interface draws with (a dot for a change, a color's hex code).
DATA = re.compile(r"^[\d\s.,:;×x%#()+\-−/–·*●…|]*(mm|cm|in|px|dpi|pt|mm²|in²|°|L\*|ΔE00|Mpx|s|×|%)?[\d\s.,]*$")
FILE = re.compile(r"\S+\.(png|jpg|tessellatum|pdf|swatches|ase|gpl)$", re.I)


def _untranslated(text: str) -> bool:
    text = (text or "").strip()
    if not text or DATA.match(text) or FILE.match(text):
        return False
    # Words outside the brackets the pseudo-locale puts round each string it translates (a translated sentence may
    # hold translated words, brackets in brackets): words it was never given.
    outside, before = text, None
    while outside != before:
        before, outside = outside, re.sub(r"\[[^\[\]]*\]", "", outside)
    return bool(re.search(r"[A-Za-z]{2,}", outside))


# Lists of the page's own data: its colors (named as the legend names them, in English with the artwork) and regions.
DATA_LISTS = {"paletteList"}


def _shown(root: QWidget):
    """(what, where, text) for every string ``root`` and its children show or say."""
    for widget in [root, *root.findChildren(QWidget)]:
        where = f"{type(widget).__name__}:{widget.objectName()}"
        texts = {"tip": widget.toolTip(), "name": widget.accessibleName(), "description": widget.accessibleDescription()}
        if isinstance(widget, (QLabel, QAbstractButton, QLineEdit)):
            texts["text"] = widget.text()
        if isinstance(widget, QLineEdit):
            texts["placeholder"] = widget.placeholderText()
        if isinstance(widget, QGroupBox):
            texts["title"] = widget.title()
        if widget.isWindow():
            texts["window"] = widget.windowTitle()
        if isinstance(widget, QListWidget) and widget.objectName() not in DATA_LISTS:
            for index in range(widget.count()):
                texts[f"item {index}"] = widget.item(index).text()
                texts[f"item tip {index}"] = widget.item(index).toolTip()
        if isinstance(widget, QTableWidget):
            for column in range(widget.columnCount()):
                header = widget.horizontalHeaderItem(column)
                texts[f"header {column}"] = header.text() if header is not None else ""
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                texts[f"item {index}"] = QComboBox.itemText(widget, index)
                texts[f"item tip {index}"] = widget.itemData(index, Qt.ToolTipRole) or ""
        for what, text in texts.items():
            yield what, where, text
    for action in root.findChildren(QAction):
        for what, text in (("action", action.text()), ("action tip", action.toolTip()), ("status", action.statusTip())):
            yield what, f"QAction:{action.objectName()}", text.replace("&", "")
    for menu in root.findChildren(QMenu):
        yield "menu", f"QMenu:{menu.objectName()}", menu.title().replace("&", "")


def test_the_pseudo_locale_leaves_no_english_shown(monkeypatch, translated):
    window = _window(monkeypatch)
    try:
        roots = [window]
        found = [s for s in _shown(window) if _untranslated(s[2])]  # the welcome
        _previewed(window)
        window.open_preferences()
        window.open_shortcuts()
        window.commands["export"].trigger()
        _app.processEvents()
        roots += [window.preferences_dialog, window.shortcuts_dialog, window.export_dialog]
        for root in roots:
            found += [s for s in _shown(root) if _untranslated(s[2])]
        found = sorted(set(found))
        assert not found, "\n".join(map(str, found))
    finally:
        _close(window)


def test_every_string_asked_for_is_in_the_catalog(monkeypatch, translated):
    window = _window(monkeypatch)
    try:
        _previewed(window)
        window.open_preferences()
        window.open_shortcuts()
        window.commands["export"].trigger()
        window.show_activity(main_window.ANALYSIS)
        _wait(window)
        window.inspector.list.setCurrentRow(0)
        _app.processEvents()
    finally:
        _close(window)
    catalog = _tool().catalog()
    ours = set(catalog) | {path.stem for path in (ROOT / "src" / "tessellatum" / "gui").glob("*.py")}
    asked = {(context, text) for context, text in translated.asked
             if context in ours and not i18n.is_pseudo(text) and re.search(r"[A-Za-z]{2,}", text)
             and not re.fullmatch(r"[a-z0-9_./:-]+", text)}
    missing = sorted((context, text) for context, text in asked if text not in catalog.get(context, ()))
    assert not missing, "\n".join(map(str, missing))
    assert len(asked) > 400  # the walk reached the interface, not a corner of it


def test_a_translation_written_compiled_and_installed_is_shown(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "TS_DIR", tmp_path / "ts")
    monkeypatch.setattr(tool, "QM_DIR", tmp_path / "qm")
    messages = tool.catalog()
    path = tool.TS_DIR / "tessellatum_de.ts"
    assert tool.write_ts(path, "de", messages)[1] == sum(len(texts) for texts in messages.values())
    # A translator's work: two strings translated.
    text = path.read_text(encoding="utf-8")
    for source, german in (("Search settings", "Einstellungen suchen"), ("Up to date", "Aktuell")):
        text = re.sub(rf"(<source>{source}</source>\s*)<translation type=\"unfinished\" />",
                      rf"\g<1><translation>{german}</translation>", text)
    path.write_text(text, encoding="utf-8")
    # Updated again with a string gone: the translations kept, the string kept as vanished.
    smaller = {context: set(texts) for context, texts in messages.items()}
    smaller["actions"].discard("&Help")
    kept, added, obsolete = tool.write_ts(path, "de", smaller)
    assert kept >= 2 and added == 0 and obsolete == 1 and "Einstellungen suchen" in path.read_text(encoding="utf-8")
    assert tool.release() == 1
    monkeypatch.setattr(i18n, "TRANSLATIONS_DIR", tmp_path / "qm")
    try:
        assert i18n.install(_app, GERMAN) == "tessellatum_de.qm"
        assert i18n.translate("search", "Search settings") == "Einstellungen suchen"
        assert i18n.translate("freshness", "Up to date") == "Aktuell"
        assert i18n.translate("freshness", "Out of date") == "Out of date"  # not translated yet: English
    finally:
        i18n.uninstall(_app)
    assert i18n.translate("search", "Search settings") == "Search settings"


def test_the_pseudo_locale_is_installed_when_asked_for(monkeypatch):
    monkeypatch.setenv(i18n.PSEUDO_ENV, "1")
    try:
        assert i18n.install(_app) == "pseudo"
        assert i18n.is_pseudo(i18n.translate("actions", "Find Setting"))
    finally:
        i18n.uninstall(_app)


# -- Choices by value -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", [ChoiceCombo, SegmentedControl])
def test_a_choice_is_known_by_its_value_whatever_its_words(translated, kind):
    choices = kind()
    choices.addItems(["A4", "Letter", "A3"])
    said = []
    (choices.valueChanged if kind is ChoiceCombo else choices.currentTextChanged).connect(said.append)
    shown = QComboBox.itemText(choices, 1) if kind is ChoiceCombo else choices.buttons()[1].text()
    assert i18n.is_pseudo(shown)
    assert choices.itemText(1) == "Letter" and choices.findText("Letter") == 1
    choices.setCurrentText("A3")
    assert choices.currentText() == "A3" and said == ["A3"]


# -- Numbers ----------------------------------------------------------------------------------------------------------
def test_numbers_are_written_the_locale_s_way():
    text = i18n.translate("status", "{0:,} × {1:,} px · {mm:.1f} mm · {g:g} pt · {hex:02X}")
    assert text.format(1448, 1086, mm=2.25, g=0.5, hex=255) == "1,448 × 1,086 px · 2.2 mm · 0.5 pt · FF"
    QLocale.setDefault(GERMAN)
    try:
        assert text.format(1448, 1086, mm=2.25, g=0.5, hex=255) == "1.448 × 1.086 px · 2,2 mm · 0,5 pt · FF"
    finally:
        QLocale.setDefault(US)


def test_a_german_window_shows_and_takes_numbers_the_german_way(monkeypatch):
    QLocale.setDefault(GERMAN)
    window = _window(monkeypatch)
    try:
        _previewed(window)
        # The page's size in pixels with a German group separator; the difficulty's description in German numbers.
        size = window.status.size_label.text()
        assert re.search(r"\d\.\d{3} × \d\.\d{3} px", size), size
        description = window.controls.preset_description.text()
        assert "ΔE00" in description and re.search(r"\d,\d", description), description
        brush = window.controls.controls["min_width_mm"]
        assert brush.type_text("1,5") and brush.value() == pytest.approx(1.5) and brush.text() == "1,5 mm"
        _wait(window)
    finally:
        _close(window)
        QLocale.setDefault(US)


# -- Paths with any characters ------------------------------------------------------------------------------------------
def _save_to(monkeypatch, path):
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))


def test_files_with_any_characters_in_their_paths(monkeypatch, tmp_path):
    folder = tmp_path / "Löwen 猫 ü — Ω"
    folder.mkdir()
    picture = folder / "Bild é 画.png"
    image = QImage(240, 160, QImage.Format_RGB32)
    image.fill(Qt.darkCyan)
    assert image.save(str(picture))
    window = _window(monkeypatch)
    try:
        window.open_path(picture)
        _app.processEvents()
        assert window.current_image_bgr is not None and window.document.source().name == picture.name
        window.generate_preview()
        _wait(window)
        for radio, name in ((window.controls.png_radio, "Seite ü.png"), (window.controls.pdf_radio, "Seite 画.pdf"),
                            (window.controls.palette_radio, "Palette é.gpl")):
            radio.setChecked(True)
            if radio is window.controls.palette_radio:
                window.controls.palette_combo.setCurrentIndex(window.controls.palette_combo.findData(".gpl"))
            _save_to(monkeypatch, folder / name)
            window.export_page()
            _wait(window)
            assert (folder / name).stat().st_size > 0, name
        project = folder / f"Projekt ö 猫{projects.SUFFIX}"
        assert window._write_project(project)
        assert window.open_project(project) and window.document.source().name == picture.name
    finally:
        _close(window)
    store = RecoveryStore(folder / "Wiederherstellung ä")
    data = picture.read_bytes()
    store.write(DocumentSettings(), picture.name, data, projects.decode_image(data), picture.name)
    store._lock.unlock()  # the session ends without closing, as a crash leaves it
    store._locked = False
    [found] = RecoveryStore(folder / "Wiederherstellung ä").pending()
    assert found.name == picture.name


# -- Right to left -----------------------------------------------------------------------------------------------------
def test_right_to_left_the_artwork_s_coordinates_are_the_same(monkeypatch, tmp_path):
    exported, clicks = {}, {}
    for direction in (Qt.LeftToRight, Qt.RightToLeft):
        _app.setLayoutDirection(direction)
        window = _window(monkeypatch)
        try:
            assert window.layoutDirection() == direction
            _previewed(window)
            window.preview.fit_to_window()
            _app.processEvents()
            point = QPointF(100.5, 60.5)  # in the page's pixels
            on_screen = window.preview.mapFromScene(point)
            got = []
            window.preview.clicked.connect(got.append)
            QTest.mouseClick(window.preview.viewport(), Qt.LeftButton, Qt.NoModifier, on_screen)
            _app.processEvents()
            clicks[direction] = (got[-1].x(), got[-1].y()) if got else None
            assert window.region_at(got[-1]) == window.region_at(point)
            window.controls.png_radio.setChecked(True)
            _save_to(monkeypatch, tmp_path / f"{direction.name}.png")
            window.export_page()
            _wait(window)
            exported[direction] = (tmp_path / f"{direction.name}.png").read_bytes()
        finally:
            _close(window)
            _app.setLayoutDirection(Qt.LeftToRight)
    assert clicks[Qt.LeftToRight] == pytest.approx(clicks[Qt.RightToLeft], abs=1.0)
    assert exported[Qt.LeftToRight] == exported[Qt.RightToLeft]

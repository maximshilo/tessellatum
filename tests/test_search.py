"""W6: settings search. A word finds a setting by its name, the words it may be called by, what it does or where it is,
best first; a result says where the setting is and what it is set to; choosing one opens its page, unfolds its section
(the Advanced group too), scrolls to the control itself, focuses it and outlines it a moment; and all of it works from
the keyboard alone."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import pipeline, settings  # noqa: E402
from tessellatum.gui import main_window, search  # noqa: E402
from tessellatum.gui.controls_panel import ControlsPanel, advanced_key  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _names(text: str) -> list[str]:
    return [match.entry.name for match in search.search(text)]


# -- Finding ------------------------------------------------------------------------------------
def test_a_word_finds_a_setting_by_its_name_first():
    assert _names("brush")[0] == "min_width_mm"  # Brush width
    assert _names("Brush width")[0] == "min_width_mm"
    assert _names("smooth")[:2] == ["blur_sigma", "line_smoothing_mm"]  # "Smoothing" before "Line smoothing"
    assert _names("") == [] and _names("   ") == []
    assert _names("xyzzy") == []


def test_the_words_a_setting_may_be_called_by_find_it():
    assert _names("outline")[0] == "line_width_mm"
    assert "line_width_mm" in _names("stroke")
    # "Darkest black on paper", and the dark marks, before the grays (whose aliases say dark) and the tone.
    assert _names("dark")[:3] == ["paper_black_lstar", "mark_contrast", "mark_length_mm"]
    assert {"line_gray", "tone"} <= set(_names("dark"))
    assert set(_names("numbers")[:2]) == {"label_gray", "min_label_pt"}  # a plural finds its singular
    assert "medium" in _names("procreate")  # Coloring on: a choice, by its tooltip's words
    assert "mark_contrast" in _names("pupils")


def test_every_word_must_match():
    assert _names("dark marks")[:2] == ["mark_contrast", "mark_length_mm"]
    assert "black_point_lstar" not in _names("dark marks")


def test_every_entry_finds_itself_by_its_label():
    for entry in settings.entries():
        assert _names(entry.label)[0] == entry.name or entry.name in _names(entry.label)[:3], entry.name


# -- The results ---------------------------------------------------------------------------------
def test_a_result_says_where_it_is_and_what_it_is_set_to():
    panel = ControlsPanel()
    panel.search.field.setText("brush")
    first = panel.search.results.item(0).text()
    assert first.startswith("Brush width") and panel.controls["min_width_mm"].text() in first
    assert "Generation · Detail" in first
    panel.search.field.setText("mark contrast")
    assert "Generation · Advanced" in panel.search.results.item(0).text()
    panel.search.field.setText("paper")
    assert "A4" in panel.search.results.item(0).text()
    panel.search.field.setText("xyzzy")
    assert panel.search.names() == [] and panel.search.results.item(0).text() == "No setting matches."
    panel.search.clear()
    assert panel.search.results.isHidden()


def test_choosing_a_result_opens_its_page_unfolds_it_and_focuses_the_control(monkeypatch):
    panel = ControlsPanel()
    panel.resize(360, 500)
    panel.show()
    _app.processEvents()
    assert panel.category() == settings.GENERATION
    group = panel.sections[advanced_key(settings.DOCUMENT)]
    assert not group.is_expanded()
    panel.search.field.setText("leader reach")
    panel.search.choose_current()
    _app.processEvents()
    field = panel.controls["leader_reach_mm"]
    assert panel.category() == settings.DOCUMENT and group.is_expanded() and not group.body.isHidden()
    assert QApplication.focusWidget() in (field.slider, field.entry)
    assert field.property("found") is True and panel.search.field.text() == ""
    # It is scrolled into sight.
    scroll = panel.scrolls[settings.DOCUMENT]
    top = field.mapTo(scroll.widget(), field.rect().topLeft()).y()
    bar = scroll.verticalScrollBar()
    assert bar.value() <= top <= bar.value() + scroll.viewport().height() - field.height()
    # The outline goes after a moment.
    panel._unflash()
    assert not field.property("found")
    # A choice is found as well, and focused.
    panel.search.field.setText("paper")
    panel.search.choose_current()
    _app.processEvents()
    assert panel.category() == settings.DOCUMENT
    assert QApplication.focusWidget() is panel.paper_combo or panel.paper_combo.isAncestorOf(QApplication.focusWidget())


# -- The keyboard --------------------------------------------------------------------------------
@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    window.activateWindow()
    _app.processEvents()
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_the_search_is_done_by_keyboard_alone(window):
    window.set_sidebar_visible(False)
    window.commands["find"].trigger()  # Ctrl+F: the sidebar back, the focus in the field
    _app.processEvents()
    field = window.controls.search.field
    assert window.sidebar_visible() and QApplication.focusWidget() is field
    assert window.commands["find"].shortcut().toString() == "Ctrl+F"
    QTest.keyClicks(field, "smooth")
    assert window.controls.search.names()[:2] == ["blur_sigma", "line_smoothing_mm"]
    QTest.keyClick(field, Qt.Key_Down)  # the second: Line smoothing
    QTest.keyClick(field, Qt.Key_Return)
    _app.processEvents()
    target = window.controls.controls["line_smoothing_mm"]
    assert window.controls.category() == settings.DOCUMENT
    assert QApplication.focusWidget() in (target.slider, target.entry)
    # Esc in the field clears it.
    window.commands["find"].trigger()
    QTest.keyClicks(field, "brush")
    QTest.keyClick(field, Qt.Key_Escape)
    assert field.text() == "" and window.controls.search.results.isHidden()
    # The results give the sidebar back when the focus moves on, and come back with it.
    QTest.keyClicks(field, "brush")
    assert window.controls.search.results.isVisible()
    window.preview.setFocus()
    _app.processEvents()
    assert window.controls.search.results.isHidden() and field.text() == "brush"
    window.commands["find"].trigger()
    _app.processEvents()
    assert window.controls.search.results.isVisible()


def test_find_leaves_focus_mode(window):
    window.set_focus_mode(True)
    window.commands["find"].trigger()
    _app.processEvents()
    assert window.sidebar_visible() and QApplication.focusWidget() is window.controls.search.field

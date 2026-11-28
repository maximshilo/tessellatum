"""X2: the keyboard reference and the keys' safety. Help > Keyboard Shortcuts lists every command's keys in the
platform's notation, and the keys that work in one place, searchable by name or by keys; no two commands share a key
(``test_actions``), no command takes a key a widget uses on its own, typing in a text field never triggers one, and
Esc in a field or a list does that field's or list's job, not Cancel's, while a page is being made."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import main_window, shortcuts  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])

# Keys a widget uses on its own: the canvas (Space, the arrows), lists (arrows, Enter), fields (steps), Tab.
WIDGET_KEYS = {Qt.Key_Space, Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown,
               Qt.Key_Home, Qt.Key_End, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Backspace,
               Qt.Key_Delete}
TYPING = Qt.NoModifier | Qt.ShiftModifier | Qt.KeypadModifier


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    window.activateWindow()
    _app.processEvents()
    yield window
    if window.shortcuts_dialog is not None:
        window.shortcuts_dialog.reject()
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _sequences(window):
    return [(action, sequence) for action in window.commands for sequence in action.shortcuts() if not sequence.isEmpty()]


def _triggered(window) -> list[str]:
    fired: list[str] = []
    for action in window.commands:
        action.triggered.connect(lambda _checked=False, a=action: fired.append(a.objectName()))
    return fired


# -- The reference ------------------------------------------------------------------------------
def test_it_lists_every_command_s_keys_as_the_platform_writes_them(window):
    window.commands["shortcuts"].trigger()
    dialog = window.shortcuts_dialog
    _app.processEvents()
    assert dialog.isVisible() and _app.focusWidget() is dialog.reference.search
    rows = {command: (keys, where) for command, keys, where in dialog.reference.rows()}
    for action, _sequence in _sequences(window):
        name = action.text().replace("&", "").rstrip("…")
        keys, where = rows[name]
        for sequence in action.shortcuts():
            assert sequence.toString(QKeySequence.NativeText) in keys
        assert where == action.property("menu")
    assert rows["Find Setting"][0].startswith(QKeySequence("Ctrl+F").toString(QKeySequence.NativeText))
    assert rows["Keyboard Shortcuts"][1] == "Help"
    # And the keys that work in one place.
    for what, keys, where in shortcuts.context_keys():
        assert rows[what] == (keys, where)


def test_help_keyboard_shortcuts_has_a_key_of_its_own(window):
    assert window.commands["shortcuts"].shortcut() == QKeySequence("Ctrl+/")
    assert window.commands["shortcuts"] in window.menuBar().actions()[-1].menu().actions()


def test_it_is_searched_by_name_by_keys_or_by_place(window):
    reference = shortcuts.ShortcutReference(window.commands)

    def shown(text):
        reference.search.setText(text)
        return [row[0] for row in reference.rows()]

    assert shown("find setting") == ["Find Setting", "Through the settings found; go to one", "Clear the search"]
    assert "Find Setting" in shown("ctrl+f")  # in Qt's portable words, on macOS too
    assert "Find Setting" in shown(QKeySequence("Ctrl+F").toString(QKeySequence.NativeText).lower())
    palette = shown("palette")
    assert "List the color's regions" in palette and "Back to the palette" in palette  # by place, or by name
    assert shown("no such thing") == [] and not reference.empty.isHidden()
    assert len(shown("")) == reference.table.rowCount() and reference.empty.isHidden()


def test_the_modifier_before_a_wheel_is_the_platform_s():
    zoom = next(keys for what, keys, _where in shortcuts.context_keys() if what == "Zoom about the pointer")
    written = QKeySequence("Ctrl+A").toString(QKeySequence.NativeText)[:-1]  # "Ctrl+" here, "⌘" on macOS
    assert f"{written}wheel" in zoom


def test_preferences_shows_the_same_reference(window):
    window.open_preferences()
    dialog = window.preferences_dialog
    assert isinstance(dialog.shortcut_reference, shortcuts.ShortcutReference)
    assert dialog.shortcuts is dialog.shortcut_reference.table
    dialog.reject()


# -- Conflicts and text-field safety --------------------------------------------------------------
def test_no_command_takes_a_key_a_widget_uses_on_its_own(window):
    for action, sequence in _sequences(window):
        combination = sequence[0]
        if combination.keyboardModifiers() & ~TYPING:
            continue  # with Ctrl, Alt or Meta: no widget's own
        assert combination.key() not in WIDGET_KEYS, (action.objectName(), sequence.toString())


def test_no_command_is_a_key_that_types(window):
    # A command on a key that types (a letter, a digit, a sign, Shift with one) would fire while a field has the focus.
    for action, sequence in _sequences(window):
        combination = sequence[0]
        types = not (combination.keyboardModifiers() & ~TYPING) and int(combination.key()) < 0x01000000
        assert not types, (action.objectName(), sequence.toString())


def test_typing_in_a_field_triggers_nothing_and_space_types_a_space(window):
    fired = _triggered(window)
    field = window.controls.search.field
    field.setFocus()
    _app.processEvents()
    QTest.keyClicks(field, "rfg 0+-=/")
    _app.processEvents()
    assert field.text() == "rfg 0+-=/" and fired == []
    assert not window.preview.space_held()


def test_esc_in_a_field_or_a_list_is_its_own_while_a_page_is_being_made(window):
    # While a page is made, Esc is Cancel's: a field or a list that uses Esc keeps it.
    fired = _triggered(window)
    window.commands["cancel"].setEnabled(True)
    search = window.controls.search.field
    search.setFocus()
    QTest.keyClicks(search, "brush")
    QTest.keyClick(search, Qt.Key_Escape)
    _app.processEvents()
    assert search.text() == "" and "cancel" not in fired
    # A value being typed is put back.
    from tessellatum.gui.fields import SettingField

    field = next(f for f in window.controls.findChildren(SettingField) if f.isEnabled())  # not one that's off
    before = field.entry.text()
    field.entry.setFocus()
    field.entry.selectAll()
    QTest.keyClicks(field.entry, "7")
    assert field.entry.isModified()
    QTest.keyClick(field.entry, Qt.Key_Escape)
    _app.processEvents()
    assert field.entry.text() == before and "cancel" not in fired
    # With nothing of its own to do, Esc is Cancel's.
    QTest.keyClick(search, Qt.Key_Escape)
    _app.processEvents()
    assert fired.count("cancel") == 1

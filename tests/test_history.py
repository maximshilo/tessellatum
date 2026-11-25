"""H1: undo and redo. Each change to the settings or the picture is one step back, named for what it did ("Undo Brush
width"); a slider dragged is one step, and so is a preset or a reset that moves many settings; a new change after an
undo drops what could be redone; the view (zoom, panels, the version shown) is never in it; and any mix of edits,
undos and redos ends where a replay of them ends."""

import os
import random
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty, pipeline, settings  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()  # no store: nothing kept
    window.resize(1280, 840)
    window.show()
    _app.processEvents()
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _settle() -> None:
    for _ in range(3):
        _app.processEvents()


def _undo_text(window) -> str:
    return window.commands["undo"].text().replace("&", "")


def test_an_edit_is_undone_and_redone_by_its_name(window):
    controls = window.controls
    assert not window.commands["undo"].isEnabled() and not window.commands["redo"].isEnabled()
    start = controls.snapshot()
    controls.controls["min_width_mm"].type_text("1.2")
    _settle()
    assert _undo_text(window) == "Undo Brush width" and window.commands["undo"].isEnabled()
    edited = controls.snapshot()
    window.commands["undo"].trigger()
    assert controls.snapshot() == start and controls.preset_combo.currentText() == "Medium"
    assert window.document.settings() == start
    assert window.commands["redo"].text().replace("&", "") == "Redo Brush width"
    window.commands["redo"].trigger()
    assert controls.snapshot() == edited and controls.preset_combo.currentText() == "Custom"
    # Undoing records nothing of its own.
    assert window.history.count() == 1


def test_a_slider_dragged_is_one_step(window):
    field = window.controls.controls["blur_sigma"]
    start = window.controls.snapshot()
    field.slider.sliderPressed.emit()
    for _ in range(6):
        field.step(1)
        _settle()
    field.slider.sliderReleased.emit()
    _settle()
    assert window.history.count() == 1 and _undo_text(window) == "Undo Smoothing"
    # The next edit of the same setting is a step of its own.
    field.step(1)
    _settle()
    assert window.history.count() == 2
    window.undo()
    window.undo()
    assert window.controls.snapshot() == start


def test_a_preset_a_reset_and_the_paints_are_each_one_step(window):
    controls = window.controls
    start = controls.snapshot()
    controls.preset_combo.setCurrentText("Hard")
    _settle()
    assert window.history.count() == 1 and _undo_text(window) == "Undo Difficulty: Hard"
    controls.controls["min_width_mm"].step(2)
    controls.controls["num_colors"].step(-2)
    _settle()
    assert window.history.count() == 2  # two edits in one go: one step
    controls.sections["Detail"].reset_button.click()
    _settle()
    assert _undo_text(window) == "Undo Reset Detail" and controls.preset_combo.currentText() == "Hard"
    controls.tone_combo.setCurrentText("Dark")
    _settle()
    assert _undo_text(window) == "Undo Tone: Dark"
    controls.set_paints((("Titanium white", (245, 245, 240)), ("Mars black", (30, 30, 30))))
    _settle()
    assert _undo_text(window) == "Undo Paints"
    window.commands["reset"].trigger()
    _settle()
    assert _undo_text(window) == "Undo Reset All Settings"
    for _ in range(window.history.count()):
        window.undo()
    assert controls.snapshot() == start and not window.commands["undo"].isEnabled()


def test_a_new_edit_after_an_undo_drops_the_redo(window):
    controls = window.controls
    controls.controls["min_width_mm"].step(1)
    _settle()
    controls.controls["blur_sigma"].step(1)
    _settle()
    window.undo()
    assert window.history.can_redo()
    controls.controls["line_width_mm"].step(1)
    _settle()
    assert not window.history.can_redo() and window.history.count() == 2
    assert _undo_text(window) == "Undo Line width"


def test_the_clean_state(window):
    history = window.history
    assert history.is_clean()
    window.controls.controls["min_width_mm"].step(1)
    _settle()
    assert not history.is_clean()
    history.set_clean()  # as a save will (projects)
    window.controls.controls["blur_sigma"].step(1)
    _settle()
    assert not history.is_clean()
    window.undo()
    assert history.is_clean()


def test_the_view_is_never_in_the_history(window):
    window.current_image_bgr = np.full((300, 400, 3), 128, np.uint8)
    window.generate_preview()
    deadline = time.monotonic() + 60
    while not window.jobs.is_idle() or window.current_page is None:
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)
    before = window.history.count()
    window.commands["zoom_in"].trigger()
    window.commands["fit"].trigger()
    window.commands["toggle_sidebar"].trigger()
    window.commands["toggle_sidebar"].trigger()
    window.commands["toggle_inspector"].trigger()
    window.show_activity(settings.DOCUMENT)
    window.version_bar.buttons[main_window.Version.COMPLETED].click()
    window.inspector.list.setCurrentRow(0)
    window.controls.sections["Detail"].toggle.click()
    window.set_diagnostics_visible(True)
    window.set_focus_mode(True)
    window.set_focus_mode(False)
    _settle()
    assert window.history.count() == before == 0


def test_restoring_settings_the_preview_wasn_t_made_with_says_it_is_out_of_date(window):
    window.current_image_bgr = np.full((300, 400, 3), 128, np.uint8)
    window.controls.controls["min_width_mm"].step(2)
    _settle()
    window.generate_preview()
    deadline = time.monotonic() + 60
    while not window.jobs.is_idle() or window.current_page is None:
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)
    window.undo()
    assert window.status.text().startswith("The preview is out of date")
    window.redo()  # the settings it was made with again
    assert window.document.settings().changes_from(window.document.result_settings()) == ()


def test_a_picture_replaced_is_one_step(window, tmp_path, monkeypatch):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (300, 200), (200, 40, 40)).save(first)
    Image.new("RGB", (400, 200), (40, 40, 200)).save(second)
    for path in (first, second):
        monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, p=path, **k: (str(p), "")))
        window.commands["open"].trigger()
    assert window.history.count() == 1 and _undo_text(window) == "Undo Open second.png"  # the first opened nothing
    window.undo()
    assert window.document.source().path == first and window.current_image_bgr.shape[:2] == (200, 300)
    assert window.controls.picture_label.text().startswith("first.png")
    window.redo()
    assert window.document.source().path == second and window.current_image_bgr.shape[:2] == (200, 400)


# -- Any mix of edits, undos and redos ends where a replay of them ends --------------------------
def _random_edit(window, rng: random.Random) -> None:
    controls = window.controls
    kind = rng.random()
    if kind < 0.6:
        s = rng.choice(settings.SETTINGS)
        field = controls.controls[s.name]
        if not field.isEnabled():
            return
        field.step(rng.choice((-3, -1, 1, 2)))
    elif kind < 0.75:
        controls.preset_combo.setCurrentText(rng.choice(difficulty.preset_names()[:-1]))
    elif kind < 0.85:
        controls.tone_combo.setCurrentText(rng.choice(("Light", "Medium", "Dark")))
    elif kind < 0.95:
        controls.paper_combo.setCurrentText(rng.choice(("A4", "A3", "A2")))
    else:
        controls.medium_combo.setCurrentText(rng.choice(difficulty.MEDIA))


def test_any_mix_of_edits_undos_and_redos_ends_where_a_replay_ends(window):
    rng = random.Random(20261125)
    for _sequence in range(100):
        window.history.clear()
        states = [window.controls.snapshot()]
        at = 0
        for _step in range(12):
            action = rng.random()
            if action < 0.55:
                _random_edit(window, rng)
                _settle()
                now = window.controls.snapshot()
                if now != states[at]:
                    del states[at + 1 :]
                    states.append(now)
                    at += 1
            elif action < 0.8:
                window.commands["undo"].trigger()
                at = max(0, at - 1)
            else:
                window.commands["redo"].trigger()
                at = min(len(states) - 1, at + 1)
            assert window.history.count() == len(states) - 1
        assert window.controls.snapshot() == states[at]
        assert window.document.settings() == states[at]


def test_a_history_torn_down_with_changes_in_it_raises_nothing(monkeypatch):
    """Its stack clears itself as it goes, after the history: nothing may still be connected to call back."""
    import sys

    from PySide6.QtCore import QObject

    from tessellatum.app.history import History

    raised = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: raised.append(exc))
    owner = QObject()
    history = History(lambda _s: None, lambda _s: None, owner)
    history.record_settings(1, 2, "one")
    history.record_settings(2, 3, "two")
    history.undo()
    owner.deleteLater()
    _app.processEvents()
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert raised == []

"""G2: the preview's state (current, modified, processing, cancelled, failed) and when a change is applied: the legend
at once, the lines, numbers and tone a moment after the last change, everything else with Generate Preview."""

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import pipeline, settings  # noqa: E402
from tessellatum.core.pipeline import PipelineCancelled  # noqa: E402
from tessellatum.gui import freshness, main_window, preferences  # noqa: E402
from tessellatum.gui.freshness import CANCELLED, CURRENT, FAILED, MODIFIED, NO_PREVIEW, PROCESSING  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120


# -- The registry's policies ---------------------------------------------------------------------------------------
def test_the_lines_numbers_and_tone_redraw_the_legend_applies_at_once_the_rest_waits_for_generate():
    debounced = {e.name for e in settings.entries() if e.policy == settings.DEBOUNCED}
    style_sliders = {s.name for s in settings.SETTINGS if s.owner == settings.STYLE}
    assert debounced == style_sliders | {"tone"}
    assert len(style_sliders) == 10
    assert settings.choice("legend_detail").policy == settings.IMMEDIATE
    assert settings.choice("paper").policy == settings.EXPLICIT  # the page's size: made again


@pytest.mark.parametrize("names, policy", [
    ((), settings.IMMEDIATE),
    (("legend_detail",), settings.IMMEDIATE),
    (("line_width_mm",), settings.DEBOUNCED),
    (("tone", "line_gray", "label_gray", "leader_gray"), settings.DEBOUNCED),
    (("line_width_mm", "num_colors"), settings.EXPLICIT),
    (("preset",), settings.EXPLICIT),
    (("settings",), settings.EXPLICIT),  # a difference in nothing named waits for Generate
])
def test_a_change_is_applied_as_its_slowest_entry_asks(names, policy):
    assert settings.update_policy(names) == policy


# -- The states ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("has_page, making, changed, outcome, state", [
    (False, False, (), None, NO_PREVIEW),
    (True, False, (), None, CURRENT),
    (True, False, ("Line width",), None, MODIFIED),
    (True, True, ("Line width",), None, PROCESSING),
    (False, True, (), None, PROCESSING),
    (True, False, (), CANCELLED, CANCELLED),
    (True, False, ("Colors",), FAILED, FAILED),
    (False, False, (), FAILED, FAILED),
])
def test_the_preview_state(has_page, making, changed, outcome, state):
    assert freshness.preview_state(has_page, making, changed, outcome) == state


def test_a_cancelled_or_failed_preview_says_the_page_shown_is_the_earlier_one():
    assert freshness.state_text(CANCELLED) == "Cancelled · earlier preview"
    assert freshness.state_text(FAILED, has_page=False) == "Failed"
    assert freshness.state_text(MODIFIED) == "Out of date"


# -- The window ----------------------------------------------------------------------------------------------------
class _Window:
    """A window (no settings kept) counting the pages the pipeline makes; ``hold`` keeps the next ones from finishing
    until released, and ``fail`` makes them fail."""

    def __init__(self, monkeypatch):
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
        self.made: list[float] = []  # each page made, its line width
        self.hold: threading.Event | None = None
        self.started = threading.Event()
        self.fail = False
        real = pipeline.generate

        def counting(*args, **kwargs):
            self.started.set()
            if self.hold is not None:
                should_cancel = kwargs.get("should_cancel")
                while not self.hold.wait(0.005):
                    if should_cancel is not None and should_cancel():
                        raise PipelineCancelled()
            if self.fail:
                raise MemoryError("out of memory")
            self.made.append(kwargs["style"].line_width_mm)
            return real(*args, **kwargs)

        monkeypatch.setattr(pipeline, "generate", counting)
        self.window = MainWindow()
        self.window.preferences.set_value(preferences.REDRAW_KEY, "After 0.2 s")
        picture = np.zeros((200, 300, 3), dtype=np.uint8)
        picture[:, :150] = (200, 40, 40)
        picture[:100, 150:] = (40, 170, 40)
        picture[100:, 150:] = (40, 40, 200)
        self.window.current_image_bgr = picture

    def settle(self, seconds: float = 0.0) -> None:
        """Run the event loop until no job is wanted and no redraw waits, and at least ``seconds``."""
        window = self.window
        start = time.monotonic()
        deadline = start + TIMEOUT_S
        while True:
            _app.processEvents()
            done = window.jobs.is_idle() and not window._redraw_timer.isActive()
            if done and time.monotonic() - start >= seconds:
                _app.processEvents()
                if window.jobs.is_idle() and not window._redraw_timer.isActive():
                    return
            assert time.monotonic() < deadline, "the window didn't settle"
            time.sleep(0.005)

    def state(self) -> str:
        return self.window.freshness.state()

    def preview(self) -> None:
        self.window.generate_preview()
        self.settle()


@pytest.fixture
def win(monkeypatch):
    w = _Window(monkeypatch)
    yield w
    w.window.close()
    w.window.deleteLater()
    _app.processEvents()


def test_no_preview_until_one_is_made_and_then_it_is_up_to_date(win):
    assert win.state() == NO_PREVIEW and "Generate Preview" in win.window.freshness.toolTip()
    win.preview()
    assert win.state() == CURRENT and win.window.freshness.text() == "Up to date"
    assert len(win.made) == 1


def test_a_slider_dragged_through_the_lines_draws_the_page_once_when_it_stops(win):
    win.preview()
    made = len(win.made)
    line = win.window.controls.controls["line_width_mm"]
    for i in range(50):  # 50 ticks of a drag, well inside the delay
        line.step(1 if i % 2 == 0 else -1) if i < 49 else line.step(1)
        _app.processEvents()
    assert len(win.made) == made  # no page on a tick
    assert win.state() == MODIFIED and "drawn again in a moment" in win.window.freshness.toolTip()
    win.settle(0.3)
    assert len(win.made) == made + 1  # one, after the last
    assert win.state() == CURRENT and win.window.document.result_is_current()
    assert win.made[-1] == pytest.approx(win.window.controls.snapshot().values["line_width_mm"])


def test_the_tone_redraws_by_itself_too(win):
    win.preview()
    made = len(win.made)
    win.window.controls.tone_combo.setCurrentText("Dark")
    win.settle(0.3)
    assert len(win.made) == made + 1 and win.state() == CURRENT


def test_a_change_that_makes_the_page_again_waits_for_generate(win):
    win.preview()
    made = len(win.made)
    win.window.controls.controls["num_colors"].step(-1)
    win.settle(0.5)
    assert len(win.made) == made  # never by itself
    assert win.state() == MODIFIED and win.window.freshness.text() == "Out of date"
    assert "Generate Preview (F5)" in win.window.freshness.toolTip()
    # A line width moved as well: the page still waits, as one of its changes needs Generate.
    win.window.controls.controls["line_width_mm"].step(1)
    win.settle(0.5)
    assert len(win.made) == made
    win.preview()
    assert len(win.made) == made + 1 and win.state() == CURRENT


def test_the_legend_s_detail_applies_at_once_and_leaves_the_preview_up_to_date(win):
    win.preview()
    made = len(win.made)
    combo = win.window.controls.legend_combo
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    win.settle(0.3)
    assert len(win.made) == made and win.state() == CURRENT


def test_redrawing_by_itself_can_be_turned_off_in_preferences(win):
    win.preview()
    made = len(win.made)
    win.window.preferences.set_value(preferences.REDRAW_KEY, "Only with Generate")
    win.window.controls.controls["line_width_mm"].step(1)
    win.settle(0.5)
    assert len(win.made) == made and win.state() == MODIFIED
    assert "Generate Preview" in win.window.status.text()


def test_putting_a_change_back_before_the_redraw_cancels_it(win):
    win.preview()
    made = len(win.made)
    line = win.window.controls.controls["line_width_mm"]
    line.step(1)
    _app.processEvents()
    line.step(-1)
    win.settle(0.4)
    assert len(win.made) == made and win.state() == CURRENT


def test_a_redraw_asked_for_while_a_page_is_made_supersedes_it_and_the_last_page_is_current(win):
    win.preview()
    win.hold = threading.Event()
    win.started.clear()
    win.window.controls.controls["num_colors"].step(-1)
    win.window.generate_preview()
    assert win.started.wait(TIMEOUT_S)
    _app.processEvents()
    assert win.state() == PROCESSING and win.window.freshness.text() == "Generating…"
    win.window.controls.controls["line_width_mm"].step(1)
    time.sleep(0.3)
    _app.processEvents()  # the redraw is due: it cancels the page being made and starts again
    win.hold.set()
    win.settle(0.3)
    assert win.state() == CURRENT and win.window.document.result_is_current()


def test_a_cancelled_preview_keeps_the_earlier_page_labeled_until_something_changes(win):
    win.preview()
    shown = win.window.current_page
    win.hold = threading.Event()
    win.started.clear()
    win.window.controls.controls["num_colors"].step(-1)
    win.window.generate_preview()
    assert win.started.wait(TIMEOUT_S)
    win.window.abort_generation()
    win.settle()
    assert win.window.current_page is shown
    assert win.state() == CANCELLED and win.window.freshness.text() == "Cancelled · earlier preview"
    assert "made before it" in win.window.freshness.toolTip()
    win.hold = None
    win.window.controls.controls["num_colors"].step(-1)
    win.settle(0.3)
    assert win.state() == MODIFIED


def test_a_failed_preview_keeps_the_earlier_page_labeled(win):
    win.preview()
    shown = win.window.current_page
    win.fail = True
    win.window.controls.controls["num_colors"].step(-1)
    win.preview()
    assert win.window.current_page is shown
    assert win.state() == FAILED and win.window.freshness.text() == "Failed · earlier preview"


def test_a_new_picture_has_no_preview(win):
    win.preview()
    win.window.current_image_bgr = np.full((100, 100, 3), 90, np.uint8)
    _app.processEvents()
    assert win.state() == NO_PREVIEW


def test_freshness_is_apart_from_unsaved_changes(win):
    win.preview()
    win.window.history.set_clean()
    win.window.controls.controls["num_colors"].step(-1)
    win.settle(0.1)
    assert not win.window.history.is_clean() and win.state() == MODIFIED
    win.preview()
    # Making the page saves nothing: the document still has its change, though the preview is up to date.
    assert not win.window.history.is_clean() and win.state() == CURRENT
    win.window.undo()
    win.settle(0.1)
    assert win.window.history.is_clean() and win.state() == MODIFIED

"""H6: a change that makes the page again (the colors, the difficulty, the paper, what is done with line art, faces and
text) may make it by itself, as a preference: Off (the default), When it's quick (while the picture's last page took
under 2 s), or Always. One page a second after the last change; it supersedes a page being made, waits for an export,
and is called off when the change is put back; a page kept is shown at once whatever the preference."""

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app.jobs import PREVIEW, PRINT_LAYOUT, SAVE, Job  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import freshness, preferences  # noqa: E402
from test_results import _Window  # noqa: E402  (the counting window)

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120
AFTER_S = preferences.AUTO_AFTER_MS / 1000


class _Auto(_Window):
    def settle(self, seconds: float = 0.0) -> None:
        """As the window's, waiting for the automatic page's timer too."""
        start = time.monotonic()
        while True:
            super().settle()
            if not self.window._auto_timer.isActive() and time.monotonic() - start >= seconds:
                _app.processEvents()
                if self.window.jobs.is_idle() and not self.window._auto_timer.isActive():
                    return
            assert time.monotonic() - start < TIMEOUT_S, "the window didn't settle"
            time.sleep(0.003)

    def mode(self, mode: str) -> None:
        self.window.preferences.set_value(preferences.AUTO_KEY, mode)


@pytest.fixture
def win(monkeypatch):
    w = _Auto(monkeypatch)
    yield w
    w.window.close()
    w.window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_off_is_the_default_and_a_change_waits_for_generate(win):
    assert preferences.Preferences().auto_generate() == preferences.OFF
    win.preview()
    win.colors(-3)
    win.settle(AFTER_S + 0.3)
    assert win.made == 1 and win.state() == freshness.MODIFIED
    assert "Generate Preview (F5) makes it again." in win.window.freshness.toolTip()


def test_always_makes_one_page_a_second_after_a_burst_of_changes(win):
    win.mode(preferences.ALWAYS)
    win.preview()
    colors = win.window.controls.controls["num_colors"]
    for i in range(50):  # 50 ticks of a drag, well inside the delay
        colors.step(-1 if i % 2 == 0 else 1) if i < 49 else colors.step(-1)
        _app.processEvents()
    assert win.made == 1 and win.state() == freshness.MODIFIED
    assert "made again in a moment" in win.window.freshness.toolTip()
    assert "made again in a moment" in win.window.status.text()
    win.settle(AFTER_S + 0.2)
    assert win.made == 2 and win.state() == freshness.CURRENT and win.window.document.result_is_current()


def test_when_it_s_quick_follows_the_picture_s_last_page(win, monkeypatch):
    win.mode(preferences.QUICK)
    win.colors(-1)
    win.settle(AFTER_S + 0.2)
    assert win.made == 0  # no page yet: nothing to make by itself, nor a time to judge it by
    monkeypatch.setattr(preferences, "QUICK_S", 1000.0)  # every page quick
    win.preview()
    win.colors(-3)
    win.settle(AFTER_S + 0.2)
    assert win.made == 2 and win.state() == freshness.CURRENT
    monkeypatch.setattr(preferences, "QUICK_S", 0.0)  # every page slow
    win.colors(-3)
    win.settle(AFTER_S + 0.3)
    assert win.made == 2 and win.state() == freshness.MODIFIED
    tip = win.window.freshness.toolTip()
    assert "too long to make by itself" in tip and f"{win.window._last_full_s:.1f} s" in tip
    assert "too long to make by itself" in win.window.status.text()


def test_a_redraw_s_time_doesn_t_count_as_a_page_made_afresh(win, monkeypatch):
    win.mode(preferences.QUICK)
    win.preview()
    full = win.window._last_full_s
    win.window.controls.controls["line_width_mm"].step(1)  # a redraw, by itself
    win.settle(0.6)
    assert win.made == 2 and win.window._last_full_s == full


def test_a_new_picture_forgets_the_last_page_s_time(win):
    win.preview()
    assert win.window._last_full_s is not None
    win.window.current_image_bgr = win.window.current_image_bgr.copy()
    assert win.window._last_full_s is None


def test_putting_the_change_back_calls_it_off(win):
    win.mode(preferences.ALWAYS)
    win.preview()
    colors = win.window.controls.controls["num_colors"]
    colors.step(-2)
    _app.processEvents()
    assert win.window._auto_timer.isActive()
    colors.step(2)
    _app.processEvents()
    assert not win.window._auto_timer.isActive()
    win.settle(AFTER_S + 0.2)
    assert win.made == 1 and win.state() == freshness.CURRENT


def test_it_supersedes_a_page_being_made_and_the_last_page_is_current(win):
    win.mode(preferences.ALWAYS)
    win.preview()
    win.hold = threading.Event()
    win.started.clear()
    win.colors(-2)
    win.window.generate_preview()
    assert win.started.wait(TIMEOUT_S)
    _app.processEvents()
    win.colors(-2)
    deadline = time.monotonic() + TIMEOUT_S
    while win.window._auto_timer.isActive():  # due: it cancels the page being made and starts again
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.003)
    _app.processEvents()
    job = win.window.jobs.current(PREVIEW)
    assert job is not None and job.settings == win.window.controls.snapshot()
    win.hold.set()
    win.settle()
    assert win.state() == freshness.CURRENT and win.window.document.result_is_current()


def test_it_waits_for_an_export_and_goes_ahead_after(win, monkeypatch):
    win.mode(preferences.ALWAYS)
    win.preview()
    monkeypatch.setattr(win.window, "_pending_export_path", Path("page.png"))  # an export waiting for its page
    win.colors(-3)
    win.settle(AFTER_S + 0.3)
    assert win.made == 1
    monkeypatch.setattr(win.window, "_pending_export_path", None)
    win.window._on_job_finished(Job(SAVE, lambda *_: None))  # the export written
    assert win.window._auto_timer.isActive()
    win.settle(AFTER_S + 0.2)
    assert win.made == 2 and win.state() == freshness.CURRENT


@pytest.mark.parametrize("mode", [preferences.OFF, preferences.QUICK, preferences.ALWAYS])
def test_a_page_kept_is_shown_at_once_under_every_setting(win, mode):
    win.mode(mode)
    win.preview()
    first = win.window.current_page
    win.colors(-3)
    win.preview()
    made = win.made
    win.colors(3)
    assert win.window.current_page is first and _nothing_made(win.window)
    assert not win.window._auto_timer.isActive()
    win.settle(AFTER_S + 0.2)
    assert win.made == made and win.state() == freshness.CURRENT


def test_the_preference_is_kept_and_applies_at_once(win, tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    prefs = preferences.Preferences(store)
    prefs.set_value(preferences.AUTO_KEY, preferences.QUICK)
    assert preferences.Preferences(store).auto_generate() == preferences.QUICK
    win.preview()
    win.colors(-3)
    win.settle(0.2)
    assert win.made == 1
    win.mode(preferences.ALWAYS)  # turned on while out of date: made, a second later
    win.settle(AFTER_S + 0.2)
    assert win.made == 2 and win.state() == freshness.CURRENT


def _nothing_made(window) -> bool:
    """No page being made or laid out: the inspectors' measuring of a page shown is not making one."""
    return not window.jobs.is_running(PREVIEW) and not window.jobs.is_running(PRINT_LAYOUT)

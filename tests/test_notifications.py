"""W2: the status bar, the notifications over the page, and the diagnostics panel. What the app tells is never a
modal dialog in the way, never takes the keyboard focus, and the diagnostics never open by themselves."""

import logging
import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import jobs  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import main_window, notifications, status  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.notifications import ERROR, INFO, SUCCESS, WARNING, DiagnosticsPanel, NotificationBar  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    _app.processEvents()
    yield window
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


def _until(condition, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)


def _blocks() -> np.ndarray:
    picture = np.zeros((200, 300, 3), np.uint8)
    picture[:, 150:] = (40, 170, 40)
    picture[100:, :150] = (40, 40, 200)
    return picture


# -- The notification bar -------------------------------------------------------------
def test_a_success_goes_by_itself_and_an_error_stays(monkeypatch):
    monkeypatch.setattr(notifications, "SUCCESS_MS", 50)
    bar = NotificationBar()
    bar.notify(SUCCESS, "Exported page.png.")
    assert not bar.isHidden() and bar.level == SUCCESS
    _until(bar.isHidden)

    bar.notify(ERROR, "Export failed: no room")
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert not bar.isHidden() and "no room" in bar.text.text()
    bar.close_button.click()
    assert bar.isHidden()


def test_a_notification_replaces_the_last_and_its_actions_run_and_close_it():
    bar = NotificationBar()
    ran = []
    bar.notify(WARNING, "First", [("Again", lambda: ran.append("again"))])
    bar.notify(INFO, "Second", [("Open Image…", lambda: ran.append("open")), ("Details", lambda: ran.append("d"))])
    _app.processEvents()
    assert bar.text.text() == "Second" and bar.action_labels() == ["Open Image…", "Details"]
    assert bar.accessibleName() == "Info: Second"
    bar.trigger("Open Image…")
    assert ran == ["open"] and bar.isHidden()
    with pytest.raises(ValueError):
        bar.notify("loud", "no such level")


def test_a_notification_never_takes_the_keyboard_focus_from_a_text_field(window):
    window.show_activity("Document")
    field = window.controls.controls["line_width_mm"].entry  # a text field: typing in it goes on as the banner shows
    window.activateWindow()
    field.setFocus()
    _app.processEvents()
    had = QApplication.focusWidget()
    assert had is not None and (had is field or field.isAncestorOf(had))
    for level in (SUCCESS, ERROR):
        window.notify(level, "Something happened", [("Details", window.show_last_error)])
        _app.processEvents()
        assert QApplication.focusWidget() is had
    assert not window.notification.isHidden() and window.diagnostics.isHidden()


# -- The window's feedback -------------------------------------------------------------------
def test_an_export_tells_what_it_wrote_and_can_show_it(window, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(main_window, "show_in_folder", shown.append)
    monkeypatch.setattr(
        main_window.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "page.png"), ""))
    )
    window.controls.set_export_dpi(150)
    window.current_image_bgr = _blocks()
    window.generate_preview()
    assert window.status.state() == status.BUSY and not window.status.progress.isHidden()
    _wait(window)
    assert window.status.state() == status.DONE and window.status.progress.isHidden()
    page = window.current_page
    assert f"{page.num_colors_used} colors" in window.status.page_label.text()
    assert "mm on A4" in window.status.size_label.text() and window.status.zoom_label.text().endswith("%")

    window.export_page()
    _wait(window)
    assert window.notification.level == SUCCESS and window.notification.text.text().startswith("Exported page.png: ")
    window.notification.trigger("Show in Folder")
    assert shown == [tmp_path / "page.png"]
    assert window.diagnostics.isHidden()  # what went right is told, never put in the way
    # The log holds each job, how long it took and how it ended.
    texts = [entry.text for entry in window.diagnostics.entries]
    assert any(text.startswith("Preview at") and ": done in " in text for text in texts)
    assert any(text.startswith("PNG page.png") and ": done in " in text for text in texts)


def test_a_failure_is_told_with_its_details_a_click_away(window, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("no luck")

    monkeypatch.setattr(pipeline, "generate", fail)
    window.current_image_bgr = _blocks()
    window.generate_preview()
    _wait(window)
    assert window.status.state() == status.FAILED
    assert window.notification.level == ERROR and window.notification.text.text() == "The page couldn't be made: no luck"
    # The diagnostics count it, but don't open by themselves.
    assert window.diagnostics.isHidden() and window.diagnostics.counts() == (0, 1)
    assert window.status.diagnostics_button.text() == "1"
    window.notification.trigger("Details")
    assert not window.diagnostics.isHidden() and window.commands["toggle_diagnostics"].isChecked()
    details = window.diagnostics.details.toPlainText()
    assert "no luck" in details and "Traceback" in details and "RuntimeError" in details

    window.diagnostics.copy()
    assert "no luck" in QGuiApplication.clipboard().text()
    window.diagnostics.clear()
    assert window.diagnostics.counts() == (0, 0) and window.status.diagnostics_button.text() == ""


def test_an_open_that_fails_is_told_not_shown_in_a_dialog(window, tmp_path, monkeypatch):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not a picture")
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    window.open_image()
    assert window.notification.level == ERROR and "broken.png" in window.notification.text.text()
    assert window.diagnostics.counts() == (0, 1) and window.status.state() == status.FAILED


def test_the_diagnostics_toggle_and_go_with_focus_mode(window):
    assert window.diagnostics.isHidden()
    window.commands["toggle_diagnostics"].trigger()
    assert not window.diagnostics.isHidden()
    window.commands["focus_mode"].trigger()
    assert window.diagnostics.isHidden()
    window.commands["focus_mode"].trigger()
    assert not window.diagnostics.isHidden()
    window.diagnostics.close_button.click()
    assert window.diagnostics.isHidden() and not window.commands["toggle_diagnostics"].isChecked()
    window.status.diagnostics_button.click()
    assert not window.diagnostics.isHidden()


# -- The log ---------------------------------------------------------------------------------
def test_the_app_s_warnings_reach_the_log_from_any_thread_and_jobs_are_logged_once(window):
    log = logging.getLogger("tessellatum.core.somewhere")
    worker = threading.Thread(target=lambda: log.warning("a thread's warning"))
    worker.start()
    worker.join()
    _until(lambda: any(e.text == "a thread's warning" for e in window.diagnostics.entries))
    assert window.diagnostics.counts() == (1, 0)
    # A job's failure is logged by the job manager too: the log has it once, as the job's own entry.
    logging.getLogger(jobs.__name__).warning("Preview failed: x")
    _app.processEvents()
    assert not any(e.text == "Preview failed: x" for e in window.diagnostics.entries)


def test_a_closed_window_stops_listening_to_the_log(window):
    handler = window._log_handler
    assert handler in logging.getLogger("tessellatum").handlers
    window.close()
    assert handler not in logging.getLogger("tessellatum").handlers


def test_the_log_keeps_its_newest_entries():
    panel = DiagnosticsPanel()
    panel.MAX_ENTRIES = 3
    for i in range(5):
        panel.add(WARNING, f"w{i}")
    assert [e.text for e in panel.entries] == ["w2", "w3", "w4"] and panel.list.count() == 3
    assert panel.counts() == (3, 0)


def test_the_status_bar_says_what_it_does():
    bar = status.StatusBar()
    bar.set_state("Generating preview…", status.BUSY)
    assert bar.progress.maximum() == 0  # it can't say yet: indeterminate
    bar.set_progress(40)
    assert bar.progress.maximum() == 100 and bar.progress.value() == 40
    bar.set_state("Cancelled.", status.CANCELLED)
    assert bar.progress.isHidden() and bar.text() == "Cancelled."
    bar.set_counts(2, 0)
    assert bar.diagnostics_button.text() == "2" and "2 warnings" in bar.diagnostics_button.toolTip()
    with pytest.raises(ValueError):
        bar.set_state("?", "sleeping")


def test_no_modal_dialog_tells_what_happened():
    """Message boxes are kept for About and for a decision (replacing files): what happened is told in the
    notifications."""
    source = open(main_window.__file__, encoding="utf-8").read()
    assert "QMessageBox.about(" in source
    for telling in ("information(", "warning(", "critical("):
        assert f"QMessageBox.{telling}" not in source


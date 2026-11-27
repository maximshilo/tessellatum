"""G4: what a page being made is doing (its stages, by name), cancelling said at once with what it waits for, the time
to stop and each stage's time in the diagnostics, and failures said with what to try."""

import errno
import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import errors, jobs  # noqa: E402
from tessellatum.app.jobs import PREVIEW, SAVE, JobManager, page_task  # noqa: E402
from tessellatum.core import export, pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.pipeline import PipelineCancelled, StageEvent  # noqa: E402
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui.notifications import ERROR, stage_times  # noqa: E402

from difficulty_levels import MIDDLE  # noqa: E402
from test_main_window import _blocks, _Drive  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120


def _until(condition, timeout=TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while not condition():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.005)


# -- Failures said with what to try ---------------------------------------------------------------------------------
def test_running_out_of_memory_on_a2_suggests_a3():
    text = errors.explain(errors.MAKING, MemoryError, "", paper="A2")
    assert "enough memory" in text and "A3" in text and "easier difficulty" in text


def test_running_out_of_memory_on_a4_suggests_no_smaller_paper():
    text = errors.explain(errors.MAKING, MemoryError, "", paper="A4")
    assert "smaller paper" not in text and "easier difficulty" in text


def test_numpy_and_opencv_out_of_memory_are_memory_errors():
    numpy_error = None
    try:
        np.empty(2**62, dtype=np.uint8)
    except MemoryError as exc:
        numpy_error = type(exc)
    assert numpy_error is not None and errors.is_memory_error(numpy_error, "")
    assert errors.is_memory_error(RuntimeError, "OpenCV(4.10) error: (-4:Insufficient memory) Failed to allocate")
    assert not errors.is_memory_error(ValueError, "no luck")


def test_a_png_too_large_for_memory_suggests_a_lower_resolution_and_a_smaller_paper():
    text = errors.explain(errors.WRITING, MemoryError, "", paper="A2", file_format="PNG", file_name="page.png")
    assert "lower resolution" in text and "A3" in text


@pytest.mark.parametrize("error_type, message, words", [
    (PermissionError, "[Errno 13] Permission denied: 'page.png'", "open in another app"),
    (OSError, f"[Errno {errno.ENOSPC}] No space left on device", "disk is full"),
    (OSError, "[WinError 112] There is not enough space on the disk", "disk is full"),
    (FileNotFoundError, "[Errno 2] No such file or directory: 'x/page.png'", "folder no longer exists"),
])
def test_a_file_that_can_t_be_written_says_why_and_what_to_do(error_type, message, words):
    text = errors.explain(errors.WRITING, error_type, message, file_format="PNG", file_name="page.png")
    assert text.startswith("Couldn't write page.png") and words in text


def test_any_other_failure_says_its_message():
    assert errors.explain(errors.MAKING, ValueError, "no luck") == "The page couldn't be made: no luck"


# -- The jobs pass the stages on -------------------------------------------------------------------------------------
def test_a_job_s_stages_are_passed_on_in_order_and_kept_with_their_times():
    manager = JobManager()
    seen = []
    manager.stage.connect(lambda job, event: seen.append((event.stage, event.done)))

    def task(progress, _should_cancel):
        for stage in ("resize", "quantize"):
            progress.stage(StageEvent(stage, pipeline.STAGES[stage]))
            progress(50)
            progress.stage(StageEvent(stage, pipeline.STAGES[stage], done=True, seconds=0.25, cached=stage == "resize"))
        return "page"

    job = manager.start(PREVIEW, task)
    _until(manager.is_idle)
    assert seen == [("resize", False), ("resize", True), ("quantize", False), ("quantize", True)]
    assert [(e.stage, e.seconds, e.cached) for e in job.stages] == [("resize", 0.25, True), ("quantize", 0.25, False)]
    assert stage_times(job.stages) == "Sizing the picture: 0.25 s (kept from before)\nChoosing the colors: 0.25 s"


def test_a_plain_callable_task_still_works_and_a_cancelled_job_says_how_long_it_took_to_stop():
    manager = JobManager()
    seen = []
    manager.stage.connect(lambda job, event: seen.append(event.stage))
    asked = threading.Event()

    def task(progress, should_cancel):
        progress(10)  # a plain call, as tasks that know nothing of stages make
        progress.stage(StageEvent("regions", pipeline.STAGES["regions"]))
        asked.wait(TIMEOUT_S)
        time.sleep(0.05)
        progress.stage(StageEvent("regions", pipeline.STAGES["regions"], done=True, seconds=0.1))
        if should_cancel():
            raise PipelineCancelled()
        return None

    job = manager.start(PREVIEW, task)
    _until(lambda: seen == ["regions"])
    manager.cancel(PREVIEW)
    asked.set()
    _until(manager.is_idle)
    assert job.state == jobs.CANCELLED
    assert seen == ["regions"]  # nothing passed on once it was asked to stop
    assert len(job.stages) == 1  # though kept, for the log
    assert job.seconds_to_stop() is not None and job.seconds_to_stop() >= 0.03  # the clock ticks in ~15 ms on Windows


def test_a_preview_task_names_every_stage_in_order_and_paints_its_versions_last(sample_image_bgr):
    events = []

    class Progress:
        def __call__(self, percent):
            pass

        def stage(self, event):
            events.append(event)

    pipeline.clear_cache()
    page_task(sample_image_bgr, MIDDLE, 300, PageStyle(), versions=tuple(Version))(Progress(), lambda: False)
    starts = [e.stage for e in events if not e.done]
    assert starts == ["resize", "ink", "detect", "quantize", "text", "regions", "contours", "render", jobs.VERSIONS]
    assert events[-1].done and events[-1].label == "Painting the versions"


# -- The window -------------------------------------------------------------------------------------------------------
@pytest.fixture
def drive(monkeypatch):
    drive = _Drive(monkeypatch)
    yield drive
    drive.window.close()
    drive.window.deleteLater()
    _app.processEvents()


def test_the_status_bar_says_each_stage_as_the_page_is_made(drive):
    window = drive.window
    said = []
    window.jobs.stage.connect(lambda job, event: not event.done and said.append(window.status.text()))
    window.current_image_bgr = _blocks((300, 200))
    pipeline.clear_cache()
    window.generate_preview()
    drive.wait()
    labels = [text.split("…")[0] for text in said]
    assert labels[:3] == ["Sizing the picture", "Looking for line art", "Finding faces and the subject"]
    assert "Building the regions" in labels and labels[-1] == "Painting the versions"
    assert all(text.endswith("%") for text in said)
    assert window.status.text() == "Preview ready."


def test_cancelling_is_said_at_once_with_the_step_it_waits_for(drive, monkeypatch):
    window = drive.window
    release, reached = threading.Event(), threading.Event()
    real = pipeline.generate

    def held_in_quantize(*args, stage_callback=None, should_cancel=None, **kwargs):
        stage_callback(StageEvent("quantize", pipeline.STAGES["quantize"]))
        reached.set()
        while not release.wait(0.005):
            if should_cancel():
                time.sleep(0.05)  # the stage runs to its end
                raise PipelineCancelled()
        return real(*args, stage_callback=stage_callback, should_cancel=should_cancel, **kwargs)

    monkeypatch.setattr(pipeline, "generate", held_in_quantize)
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    assert reached.wait(TIMEOUT_S)
    _until(lambda: window.status.text().startswith("Choosing the colors"))
    window.abort_generation()
    # In the same turn of the event loop: nothing processed in between.
    assert window.status.text() == "Cancelling: it stops after the current step (choosing the colors)…"
    drive.wait()
    assert window.status.text() == "Cancelled."
    entry = [e for e in window.diagnostics.entries if e.text.startswith("Preview at")][-1]
    assert ": cancelled in " in entry.text and "s after it was asked to stop" in entry.text


def test_the_diagnostics_give_each_stage_s_time_and_say_which_were_kept(drive):
    window = drive.window
    window.current_image_bgr = _blocks((300, 200))
    pipeline.clear_cache()
    window.generate_preview()
    drive.wait()
    first = [e for e in window.diagnostics.entries if e.text.startswith("Preview at")][-1]
    assert "Building the regions: " in first.details and "kept from before" not in first.text
    window.controls.controls["line_width_mm"].step(2)
    _until(lambda: len([e for e in window.diagnostics.entries if e.text.startswith("Redraw at")]) == 1)
    redraw = [e for e in window.diagnostics.entries if e.text.startswith("Redraw at")][-1]
    assert "stages kept from before)" in redraw.text
    assert "Building the regions: " in redraw.details and "(kept from before)" in redraw.details


def test_a_page_too_large_for_memory_says_to_try_a_smaller_paper(drive, monkeypatch):
    def out_of_memory(*_args, **_kwargs):
        raise MemoryError()

    monkeypatch.setattr(pipeline, "generate", out_of_memory)
    window = drive.window
    window.controls.paper_combo.setCurrentText("A2")
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    drive.wait()
    level, text = drive.dialogs[-1]
    assert level == ERROR and "enough memory" in text and "A3" in text


def test_a_png_is_written_with_a_busy_bar_that_doesn_t_pretend_to_measure_and_a_write_failure_says_why(
    drive, monkeypatch, tmp_path
):
    window = drive.window
    release, writing = threading.Event(), threading.Event()

    def held_png(*_args, **_kwargs):
        writing.set()
        release.wait(TIMEOUT_S)
        raise PermissionError(errno.EACCES, "Permission denied", str(tmp_path / "page.png"))

    monkeypatch.setattr(export, "export_png", held_png)
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    drive.wait()
    drive.save_path = str(tmp_path / "page.png")
    window.export_page(previewed=True)
    _until(writing.is_set)
    _until(lambda: window.jobs.is_running(SAVE))
    assert window.status.progress.maximum() == 0 and window.controls.progress_bar.maximum() == 0  # indeterminate
    assert not window.controls.abort_button.isEnabled()
    release.set()
    drive.wait()
    level, text = drive.dialogs[-1]
    assert level == ERROR and text.startswith("Couldn't write page.png") and "open in another app" in text

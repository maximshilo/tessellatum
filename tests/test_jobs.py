"""The job manager (G1): identity, stale results, cancellation and shutdown, driven by tasks the tests hold and release.

Every wait is for a condition (a gate opened, an outcome delivered), with a deadline: none sleeps and hopes.
"""

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import jobs  # noqa: E402
from tessellatum.app.document import Document  # noqa: E402
from tessellatum.app.jobs import EXPORT, PREVIEW, RECIPES, JobManager  # noqa: E402
from tessellatum.core import difficulty, pipeline  # noqa: E402
from tessellatum.core.pipeline import PipelineCancelled  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 30


def until(condition, timeout: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        _app.processEvents()
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)
    _app.processEvents()


class Fake:
    """A task held at a gate until released, recording that it ran; it stops when cancelled if ``obeys``."""

    def __init__(self, result="page", obeys=True, fails: Exception | None = None):
        self.result, self.obeys, self.fails = result, obeys, fails
        self.gate = threading.Event()
        self.running = threading.Event()
        self.ran = False

    def __call__(self, progress, should_cancel):
        self.ran = True
        self.running.set()
        progress(10)
        while not self.gate.wait(0.002):
            if self.obeys and should_cancel():
                raise PipelineCancelled()
        if self.fails is not None:
            raise self.fails
        progress(100)
        return self.result


class Record:
    """Every signal a manager sent, in order."""

    def __init__(self, manager: JobManager):
        self.events = []
        manager.succeeded.connect(lambda job, result: self.events.append(("succeeded", job, result)))
        manager.failed.connect(lambda job, message: self.events.append(("failed", job, message)))
        manager.cancelled.connect(lambda job: self.events.append(("cancelled", job)))
        manager.dropped.connect(lambda job: self.events.append(("dropped", job)))
        manager.progress.connect(lambda job, percent: self.events.append(("progress", job, percent)))

    def outcomes(self):
        return [event[:2] if event[0] != "succeeded" else event for event in self.events if event[0] != "progress"]


@pytest.fixture
def document():
    doc = Document()
    doc.set_source(np.full((40, 60, 3), 100, np.uint8))
    yield doc
    doc.deleteLater()


@pytest.fixture
def manager(document):
    manager = JobManager(document)
    yield manager
    manager.shutdown()
    assert jobs.wait_for_all(TIMEOUT_S)
    manager.deleteLater()
    _app.processEvents()


def test_a_job_s_result_is_delivered_with_what_it_was_started_for(manager, document):
    record = Record(manager)
    fake = Fake()
    job = manager.start(PREVIEW, fake)
    assert job.revision == document.revision() and job.source_token == document.source().token
    assert job.settings is document.settings() and manager.is_running(PREVIEW)
    fake.running.wait(TIMEOUT_S)
    fake.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("succeeded", job, "page")] and job.state == jobs.SUCCEEDED
    assert ("progress", job, 10) in record.events and job.seconds() >= 0


def test_a_newer_job_of_a_kind_supersedes_the_running_one_and_only_its_result_is_delivered(manager):
    # A starts, B starts, A (which ignores cancellation) finishes, then B: A's page is dropped, never shown.
    record = Record(manager)
    a, b = Fake("A", obeys=False), Fake("B")
    job_a = manager.start(PREVIEW, a)
    a.running.wait(TIMEOUT_S)
    job_b = manager.start(PREVIEW, b)
    assert job_b.state == jobs.QUEUED and not b.ran  # waits for A to stop: never two side by side
    assert manager.current(PREVIEW) is job_b
    a.gate.set()
    until(lambda: b.running.is_set())
    b.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("dropped", job_a), ("succeeded", job_b, "B")]


def test_jobs_of_different_kinds_run_side_by_side_and_each_is_delivered(manager):
    record = Record(manager)
    preview, export = Fake("preview"), Fake("export")
    job_p = manager.start(PREVIEW, preview)
    job_e = manager.start(EXPORT, export)
    preview.running.wait(TIMEOUT_S) and export.running.wait(TIMEOUT_S)
    export.gate.set()
    until(lambda: job_e.state == jobs.SUCCEEDED)
    preview.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("succeeded", job_e, "export"), ("succeeded", job_p, "preview")]


def test_cancel_before_during_and_after_a_job(manager):
    record = Record(manager)
    # During: it stops, and is announced cancelled.
    running = Fake()
    job = manager.start(PREVIEW, running)
    running.running.wait(TIMEOUT_S)
    assert manager.cancel(PREVIEW)
    until(manager.is_idle)
    assert record.outcomes() == [("cancelled", job)]

    # Before it starts (queued behind one that ignores cancellation): it never runs.
    stubborn, queued = Fake(obeys=False), Fake()
    first = manager.start(EXPORT, stubborn)
    stubborn.running.wait(TIMEOUT_S)
    second = manager.start(EXPORT, queued)
    assert manager.cancel(EXPORT)
    assert second.state == jobs.CANCELLED and not queued.ran
    stubborn.gate.set()
    until(manager.is_idle)
    assert not queued.ran and first.state == jobs.DROPPED

    # After it finished: nothing to cancel, its result stands.
    done = Fake("kept")
    done.gate.set()
    job = manager.start(PREVIEW, done)
    until(manager.is_idle)
    assert not manager.cancel(PREVIEW) and job.state == jobs.SUCCEEDED


def test_a_result_that_finished_as_it_was_cancelled_is_not_used(manager):
    record = Record(manager)
    fake = Fake(obeys=False)
    job = manager.start(PREVIEW, fake)
    fake.running.wait(TIMEOUT_S)
    manager.cancel(PREVIEW)
    fake.gate.set()  # it finishes anyway
    until(manager.is_idle)
    assert record.outcomes() == [("cancelled", job)]


def test_a_job_that_can_t_be_cancelled_runs_to_its_end(manager):
    record = Record(manager)
    fake = Fake("file")
    job = manager.start(jobs.SAVE, fake, cancellable=False)
    fake.running.wait(TIMEOUT_S)
    assert not manager.cancel(jobs.SAVE)
    fake.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("succeeded", job, "file")]


def test_settings_changed_during_a_job_still_deliver_it_with_the_settings_it_was_made_with(manager, document):
    # Undo during a job (H1) is the same: the revision moves on, the page is of the same picture (D-011).
    record = Record(manager)
    fake = Fake("page")
    before = document.settings()
    job = manager.start(PREVIEW, fake)
    fake.running.wait(TIMEOUT_S)
    document.set_settings(before.replace(preset="Hard"))
    assert document.revision() != job.revision
    fake.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("succeeded", job, "page")] and job.settings is before
    document.set_result("page", job.settings)
    assert not document.result_is_current()  # shown as made with the settings before the edit
    document.set_settings(before)  # and undone
    assert document.result_is_current()


def test_a_result_of_a_picture_replaced_during_the_job_is_dropped(manager, document):
    record = Record(manager)
    fake = Fake()
    job = manager.start(PREVIEW, fake)
    fake.running.wait(TIMEOUT_S)
    document.set_source(np.zeros((10, 10, 3), np.uint8))
    fake.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("dropped", job)]


def test_a_job_s_own_check_can_refuse_its_result(manager):
    record = Record(manager)
    wanted = {"yes": True}
    fake = Fake()
    fake.gate.set()
    job = manager.start(jobs.PRINT_LAYOUT, fake, accept=lambda job: wanted["yes"])
    wanted["yes"] = False  # the preview it was drawn from was replaced
    until(manager.is_idle)
    assert record.outcomes() == [("dropped", job)]


def test_a_failure_becomes_a_failed_state_with_its_details(manager):
    record = Record(manager)
    fake = Fake(fails=ValueError("no luck"))
    fake.gate.set()
    job = manager.start(PREVIEW, fake)
    until(manager.is_idle)
    assert record.outcomes() == [("failed", job)] and record.events[-1][2] == "no luck"
    assert job.state == jobs.FAILED and job.error == "no luck" and "ValueError: no luck" in job.details


def test_fifty_rapid_starts_run_the_first_and_the_last_only(manager):
    record = Record(manager)
    fakes = [Fake(f"page {i}") for i in range(50)]
    for fake in fakes:
        fake.gate.set()
    fakes[0].gate.clear()
    started = [manager.start(PREVIEW, fake) for fake in fakes]
    fakes[0].running.wait(TIMEOUT_S)
    until(manager.is_idle)
    assert [fake.ran for fake in fakes] == [True] + [False] * 48 + [True]
    delivered = [event for event in record.events if event[0] == "succeeded"]
    assert delivered == [("succeeded", started[-1], "page 49")]
    assert {job.state for job in started[1:-1]} == {jobs.DROPPED}


def test_closing_during_a_job_returns_at_once_and_delivers_nothing(document):
    manager = JobManager(document)
    record = Record(manager)
    fake = Fake(obeys=False)  # as a PNG being drawn: it can't be stopped
    manager.start(jobs.SAVE, fake, cancellable=False)
    fake.running.wait(TIMEOUT_S)
    began = time.monotonic()
    manager.shutdown()
    manager.deleteLater()  # the window goes, its thread running on
    _app.processEvents()
    assert time.monotonic() - began < 1.0
    assert jobs.live_threads() == 1
    fake.gate.set()
    assert jobs.wait_for_all(TIMEOUT_S)
    until(lambda: jobs.live_threads() == 0)
    assert record.events == []


def test_wait_for_all_stops_the_jobs_that_can_be(document):
    manager = JobManager(document)
    fake = Fake()
    manager.start(RECIPES, fake)
    fake.running.wait(TIMEOUT_S)
    assert jobs.wait_for_all(TIMEOUT_S) and jobs.live_threads() == 0
    manager.deleteLater()
    _app.processEvents()


def test_busy_follows_the_jobs(manager):
    busy = []
    manager.busy_changed.connect(busy.append)
    fake = Fake()
    fake.gate.set()
    manager.start(PREVIEW, fake)
    until(manager.is_idle)
    assert busy == [True, False]


def test_the_real_pipeline_runs_as_a_job(manager, document):
    record = Record(manager)
    picture = np.zeros((120, 160, 3), np.uint8)
    picture[:, 80:] = (40, 170, 40)
    document.set_source(picture)
    params = difficulty.params_for_preset("Beginner")
    job = manager.start(PREVIEW, lambda progress, should_cancel: pipeline.generate(
        picture, params, 400, progress_callback=progress, should_cancel=should_cancel,
    ))
    until(manager.is_idle, timeout=120)
    (outcome, delivered, page), = record.outcomes()
    assert outcome == "succeeded" and delivered is job and page.page.size == (400, 300)
    assert [event[2] for event in record.events if event[0] == "progress"][-1] == 100


def test_a_dropped_job_is_not_announced_cancelled_and_its_result_is_not_delivered(manager):
    record = Record(manager)
    fake = Fake(obeys=False)
    job = manager.start(jobs.PRINT_LAYOUT, fake)
    fake.running.wait(TIMEOUT_S)
    assert manager.drop(jobs.PRINT_LAYOUT) and not manager.is_running(jobs.PRINT_LAYOUT)
    replacement = Fake("new")
    second = manager.start(jobs.PRINT_LAYOUT, replacement)
    assert second.state == jobs.QUEUED  # still after the dropped one's thread
    fake.gate.set()
    replacement.gate.set()
    until(manager.is_idle)
    assert record.outcomes() == [("dropped", job), ("succeeded", second, "new")]

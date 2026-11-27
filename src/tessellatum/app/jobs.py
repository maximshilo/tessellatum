"""Background jobs: each one an identity, a kind, what it was started for, and a check on delivery.

Every slow step the app takes off the GUI thread -- making a preview, laying it out to print, making a page to
export, writing the file, mixing the legend's recipes -- is one ``Job``, run by the window's ``JobManager``:

- A job records the document's revision, its picture (the source's token) and its settings snapshot when it starts.
- Its task runs on a thread of its own and never touches a widget: it gets a progress callback and a cancellation
  flag, and returns its result or raises. The progress callback also takes the pipeline's stages as they start and
  end (``progress.stage(event)``, see ``pipeline.StageEvent``): the job keeps the ones done, with their times, and
  the manager passes them on, so the window can say what is being done and the diagnostics how long each part took.
- Its outcome comes back to the GUI thread through a queued signal and is checked there. A result is delivered only
  if its job is still the newest of its kind, wasn't cancelled, and is of the picture the document still holds, and
  if the job's own ``accept`` check (if any) agrees; otherwise it is dropped. A result made with settings that have
  changed since is delivered, with the settings it was made with (D-011).
- One job of each kind runs at a time: a new one cancels the last and waits for it to stop, so two jobs never compute
  the same stage side by side.
- Closing cancels everything at once without waiting; the threads outlive the window, and ``wait_for_all`` bounds
  the wait when the application quits.

QtCore only, so it is tested headless (``tests/test_jobs.py``).
"""

from __future__ import annotations

import itertools
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal

from tessellatum.core import pipeline
from tessellatum.core.pipeline import PipelineCancelled, StageEvent

log = logging.getLogger(__name__)

PREVIEW = "preview"  # the page made of the picture at its working size, with its versions
PRINT_LAYOUT = "print layout"  # the preview's page laid out to print, for the zoom and the export
EXPORT = "export"  # a page made afresh to export, when the preview isn't what is exported
SAVE = "save"  # writing a file: the PDF, the PNG drawn in pixels, a palette
RECIPES = "recipes"  # how to mix each legend color from the paints listed
INSPECTION = "inspection"  # measuring the page shown for the inspectors: how much of it each color paints
ANALYSIS = "analysis"  # measuring the page shown as the benchmark does, for the Analysis page
KINDS = (PREVIEW, PRINT_LAYOUT, EXPORT, SAVE, RECIPES, INSPECTION, ANALYSIS)

# A job's states. QUEUED waits for its kind's last job to stop; DROPPED finished, but was no longer wanted.
QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, DROPPED = (
    "queued", "running", "succeeded", "failed", "cancelled", "dropped"
)
FINISHED = (SUCCEEDED, FAILED, CANCELLED, DROPPED)

Task = Callable[[Callable[[int], None], Callable[[], bool]], Any]

# The last stage of a preview, after the pipeline's: the page painted in, for the canvas's other versions.
VERSIONS, VERSIONS_LABEL = "versions", "Painting the versions"

_ids = itertools.count(1)


@dataclass(eq=False)
class Job:
    """One run of a background task, and what it was started for."""

    kind: str
    task: Task  # task(progress, should_cancel) -> result; raises PipelineCancelled when it stops early
    revision: int | None = None  # the document's revision when it started
    source_token: int | None = None  # the picture it is of (``Source.token``); None if of none
    settings: Any = None  # the document's settings snapshot it was started with
    cancellable: bool = True
    accept: Callable[[Job], bool] | None = None  # checked on delivery, on the GUI thread
    label: str = ""  # what it does, for the diagnostics log
    context: Any = None  # whatever the caller needs back with its outcome
    id: int = field(default_factory=lambda: next(_ids))
    state: str = QUEUED
    error: str = ""  # a failure's message
    details: str = ""  # a failure's traceback
    error_type: type | None = None  # a failure's exception class
    stages: list = field(default_factory=list)  # the pipeline's stages it went through, done (``StageEvent``)
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    cancelled_at: float | None = None  # when it was asked to stop
    _cancel_requested: bool = False

    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def seconds(self) -> float | None:
        """How long it ran, once finished."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def seconds_to_stop(self) -> float | None:
        """How long it took to stop once asked to, if it was asked to and has finished."""
        if self.cancelled_at is None or self.finished_at is None or self.started_at is None:
            return None
        return max(0.0, self.finished_at - self.cancelled_at)

    def __repr__(self) -> str:
        return f"Job({self.id}, {self.kind!r}, {self.state})"


class Progress:
    """What a task reports how far along it is through: called with a percentage (0-100), and ``stage(event)`` with
    a stage of the pipeline starting or done (``pipeline.StageEvent``). A task that knows nothing of stages calls it
    as a plain function."""

    def __init__(self, percent: Callable[[int], None], stage: Callable[[StageEvent], None]):
        self._percent = percent
        self.stage = stage

    def __call__(self, percent: int) -> None:
        self._percent(percent)


class _Runner(QThread):
    """The thread a job runs on. Not parented to anything, and kept in ``_live`` until it ends, so that a window
    deleted while it runs never destroys a running thread."""

    progressed = Signal(int)
    staged = Signal(object)  # a StageEvent
    done = Signal(str, object)  # SUCCEEDED / FAILED / CANCELLED, and the result or (message, traceback, class)

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def run(self) -> None:
        job = self.job
        try:
            result = job.task(Progress(self.progressed.emit, self.staged.emit), self.isInterruptionRequested)
        except PipelineCancelled:
            self.done.emit(CANCELLED, None)
            return
        # Every failure becomes the job's failed state, a MemoryError too: it is said in words (see ``app.errors``).
        except Exception as exc:  # noqa: BLE001
            self.done.emit(FAILED, (str(exc) or type(exc).__name__, traceback.format_exc(), type(exc)))
            return
        self.done.emit(SUCCEEDED, result)


_live: set[_Runner] = set()


def _forget(runner: _Runner) -> None:
    _live.discard(runner)
    runner.deleteLater()


def live_threads() -> int:
    """How many job threads are still running, in every manager."""
    return sum(1 for runner in _live if runner.isRunning())


def wait_for_all(timeout_s: float = 10.0) -> bool:
    """Ask every running job to stop and wait for them, at most ``timeout_s`` in all: when the application quits, so
    that no thread is destroyed running. True if they all stopped."""
    deadline = time.monotonic() + timeout_s
    for runner in list(_live):
        runner.requestInterruption()
    for runner in list(_live):
        left = max(0, int((deadline - time.monotonic()) * 1000))
        if runner.isRunning() and not runner.wait(left):
            log.warning("a %s job was still running at exit", runner.job.kind)
            return False
    return True


class JobManager(QObject):
    """Runs the window's jobs, one of each kind at a time, and delivers only the results still wanted.

    ``document``, if given, is what a job's picture is checked against: a result of a picture since replaced is
    dropped. Every signal comes on the GUI thread.
    """

    started = Signal(object)  # Job
    progress = Signal(object, int)  # Job, 0-100: the current job of its kind only
    stage = Signal(object, object)  # Job, StageEvent: a stage starting or done, of the current job of its kind only
    succeeded = Signal(object, object)  # Job, its result
    failed = Signal(object, str)  # Job, its message (the traceback in ``job.details``)
    cancelled = Signal(object)  # Job: cancelled while it was the current one of its kind
    dropped = Signal(object)  # Job: finished, but superseded or no longer of the document's picture
    finished = Signal(object)  # Job: every job ends with this, whatever its outcome
    busy_changed = Signal(bool)

    def __init__(self, document=None, parent: QObject | None = None):
        super().__init__(parent)
        self._document = document
        self._current: dict[str, Job] = {}  # the newest job of each kind, queued or running
        self._runners: dict[int, _Runner] = {}  # by job id, while running
        self._queued: dict[str, Job] = {}  # the job waiting for its kind's last one to stop
        self._closed = False
        self._busy = False

    # -- Starting and stopping -------------------------------------------------
    def start(
        self,
        kind: str,
        task: Task,
        *,
        cancellable: bool = True,
        accept: Callable[[Job], bool] | None = None,
        label: str = "",
        context: Any = None,
    ) -> Job:
        """Run ``task`` as a job of ``kind``, stamped with the document's revision, picture and settings now. A job
        of the same kind still running is cancelled first; this one starts once it has stopped."""
        if kind not in KINDS:
            raise ValueError(f"no job kind {kind!r}")
        document = self._document
        source = document.source() if document is not None else None
        job = Job(
            kind, task,
            revision=document.revision() if document is not None else None,
            source_token=source.token if source is not None else None,
            settings=document.settings() if document is not None else None,
            cancellable=cancellable, accept=accept, label=label or kind, context=context,
        )
        if self._closed:
            job.state = DROPPED
            return job
        previous = self._current.get(kind)
        self._current[kind] = job
        if previous is not None:
            self._cancel(previous)
        if self._running_of(kind) is not None:
            replaced = self._queued.pop(kind, None)
            if replaced is not None:
                self._end(replaced, DROPPED)
            self._queued[kind] = job
        else:
            self._run(job)
        self._busy_maybe_changed()
        return job

    def cancel(self, kind: str | None = None) -> bool:
        """Cancel the current job of ``kind`` (every kind if None); True if one was asked to stop. A job that can't
        be cancelled runs to its end, and its result is dropped."""
        asked = False
        for job in list(self._current.values()):
            if kind is None or job.kind == kind:
                asked |= self._cancel(job)
        return asked

    def drop(self, kind: str) -> bool:
        """Stop wanting the current job of ``kind``: it is asked to stop, and whatever it comes to is dropped, not
        announced as cancelled. True if there was one."""
        job = self._current.pop(kind, None)
        if job is None:
            return False
        if self._queued.get(kind) is job:
            del self._queued[kind]
            self._end(job, DROPPED)
        else:
            runner = self._runners.get(job.id)
            if runner is not None and job.cancellable:
                runner.requestInterruption()
        self._busy_maybe_changed()
        return True

    def shutdown(self) -> None:
        """Cancel everything and deliver nothing more: the window is closing. Returns at once."""
        self._closed = True
        self.cancel()
        for job in list(self._queued.values()):
            self._end(job, DROPPED)
        self._queued.clear()
        self._current.clear()
        self._busy_maybe_changed()

    # -- What runs ---------------------------------------------------------------
    def current(self, kind: str) -> Job | None:
        """The newest job of ``kind`` not yet finished, queued or running."""
        return self._current.get(kind)

    def is_running(self, kind: str | None = None) -> bool:
        """Whether a job of ``kind`` (of any kind if None) is wanted and not finished."""
        return any(kind is None or job.kind == kind for job in self._current.values())

    def is_idle(self) -> bool:
        """No job wanted, none queued, and every thread's outcome delivered."""
        return not self._current and not self._queued and not self._runners

    # -- Inside -------------------------------------------------------------------
    def _running_of(self, kind: str) -> _Runner | None:
        for runner in self._runners.values():
            if runner.job.kind == kind:
                return runner
        return None

    def _run(self, job: Job) -> None:
        runner = _Runner(job)
        runner.progressed.connect(lambda percent, job=job: self._on_progress(job, percent))
        runner.staged.connect(lambda event, job=job: self._on_stage(job, event))
        runner.done.connect(lambda outcome, payload, runner=runner: self._on_done(runner, outcome, payload))
        runner.finished.connect(lambda runner=runner: _forget(runner))
        _live.add(runner)
        self._runners[job.id] = runner
        job.state = RUNNING
        job.started_at = time.monotonic()
        runner.start()
        self.started.emit(job)

    def _cancel(self, job: Job) -> bool:
        if job.state in FINISHED or job._cancel_requested:
            return False
        if job.state == RUNNING and not job.cancellable:
            return False  # it runs to its end; superseded, its result is dropped
        job._cancel_requested = True
        job.cancelled_at = time.monotonic()
        if job.state == QUEUED:
            if self._queued.get(job.kind) is job:
                del self._queued[job.kind]
            if self._current.get(job.kind) is job:
                del self._current[job.kind]
                self._end(job, CANCELLED, announce=True)
            else:
                self._end(job, DROPPED)
            self._busy_maybe_changed()
            return True
        runner = self._runners.get(job.id)
        if runner is not None:
            runner.requestInterruption()
        return True

    def _wanted(self, job: Job) -> bool:
        if self._closed or job._cancel_requested or self._current.get(job.kind) is not job:
            return False
        if self._document is not None:
            source = self._document.source()
            if (source.token if source is not None else None) != job.source_token:
                return False
        return job.accept is None or job.accept(job)

    def _on_progress(self, job: Job, percent: int) -> None:
        if not self._closed and self._current.get(job.kind) is job and not job._cancel_requested:
            self.progress.emit(job, percent)

    def _on_stage(self, job: Job, event: StageEvent) -> None:
        if event.done:
            job.stages.append(event)  # kept whatever becomes of the job, for the diagnostics
        if not self._closed and self._current.get(job.kind) is job and not job._cancel_requested:
            self.stage.emit(job, event)

    def _on_done(self, runner: _Runner, outcome: str, payload) -> None:
        job = runner.job
        self._runners.pop(job.id, None)
        current = self._current.get(job.kind) is job
        if current:
            del self._current[job.kind]
        if outcome == FAILED:
            job.error, job.details, job.error_type = payload
        if current and job._cancel_requested:
            self._end(job, CANCELLED, announce=True)  # asked to stop: whatever it came to isn't used
        elif current and outcome != CANCELLED and self._wanted_after_done(job):
            if outcome == SUCCEEDED:
                self._end(job, SUCCEEDED, result=payload, announce=True)
            else:
                log.warning("%s failed: %s\n%s", job.label, job.error, job.details)
                self._end(job, FAILED, announce=True)
        else:
            self._end(job, DROPPED)
        queued = self._queued.pop(job.kind, None)
        if queued is not None and not self._closed:
            self._run(queued)
        self._busy_maybe_changed()

    def _wanted_after_done(self, job: Job) -> bool:
        # _on_done already took it off _current: check the rest as _wanted would.
        self._current[job.kind] = job
        try:
            return self._wanted(job)
        finally:
            del self._current[job.kind]

    def _end(self, job: Job, state: str, result=None, announce: bool = False) -> None:
        job.state = state
        job.finished_at = time.monotonic()
        if self._closed:
            return
        if state == SUCCEEDED:
            self.succeeded.emit(job, result)
        elif state == FAILED:
            self.failed.emit(job, job.error)
        elif state == CANCELLED and announce:
            self.cancelled.emit(job)
        else:
            self.dropped.emit(job)
        self.finished.emit(job)

    def _busy_maybe_changed(self) -> None:
        busy = not self.is_idle() and not self._closed
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)


def page_task(
    image_bgr,
    params,
    long_edge: int,
    style=None,
    handling=None,
    versions=(),
    drawn_from=None,
    vector: bool = False,
) -> Task:
    """A task making a page of ``image_bgr`` at ``long_edge`` (see ``pipeline.generate``), and drawing ``versions`` of
    it too (see ``painting``), off the GUI thread, before it is handed over: ``GeneratedPage.image`` keeps them. With
    ``drawn_from``, a page of the same picture made with these settings at a smaller size or at ``long_edge``, the page
    is that one drawn at ``long_edge`` (see ``pipeline.draw_page``) rather than made again. With ``vector``, it is laid
    out to be drawn as vectors, a PDF."""
    versions = tuple(versions)

    def run(progress, should_cancel):
        stage = getattr(progress, "stage", None)  # a Progress tells the stages too; a plain callable doesn't
        if drawn_from is not None:
            page = pipeline.draw_page(
                drawn_from, image_bgr, long_edge, progress_callback=progress, should_cancel=should_cancel,
                vector=vector, stage_callback=stage,
            )
        else:
            kwargs = {k: v for k, v in (("style", style), ("handling", handling)) if v is not None}
            page = pipeline.generate(
                image_bgr, params, long_edge, progress_callback=progress, should_cancel=should_cancel,
                vector=vector, stage_callback=stage, **kwargs,
            )
        if versions and stage is not None:
            stage(StageEvent(VERSIONS, VERSIONS_LABEL))
        started = time.perf_counter()
        for version in versions:
            if should_cancel():
                raise PipelineCancelled()
            page.image(version)
        if versions and stage is not None:
            stage(StageEvent(VERSIONS, VERSIONS_LABEL, done=True, seconds=time.perf_counter() - started))
        return page

    return run

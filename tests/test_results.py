"""H2: the result cache. Pages made are kept by the picture's content, the settings' generation key and the pipeline's
version, within a memory budget, the page shown and the ones undo and redo go to protected; undo, redo and a setting
put back show a kept page at once, with no page made, and a state that had a page no longer kept is made again in the
background."""

import os
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum import __version__  # noqa: E402
from tessellatum.app import document, results  # noqa: E402
from tessellatum.app.jobs import PREVIEW, PRINT_LAYOUT  # noqa: E402
from tessellatum.app.results import Entry, ResultCache, ResultKey  # noqa: E402
from tessellatum.app.snapshot import CUSTOM, MY_PAINTS, DocumentSettings, ExportOptions  # noqa: E402
from tessellatum.core import pipeline, settings  # noqa: E402
from tessellatum.core.pipeline import PipelineCancelled  # noqa: E402
from tessellatum.gui import freshness, main_window, preferences  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120
MB = 1024 ** 2


# -- Keys --------------------------------------------------------------------------------------------------------------
def test_equal_settings_however_reached_have_one_key():
    medium = DocumentSettings()
    key = ResultKey.of("abc", medium)
    # Neither the legend's detail nor the export's options make the page again.
    assert ResultKey.of("abc", medium.replace(legend="Name, hex and RGB")) == key
    assert ResultKey.of("abc", medium.replace(export=ExportOptions(dpi=300))) == key
    # A difficulty's own values, as Custom or as a preset of one's own, are the same page as the difficulty.
    assert ResultKey.of("abc", medium.replace(preset=CUSTOM)) == key
    assert ResultKey.of("abc", medium.replace(preset="My own")) == key
    # The paints listed matter only when the colors come from them.
    assert ResultKey.of("abc", medium.replace(paints=(("Red", (200, 0, 0)),), mix=True)) == key
    assert hash(key) == hash(ResultKey.of("abc", DocumentSettings()))


def _other(setting) -> float:
    return setting.maximum if setting.default != setting.maximum else setting.minimum


def _changes():
    """Every change that makes another page: each slider moved, each choice switched."""
    base = DocumentSettings(preset=CUSTOM)  # every difficulty value is the page's
    for s in settings.SETTINGS:
        yield s.name, base, base.with_values(**{s.name: _other(s)})
    yield "preset", DocumentSettings(), DocumentSettings(preset="Hard")
    yield "paper", base, base.replace(paper="A3")
    yield "medium", base, base.replace(medium="Screen")
    for name in ("line_art", "detail", "text"):
        yield name, base, base.replace(**{name: not getattr(base, name)})
    paints = (("Red", (200, 20, 20)), ("Blue", (20, 20, 200)), ("White", (250, 250, 250)))
    painted = base.replace(paints=paints, colors_from=MY_PAINTS)
    yield "colors_from", base.replace(paints=paints), painted
    yield "paints", painted, painted.replace(paints=paints[:2])
    yield "mix", painted, painted.replace(mix=True)


@pytest.mark.parametrize("name, before, after", list(_changes()), ids=lambda v: v if isinstance(v, str) else "")
def test_every_change_that_makes_another_page_has_another_key(name, before, after):
    assert after.changed_entries(before), name  # the panel says it changes the page...
    assert ResultKey.of("abc", after) != ResultKey.of("abc", before)  # ...and the cache agrees


def test_another_picture_or_another_build_has_another_key(monkeypatch):
    medium = DocumentSettings()
    key = ResultKey.of("abc", medium)
    assert ResultKey.of("abd", medium) != key
    assert ResultKey.of(None, medium) is None  # the picture's hash not known yet: nothing is looked up
    assert key.version.startswith(__version__ + "+")
    monkeypatch.setattr(results, "_version", "9.9.9+other-models")
    assert ResultKey.of("abc", medium) != key


# -- Measuring, freezing ----------------------------------------------------------------------------------------------
@dataclass
class _Page:
    """Stands in for a page: arrays and images in dataclasses, lists and dicts."""

    pixels: np.ndarray
    image: Image.Image
    parts: list = field(default_factory=list)
    kept: dict = field(default_factory=dict)


def _page(mb: float = 1.0, seed: int = 0) -> _Page:
    n = int(mb * MB)
    shared = np.zeros(1000, np.uint8)
    return _Page(
        np.full(n - 1000 - 100 * 100 * 3, seed % 255, np.uint8), Image.new("RGB", (100, 100)),
        parts=[shared, (shared,)], kept={"again": shared},
    )


def test_a_page_is_measured_from_its_buffers_each_counted_once():
    page = _page(1.0)
    assert results.nbytes(page) == MB  # the array shared three times counted once, the image as its pixels
    assert results.nbytes(page, page) == MB
    assert results.nbytes(None) == 0


def test_a_page_kept_is_made_read_only():
    page = _page(0.1)
    cache = ResultCache(10 * MB)
    cache.put(ResultKey("v", "a", 1), page)
    assert not page.pixels.flags.writeable and not page.parts[0].flags.writeable
    with pytest.raises(ValueError):
        page.pixels[0] = 1


# -- The cache --------------------------------------------------------------------------------------------------------
def _key(n: int) -> ResultKey:
    return ResultKey("v", "picture", n)


def test_the_least_recently_used_page_goes_first():
    cache = ResultCache(3 * MB)
    for n in range(3):
        cache.put(_key(n), _page())
    assert cache.get(_key(0)) is not None  # used: now the most recent
    cache.put(_key(3), _page())
    assert cache.keys() == [_key(2), _key(0), _key(3)]  # 1 went, the least recently used
    assert cache.size() == 3 * MB
    assert cache.get(_key(1)) is None and cache.was_made(_key(1))  # gone, but it had a page
    assert not cache.was_made(_key(9))


def test_protected_pages_are_kept_past_the_budget_and_go_once_unprotected():
    cache = ResultCache(2 * MB)
    cache.put(_key(0), _page())
    cache.put(_key(1), _page())
    cache.protect([_key(0), _key(1), None])
    cache.put(_key(2), _page())
    assert set(cache.keys()) == {_key(0), _key(1)}  # the new one is the only one that may go
    cache.protect([_key(0), _key(1), _key(2)])
    cache.put(_key(2), _page())
    assert len(cache) == 3 and cache.size() == 3 * MB  # past the budget: all protected
    cache.protect([_key(2)])
    assert cache.keys() == [_key(2)] or cache.size() <= 2 * MB
    assert _key(2) in cache and cache.size() <= 2 * MB


def test_the_budget_counts_print_layouts_and_shrinking_it_evicts():
    cache = ResultCache(10 * MB)
    page, drawn = _page(1.0), _page(2.0, seed=1)
    cache.put(_key(0), page)
    assert cache.attach_print_page(_key(0), page, drawn)
    assert cache.size() == 3 * MB and cache.get(_key(0)).print_page is drawn
    assert not cache.attach_print_page(_key(0), _page(), _page())  # not the page kept: not attached
    cache.put(_key(0), page)  # the same page again keeps its print layout
    assert cache.get(_key(0)).print_page is drawn and cache.size() == 3 * MB
    cache.put(_key(1), _page(4.0))
    cache.set_budget(4 * MB)
    assert cache.keys() == [_key(1)] and cache.size() == 4 * MB
    cache.clear()
    assert len(cache) == 0 and cache.size() == 0 and cache.was_made(_key(0))


def test_pages_put_and_looked_up_from_many_threads_stay_counted_and_within_the_budget():
    cache = ResultCache(8 * MB)
    pages = [_page(0.5, seed=n) for n in range(40)]
    errors = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(300):
                n = rng.randrange(40)
                if rng.random() < 0.5:
                    cache.put(_key(n), pages[n])
                else:
                    entry = cache.get(_key(n))
                    assert entry is None or entry.page is pages[n]
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert cache.size() == sum(results.nbytes(cache.get(key).page) for key in cache.keys()) <= 8 * MB


def test_the_automatic_budget_is_a_quarter_of_the_memory_at_most_2_gb():
    assert results.automatic_budget(4 * 1024 * MB) == 1024 * MB
    assert results.automatic_budget(64 * 1024 * MB) == 2048 * MB
    memory = results.total_memory()
    assert memory is None or memory > 256 * MB
    assert 0 < results.automatic_budget() <= results.AUTOMATIC_CAP


def test_the_budget_is_a_preference(tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    prefs = preferences.Preferences(store)
    assert prefs.value(preferences.RESULTS_KEY) == preferences.AUTOMATIC
    assert prefs.result_memory() == results.automatic_budget()
    prefs.set_value(preferences.RESULTS_KEY, "512 MB")
    assert preferences.Preferences(store).result_memory() == 512 * MB  # kept between runs


# -- The window -------------------------------------------------------------------------------------------------------
class _Window:
    """A window (nothing kept between runs) counting the pages the pipeline makes; ``hold`` keeps the next ones from
    finishing until released."""

    def __init__(self, monkeypatch, size=(160, 240)):
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
        self.made = 0
        self.hold: threading.Event | None = None
        self.started = threading.Event()
        real = pipeline.generate

        def counting(*args, **kwargs):
            self.started.set()
            if self.hold is not None:
                should_cancel = kwargs.get("should_cancel")
                while not self.hold.wait(0.005):
                    if should_cancel is not None and should_cancel():
                        raise PipelineCancelled()
                if should_cancel is not None and should_cancel():
                    raise PipelineCancelled()  # released after it was asked to stop
            self.made += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(pipeline, "generate", counting)
        self.window = MainWindow()
        self.window.current_image_bgr = _picture(size)
        self.identified()

    def identified(self) -> None:
        """Wait for the picture's hash, worked out in the background."""
        deadline = time.monotonic() + TIMEOUT_S
        while self.window.document.source().content_hash is None:
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.002)

    def settle(self, seconds: float = 0.0) -> None:
        window = self.window
        start = time.monotonic()
        while True:
            _app.processEvents()
            done = window.jobs.is_idle() and not window._redraw_timer.isActive()
            if done and time.monotonic() - start >= seconds:
                _app.processEvents()
                if window.jobs.is_idle() and not window._redraw_timer.isActive():
                    return
            assert time.monotonic() - start < TIMEOUT_S, "the window didn't settle"
            time.sleep(0.003)

    def preview(self) -> None:
        self.window.generate_preview()
        self.settle()

    def colors(self, step: int) -> None:
        """Move the number of colors, as one undoable edit."""
        self.window.controls.controls["num_colors"].step(step)
        _app.processEvents()
        self.window._record_settings()

    def state(self) -> str:
        return self.window.freshness.state()


def _picture(size) -> np.ndarray:
    height, width = size
    rng = np.random.default_rng(1)
    picture = np.zeros((height, width, 3), np.uint8)
    picture[:, : width // 2] = (200, 40, 40)
    picture[: height // 2, width // 2:] = (40, 170, 40)
    picture[height // 2:, width // 2:] = (40, 40, 200)
    return np.clip(picture + rng.integers(0, 30, picture.shape), 0, 255).astype(np.uint8)


@pytest.fixture
def win(monkeypatch):
    w = _Window(monkeypatch)
    yield w
    w.window.close()
    w.window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_undo_and_redo_show_the_pages_kept_at_once_with_no_page_made(win):
    win.preview()
    first = win.window.current_page
    win.colors(-4)
    win.preview()
    second = win.window.current_page
    assert win.made == 2 and len(win.window.results) == 2
    win.window.undo()
    assert win.window.jobs.is_idle()  # nothing started: the page is shown in this very call
    assert win.window.current_page is first and win.state() == freshness.CURRENT
    assert win.window.last_restore_ms < 200
    assert "made before" in win.window.status.text()
    win.window.redo()
    assert win.window.current_page is second and win.state() == freshness.CURRENT
    win.settle()
    assert win.made == 2


def test_a_setting_put_back_by_hand_shows_its_page_kept(win):
    win.preview()
    first = win.window.current_page
    win.colors(-4)
    win.preview()
    win.colors(4)  # back by hand, not by undo
    assert win.window.current_page is first and win.window.jobs.is_idle()
    win.settle(0.3)
    assert win.made == 2 and win.state() == freshness.CURRENT


def test_a_page_kept_brings_its_print_layout(win):
    win.preview()
    win.window._draw_print_page()
    win.settle()
    drawn = win.window._print_page
    assert drawn is not None
    win.colors(-4)
    win.preview()
    assert win.window._print_page is None
    win.window.undo()
    assert win.window._print_page is drawn and win.window.jobs.current(PRINT_LAYOUT) is None
    assert win.window.results.size() >= results.nbytes(drawn)


def test_a_state_that_had_a_page_no_longer_kept_is_made_again_in_the_background(win):
    win.preview()
    win.colors(-4)
    win.preview()
    second = win.window.current_page
    win.window.results.clear()
    win.window.undo()
    # The settings are back at once, the page shown kept and labeled while its page is made again.
    assert win.window.current_page is second
    assert win.window.jobs.current(PREVIEW) is not None and win.state() == freshness.PROCESSING
    assert "no longer kept" in win.window.status.text()
    win.settle()
    assert win.made == 3 and win.state() == freshness.CURRENT and win.window.current_page is not second


def test_a_state_that_never_had_a_page_waits_for_generate(win):
    win.colors(-4)
    win.preview()
    win.window.undo()  # the settings before any page was made
    win.settle(0.2)
    assert win.made == 1 and win.state() == freshness.MODIFIED


def test_a_preview_being_made_with_other_settings_is_dropped_when_a_page_kept_is_shown(win):
    win.preview()
    first = win.window.current_page
    win.hold = threading.Event()
    win.started.clear()
    win.colors(-4)
    win.window.generate_preview()
    assert win.started.wait(TIMEOUT_S)
    _app.processEvents()
    win.window.undo()
    assert win.window.current_page is first and win.state() == freshness.CURRENT
    assert not win.window.jobs.is_running(PREVIEW) and win.window.controls.progress_row.isHidden()
    win.hold.set()
    win.settle()
    assert win.window.current_page is first and win.made == 1  # the page dropped was never made


def test_a_branch_keeps_the_pages_of_the_one_left(win):
    win.preview()
    win.colors(-4)
    win.preview()
    b = win.window.current_page
    win.window.undo()
    win.colors(2)  # a new branch: what could be redone is dropped from the history...
    win.preview()
    assert not win.window.history.can_redo()
    win.colors(-6)  # ...but its page is kept: back to b's settings by hand, it is shown
    assert win.window.current_page is b and win.window.jobs.is_idle()
    assert win.made == 3


def test_undoing_an_open_shows_the_first_picture_s_page_kept(win, tmp_path, monkeypatch):
    paths = []
    for n, color in enumerate(((200, 40, 40), (40, 40, 200))):
        path = tmp_path / f"picture{n}.png"
        image = Image.fromarray(_picture((150 + 20 * n, 200))[:, :, ::-1])
        image.save(path)
        paths.append(path)
    for path in paths:
        monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, p=path, **k: (str(p), "")))
        win.window.commands["open"].trigger()
        win.identified()
        win.preview()
    first_key, second_page = win.window.results.keys()[-2], win.window.current_page
    win.window.undo()
    assert win.window.document.source().path == paths[0] and win.window.jobs.is_idle()
    assert win.window.current_page is win.window.results.get(first_key).page
    win.window.redo()
    assert win.window.current_page is second_page
    win.settle()
    assert win.made == 2  # one page of each picture opened, none made again


def test_the_page_shown_and_its_undo_and_redo_neighbors_are_kept_past_the_budget(win):
    win.window.results.set_budget(0)
    win.preview()
    for _ in range(3):
        win.colors(-2)
        win.preview()
    kept = set(win.window.results.keys())
    assert kept == set(win.window.results.protected()) and len(kept) == 2  # the page shown and undo's
    win.window.undo()
    win.settle()
    # Back one: the page shown and redo's are kept; undo's own went before it was a neighbor.
    kept = set(win.window.results.keys())
    assert kept <= set(win.window.results.protected()) and len(kept) == 2
    assert win.window.current_page is win.window.results.get(win.window._result_key()).page


def test_a_page_made_before_the_picture_s_hash_is_known_is_kept_once_it_is(monkeypatch):
    gate = threading.Event()
    real = document.content_hash
    monkeypatch.setattr(document, "content_hash", lambda image: gate.wait(TIMEOUT_S) and real(image))
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    try:
        window.current_image_bgr = _picture((120, 160))
        window.generate_preview()
        deadline = time.monotonic() + TIMEOUT_S
        while window.current_page is None:
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.003)
        assert len(window.results) == 0  # no key yet
        gate.set()
        while window.document.source().content_hash is None:
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.003)
        _app.processEvents()
        assert window.results.get(window._result_key()).page is window.current_page
    finally:
        gate.set()
        window.close()
        window.deleteLater()
        _app.processEvents()


def test_the_memory_kept_stays_within_the_budget_over_50_cycles(win):
    win.preview()
    page_size = results.nbytes(win.window.current_page)
    budget = 4 * page_size
    win.window.results.set_budget(budget)
    rng = random.Random(3)
    for cycle in range(50):
        if cycle % 3 == 2 and win.window.history.can_undo():
            win.window.undo()
        elif cycle % 3 == 1 and win.window.history.can_redo():
            win.window.redo()
        else:
            win.colors(rng.choice((-3, -2, -1, 1, 2, 3)))
            win.window.generate_preview()
        win.settle()
        protected = sum(win.window.results.get(key).size for key in win.window.results.protected()
                        if key in win.window.results)
        assert win.window.results.size() <= max(budget, protected) + 2 * page_size
        assert win.window.results.size() == sum(
            entry.size for entry in (win.window.results.get(key) for key in win.window.results.keys()))
    assert len(win.window.results) <= budget // page_size + 3


def test_clear_now_forgets_the_pages_kept(win):
    win.preview()
    assert len(win.window.results) == 1
    win.window.open_preferences()
    win.window.preferences_dialog.clear_cache_button.click()
    assert len(win.window.results) == 0
    win.window.preferences.set_value(preferences.RESULTS_KEY, "256 MB")
    assert win.window.results.budget() == 256 * MB
    win.window.preferences_dialog.close()

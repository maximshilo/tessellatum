"""H4: recovery. While the work has unsaved changes it is kept in the app's data folder; a clean exit, a save and
closing remove it; a session that ended without closing (a real crash in a child process here) leaves it, and the next
start offers it back, unsaved; snapshots of windows still open are never offered; old ones are pruned."""

import errno
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import project as projects  # noqa: E402
from tessellatum.app import recovery  # noqa: E402
from tessellatum.app.recovery import RecoveryStore  # noqa: E402
from tessellatum.app.snapshot import CUSTOM, DocumentSettings  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import main_window, preferences  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.recovery_dialog import RecoveryDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120


def _picture_bytes(size=(90, 120)) -> bytes:
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    rgb[: size[0] // 2] = (200, 60, 60)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, "PNG")
    return out.getvalue()


def _write(store: RecoveryStore, name="lion.png", settings=None) -> bytes:
    data = _picture_bytes()
    store.write(settings or DocumentSettings().with_values(num_colors=9), name, data, projects.decode_image(data),
                name)
    return data


_CRASH = """
import os, sys
sys.path.insert(0, sys.argv[2])
from tessellatum.app.recovery import RecoveryStore
from tessellatum.app import project as projects
from tessellatum.app.snapshot import CUSTOM, DocumentSettings
data = open(sys.argv[3], "rb").read()
store = RecoveryStore(sys.argv[1])
settings = DocumentSettings(preset=CUSTOM).with_values(num_colors=11)
store.write(settings, "crashed.png", data, projects.decode_image(data), "crashed.png")
print(store.session, flush=True)
os._exit(3)  # no close: as a crash ends it
"""


def _crash(folder: Path, tmp_path: Path) -> str:
    """A session in a child process that writes a snapshot and dies without closing; its id."""
    picture = tmp_path / "crash-picture.png"
    picture.write_bytes(_picture_bytes())
    src = Path(__file__).resolve().parents[1] / "src"
    done = subprocess.run([sys.executable, "-c", _CRASH, str(folder), str(src), str(picture)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 3, done.stderr
    return done.stdout.strip()


# -- The store ---------------------------------------------------------------------------------------------------------
def test_a_crash_leaves_its_snapshot_and_the_next_start_finds_it(tmp_path):
    folder = tmp_path / "recovery"
    session = _crash(folder, tmp_path)
    assert (folder / f"{session}{projects.SUFFIX}").exists()
    store = RecoveryStore(folder)
    [found] = store.pending()
    assert found.session == session and found.name == "crashed.png" and found.project is None
    assert time.time() - found.saved_at < 300
    assert projects.read_project(found.path).settings == DocumentSettings(preset=CUSTOM).with_values(num_colors=11)
    assert found.thumbnail()


def test_a_clean_exit_leaves_nothing(tmp_path):
    folder = tmp_path / "recovery"
    store = RecoveryStore(folder)
    _write(store)
    store.close()
    assert list(folder.iterdir()) == []
    assert RecoveryStore(folder).pending() == []


def test_a_window_still_open_is_never_offered(tmp_path):
    folder = tmp_path / "recovery"
    running = RecoveryStore(folder)
    _write(running)
    other = RecoveryStore(folder)
    assert other.pending() == []
    other.prune(now=time.time() + 365 * 86400)  # nor pruned, however old
    assert running.has_snapshot()
    running.close()


def test_the_lock_of_a_session_that_ended_with_nothing_to_offer_is_pruned(tmp_path):
    folder = tmp_path / "recovery"
    # An unclean exit before any snapshot was written: its lock is left behind, its process gone.
    src = Path(__file__).resolve().parents[1] / "src"
    script = ("import os, sys; sys.path.insert(0, sys.argv[2]); from tessellatum.app.recovery import RecoveryStore; "
              "store = RecoveryStore(sys.argv[1]); print(store.session, flush=True); os._exit(3)")
    done = subprocess.run([sys.executable, "-c", script, str(folder), str(src)], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 3, done.stderr
    orphan = folder / f"{done.stdout.strip()}.lock"
    assert orphan.exists()
    running = RecoveryStore(folder)  # a window still open, with no snapshot either
    store = RecoveryStore(folder)
    assert store.prune() == 0
    assert not orphan.exists()
    assert (folder / f"{running.session}.lock").exists() and (folder / f"{store.session}.lock").exists()
    running.close()
    store.close()


def test_snapshots_are_written_atomically_and_replace_the_last(tmp_path, monkeypatch):
    store = RecoveryStore(tmp_path)
    _write(store, settings=DocumentSettings())
    before = store.path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(recovery.os, "replace", broken)
    with pytest.raises(OSError):
        _write(store, settings=DocumentSettings().with_values(num_colors=5))
    monkeypatch.undo()
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
    _write(store, settings=DocumentSettings().with_values(num_colors=5))
    assert projects.read_project(store.path).settings == DocumentSettings().with_values(num_colors=5)
    store.close()


def test_discard_removes_a_snapshot_for_good(tmp_path):
    folder = tmp_path / "recovery"
    _crash(folder, tmp_path)
    store = RecoveryStore(folder)
    [found] = store.pending()
    store.discard(found)
    assert store.pending() == [] and sorted(p.name for p in folder.iterdir()) == [f"{store.session}.lock"]


def test_old_snapshots_and_too_many_are_pruned(tmp_path, monkeypatch):
    folder = tmp_path / "recovery"
    for _ in range(3):
        _crash(folder, tmp_path)
    store = RecoveryStore(folder)
    assert len(store.pending()) == 3
    monkeypatch.setattr(recovery, "KEEP", 2)
    assert store.prune() == 1 and len(store.pending()) == 2
    assert store.prune(now=time.time() + (recovery.KEEP_DAYS + 1) * 86400) == 2
    assert store.pending() == []


def test_a_snapshot_whose_note_is_damaged_is_still_offered(tmp_path):
    folder = tmp_path / "recovery"
    session = _crash(folder, tmp_path)
    (folder / f"{session}.json").write_text("{damaged", encoding="utf-8")
    [found] = RecoveryStore(folder).pending()
    assert found.name == "Untitled"


def test_a_file_another_window_removes_meanwhile_is_passed_over(tmp_path, monkeypatch):
    folder = tmp_path / "recovery"
    session = _crash(folder, tmp_path)
    (folder / f"{session}.json").write_text("{damaged", encoding="utf-8")  # so its snapshot's time is asked for
    (folder / ".left-1.tmp").write_bytes(b"x")
    store = RecoveryStore(folder)
    real_stat, real_is_file = Path.stat, Path.is_file
    removed = (".left-1.tmp", f"{session}{projects.SUFFIX}")

    def gone(path, *args, **kwargs):  # listed, then removed by another window before it is looked at
        if path.name in removed:
            raise FileNotFoundError(errno.ENOENT, "removed meanwhile", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", gone)
    monkeypatch.setattr(Path, "is_file", lambda path: path.name in removed or real_is_file(path))  # seen, then gone
    assert store.pending() == []
    store.prune()  # no error: nothing to offer, and no recovery dialog lost to it


# -- The dialog --------------------------------------------------------------------------------------------------------
def test_when_it_was_kept_is_said_in_the_locales_words_and_formats():
    from tessellatum.gui.i18n import PseudoTranslator, is_pseudo
    from tessellatum.gui.recovery_dialog import when

    now = time.time()
    assert when(now).startswith("today at ")
    assert "today" not in when(now - 3 * 86400) and " at " in when(now - 3 * 86400)
    translator = PseudoTranslator(_app)
    _app.installTranslator(translator)
    try:
        assert is_pseudo(when(now)) and is_pseudo(when(now - 3 * 86400))
    finally:
        _app.removeTranslator(translator)


def test_the_dialog_lists_restores_and_discards(tmp_path):
    folder = tmp_path / "recovery"
    _crash(folder, tmp_path)
    _crash(folder, tmp_path)
    store = RecoveryStore(folder)
    pending = store.pending()
    dialog = RecoveryDialog(pending, store)
    assert dialog.list.count() == 2 and "crashed.png" in dialog.list.item(0).text()
    assert not dialog.list.item(0).icon().isNull()  # its thumbnail
    dialog.discard_button.click()
    assert dialog.list.count() == 1 and len(store.pending()) == 1
    dialog.restore_button.click()
    assert dialog.result() == dialog.DialogCode.Accepted and dialog.chosen == pending[1]
    dialog.deleteLater()


def test_keep_for_later_leaves_them_all(tmp_path):
    folder = tmp_path / "recovery"
    _crash(folder, tmp_path)
    store = RecoveryStore(folder)
    dialog = RecoveryDialog(store.pending(), store)
    dialog.later_button.click()
    assert dialog.chosen is None and len(store.pending()) == 1
    dialog.deleteLater()


# -- The window --------------------------------------------------------------------------------------------------------
class _Window:
    def __init__(self, monkeypatch, tmp_path, folder=None):
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
        self.monkeypatch, self.tmp = monkeypatch, tmp_path
        self.folder = folder or tmp_path / "recovery"
        self.store = RecoveryStore(self.folder)
        self.dialogs: list[RecoveryDialog] = []
        self.choose = None  # what the recovery dialog answers: a callable given the dialog, or None for Keep for Later
        monkeypatch.setattr(MainWindow, "_run_recovery_dialog", lambda window, dialog: self._answer(dialog))
        self.window = MainWindow(recovery=self.store)

    def _answer(self, dialog) -> bool:
        self.dialogs.append(dialog)
        if self.choose is None:
            return False
        self.choose(dialog)
        return dialog.chosen is not None

    def open(self, path: Path) -> None:
        self.monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName",
                                 staticmethod(lambda *a, **k: (str(path), "")))
        self.window.commands["open"].trigger()
        self.settle()

    def settle(self) -> None:
        deadline = time.monotonic() + TIMEOUT_S
        while not self.window.jobs.is_idle() or self.window._redraw_timer.isActive():
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.003)
        _app.processEvents()

    def edit(self, step=-3) -> None:
        self.window.controls.controls["num_colors"].step(step)
        _app.processEvents()
        self.window._record_settings()

    def close(self) -> None:
        self.window.close()
        self.window.deleteLater()
        _app.processEvents()


@pytest.fixture
def win(monkeypatch, tmp_path):
    w = _Window(monkeypatch, tmp_path)
    picture = tmp_path / "lion cub.png"
    picture.write_bytes(_picture_bytes())
    w.picture = picture
    yield w
    w.close()
    pipeline.clear_cache()


def test_unsaved_work_is_kept_only_while_it_is_unsaved(win):
    win.open(win.picture)
    assert not win.window._keep_for_recovery() and not win.store.has_snapshot()  # nothing changed yet
    win.edit()
    assert win.window._keep_for_recovery() and win.store.has_snapshot()
    assert not win.window._keep_for_recovery()  # nothing new since: not written again
    win.edit()
    assert win.window._keep_for_recovery()
    kept = projects.read_project(win.store.path)
    assert kept.settings == win.window.controls.snapshot() and kept.source_bytes == win.picture.read_bytes()
    win.window.undo()
    win.window.undo()
    win.window._keep_for_recovery()  # back as it was opened: nothing to keep
    assert not win.store.has_snapshot()


def test_saving_and_closing_remove_the_snapshot(win):
    win.open(win.picture)
    win.edit()
    win.window._keep_for_recovery()
    path = win.tmp / "cub.tessellatum"
    win.monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    assert win.window.save_project()
    assert not win.store.has_snapshot()
    win.edit()
    win.window._keep_for_recovery()
    assert win.window.close_document() and not win.store.has_snapshot()


def test_a_clean_exit_leaves_nothing_to_recover(win):
    win.open(win.picture)
    win.edit()
    win.window._keep_for_recovery()
    win.close()  # unsaved changes let go (the tests' answer)
    assert list(win.folder.iterdir()) == []


def test_the_timer_follows_the_preference(win):
    assert win.window._recovery_timer.isActive()
    assert win.window._recovery_timer.interval() == preferences.RECOVERY[preferences.DEFAULT_RECOVERY]
    win.window.preferences._values[preferences.RECOVERY_KEY] = "Off"
    win.window._apply_preference(preferences.RECOVERY_KEY)
    assert not win.window._recovery_timer.isActive()
    win.window.preferences._values[preferences.RECOVERY_KEY] = "Every 1 min"
    win.window._apply_preference(preferences.RECOVERY_KEY)
    assert win.window._recovery_timer.interval() == 60_000


def test_after_a_crash_the_work_is_offered_back_unsaved(monkeypatch, tmp_path):
    folder = tmp_path / "recovery"
    _crash(folder, tmp_path)
    w = _Window(monkeypatch, tmp_path, folder)
    try:
        w.choose = lambda dialog: dialog.restore()
        w.window.offer_recovery()
        w.settle()
        assert len(w.dialogs) == 1
        window = w.window
        assert window.controls.snapshot().values["num_colors"] == 11
        assert window.document.source().name == "crashed.png"
        assert window.has_unsaved_work() and window.isWindowModified()  # recovered, not saved
        assert window.document.project_path() is None
        assert "Recovered crashed.png" in window.notification.text.text()
        assert window.document.result_is_current()  # its page made
        # Kept again as this session's, and the crashed session's snapshot let go: offered once only.
        assert w.store.has_snapshot() and w.store.pending() == []
    finally:
        w.close()
    assert RecoveryStore(folder).pending() == []


def test_keep_for_later_offers_it_again_at_the_next_start(monkeypatch, tmp_path):
    folder = tmp_path / "recovery"
    _crash(folder, tmp_path)
    w = _Window(monkeypatch, tmp_path, folder)
    w.window.offer_recovery()
    assert len(w.dialogs) == 1 and w.window.current_image_bgr is None
    w.close()
    assert len(RecoveryStore(folder).pending()) == 1


def test_nothing_to_recover_shows_no_dialog(win):
    win.window.offer_recovery()
    assert win.dialogs == []


def test_the_preference_is_kept(tmp_path):
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    prefs = preferences.Preferences(store)
    assert prefs.recovery_every_ms() == 120_000
    store.setValue(preferences.RECOVERY_KEY, "Off")
    assert preferences.Preferences(store).recovery_every_ms() is None

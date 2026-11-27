"""Recovery: snapshots of the work while it has unsaved changes, offered back after the app ended without closing.

Each window is a **session** with its own id. While its work has changes that aren't saved, the window writes a
snapshot of it every few minutes (Preferences > Workspace > Keep for recovery) to the app's data folder
(``recovery/``), never beside or over a project:

- ``<session>.tessellatum``: the work, as a project (``app.project``: the picture's bytes and every setting);
- ``<session>.json``: when it was written, the work's name, and the project it belongs to, if any.

Both are written atomically (a temporary file, then ``os.replace``). A session holds ``<session>.lock`` (a
``QLockFile``) while it runs: a snapshot whose session's lock is held belongs to a window still open, and is not
offered. A clean exit, a save and closing the picture remove the session's snapshot; so a snapshot found at the start,
its session gone, is work an unclean exit left. Snapshots older than ``KEEP_DAYS``, or past the newest ``KEEP``, are
pruned.

QtCore only (``QLockFile``), tested headless.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QLockFile

from tessellatum import __version__
from tessellatum.app import project as projects

FOLDER = "recovery"
KEEP_DAYS = 14
KEEP = 20
_META = ".json"
_LOCK = ".lock"


@dataclass(frozen=True)
class Snapshot:
    """A snapshot left by a session that ended: where it is, when it was written, the work's name and project."""

    path: Path  # the snapshot itself, a project file
    session: str
    saved_at: float  # seconds since the epoch
    name: str
    project: Path | None

    def thumbnail(self) -> bytes | None:
        return projects.read_thumbnail(self.path)


def _write_atomically(path: Path, data: bytes) -> None:
    handle, temp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _lock(path: Path) -> QLockFile:
    lock = QLockFile(str(path))
    lock.setStaleLockTime(0)  # held for as long as the session runs: stale only once its process is gone
    return lock


class RecoveryStore:
    """The recovery folder, for the session ``session`` (a new id by default), which it locks as long as it lives."""

    def __init__(self, folder: Path, session: str | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session = session or uuid.uuid4().hex
        self._lock = _lock(self.folder / f"{self.session}{_LOCK}")
        self._locked = self._lock.tryLock(0)

    # -- This session's snapshot ----------------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.folder / f"{self.session}{projects.SUFFIX}"

    def write(self, snapshot, source_name: str, source_bytes: bytes, image, name: str,
              project: Path | None = None) -> Path:
        """Write this session's snapshot of the work (see the module's docstring), replacing the last one."""
        _write_atomically(self.path, projects.project_bytes(snapshot, source_name, source_bytes, image))
        meta = {"saved_at": time.time(), "name": name, "project": None if project is None else str(project),
                "app_version": __version__}
        _write_atomically(self.folder / f"{self.session}{_META}", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        return self.path

    def has_snapshot(self) -> bool:
        return self.path.exists()

    def clear(self) -> None:
        """Remove this session's snapshot: the work was saved, closed, or let go."""
        self._remove(self.session, lock=False)

    def close(self) -> None:
        """A clean exit: the snapshot removed, the session's lock let go."""
        self.clear()
        if self._locked:
            self._lock.unlock()
            self._locked = False

    # -- Snapshots left by other sessions -----------------------------------------------------------------------
    def _alive(self, session: str) -> bool:
        if session == self.session:
            return True
        other = _lock(self.folder / f"{session}{_LOCK}")
        if other.tryLock(0):
            other.unlock()  # nobody held it: the session is gone (its lock file goes with the unlock)
            return False
        return other.error() == QLockFile.LockFailedError

    def pending(self) -> list[Snapshot]:
        """The snapshots of sessions that ended without closing, the latest first."""
        found = []
        for meta_path in self.folder.glob(f"*{_META}"):
            session = meta_path.stem
            snapshot = self.folder / f"{session}{projects.SUFFIX}"
            if session == self.session or not snapshot.is_file() or self._alive(session):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                saved_at = float(meta["saved_at"])
                name = str(meta.get("name") or "Untitled")
                project = Path(meta["project"]) if meta.get("project") else None
            except (OSError, ValueError, KeyError, TypeError):
                saved_at, name, project = snapshot.stat().st_mtime, "Untitled", None
            found.append(Snapshot(snapshot, session, saved_at, name, project))
        return sorted(found, key=lambda s: s.saved_at, reverse=True)

    def discard(self, snapshot: Snapshot) -> None:
        """Remove ``snapshot``, of a session that ended."""
        if snapshot.session != self.session and not self._alive(snapshot.session):
            self._remove(snapshot.session, lock=True)

    def prune(self, now: float | None = None) -> int:
        """Remove the snapshots older than ``KEEP_DAYS`` and past the newest ``KEEP``, what failed writes left, and the
        locks of sessions that ended without a snapshot to offer (an unclean exit before any was written); how many
        snapshots went."""
        now = time.time() if now is None else now
        pending = self.pending()
        gone = 0
        for index, snapshot in enumerate(pending):
            if index >= KEEP or now - snapshot.saved_at > KEEP_DAYS * 86400:
                self.discard(snapshot)
                gone += 1
        for lock in self.folder.glob(f"*{_LOCK}"):
            session = lock.stem
            if session == self.session or (self.folder / f"{session}{projects.SUFFIX}").exists():
                continue  # this session's, or one whose snapshot is offered: it goes with the snapshot
            if not self._alive(session):  # the session is gone: trying its lock took it over and let it go
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        for temp in self.folder.glob(".*.tmp"):
            if now - temp.stat().st_mtime > 3600:
                try:
                    temp.unlink()
                except OSError:
                    pass
        return gone

    def _remove(self, session: str, lock: bool) -> None:
        suffixes = (projects.SUFFIX, _META) + ((_LOCK,) if lock else ())
        for suffix in suffixes:
            try:
                (self.folder / f"{session}{suffix}").unlink()
            except FileNotFoundError:
                pass

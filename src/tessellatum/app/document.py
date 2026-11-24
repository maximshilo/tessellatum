"""The document: the picture being worked on, its settings, and the page made of them, apart from any widget.

What the user edits lives here, once: the panel shows it and writes its edits into it, the window reads the settings
to make and export pages from it, and later undo, projects and recovery (H1, H3) save and restore it. The workspace
(zoom, the version shown, panels) is not part of it.

Every change of the settings is one ``settings_changed`` with the new revision; an edit that changes nothing emits
nothing. The picture's content hash, its identity for caches and stale-result checks, is worked out off the GUI
thread and announced with ``source_identified``.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from tessellatum.app.snapshot import DocumentSettings

_tokens = itertools.count(1)


@dataclass(eq=False)
class Source:
    """The picture: its pixels (BGR), where it came from, and its content hash once worked out."""

    image: np.ndarray
    path: Path | None = None
    token: int = field(default_factory=lambda: next(_tokens))  # this load of it, whatever its content
    content_hash: str | None = None


def content_hash(image: np.ndarray) -> str:
    """A hash of the picture's pixels and shape: equal for equal pictures, wherever they came from."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr((image.shape, image.dtype.str)).encode("ascii"))
    digest.update(np.ascontiguousarray(image).data)
    return digest.hexdigest()


class Document(QObject):
    settings_changed = Signal(int)  # the new revision
    source_changed = Signal()
    source_identified = Signal(str)  # the picture's content hash, once worked out
    result_changed = Signal()
    dirty_changed = Signal(bool)
    _hashed = Signal(int, str)  # from the hashing thread: the source's token and its hash

    def __init__(self, settings: DocumentSettings | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = settings if settings is not None else DocumentSettings()
        self._revision = 0
        self._saved_revision = 0
        self._source: Source | None = None
        self._result = None
        self._result_settings: DocumentSettings | None = None
        self._hashed.connect(self._on_hashed)

    # -- Settings ---------------------------------------------------------------
    def settings(self) -> DocumentSettings:
        return self._settings

    def revision(self) -> int:
        """How many changes the document has had: settings and pictures alike."""
        return self._revision

    def set_settings(self, settings: DocumentSettings) -> bool:
        """Make ``settings`` the document's; True if they differ from what it had (and ``settings_changed`` was
        emitted), False if they're the same."""
        if settings == self._settings:
            return False
        was_dirty = self.is_dirty()
        self._settings = settings
        self._revision += 1
        self.settings_changed.emit(self._revision)
        self._dirty_maybe_changed(was_dirty)
        return True

    # -- The picture ------------------------------------------------------------
    def source(self) -> Source | None:
        return self._source

    def set_source(self, image: np.ndarray | None, path: Path | None = None) -> Source | None:
        """Work on ``image`` (BGR) now, read from ``path`` if it was; None for no picture. The result made of the last
        picture is dropped. Its hash is worked out in the background (``source_identified``)."""
        was_dirty = self.is_dirty()
        self._source = None if image is None else Source(image, path)
        self._revision += 1
        self.set_result(None)
        self.source_changed.emit()
        self._dirty_maybe_changed(was_dirty)
        if self._source is not None:
            source = self._source
            threading.Thread(
                target=lambda: self._hashed.emit(source.token, content_hash(source.image)),
                name="source-hash", daemon=True,
            ).start()
        return self._source

    def _on_hashed(self, token: int, digest: str) -> None:
        source = self._source
        if source is None or source.token != token:
            return  # a picture since replaced
        source.content_hash = digest
        self.source_identified.emit(digest)

    # -- The page made ----------------------------------------------------------
    def result(self):
        """The page last made of the picture (a ``GeneratedPage``), or None."""
        return self._result

    def result_settings(self) -> DocumentSettings | None:
        """The settings the result was made with."""
        return self._result_settings

    def set_result(self, page, settings: DocumentSettings | None = None) -> None:
        """``page``, made of the current picture with ``settings`` (by default, the document's own), or None."""
        if page is None and self._result is None:
            return
        self._result = page
        self._result_settings = None if page is None else (settings if settings is not None else self._settings)
        self.result_changed.emit()

    def result_is_current(self) -> bool:
        """Whether the result is what the settings would make now (the legend and export options aside)."""
        if self._result is None or self._result_settings is None:
            return False
        return self._result_settings.generation_key() == self._settings.generation_key()

    # -- Saved or not ---------------------------------------------------------------
    def is_dirty(self) -> bool:
        """Whether it changed since it was last saved (or made)."""
        return self._revision != self._saved_revision

    def mark_saved(self) -> None:
        was_dirty = self.is_dirty()
        self._saved_revision = self._revision
        self._dirty_maybe_changed(was_dirty)

    def _dirty_maybe_changed(self, was_dirty: bool) -> None:
        if self.is_dirty() != was_dirty:
            self.dirty_changed.emit(self.is_dirty())

"""The result cache: pages already made, kept so that undo, redo and a setting put back show them at once.

Each entry is a page made of a picture (a ``GeneratedPage``) and, once drawn, its print layout, the same page laid out
at print size for the zoom and the export. It is found by a ``ResultKey``:

- the picture's content hash (``document.content_hash``), so the same pixels find their pages wherever they were opened
  from, an Open undone included;
- the settings' generation key (``DocumentSettings.generation_key``): the pipeline's own settings objects, so equal
  settings, however they were reached, give one key, and the legend's detail and the export's options, which never make
  the page again, are not in it;
- the pipeline's version: the app's version and a hash of the model files the pipeline runs, so a page made by another
  build is never shown as this one's.

Generation is deterministic (seeded k-means; the distance transforms quantized before any choice is made of them), so
a key always stands for one page and a page kept is the page that would be made again.

**Memory.** Each entry is measured from its buffers (the arrays' ``nbytes``, the images' pixels; an array shared
within the entry counted once, one shared with the stage cache counted here too, so the count errs high). The cache
holds at most its budget, evicting the least recently used entries first, except the protected ones: the page shown
and the pages of the states undo and redo go to next. Those are kept even past the budget (at most a few pages). The
history keeps settings and pictures, never pages, and a job's page is counted once it is delivered, so the cache and
the document's page are where pages are held. The budget is automatic by default: a quarter of the computer's memory,
at most 2 GB (``automatic_budget``), or the size picked in Preferences.

**Immutability.** A page put in the cache is shared by everyone who shows, exports or restores it, so its arrays are
made read-only (``freeze``): a change in place raises rather than altering a page kept.

Thread-safe: jobs' threads may look up and put while the GUI thread does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import threading
from collections import OrderedDict
from enum import Enum
from importlib import resources
from typing import Any, Hashable, NamedTuple

import numpy as np
from PIL import Image

from tessellatum import __version__

AUTOMATIC_SHARE = 0.25  # of the computer's memory, for the automatic budget
AUTOMATIC_CAP = 2 * 1024 ** 3  # bytes: the automatic budget's ceiling
ASSUMED_MEMORY = 8 * 1024 ** 3  # bytes: the computer's memory when it can't be read
SEEN_KEYS = 4096  # how many keys of pages made are remembered once their pages are evicted (``was_made``)
MODEL_SUFFIXES = (".onnx", ".xml", ".txt")  # the resources the pipeline reads: models, a cascade, the color names

_version: str | None = None
_version_lock = threading.Lock()


def pipeline_version() -> str:
    """The app's version and a hash of the model files the pipeline runs: a page made by another build has another."""
    global _version
    with _version_lock:
        if _version is None:
            digest = hashlib.blake2b(digest_size=8)
            folder = resources.files("tessellatum").joinpath("resources")
            for item in sorted(folder.iterdir(), key=lambda i: i.name):
                if item.is_file() and item.name.endswith(MODEL_SUFFIXES):
                    digest.update(item.name.encode("utf-8"))
                    digest.update(item.read_bytes())
            _version = f"{__version__}+{digest.hexdigest()}"
        return _version


class ResultKey(NamedTuple):
    """What finds a page: the pipeline's version, the picture's content hash, and the settings' generation key."""

    version: str
    content_hash: str
    settings: Hashable

    @classmethod
    def of(cls, content_hash: str | None, settings) -> ResultKey | None:
        """The key of the page ``settings`` (a ``DocumentSettings``) make of the picture with ``content_hash``; None
        while the picture's hash isn't known."""
        if content_hash is None or settings is None:
            return None
        return cls(pipeline_version(), content_hash, settings.generation_key())


def total_memory() -> int | None:
    """The computer's physical memory in bytes, or None if it can't be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
            return None
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(size) if pages > 0 and size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def automatic_budget(memory: int | None = None) -> int:
    """The automatic budget: a quarter of the computer's memory (``memory`` bytes, read if not given), at most 2 GB."""
    if memory is None:
        memory = total_memory() or ASSUMED_MEMORY
    return int(min(memory * AUTOMATIC_SHARE, AUTOMATIC_CAP))


# -- Measuring and freezing a page ----------------------------------------------------------------------------------
_LEAVES = (str, bytes, int, float, bool, complex, type(None), Enum, type)


def _walk(value: Any, seen: set[int], visit) -> None:
    """Call ``visit`` on every array and image reachable from ``value`` through dataclasses, mappings and sequences,
    each object once."""
    if isinstance(value, _LEAVES) or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (np.ndarray, Image.Image)):
        visit(value)
        if isinstance(value, np.ndarray) and value.dtype == object:
            for item in value.flat:
                _walk(item, seen, visit)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _walk(getattr(value, f.name, None), seen, visit)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk(key, seen, visit)
            _walk(item, seen, visit)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _walk(item, seen, visit)
        return
    attrs = getattr(value, "__dict__", None)  # a plain object holding arrays (a polygon, a label's shape)
    if isinstance(attrs, dict):
        for item in attrs.values():
            _walk(item, seen, visit)


def _image_bytes(image: Image.Image) -> int:
    return image.width * image.height * len(image.getbands()) * (4 if image.mode in ("I", "F", "I;16") else 1)


def nbytes(*values: Any) -> int:
    """The bytes held by the arrays and images reachable from ``values`` (see ``_walk``), each counted once. A view
    counts its own extent, so a view and its base both counted err high."""
    total = 0

    def visit(item) -> None:
        nonlocal total
        total += item.nbytes if isinstance(item, np.ndarray) else _image_bytes(item)

    seen: set[int] = set()
    for value in values:
        _walk(value, seen, visit)
    return total


def freeze(*values: Any) -> None:
    """Make every array reachable from ``values`` read-only: they are shared from now on."""

    def visit(item) -> None:
        if isinstance(item, np.ndarray) and item.flags.writeable:
            item.flags.writeable = False

    seen: set[int] = set()
    for value in values:
        _walk(value, seen, visit)


# -- The cache ------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Entry:
    """A page kept: the page, its print layout once drawn, and the bytes they hold."""

    page: Any
    print_page: Any = None
    size: int = 0


class ResultCache:
    """Pages made, by ``ResultKey``, within a memory budget in bytes (see the module's docstring)."""

    def __init__(self, budget: int | None = None):
        self._lock = threading.Lock()
        self._entries: OrderedDict[ResultKey, Entry] = OrderedDict()
        self._budget = automatic_budget() if budget is None else int(budget)
        self._size = 0
        self._protected: frozenset = frozenset()
        self._seen: OrderedDict[ResultKey, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    # -- Looking up ------------------------------------------------------------------------------------------------
    def get(self, key: ResultKey | None) -> Entry | None:
        """The entry kept for ``key``, made the most recently used; None if there is none (a miss is counted)."""
        if key is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._entries

    def was_made(self, key: ResultKey | None) -> bool:
        """Whether a page was made for ``key``, kept or since evicted: a state that had a page."""
        if key is None:
            return False
        with self._lock:
            return key in self._entries or key in self._seen

    # -- Keeping -------------------------------------------------------------------------------------------------
    def put(self, key: ResultKey | None, page, print_page=None) -> Entry | None:
        """Keep ``page`` (and its ``print_page``, if drawn) for ``key``, its arrays made read-only. A page already kept
        for the key keeps its print layout unless another is given. Returns the entry, or None without a key."""
        if key is None or page is None:
            return None
        freeze(page, print_page)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._size -= old.size
                if print_page is None and old.page is page:
                    print_page = old.print_page
            entry = Entry(page, print_page, nbytes(page, print_page))
            self._entries[key] = entry
            self._size += entry.size
            self._remember(key)
            self._evict()
            return entry

    def attach_print_page(self, key: ResultKey | None, page, print_page) -> bool:
        """Keep ``print_page`` with the entry for ``key`` if it still holds ``page``; True if it does."""
        if key is None or print_page is None:
            return False
        freeze(print_page)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.page is not page:
                return False
            self._size -= entry.size
            entry.print_page = print_page
            entry.size = nbytes(entry.page, print_page)
            self._size += entry.size
            self._evict()
            return True

    def _remember(self, key: ResultKey) -> None:
        self._seen[key] = None
        self._seen.move_to_end(key)
        while len(self._seen) > SEEN_KEYS:
            self._seen.popitem(last=False)

    def _evict(self) -> None:
        """Forget the least recently used unprotected entries until within the budget."""
        if self._size <= self._budget:
            return
        for key in list(self._entries):
            if self._size <= self._budget:
                break
            if key in self._protected:
                continue
            self._size -= self._entries.pop(key).size

    # -- Protection, budget, clearing --------------------------------------------------------------------------------
    def protect(self, keys) -> None:
        """Keep the entries for ``keys`` whatever the budget (the page shown, its history neighbors), replacing the
        keys protected before; entries left unprotected are evicted now if over the budget."""
        with self._lock:
            self._protected = frozenset(key for key in keys if key is not None)
            self._evict()

    def protected(self) -> frozenset:
        with self._lock:
            return self._protected

    def budget(self) -> int:
        return self._budget

    def set_budget(self, budget: int) -> None:
        with self._lock:
            self._budget = max(0, int(budget))
            self._evict()

    def size(self) -> int:
        """The bytes the entries hold, as measured when they were kept."""
        with self._lock:
            return self._size

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> list[ResultKey]:
        """The keys kept, least recently used first."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Forget every page kept (the keys of pages made are remembered: ``was_made``)."""
        with self._lock:
            self._entries.clear()
            self._size = 0

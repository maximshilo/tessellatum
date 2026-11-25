"""Undo and redo: the document's history of changes.

Each change keeps the state before and after it, whole, and undoing or redoing puts one or the other back, through the
callbacks the window gives (the settings shown on the panel, the picture open). Keeping whole states, which are small
and immutable (``DocumentSettings``), rather than the steps between them, means a change can't be undone wrongly
however it was made: a slider, a preset, a reset of a section, the paints listed.

What makes one change is the window's call: the edits made in one go (a preset sets many settings at once) are one,
and a slider dragged is one from its press to its release (``begin_merge``/``end_merge``). A new change after an undo
drops what could have been redone. Only the document is in it: the zoom, the panels, the version shown and the selection
never are.

The clean state is where the document was last saved (``set_clean``), for projects (H3) to mark.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

_merge_keys = itertools.count(1)


class _Change(QUndoCommand):
    """A change from ``before`` to ``after``, put back by ``restore``; already made when it is pushed."""

    MERGING = 1

    def __init__(self, before, after, text: str, restore: Callable, merge_key: int | None):
        super().__init__(text)
        self.before, self.after = before, after
        self._restore = restore
        self.merge_key = merge_key
        self._made = True  # pushed after the change was made: the first redo has nothing to do

    def id(self) -> int:
        return self.MERGING if self.merge_key is not None else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, _Change) or other.merge_key != self.merge_key or other._restore != self._restore:
            return False
        self.after = other.after
        return True

    def redo(self) -> None:
        if self._made:
            self._made = False
            return
        self._restore(self.after)

    def undo(self) -> None:
        self._restore(self.before)


class History(QObject):
    """The document's history. ``restore_settings(settings)`` and ``restore_source(source)`` put a state back."""

    changed = Signal()  # something was done, undone or redone: what Undo and Redo say may have changed

    def __init__(self, restore_settings: Callable, restore_source: Callable, parent: QObject | None = None):
        super().__init__(parent)
        self.stack = QUndoStack(self)
        self._restore_settings = restore_settings
        self._restore_source = restore_source
        self._merge_key: int | None = None
        # Signal to signal, so that Qt cuts them with this object: the stack, deleted after it, clears itself first.
        self.stack.indexChanged.connect(self.changed)
        self.stack.cleanChanged.connect(self.changed)

    # -- Recording ----------------------------------------------------------------
    def record_settings(self, before, after, text: str) -> None:
        """The settings changed from ``before`` to ``after``, called ``text`` (``Undo <text>``). Within a merge (a
        drag), it joins the change the merge began with."""
        self.stack.push(_Change(before, after, text, self._restore_settings, self._merge_key))

    def record_source(self, before, after, text: str = "Open Image") -> None:
        """The picture changed from ``before`` to ``after`` (each whatever ``restore_source`` takes)."""
        self.stack.push(_Change(before, after, text, self._restore_source, None))

    def begin_merge(self) -> None:
        """The changes recorded until ``end_merge`` are one: a slider dragged."""
        self._merge_key = next(_merge_keys)

    def end_merge(self) -> None:
        self._merge_key = None

    # -- Undo and redo ----------------------------------------------------------------
    def undo(self) -> None:
        self.end_merge()
        self.stack.undo()

    def redo(self) -> None:
        self.end_merge()
        self.stack.redo()

    def can_undo(self) -> bool:
        return self.stack.canUndo()

    def can_redo(self) -> bool:
        return self.stack.canRedo()

    def undo_text(self) -> str:
        return self.stack.undoText()

    def redo_text(self) -> str:
        return self.stack.redoText()

    def count(self) -> int:
        return self.stack.count()

    # -- The clean state ----------------------------------------------------------------
    def set_clean(self) -> None:
        self.stack.setClean()

    def is_clean(self) -> bool:
        return self.stack.isClean()

    def clear(self) -> None:
        self.end_merge()
        self.stack.clear()

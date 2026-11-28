"""A failure said so the user can act on it: what went wrong, in words, and what to try. Qt-free.

The window says ``explain(...)`` in its notification; the exception's own message and traceback go to the
diagnostics, behind the notification's Details.
"""

from __future__ import annotations

import errno

# What failed, as the window asks for it: making a page (a preview, a page to export, its print layout) or writing a
# file.
MAKING, WRITING = "making", "writing"

# A page that ran out of memory on one of these papers is offered the next smaller of its kind: A1, A2, A3, A4; and
# Tabloid or Legal, Letter.
_SMALLER = {"A1": "A2", "A2": "A3", "A3": "A4", "Tabloid": "Letter", "Legal": "Letter"}

_MEMORY_WORDS = ("out of memory", "insufficient memory", "cannot allocate", "unable to allocate", "bad_alloc")


def is_memory_error(error_type: type | None, message: str) -> bool:
    """Whether a failure was running out of memory: a ``MemoryError`` (NumPy's included), or an OpenCV or other error
    saying so."""
    if error_type is not None and issubclass(error_type, MemoryError):
        return True
    return any(words in message.lower() for words in _MEMORY_WORDS)


def explain(
    what: str,
    error_type: type | None,
    message: str,
    paper: str = "A4",
    file_format: str | None = None,
    file_name: str = "",
) -> str:
    """The failure of ``what`` (``MAKING`` or ``WRITING``), an ``error_type`` saying ``message``, in words that say
    what to try. ``paper`` is the page's; ``file_format`` ("PNG", "PDF", "Palette") and ``file_name`` the file being
    written, if one was."""
    smaller = _SMALLER.get(paper)
    if is_memory_error(error_type, message):
        if what == WRITING and file_format == "PNG":
            tip = f"a lower resolution, or a smaller paper ({smaller})" if smaller else "a lower resolution"
            return f"There wasn't enough memory to draw the PNG at this size. Try {tip}."
        tip = f"a smaller paper ({smaller}), " if smaller else ""
        large = " A page on A1 needs about 6 GB at Medium and 9 GB at Realistic." if paper == "A1" else ""
        return (f"There wasn't enough memory to make the page on {paper}.{large} Try {tip}an easier difficulty, or "
                f"fewer pixels across the brush (Generation > Advanced > Resolution); closing other apps can help too.")
    if what == WRITING and error_type is not None and issubclass(error_type, OSError):
        name = file_name or "the file"
        code = _errno(message)
        if issubclass(error_type, PermissionError) or code in (errno.EACCES, errno.EPERM):
            return (f"Couldn't write {name}: it may be open in another app, or the folder may not allow it. Close it, "
                    f"or export to another folder.")
        if code == errno.ENOSPC or "no space" in message.lower():
            return f"Couldn't write {name}: the disk is full. Free some space, or export to another drive."
        if issubclass(error_type, FileNotFoundError) or code == errno.ENOENT:
            return f"Couldn't write {name}: its folder no longer exists. Export to another folder."
        return f"Couldn't write {name}: {message}"
    if what == WRITING:
        return f"Couldn't write {file_name or 'the file'}: {message}"
    return f"The page couldn't be made: {message}"


def _errno(message: str) -> int | None:
    """The error number an OSError's message starts with ("[Errno 13] ..."), if it does."""
    if message.startswith("[Errno ") and "]" in message:
        try:
            return int(message[len("[Errno "): message.index("]")])
        except ValueError:
            return None
    if message.startswith("[WinError ") and "]" in message:
        code = message[len("[WinError "): message.index("]")]
        return {"5": errno.EACCES, "112": errno.ENOSPC, "3": errno.ENOENT}.get(code)
    return None

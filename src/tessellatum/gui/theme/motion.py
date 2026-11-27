"""Whether the system asks for reduced motion: animations are then skipped (their durations become 0).

Read where the platform says it: Windows' client-area animation setting, macOS's Reduce motion, GNOME's
enable-animations. None where it can't be read; a preference overrides it either way (see ``ThemeManager``).
"""

from __future__ import annotations

import subprocess
import sys

_SPI_GETCLIENTAREAANIMATION = 0x1042


def system_reduces_motion() -> bool | None:
    """True if the system asks for reduced motion, False if it doesn't, None if it can't be told."""
    try:
        if sys.platform == "win32":
            import ctypes

            enabled = ctypes.c_int(1)
            ok = ctypes.windll.user32.SystemParametersInfoW(_SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(enabled), 0)
            return None if not ok else not bool(enabled.value)
        if sys.platform == "darwin":
            out = _run(["defaults", "read", "com.apple.universalaccess", "reduceMotion"])
            return None if out is None else out.strip() == "1"
        out = _run(["gsettings", "get", "org.gnome.desktop.interface", "enable-animations"])
        return None if out is None else out.strip() == "false"
    except Exception:  # noqa: BLE001 - an unreadable setting is no setting
        return None


def _run(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None

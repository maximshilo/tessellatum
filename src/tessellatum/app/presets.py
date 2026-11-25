"""User presets: difficulties of one's own, saved by name beside the built-in ones.

A user preset holds what a built-in one does, the settings the difficulty owns (``Setting.preset_owned``): the paper,
what the page is colored on, the paints and the rest of the page's settings are never part of one, so applying it
never changes them. The presets are kept in one JSON file in the app's preferences folder, versioned:

    {"version": 1, "presets": {"Fine fur": {"num_colors": 30, "min_region_area_mm2": 4, ...}}}

A file that can't be read (broken JSON, another version, values out of range) is ignored with a warning in the log,
which the diagnostics show, and set aside as ``presets.json.bak`` the first time a preset is saved over it, so nothing
in it is lost. A preset missing a setting takes the built-in default preset's value for it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from tessellatum.core import difficulty, settings

log = logging.getLogger("tessellatum.presets")

VERSION = 1
FILE_NAME = "presets.json"


def owned() -> list[settings.Setting]:
    """The settings a preset sets."""
    return [s for s in settings.SETTINGS if s.preset_owned]


def reserved(name: str) -> bool:
    """Whether ``name`` is the built-in presets' (or Custom's), which a user preset can't take."""
    return name.strip().casefold() in {n.casefold() for n in difficulty.preset_names()}


class PresetStore:
    """The user presets, read from ``path`` and written back to it on every change; with no path, kept in memory only
    (tests, and a window that keeps nothing)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self._presets: dict[str, dict[str, float]] = {}
        self._unreadable = False
        self._read()

    # -- Reading ------------------------------------------------------------------
    def _read(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != VERSION:
                raise ValueError(f"not a version {VERSION} presets file")
            presets = data.get("presets")
            if not isinstance(presets, dict):
                raise ValueError("no presets in it")
            read = {}
            for name, values in presets.items():
                read[self._checked_name(name)] = self._checked_values(values)
        except (OSError, ValueError, TypeError) as exc:
            self._unreadable = True
            log.warning("The presets file %s can't be read, so its presets are left out: %s", self.path.name, exc)
            return
        self._presets = read

    @staticmethod
    def _checked_name(name) -> str:
        if not isinstance(name, str) or not name.strip() or reserved(name):
            raise ValueError(f"a preset can't be called {name!r}")
        return name.strip()

    @staticmethod
    def _checked_values(values) -> dict[str, float]:
        if not isinstance(values, dict):
            raise ValueError("a preset's values must be a mapping")
        defaults = settings.preset_values(difficulty.DEFAULT_PRESET)
        checked = {}
        for s in owned():
            value = values.get(s.name, defaults[s.name])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{s.name} is not a number")
            reason = s.check(float(value))
            if reason is not None:
                raise ValueError(f"{s.name}: {reason}")
            checked[s.name] = float(value)
        return checked

    # -- What it holds ----------------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._presets, key=str.casefold)

    def __contains__(self, name: str) -> bool:
        return name in self._presets

    def values(self, name: str) -> dict[str, float]:
        return dict(self._presets[name])

    def params(self, name: str) -> difficulty.DifficultyParams:
        return difficulty.custom_params(**self._presets[name])

    # -- Changing it ------------------------------------------------------------------
    def save(self, name: str, values: dict[str, float]) -> str:
        """Keep ``values`` (the preset-owned settings among them) as the preset ``name``, replacing one of that name.
        Returns the name as kept. Raises ``ValueError`` for a name a user preset can't have."""
        name = self._checked_name(name)
        self._presets[name] = self._checked_values({s.name: values[s.name] for s in owned()})
        self._write()
        return name

    def rename(self, old: str, new: str) -> str:
        new = self._checked_name(new)
        if new != old and new in self._presets:
            raise ValueError(f"there is a preset called {new} already")
        self._presets = {(new if name == old else name): values for name, values in self._presets.items()}
        self._write()
        return new

    def delete(self, name: str) -> None:
        del self._presets[name]
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._unreadable and self.path.exists():
            self.path.replace(self.path.with_name(self.path.name + ".bak"))  # what couldn't be read, kept aside
            self._unreadable = False
        data = {"version": VERSION, "presets": {name: self._presets[name] for name in self.names()}}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)

"""The release notes the release workflow publishes (``packaging/release_notes.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_notes", ROOT / "packaging" / "release_notes.py")
release_notes = importlib.util.module_from_spec(_spec)
sys.modules["release_notes"] = release_notes
_spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [1.2.0] - 2026-02-01

### Added

- A new setting, described over
  two lines.

## [1.1.1] - 2026-01-15

### Fixed

- A fix.

## [1.1.0] - 2026-01-01

- The last release.

[1.1.0]: https://example.invalid/v1.1.0
"""


def test_the_notes_of_one_version_are_its_section_unwrapped_then_the_downloads():
    text = release_notes.notes("1.2.0", CHANGELOG)
    assert text.startswith("### Added\n\n- A new setting, described over two lines.\n")
    assert "A fix" not in text
    for name in ("windows-x64.zip", "macos-arm64.zip", "linux-x64.tar.gz"):
        assert f"`Tessellatum-1.2.0-{name}`" in text


def test_the_notes_since_the_last_release_hold_every_version_after_it_newest_first():
    text = release_notes.notes("1.2.0", CHANGELOG, since="1.1.0")
    assert text.index("## 1.2.0 (2026-02-01)") < text.index("## 1.1.1 (2026-01-15)")
    assert "A fix." in text and "The last release" not in text and "example.invalid" not in text


def test_the_notes_since_the_version_before_are_its_section_alone():
    assert release_notes.notes("1.2.0", CHANGELOG, since="1.1.1") == release_notes.notes("1.2.0", CHANGELOG)


def test_a_version_missing_from_the_changelog_has_no_notes():
    with pytest.raises(SystemExit):
        release_notes.notes("1.3.0", CHANGELOG)


def test_the_changelog_has_notes_for_this_version_and_every_one_since_1_0_0():
    from tessellatum import __version__

    text = release_notes.notes(__version__, (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), since="1.0.0")
    major, minor, patch = (int(part) for part in __version__.split("."))
    if (major, minor) == (1, 0):  # every patch release since 1.0.0 has its section, none lost (as 1.0.7's once was)
        assert [f"## 1.0.{p} (" in text for p in range(1, patch + 1)] == [True] * patch

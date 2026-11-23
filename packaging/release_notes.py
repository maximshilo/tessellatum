"""Write the release notes for a version: its section of CHANGELOG.md, or every version's since the last release.

    python packaging/release_notes.py 1.0.15 notes.md
    python packaging/release_notes.py 1.0.15 notes.md --since 1.0.0

Used by the release workflow, which passes the previous release's version as ``--since``: the notes then hold every
version after it up to this one, newest first, each under its own heading, so someone updating from the last release
reads everything that changed. They end with how to download and run the app on Windows, macOS and Linux.

The notes are written to the file as UTF-8 (or printed, without a file): piped through a Windows shell, a console code
page would garble characters such as "²" and "–". Exits with an error if the changelog has no section for the version,
so a release is never published without notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

# A version's section: its heading, ``## [1.2.3] - date``, and what follows up to the next heading or link list.
SECTION = re.compile(r"^## \[([^\]]+)\]([^\n]*)\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)", re.M | re.S)


def notes(version: str, text: str, since: str | None = None) -> str:
    """The release notes for ``version`` from the changelog ``text``: its section, or with ``since``, the sections
    of every version after ``since`` up to ``version``; then the downloads (see ``downloads``)."""
    sections = {m.group(1): (m.group(2).strip(" -"), m.group(3).strip()) for m in SECTION.finditer(text)}
    if version not in sections:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    if since is None:
        shown = [version]
    else:
        shown = [v for v in sections if _key(since) < _key(v) <= _key(version)]
    if len(shown) == 1:
        body = unwrapped(sections[version][1])
    else:
        parts = [f"Everything that changed since {since}, newest first."]
        for v in shown:
            date = sections[v][0]
            parts.append(f"## {v}" + (f" ({date})" if date else "") + "\n\n" + unwrapped(sections[v][1]))
        body = "\n\n".join(parts)
    return f"{body}\n\n{downloads(version)}\n"


def downloads(version: str) -> str:
    """How to get and run the app on each platform the release workflow builds it for."""
    return "\n".join(
        [
            "## Download",
            "",
            "No installer or Python needed: unpack it anywhere and run it.",
            "",
            f"- **Windows** (64-bit): `Tessellatum-{version}-windows-x64.zip`. Unzip it and run "
            "`Tessellatum\\Tessellatum.exe`.",
            f"- **macOS** (Apple silicon): `Tessellatum-{version}-macos-arm64.zip`. Unzip it and move "
            "`Tessellatum.app` into Applications. The app isn't notarized by Apple, so macOS blocks it the first time: "
            "open it once, then click **Open Anyway** in System Settings > Privacy & Security (or run "
            "`xattr -dr com.apple.quarantine /Applications/Tessellatum.app` once).",
            f"- **Linux** (x86-64, glibc 2.35 or later, such as Ubuntu 22.04 and later): "
            f"`Tessellatum-{version}-linux-x64.tar.gz`. Unpack it with `tar -xzf` and run `Tessellatum/Tessellatum`. "
            "It uses the system's OpenGL, xkbcommon and fontconfig libraries, which desktops have.",
        ]
    )


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def unwrapped(markdown: str) -> str:
    """``markdown`` with each paragraph and list item on one line: a release page shows every line break."""
    out: list[str] = []
    for line in markdown.split("\n"):
        starts_block = not line.strip() or re.match(r"\s*(#|[-*] |\d+\. )", line)
        if out and out[-1].strip() and not out[-1].lstrip().startswith("#") and not starts_block:
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    since = None
    if "--since" in args:
        at = args.index("--since")
        if at + 1 >= len(args):
            raise SystemExit("--since needs the previous release's version")
        since = args[at + 1]
        args = args[:at] + args[at + 2 :]
    if len(args) not in (1, 2):
        raise SystemExit("usage: release_notes.py VERSION [OUTPUT_FILE] [--since PREVIOUS_VERSION]")
    text = notes(args[0], CHANGELOG.read_text(encoding="utf-8"), since)
    if len(args) == 2:
        Path(args[1]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

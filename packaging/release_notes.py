"""Write the release notes for one version: its section of CHANGELOG.md, without the heading.

    python packaging/release_notes.py 1.0.0 notes.md

Used by the release workflow. The notes are written to the file as UTF-8 (or printed, without a file): piped through
a Windows shell, a console code page would garble characters such as "²" and "–". Exits with an error if the changelog
has no section for the version, so a release is never published without notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def notes(version: str, text: str) -> str:
    match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)", text, re.M | re.S)
    if match is None:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    body = unwrapped(match.group(1).strip())
    download = (
        f"**Download:** `Tessellatum-{version}-windows-x64.zip` below. Unzip it anywhere and run "
        "`Tessellatum\\Tessellatum.exe`; no installer or Python needed."
    )
    return f"{body}\n\n{download}\n"


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
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: release_notes.py VERSION [OUTPUT_FILE]")
    text = notes(sys.argv[1], CHANGELOG.read_text(encoding="utf-8"))
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

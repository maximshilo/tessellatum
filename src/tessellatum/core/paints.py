"""The painter's own paints or pencils: read from what they type or paste, or from a palette file.

A paint is a name and the 8-bit sRGB color it paints (a swatch's color as a screen shows it). The name may be empty:
the legend then names the color as it names any other (see ``color_names``).

Typed or pasted, a paint is a line holding its color, as a hex code (``#5A6B2F`` or ``5A6B2F``, or ``#5AF`` short) or
as three numbers from 0 to 255 (``90 107 47`` or ``90, 107, 47``), and the rest of the line, before or after it, is its
name: ``Sap green #5A6B2F``, ``#5A6B2F sap green`` and ``90 107 47 Sap green`` are all one paint. Blank lines and lines
starting with ``#`` followed by a space or a letter that isn't a hex code (a comment) are skipped; a line with no color
is reported back, not guessed at.

A palette file is any of the formats the app saves (see ``swatches``): a GIMP palette (``.gpl``), an Adobe Swatch
Exchange file (``.ase``, its RGB colors; CMYK, Lab and gray ones are converted, roughly) or a Procreate palette
(``.swatches``, which carries no names).
"""

from __future__ import annotations

import colorsys
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Paint = tuple[str, tuple[int, int, int]]

_HEX = re.compile(r"(?<![0-9A-Za-z])#?([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})(?![0-9A-Za-z])")
_TRIPLE = re.compile(r"(?<![\d.])(\d{1,3})\s*[,;\s]\s*(\d{1,3})\s*[,;\s]\s*(\d{1,3})(?![\d.])")
# The most paints a set may hold: a 150-pencil box and then some. More are left off, and said so.
MOST_PAINTS = 256


@dataclass(frozen=True)
class ParsedPaints:
    """What a text held: its paints, and the lines that held no color (1-based line number, the line)."""

    paints: tuple[Paint, ...]
    rejected: tuple[tuple[int, str], ...]


def parse_paints(text: str) -> ParsedPaints:
    """The paints in ``text``, one a line (see the module's notes)."""
    paints: list[Paint] = []
    rejected: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or _is_comment(line):
            continue
        found = _color_in(line)
        if found is None:
            rejected.append((number, line))
            continue
        rgb, (start, end) = found
        name = " ".join((line[:start] + " " + line[end:]).replace("\t", " ").split()).strip(" -:,;|")
        paints.append((name, rgb))
    return ParsedPaints(tuple(paints), tuple(rejected))


def paints_text(paints: tuple[Paint, ...] | list[Paint]) -> str:
    """``paints`` as lines ``parse_paints`` reads back: ``#5A6B2F  Sap green``."""
    return "\n".join(f"#{r:02X}{g:02X}{b:02X}  {name}".rstrip() for name, (r, g, b) in paints)


def read_palette_file(path: Path) -> list[Paint]:
    """The paints in a palette file: ``.gpl``, ``.ase`` or ``.swatches``, by its suffix; anything else is read as
    text, a paint a line. Raises ValueError for a file that isn't what its suffix says."""
    path = Path(path)
    suffix = path.suffix.lower()
    data = path.read_bytes()
    try:
        if suffix == ".gpl":
            return _read_gpl(data.decode("utf-8", errors="replace"))
        if suffix == ".ase":
            return _read_ase(data)
        if suffix == ".swatches":
            return _read_procreate(data)
    except (ValueError, KeyError, IndexError, struct.error, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not a palette file the app can read: {exc}") from exc
    return list(parse_paints(data.decode("utf-8", errors="replace")).paints)


def distinct(paints: list[Paint] | tuple[Paint, ...]) -> tuple[Paint, ...]:
    """``paints`` without a color listed twice (the first name kept), ``MOST_PAINTS`` at most."""
    seen: set[tuple[int, int, int]] = set()
    kept: list[Paint] = []
    for name, rgb in paints:
        if rgb not in seen:
            seen.add(rgb)
            kept.append((name, rgb))
    return tuple(kept[:MOST_PAINTS])


def _is_comment(line: str) -> bool:
    if line.startswith("//"):
        return True
    if not line.startswith("#"):
        return False
    return _HEX.match(line) is None or line.startswith("# ")


def _color_in(line: str) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
    """The first color in ``line`` and where it lies: a hex code with its #, else three numbers 0-255."""
    hashed = re.search(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})(?![0-9A-Za-z])", line)
    if hashed is not None:
        return _from_hex(hashed.group(1)), hashed.span()
    # Three numbers: those that start or end the line, as a palette's columns do, rather than a number in a paint's
    # name ("Polychromos 120 90 107 47"); else the first three.
    triples = []
    for start in range(len(line)):
        found = _TRIPLE.match(line, start)
        if found is not None and (start == 0 or not line[start - 1].isdigit()):
            triples.append(found)
    triples = [t for t in triples if all(int(v) <= 255 for v in t.groups())]
    if triples:
        ends = [t for t in triples if t.start() == 0 or not line[t.end() :].strip()]
        triple = (ends or triples)[0]
        return tuple(int(v) for v in triple.groups()), triple.span()  # type: ignore[return-value]
    bare = _HEX.search(line)
    if bare is not None and len(bare.group(1)) == 6 and re.search(r"\d", bare.group(1)):
        return _from_hex(bare.group(1)), bare.span()  # "5A6B2F": a hex code without its #, digits in it
    return None


def _from_hex(code: str) -> tuple[int, int, int]:
    if len(code) == 3:
        code = "".join(c * 2 for c in code)
    return tuple(int(code[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _read_gpl(text: str) -> list[Paint]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "GIMP Palette":
        raise ValueError("no 'GIMP Palette' line")
    paints = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Name:") or line.startswith("Columns:"):
            continue
        parts = line.split(maxsplit=3)
        rgb = tuple(int(v) for v in parts[:3])
        if not all(0 <= v <= 255 for v in rgb):
            raise ValueError(f"a color out of range: {line!r}")
        paints.append((parts[3].strip() if len(parts) > 3 else "", rgb))
    return paints


def _read_ase(data: bytes) -> list[Paint]:
    if data[:4] != b"ASEF":
        raise ValueError("no ASEF signature")
    count = struct.unpack(">I", data[8:12])[0]
    at, paints = 12, []
    for _ in range(count):
        kind, length = struct.unpack(">HI", data[at : at + 6])
        body = data[at + 6 : at + 6 + length]
        at += 6 + length
        if kind != 0x0001:
            continue  # a group's start or end
        units = struct.unpack(">H", body[:2])[0]
        name = body[2 : 2 + 2 * units].decode("utf-16-be").rstrip("\x00")
        rest = body[2 + 2 * units :]
        model = rest[:4]
        if model == b"RGB ":
            rgb = struct.unpack(">3f", rest[4:16])
        elif model == b"CMYK":
            c, m, y, k = struct.unpack(">4f", rest[4:20])
            rgb = tuple((1 - v) * (1 - k) for v in (c, m, y))
        elif model == b"Gray":
            rgb = struct.unpack(">f", rest[4:8]) * 3
        elif model == b"LAB ":
            rgb = _lab_to_rgb(*struct.unpack(">3f", rest[4:16]))
        else:
            raise ValueError(f"an unknown color model {model!r}")
        paints.append((name, tuple(int(round(min(1.0, max(0.0, v)) * 255)) for v in rgb)))
    return paints


def _read_procreate(data: bytes) -> list[Paint]:
    import io

    with zipfile.ZipFile(io.BytesIO(data)) as package:
        document = json.loads(package.read("Swatches.json").decode("utf-8"))
    if isinstance(document, list):
        document = document[0]
    paints = []
    for swatch in document["swatches"]:
        if swatch is None:
            continue
        rgb = colorsys.hsv_to_rgb(float(swatch["hue"]), float(swatch["saturation"]), float(swatch["brightness"]))
        paints.append(("", tuple(int(round(v * 255)) for v in rgb)))
    return paints


def _lab_to_rgb(lightness: float, a: float, b: float) -> tuple[float, float, float]:
    """An ASE Lab color (L 0-1 for 0-100, a and b as they are, D50 as Adobe's are) as sRGB 0-1, roughly: D65 here."""
    from tessellatum.core.color import _SRGB_TO_XYZ, _lab_to_relative_xyz

    xyz = _lab_to_relative_xyz(np.array([lightness * 100, a, b])) * _SRGB_TO_XYZ.sum(axis=1)
    linear = np.linalg.solve(_SRGB_TO_XYZ, xyz)
    linear = np.clip(linear, 0, 1)
    rgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return tuple(float(v) for v in rgb)  # type: ignore[return-value]

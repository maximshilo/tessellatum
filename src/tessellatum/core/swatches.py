"""Save a page's legend as a palette file, to color it in a painting app with exactly its colors.

Three formats, each written as the published descriptions of it give it (Adobe and Procreate publish none of their
own; see ``wiki/User-Guide.md``):

- **GIMP palette** (``.gpl``), read by GIMP, Krita, Inkscape, Aseprite and others: text, a ``GIMP Palette`` line, then
  ``Name:`` and ``Columns:``, then a color a line, ``r g b`` (0-255) and its name (GIMP's own spec, "GIMP Palette
  Format Version 2").
- **Adobe Swatch Exchange** (``.ase``), read by Photoshop, Illustrator, InDesign, Affinity's apps, Procreate and
  others: big-endian binary, ``ASEF`` and version 1.0, a block count, then blocks of a type (``0xC001`` a group's
  start, ``0xC002`` its end, ``0x0001`` a color) and a length that leaves out the type and the length themselves. A
  name is a UTF-16BE string led by its length in code units, its terminating null counted; a color is its name, the
  model ``RGB `` and three float32 from 0 to 1, then its type, 2 for a normal (process) color.
- **Procreate swatches** (``.swatches``): a zip holding ``Swatches.json``, an object with the palette's ``name`` and
  its ``swatches``, each ``hue``, ``saturation`` and ``brightness`` from 0 to 1 (HSB), ``alpha`` 1 and ``colorSpace``
  0, as the MIT-licensed procreate-swatches library writes them. A Procreate palette holds 30 colors at most, so a
  larger legend is saved as several files of 30, each named for the colors it holds ("1-30", "31-60").

Each color is named by its number on the page and its name on the legend ("7 dusty rose"), so the app's palette reads
as the legend does, in its order.
"""

from __future__ import annotations

import colorsys
import json
import struct
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Sequence

from tessellatum.core.color_names import name_colors
from tessellatum.core.legend import plain

GPL = ".gpl"
ASE = ".ase"
PROCREATE = ".swatches"
FORMATS = {
    PROCREATE: "Procreate swatches (*.swatches)",
    ASE: "Adobe Swatch Exchange (*.ase)",
    GPL: "GIMP palette: GIMP, Krita, Inkscape (*.gpl)",
}
# A Procreate palette's colors at most: more are dropped on import, so a larger legend is split into palettes this big.
PROCREATE_MAX_SWATCHES = 30
# A GIMP palette's columns, as GIMP shows it: as many as a numbered-only legend has across an A4 sheet.
GPL_COLUMNS = 12


def entries(
    palette_rgb: Sequence[tuple[int, int, int]], names: Sequence[str] | None = None
) -> list[tuple[str, tuple[int, int, int]]]:
    """Each of ``palette_rgb`` (the legend's, in its order) with the name it is saved under: "7 dusty rose", its name
    from ``names`` (``color_names.name_colors``' if None), as the legend writes it."""
    rgb = [tuple(int(v) for v in color) for color in palette_rgb]
    names = name_colors(rgb) if names is None else [plain(name) for name in names]
    return [(f"{i + 1} {name}", color) for i, (name, color) in enumerate(zip(names, rgb))]


def gpl_bytes(title: str, colors: Sequence[tuple[str, tuple[int, int, int]]]) -> bytes:
    """A GIMP palette named ``title`` of ``colors`` (name, RGB)."""
    lines = ["GIMP Palette", f"Name: {_one_line(title)}", f"Columns: {GPL_COLUMNS}", "#"]
    lines += [f"{r:3d} {g:3d} {b:3d}\t{_one_line(name)}" for name, (r, g, b) in colors]
    return ("\n".join(lines) + "\n").encode("utf-8")


def ase_bytes(title: str, colors: Sequence[tuple[str, tuple[int, int, int]]]) -> bytes:
    """An Adobe Swatch Exchange file of ``colors`` (name, RGB), in a group named ``title``."""
    blocks = [_ase_block(0xC001, _ase_name(title))]
    for name, rgb in colors:
        values = struct.pack(">3f", *(v / 255 for v in rgb))
        blocks.append(_ase_block(0x0001, _ase_name(name) + b"RGB " + values + struct.pack(">H", 2)))
    blocks.append(_ase_block(0xC002, b""))
    return b"ASEF" + struct.pack(">HHI", 1, 0, len(blocks)) + b"".join(blocks)


def procreate_bytes(title: str, colors: Sequence[tuple[str, tuple[int, int, int]]]) -> bytes:
    """A Procreate palette named ``title`` of ``colors`` (name, RGB), at most ``PROCREATE_MAX_SWATCHES`` of them.

    Procreate keeps no names for its swatches, only their colors, in order."""
    if len(colors) > PROCREATE_MAX_SWATCHES:
        raise ValueError(f"a Procreate palette holds {PROCREATE_MAX_SWATCHES} colors at most, not {len(colors)}")
    swatches = []
    for _name, (r, g, b) in colors:
        hue, saturation, brightness = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        swatches.append(
            {"hue": hue, "saturation": saturation, "brightness": brightness, "alpha": 1, "colorSpace": 0}
        )
    document = json.dumps({"name": title, "swatches": swatches}, ensure_ascii=False)
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        # A fixed date, so that the same legend always gives the same bytes.
        package.writestr(zipfile.ZipInfo("Swatches.json", date_time=(2020, 1, 1, 0, 0, 0)), document)
    return archive.getvalue()


def save_palette(
    path: Path,
    palette_rgb: Sequence[tuple[int, int, int]],
    title: str | None = None,
    names: Sequence[str] | None = None,
) -> list[Path]:
    """Save the legend ``palette_rgb`` as a palette file of the format ``path``'s suffix names (one of ``FORMATS``),
    the palette called ``title`` (the file's name if not given), its colors ``names`` (see ``entries``). Returns the files written: one, or for a Procreate
    palette of more than 30 colors, one for each 30, named for the colors it holds ("page 1-30.swatches")."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"not a palette format: {path.suffix!r}")
    title = title or path.stem
    colors = entries(palette_rgb, names)
    if suffix == GPL:
        path.write_bytes(gpl_bytes(title, colors))
        return [path]
    if suffix == ASE:
        path.write_bytes(ase_bytes(title, colors))
        return [path]
    if len(colors) <= PROCREATE_MAX_SWATCHES:
        path.write_bytes(procreate_bytes(title, colors))
        return [path]
    written = []
    for start, target in zip(range(0, len(colors), PROCREATE_MAX_SWATCHES), palette_files(path, len(colors))):
        part = colors[start : start + PROCREATE_MAX_SWATCHES]
        span = f"{start + 1}-{start + len(part)}"
        target.write_bytes(procreate_bytes(f"{title} {span}", part))
        written.append(target)
    return written


def palette_files(path: Path, colors: int) -> list[Path]:
    """The files ``save_palette`` writes for a palette of ``colors`` colors at ``path``: ``path`` itself, or for a
    Procreate palette of more than 30 colors one for each 30, named for the colors it holds ("page 1-30.swatches")."""
    path = Path(path)
    if path.suffix.lower() != PROCREATE or colors <= PROCREATE_MAX_SWATCHES:
        return [path]
    spans = [(start + 1, min(start + PROCREATE_MAX_SWATCHES, colors)) for start in range(0, colors, PROCREATE_MAX_SWATCHES)]
    return [path.with_name(f"{path.stem} {first}-{last}{path.suffix}") for first, last in spans]


def _ase_block(kind: int, data: bytes) -> bytes:
    return struct.pack(">HI", kind, len(data)) + data


def _ase_name(name: str) -> bytes:
    """``name`` as an ASE string: its length in UTF-16 code units, the null that ends it counted, then the string."""
    encoded = name.encode("utf-16-be") + b"\x00\x00"
    return struct.pack(">H", len(encoded) // 2) + encoded


def _one_line(text: str) -> str:
    return " ".join(text.split())

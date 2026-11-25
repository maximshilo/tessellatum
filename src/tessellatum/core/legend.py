"""Build the color-index -> swatch legend shown alongside the coloring page.

Each color gets a swatch with its number in it and, as the legend's detail setting asks, its name (see
``color_names``) and its hex code and RGB values beside it, to find, buy or mix it by. The text's layout is worked out
here in millimeters, once, and drawn by both the raster legend (``render_legend``) and the PDF's legend sheet
(``export``), so the two wrap and place it alike.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.color_names import hex_code, name_colors
from tessellatum.core.print_size import MM_PER_INCH, PT_PER_INCH

# A swatch's size on paper, so that a legend prints as large on an export as on a preview of the same picture, and a
# number on it stays legible at 300 dpi: 12 mm swatches 3 mm apart, framed 0.5 mm wide, numbered 0.42 of a swatch high
# (14 pt). On an 1100 px preview that is about the 48 px swatch the legend used to have.
SWATCH_MM = 12.0
SWATCH_GAP_MM = 3.0
SWATCH_BORDER_MM = 0.5
LABEL_FONT_RATIO = 0.42

# What the legend says beside each swatch (the setting "Legend details").
NUMBER_ONLY = "Number only"
NAME = "Name"
NAME_HEX_RGB = "Name, hex and RGB"
# The name, then how to mix the color from the painter's paints (see ``mixing``), "3 Titanium white" a line; with no
# recipes to show, as NAME_HEX_RGB.
NAME_RECIPE = "Name and recipe"
LEGEND_DETAILS = (NUMBER_ONLY, NAME, NAME_HEX_RGB, NAME_RECIPE)

# A legend with colors mixed from the painter's paints (see ``quantize.mixed_palette``) has two sections, each under a
# heading in 9 pt, on a line 5 mm high: the paints, then the mixed colors.
PAINTS_HEADING = "Your paints"
MIXED_HEADING = "Mixed colors: mix each from your paints, in the parts given"
# Over recipes from the paints listed when the page's colors aren't the paints (Colors from: The picture): they are a
# guide to mixing the picture's colors, not colors the page was made of.
RECIPES_HEADING = "Recipes from your paints list: mix each in the parts given"
HEADING_PT = 9.0
HEADING_MM = 5.0
HEADING_BASELINE_MM = 4.5
DEFAULT_LEGEND_DETAIL = NAME_HEX_RGB

# The text beside a swatch, on paper: the name in 8 pt, wrapped onto two lines at most, and below it the hex code and
# RGB values in 7 pt, the smallest a legend prints (legible on a printed sheet, as small print on a label is). It
# starts 1.5 mm right of the swatch and takes a column 30.5 mm wide, which the longest values line, "#FFFFFF  RGB 255
# 255 255" (30.2 mm), fits; so a cell is 47 mm across, four to a portrait A4 sheet's 190 mm, and 64 colors take 16
# rows, 243 mm of its 277.
NAME_PT = 8.0
DETAIL_PT = 7.0
TEXT_GAP_MM = 1.5
TEXT_COLUMN_MM = 30.5
NAME_LINES = 2
# Lines one under the other are this many ems apart, baseline to baseline.
LINE_SPACING = 1.2
NAME_GRAY = 0
DETAIL_GRAY = 80

_MM_PER_PT = MM_PER_INCH / PT_PER_INCH
# The size the text is measured at, in pixels an em: large enough that rounding to whole pixels is a thousandth of one.
_MEASURING_EM_PX = 1000


@dataclass(frozen=True)
class LegendText:
    """A line of text beside a swatch: ``x_mm`` right of the swatch's left edge, its baseline ``baseline_mm`` below the
    swatch's top, ``size_mm`` an em, in the gray ``gray`` (0 black)."""

    text: str
    x_mm: float
    baseline_mm: float
    size_mm: float
    gray: int


def cell_mm(detail: str) -> float:
    """How wide a swatch's cell is with ``detail`` beside it, gap included, on paper."""
    text = TEXT_GAP_MM + TEXT_COLUMN_MM if detail != NUMBER_ONLY else 0.0
    return SWATCH_MM + SWATCH_GAP_MM + text


def detail_line(rgb: tuple[int, int, int]) -> str:
    """The hex code and RGB values of ``rgb``, as the legend writes them: ``#5A6B2F  RGB 90 107 47``."""
    return f"{hex_code(rgb)}  RGB {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}"


def swatch_texts(
    palette_rgb: Sequence[tuple[int, int, int]],
    detail: str,
    names: Sequence[str] | None = None,
    recipes: Sequence[Sequence[str]] | None = None,
) -> list[list[LegendText]]:
    """The lines of text beside each of ``palette_rgb``'s swatches with ``detail``, centered on the swatch's height:
    each color's name from ``names`` (``color_names.name_colors``' if None), in plain ASCII, as the PDF writes it, and
    with NAME_RECIPE its recipe's lines from ``recipes`` (see ``mixing.recipe_lines``), each cut to the column."""
    if detail == NUMBER_ONLY or len(palette_rgb) == 0:
        return [[] for _ in palette_rgb]
    names = [plain(name) for name in (name_colors(palette_rgb) if names is None else names)]
    if detail == NAME_RECIPE and not recipes:
        detail = NAME_HEX_RGB
    name_mm, detail_mm = NAME_PT * _MM_PER_PT, DETAIL_PT * _MM_PER_PT
    cap = _cap_height()
    texts = []
    for i, (rgb, name) in enumerate(zip(palette_rgb, names)):
        name_lines = 1 if detail == NAME_RECIPE else NAME_LINES
        lines = [(line, name_mm, NAME_GRAY) for line in _wrap(name, name_mm, TEXT_COLUMN_MM, name_lines)]
        if detail == NAME_HEX_RGB:
            lines.append((detail_line(rgb), detail_mm, DETAIL_GRAY))
        elif detail == NAME_RECIPE and recipes is not None and i < len(recipes):
            for step in recipes[i]:
                lines.append((_wrap(plain(step), detail_mm, TEXT_COLUMN_MM, 1)[0], detail_mm, DETAIL_GRAY))
        # Baselines one under the other; the block, from the first line's capitals to the last baseline, centered.
        baselines = [0.0]
        for _, next_size, _ in lines[1:]:
            baselines.append(baselines[-1] + LINE_SPACING * next_size)
        top = cap * lines[0][1]
        height = baselines[-1] + top
        offset = (SWATCH_MM - height) / 2 + top
        texts.append(
            [
                LegendText(text, SWATCH_MM + TEXT_GAP_MM, offset + baseline, size, gray)
                for (text, size, gray), baseline in zip(lines, baselines)
            ]
        )
    return texts


def plain(text: str) -> str:
    """``text`` in printable ASCII, on one line: accents dropped ("Céruléen" -> "Ceruleen"), any other letter a "?"."""
    decomposed = unicodedata.normalize("NFKD", " ".join(str(text).split()))
    kept = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c if " " <= c <= "~" else "?" for c in kept)


def text_width_mm(text: str, size_mm: float) -> float:
    """How far ``text`` runs, written ``size_mm`` an em, in the legend's font."""
    return _advance(text) * size_mm


def _wrap(text: str, size_mm: float, width_mm: float, most: int) -> list[str]:
    """``text`` broken between words into lines no wider than ``width_mm``, ``most`` of them at most. What won't fit
    on the last is left off at a word's end, marked "..." (even if the mark itself then runs over); a single word too
    long for a line is kept whole."""
    lines: list[str] = []
    words = text.split()
    for at, word in enumerate(words):
        if lines and text_width_mm(f"{lines[-1]} {word}", size_mm) <= width_mm:
            lines[-1] = f"{lines[-1]} {word}"
        elif len(lines) < most:
            lines.append(word)
        else:
            last = lines[-1]
            while " " in last and text_width_mm(f"{last} ...", size_mm) > width_mm:
                last = last.rsplit(" ", 1)[0]
            lines[-1] = f"{last} ..."
            break
    return lines or [""]


@lru_cache(maxsize=4096)
def _advance(text: str) -> float:
    """``text``'s advance, in ems."""
    return _font(_MEASURING_EM_PX).getlength(text) / _MEASURING_EM_PX


@lru_cache(maxsize=1)
def _cap_height() -> float:
    """How tall the font's capitals stand above the baseline, in ems."""
    return -_font(_MEASURING_EM_PX).getbbox("H", anchor="ls")[1] / _MEASURING_EM_PX


@lru_cache(maxsize=256)
def _font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def mixed_from(mixes: Sequence[Sequence[str] | None] | None) -> int | None:
    """Where a legend's mixed colors start (they follow its paints), or None if it has none: ``mixes`` gives each color's
    recipe as the legend writes it, None for a paint."""
    if not mixes:
        return None
    first = next((i for i, lines in enumerate(mixes) if lines is not None), None)
    if first is not None and any(lines is None for lines in mixes[first:]):
        raise ValueError("a legend's mixed colors come after its paints")
    return first


def legend_sections(
    palette_rgb: Sequence[tuple[int, int, int]],
    detail: str,
    names: Sequence[str] | None = None,
    recipes: Sequence[Sequence[str]] | None = None,
    mixes: Sequence[Sequence[str] | None] | None = None,
) -> list[tuple[str, int, int, str, list[list[LegendText]]]]:
    """A legend's sections: (heading, first color, past the last, the detail its cells have, each swatch's text) each.
    Without mixes, one section: with no heading, or under ``RECIPES_HEADING`` if it writes ``recipes``. With mixes, the
    paints under ``PAINTS_HEADING`` with ``detail``, and the mixed colors under ``MIXED_HEADING``, each with its name
    and its recipe from ``mixes``, whatever ``detail`` is (a mixed color can't be painted without it)."""
    names = list(name_colors(palette_rgb) if names is None else names)
    start = mixed_from(mixes)
    if start is None:
        heading = RECIPES_HEADING if detail == NAME_RECIPE and recipes else ""
        return [(heading, 0, len(palette_rgb), detail, swatch_texts(palette_rgb, detail, names, recipes))]
    sections = []
    if start > 0:
        paint_recipes = list(recipes[:start]) if recipes else None
        sections.append(
            (PAINTS_HEADING, 0, start, detail, swatch_texts(palette_rgb[:start], detail, names[:start], paint_recipes))
        )
    mixed_texts = swatch_texts(palette_rgb[start:], NAME_RECIPE, names[start:], list(mixes[start:]))
    sections.append((MIXED_HEADING, start, len(palette_rgb), NAME_RECIPE, mixed_texts))
    return sections


def render_legend(
    palette_bgr: np.ndarray,
    width: int,
    px_per_mm: float,
    detail: str = DEFAULT_LEGEND_DETAIL,
    names: Sequence[str] | None = None,
    recipes: Sequence[Sequence[str]] | None = None,
    mixes: Sequence[Sequence[str] | None] | None = None,
) -> Image.Image:
    """Render a wrapped grid of numbered color swatches, ``width`` pixels wide, at ``px_per_mm`` pixels a millimeter,
    with ``detail`` beside each (one of ``LEGEND_DETAILS``), the colors called ``names``, mixed as ``recipes`` say
    (see ``swatch_texts``). With ``mixes`` (each color's recipe as mixed colors are written, None for a paint), in
    two sections with headings (see ``legend_sections``): the paints, then the colors mixed from them."""
    palette_bgr = np.asarray(palette_bgr, dtype=np.uint8).reshape(-1, 3)
    if mixed_from(mixes) is None and not (detail == NAME_RECIPE and recipes):
        return _swatch_grid(palette_bgr, width, px_per_mm, detail, names, recipes)
    palette_rgb = [(int(bgr[2]), int(bgr[1]), int(bgr[0])) for bgr in palette_bgr]
    parts = []
    for heading, start, stop, cell_detail, texts in legend_sections(palette_rgb, detail, names, recipes, mixes):
        if heading:
            parts.append(_heading_strip(heading, width, px_per_mm))
        parts.append(_swatch_grid(palette_bgr[start:stop], width, px_per_mm, cell_detail, texts=texts, first=start + 1))
    legend = Image.new("RGB", (width, sum(part.height for part in parts)), "white")
    y = 0
    for part in parts:
        legend.paste(part, (0, y))
        y += part.height
    return legend


def _heading_strip(heading: str, width: int, px_per_mm: float) -> Image.Image:
    """A section's heading, ``HEADING_MM`` high on paper, in line with the swatches below it."""
    strip = Image.new("RGB", (width, max(1, round(HEADING_MM * px_per_mm))), "white")
    ImageDraw.Draw(strip).text(
        (max(2, round(SWATCH_GAP_MM * px_per_mm)), HEADING_BASELINE_MM * px_per_mm),
        heading,
        fill=(NAME_GRAY,) * 3,
        font=_font(HEADING_PT * _MM_PER_PT * px_per_mm),
        anchor="ls",
    )
    return strip


def _swatch_grid(
    palette_bgr: np.ndarray,
    width: int,
    px_per_mm: float,
    detail: str = DEFAULT_LEGEND_DETAIL,
    names: Sequence[str] | None = None,
    recipes: Sequence[Sequence[str]] | None = None,
    texts: list[list[LegendText]] | None = None,
    first: int = 1,
) -> Image.Image:
    """The swatches of ``palette_bgr`` in a wrapped grid, numbered from ``first``, with ``texts`` beside them (as
    ``swatch_texts`` gives them for ``detail``, ``names`` and ``recipes`` if None)."""
    swatch = max(8, round(SWATCH_MM * px_per_mm))
    gap = max(2, round(SWATCH_GAP_MM * px_per_mm))
    border = max(1, round(SWATCH_BORDER_MM * px_per_mm))
    cell_w = swatch + gap + round((cell_mm(detail) - SWATCH_MM - SWATCH_GAP_MM) * px_per_mm)
    cols = max(1, width // cell_w)
    rows = (len(palette_bgr) + cols - 1) // cols
    height = rows * (swatch + gap) + gap

    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default(size=round(swatch * LABEL_FONT_RATIO))
    palette_rgb = [(int(bgr[2]), int(bgr[1]), int(bgr[0])) for bgr in palette_bgr]
    if texts is None:
        texts = swatch_texts(palette_rgb, detail, names, recipes)

    for index, rgb in enumerate(palette_rgb):
        row, col = divmod(index, cols)
        x0 = gap + col * cell_w
        y0 = gap + row * (swatch + gap)
        x1, y1 = x0 + swatch, y0 + swatch

        draw.rectangle([x0, y0, x1, y1], fill=rgb, outline="black", width=border)

        text = str(index + first)
        text_color = number_fill(rgb)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + swatch / 2 - tw / 2 - bbox[0], y0 + swatch / 2 - th / 2 - bbox[1]),
            text,
            fill=text_color,
            font=font,
        )
        for line in texts[index]:
            draw.text(
                (x0 + line.x_mm * px_per_mm, y0 + line.baseline_mm * px_per_mm),
                line.text,
                fill=(line.gray,) * 3,
                font=_font(line.size_mm * px_per_mm),
                anchor="ls",
            )

    return legend


def number_fill(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """What a swatch of ``rgb`` has its number written in: white on a dark swatch, black on a light one."""
    return (255, 255, 255) if _luminance(rgb) < 140 else (0, 0, 0)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

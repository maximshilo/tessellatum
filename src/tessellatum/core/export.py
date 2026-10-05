"""Save a generated coloring page and its legend as a PNG or a PDF, sized for printing on A4 (see ``print_size``)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from tessellatum.core.boundaries import region_outlines
from tessellatum.core.ink_outline import ink_outline
from tessellatum.core.labels import baseline_bbox, font
from tessellatum.core.legend import LABEL_FONT_RATIO, SWATCH_BORDER_MM, SWATCH_GAP_MM, SWATCH_MM, number_fill
from tessellatum.core.painting import Painting, Version, tint_rgb
from tessellatum.core.print_size import A4, MM_PER_INCH, PT_PER_INCH, print_scale
from tessellatum.core.render import PageDrawing, number_origin

# The space between the page and the legend in a PNG, on paper.
LEGEND_GAP_MM = 6.0
# How wide a painted version's fills are stroked round their edge, in their own color, on paper. Two fills meet along
# the same line, but a viewer that smooths each one's edge on its own leaves a hairline of paper between them, which
# this covers. It moves an edge by half that, 0.05 mm, less than a pixel of a page at 300 dpi.
FILL_EDGE_MM = 0.1

_PT_PER_MM = PT_PER_INCH / MM_PER_INCH
# The size Pillow's measurements of a number's text box are taken at, in pixels an em: large enough that rounding them
# to whole pixels is a thousandth of the em.
_MEASURING_EM_PX = 1000


def save_png(page: Image.Image, legend: Image.Image, path: Path) -> None:
    """Stack the page and its legend into one PNG (a PNG has no pages).

    The file records the page's print resolution, so that it prints the page as
    large as the print model places it on A4.
    """
    scale = print_scale(page.size)
    gap = round(scale.mm_to_px(LEGEND_GAP_MM))
    width = max(page.width, legend.width)
    height = page.height + gap + legend.height
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(page, ((width - page.width) // 2, 0))
    combined.paste(legend, ((width - legend.width) // 2, page.height + gap))
    combined.save(path, "PNG", dpi=(scale.dpi, scale.dpi))


def save_pdf(
    drawing: PageDrawing,
    palette_rgb: Sequence[tuple[int, int, int]],
    path: Path,
    version: Version = Version.PAGE,
    painting: Painting | None = None,
) -> None:
    """Save as two A4 sheets of vector art: the coloring page, then its legend.

    The page prints as the print model places it: scaled to fill the printable
    area inside the margins, centered, on a landscape sheet when it is wider
    than tall. It is drawn again from what the page was drawn from (see
    ``render.PageDrawing``), so that a printer draws it at its own resolution
    rather than at the page's pixels:

    - the lines as paths, stroked with a round pen of the style's width on
      paper -- never widened to a pixel, as the page is where a pixel is
      coarser than the line;
    - the numbers and their leaders the same way, the numbers as text in the
      font the page writes them in, embedded;
    - the printed ink -- line art's own, the marks in a face -- as the outline
      of its pixels, filled, smoothed off their staircase without joining or
      breaking anything the page keeps apart or together (see
      ``ink_outline``);
    - the letters of signs and captions as their outline, filled: traced
      between the page's pixels, so they print as smooth as the source shows
      them (see ``text.lettering``).

    Nothing on the sheet is an image.

    ``version`` is the version of the page the first sheet prints (see
    ``painting``), painted with ``painting`` unless it is the page itself. Its
    regions are filled along a path round each one made of the page's lines
    (see ``boundaries.region_outlines``), so that two neighbors meet on the
    same line:

    - the completed version fills every region with its paint and prints the
      letters and the ink over it, solid, as above; no line and no number;
    - the tinted version fills them with their wash (``painting.tint_rgb``) and
      prints the page over it, drawn as above, in the Multiply blend mode: the
      page's grays multiplied by the wash, as on the tinted image.

    The legend gets a portrait sheet of its own, across the printable width
    from the top margin: its swatches, ``legend.SWATCH_MM`` squares in their
    exact colors, numbered as the page is. The file carries no date, so the
    same page always gives the same bytes.
    """
    document = _Document()
    numbers = document.font(_number_font())
    _page_sheet(document, drawing, numbers, version, painting)
    _legend_sheet(document, palette_rgb, numbers)
    Path(path).write_bytes(document.to_bytes())


def _sheet_size_mm(landscape: bool) -> tuple[float, float]:
    return (A4.height_mm, A4.width_mm) if landscape else (A4.width_mm, A4.height_mm)


def _page_sheet(
    document: _Document, drawing: PageDrawing, numbers: int, version: Version, painting: Painting | None
) -> None:
    """The page's sheet, drawn in the page's pixels, y down, mapped onto where the print model puts the page."""
    width, height = drawing.size
    scale = print_scale(drawing.size)
    sheet = _sheet_size_mm(scale.landscape)
    printed_w, printed_h = scale.printed_size_mm
    left, top = (sheet[0] - printed_w) / 2, (sheet[1] - printed_h) / 2
    pt_per_px = _PT_PER_MM / scale.px_per_mm
    if version is not Version.PAGE and painting is None:
        raise ValueError(f"the {version.value.lower()} version needs what the page is painted with")

    ops = [
        f"q {_num(pt_per_px, 8)} 0 0 {_num(-pt_per_px, 8)} {_num(left * _PT_PER_MM)} {_num((sheet[1] - top) * _PT_PER_MM)} cm",
        # Half of a line on the page's edge falls off the paper, as on the page.
        f"0 0 {width} {height} re W n",
    ]
    if version is Version.PAGE:
        darken = document.add(b"<< /Type /ExtGState /BM /Darken >>")
        resources = {"Font": {"F1": numbers}, "ExtGState": {"Dk": darken}}
        ops += _page_ops(drawing, scale.px_per_mm)
    elif version is Version.COMPLETED:
        resources = {}
        ops += _fill_ops(painting, painting.palette_rgb, scale.px_per_mm)
        ops += _picture_ink_ops(drawing)
    else:
        darken = document.add(b"<< /Type /ExtGState /BM /Darken >>")
        multiply = document.add(b"<< /Type /ExtGState /BM /Multiply >>")
        # The page, drawn on its own as on its sheet -- an isolated group -- then multiplied into the wash.
        page = document.stream(
            f"/Type /XObject /Subtype /Form /BBox [0 0 {width} {height}] "
            "/Group << /S /Transparency /I true /CS /DeviceRGB >> "
            f"/Resources << /Font << /F1 {numbers} 0 R >> /ExtGState << /Dk {darken} 0 R >> >> ",
            "\n".join(_page_ops(drawing, scale.px_per_mm)).encode("latin-1"),
        )
        resources = {"ExtGState": {"Mu": multiply}, "XObject": {"Pg": page}}
        ops += _fill_ops(painting, [tint_rgb(rgb) for rgb in painting.palette_rgb], scale.px_per_mm)
        ops.append("q /Mu gs /Pg Do Q")
    ops.append("Q")
    document.page(sheet, "\n".join(ops), resources)


def _page_ops(drawing: PageDrawing, px_per_mm: float) -> list[str]:
    """The page as it prints, in its pixels: its lines, letters, ink, leaders and numbers, with the resources /F1 (the
    numbers' font) and /Dk (the Darken blend mode)."""
    style = drawing.style
    line_width = style.line_width_mm * px_per_mm
    ops = [f"1 J 1 j {_num(line_width)} w"]
    if drawing.strokes:
        ops += [f"{_gray(style.line_gray)} G", _path(drawing.strokes, offset=0.5), "S"]
    if drawing.lettering:
        # The letters, the darker of them and the lines, as the page prints them.
        ops += [f"q /Dk gs {_gray(drawing.ink_gray)} g", _path(drawing.lettering, offset=0.5), "f* Q"]
    if drawing.ink is not None and drawing.ink.any():
        ops += [f"{_gray(drawing.ink_gray)} g", _path(ink_outline(drawing.ink), offset=0.5), "f*"]
    leaders = [label.leader for label in drawing.labels if label.leader is not None]
    if leaders:
        dot = line_width * style.leader_dot_ratio / 2
        ops += [f"q /Dk gs {_gray(style.label_gray)} G {_gray(style.label_gray)} g"]
        ops += [_path([np.array(leader, dtype=np.float64)], offset=0.5) + " S" for leader in leaders]
        ops += [_disk(x + 0.5, y + 0.5, dot) for _, (x, y) in leaders]
        ops.append("Q")
    if drawing.labels:
        ops.append(f"{_gray(style.label_gray)} g BT")
        for label in drawing.labels:
            x, y = number_origin(label)
            ops.append(f"/F1 {label.font_size} Tf 1 0 0 -1 {_num(x)} {_num(y)} Tm {_text(label.text)} Tj")
        ops.append("ET")
    return ops


def _fill_ops(painting: Painting, colors: Sequence[tuple[int, int, int]], px_per_mm: float) -> list[str]:
    """Every region filled in its paint's entry of ``colors``, one path a paint, its edge stroked ``FILL_EDGE_MM`` wide.

    A paint's path is the rings of all its regions, filled by the even-odd rule: regions never overlap, so a point in
    one is inside its rings alone, and two regions of one paint are both filled even where they touch.
    """
    rings: dict[int, list[np.ndarray]] = {}
    region_color = np.asarray(painting.region_color)
    for region, outline in sorted(region_outlines(painting.region_id_map).items()):
        rings.setdefault(int(region_color[region]), []).extend(outline)
    ops = [f"1 J 1 j {_num(FILL_EDGE_MM * px_per_mm)} w"]
    for paint in sorted(rings):
        rgb = " ".join(_gray(v) for v in colors[paint])
        ops += [f"{rgb} rg {rgb} RG", _path(rings[paint], offset=0.5), "B*"]
    return ops


def _picture_ink_ops(drawing: PageDrawing) -> list[str]:
    """What the page prints of the picture itself, solid in its ink's gray: the letters, then the ink."""
    ops = []
    if drawing.lettering:
        ops += [f"{_gray(drawing.ink_gray)} g", _path(drawing.lettering, offset=0.5), "f*"]
    if drawing.ink is not None and drawing.ink.any():
        ops += [f"{_gray(drawing.ink_gray)} g", _path(ink_outline(drawing.ink), offset=0.5), "f*"]
    return ops


def _legend_sheet(document: _Document, palette_rgb: Sequence[tuple[int, int, int]], numbers: int) -> None:
    """The legend's portrait sheet, drawn in millimeters from the top-left margin, y down."""
    area_w, area_h = A4.printable_mm()
    cell = SWATCH_MM + SWATCH_GAP_MM
    columns = max(1, int(area_w // cell))
    rows = -(-len(palette_rgb) // columns)
    fit = min(1.0, area_h / (rows * cell + SWATCH_GAP_MM))  # a legend never runs past the sheet; 40 colors take a quarter of it
    scale = _PT_PER_MM * fit
    em = SWATCH_MM * LABEL_FONT_RATIO
    ops = [f"q {_num(scale, 8)} 0 0 {_num(-scale, 8)} {_num(A4.margin_mm * _PT_PER_MM)} {_num((A4.height_mm - A4.margin_mm) * _PT_PER_MM)} cm"]
    for index, rgb in enumerate(palette_rgb):
        row, column = divmod(index, columns)
        x, y = SWATCH_GAP_MM + column * cell, SWATCH_GAP_MM + row * cell
        inset = SWATCH_BORDER_MM / 2  # the frame lies inside the swatch, as on the page's legend
        ops.append(f"{' '.join(_gray(v) for v in rgb)} rg {_num(x)} {_num(y)} {_num(SWATCH_MM)} {_num(SWATCH_MM)} re f")
        ops.append(
            f"0 G {_num(SWATCH_BORDER_MM)} w {_num(x + inset)} {_num(y + inset)} "
            f"{_num(SWATCH_MM - SWATCH_BORDER_MM)} {_num(SWATCH_MM - SWATCH_BORDER_MM)} re S"
        )
        # The number's text box is centered on the swatch, as on the page's legend: Pillow's box, which runs across from
        # where the pen starts to where it ends, and up and down over the ink.
        text = str(index + 1)
        box_x0, box_y0, box_x1, box_y1 = (v / _MEASURING_EM_PX for v in baseline_bbox(text, _MEASURING_EM_PX))
        origin_x = x + SWATCH_MM / 2 - (box_x0 + box_x1) / 2 * em
        origin_y = y + SWATCH_MM / 2 - (box_y0 + box_y1) / 2 * em
        ops.append(
            f"{_gray(number_fill(rgb)[0])} g BT /F1 {_num(em)} Tf 1 0 0 -1 {_num(origin_x)} {_num(origin_y)} Tm "
            f"{_text(text)} Tj ET"
        )
    ops.append("Q")
    document.page(_sheet_size_mm(False), "\n".join(ops), {"Font": {"F1": numbers}})


def _path(paths: list[np.ndarray], offset: float) -> str:
    """Path construction for ``paths``, Nx2 (x, y) points each, one subpath apiece, moved by ``offset`` in x and y.

    The page's lines, leaders and ink outline put pixel centers at integer
    coordinates; the sheet's image of the page puts them half a pixel in, so
    their ``offset`` is 0.5. Two decimals: a hundredth of a pixel is a
    hundredth of 0.1-0.4 mm, far finer than a printer's dot.
    """
    points = np.concatenate(paths) + offset
    words = [f"{x:.2f} {y:.2f} l" for x, y in points.tolist()]
    at = 0
    for path in paths:
        words[at] = words[at][:-1] + "m"
        at += len(path)
    return "\n".join(words)


def _disk(x: float, y: float, radius: float) -> str:
    """A filled disk of ``radius`` around (x, y), as four Bézier quarters."""
    r, k = radius, radius * 0.5522847498  # k: the control points' distance that best fits a quarter circle
    points = [
        (x + r, y),
        (x + r, y + k), (x + k, y + r), (x, y + r),
        (x - k, y + r), (x - r, y + k), (x - r, y),
        (x - r, y - k), (x - k, y - r), (x, y - r),
        (x + k, y - r), (x + r, y - k), (x + r, y),
    ]  # fmt: skip
    words = [f"{_num(px)} {_num(py)}" for px, py in points]
    quarters = [" ".join(words[i : i + 3]) + " c" for i in range(1, 13, 3)]
    return f"{words[0]} m " + " ".join(quarters) + " f"


def _gray(level: int) -> str:
    """A level of 0-255 as a PDF color component, 0-1, to as many places as round back to the same level."""
    return _num(int(level) / 255)


def _text(text: str) -> str:
    """``text`` as a PDF string: the page's numbers are digits, which the embedded font's widths cover."""
    if not text.isdigit():
        raise ValueError(f"the PDF writes digits only, got {text!r}")
    return f"({text})"


def _num(value: float, places: int = 4) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


@dataclass(frozen=True)
class _TrueType:
    """What a PDF needs to know about a TrueType font to embed it: its bytes, metrics and the digits' advances."""

    data: bytes
    name: str  # its PostScript name
    units_per_em: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1, in font units
    ascent: int
    descent: int
    cap_height: int
    digit_advances: tuple[int, ...]  # the advance widths of "0" to "9", in font units

    @classmethod
    def parse(cls, data: bytes) -> _TrueType:
        tables = {}
        for i in range(struct.unpack(">H", data[4:6])[0]):
            tag, _checksum, offset, length = struct.unpack(">4sIII", data[12 + 16 * i : 28 + 16 * i])
            tables[tag.decode("latin-1")] = (offset, length)
        head, hhea, hmtx, os2 = (tables[t][0] for t in ("head", "hhea", "hmtx", "OS/2"))
        units_per_em = struct.unpack(">H", data[head + 18 : head + 20])[0]
        bbox = struct.unpack(">4h", data[head + 36 : head + 44])
        ascent, descent = struct.unpack(">hh", data[hhea + 4 : hhea + 8])
        metrics = struct.unpack(">H", data[hhea + 34 : hhea + 36])[0]
        os2_version = struct.unpack(">H", data[os2 : os2 + 2])[0]
        cap_height = struct.unpack(">h", data[os2 + 88 : os2 + 90])[0] if os2_version >= 2 else round(0.7 * units_per_em)
        glyphs = _cmap_bmp(data, tables["cmap"][0])
        advances = []
        for digit in "0123456789":
            glyph = min(glyphs[ord(digit)], metrics - 1)  # glyphs past the last metric take its advance
            advances.append(struct.unpack(">H", data[hmtx + 4 * glyph : hmtx + 4 * glyph + 2])[0])
        return cls(data, _postscript_name(data, tables["name"][0]), units_per_em, bbox, ascent, descent, cap_height, tuple(advances))

    def em(self, units: float) -> str:
        """A length in font units, in the thousandths of an em a PDF gives a font's metrics in."""
        return _num(units * 1000 / self.units_per_em, 3)


def _cmap_bmp(data: bytes, cmap: int) -> dict[int, int]:
    """The glyph of each character in the font's Windows Unicode (format 4) character map."""
    for i in range(struct.unpack(">H", data[cmap + 2 : cmap + 4])[0]):
        platform, encoding, offset = struct.unpack(">HHI", data[cmap + 4 + 8 * i : cmap + 12 + 8 * i])
        if (platform, encoding) == (3, 1):
            break
    else:
        raise ValueError("the font has no Windows Unicode character map")
    at = cmap + offset
    if struct.unpack(">H", data[at : at + 2])[0] != 4:
        raise ValueError("the font's Windows Unicode character map is not format 4")
    segments = struct.unpack(">H", data[at + 6 : at + 8])[0] // 2
    ends_at, starts_at = at + 14, at + 16 + 2 * segments
    deltas_at, ranges_at = starts_at + 2 * segments, starts_at + 4 * segments
    ends = struct.unpack(f">{segments}H", data[ends_at : ends_at + 2 * segments])
    starts = struct.unpack(f">{segments}H", data[starts_at : starts_at + 2 * segments])
    deltas = struct.unpack(f">{segments}H", data[deltas_at : deltas_at + 2 * segments])
    ranges = struct.unpack(f">{segments}H", data[ranges_at : ranges_at + 2 * segments])
    glyphs = {}
    for k in range(segments):
        for char in range(starts[k], ends[k] + 1):
            if char == 0xFFFF:
                continue
            if ranges[k] == 0:
                glyphs[char] = (char + deltas[k]) & 0xFFFF
            else:
                where = ranges_at + 2 * k + ranges[k] + 2 * (char - starts[k])
                glyph = struct.unpack(">H", data[where : where + 2])[0]
                glyphs[char] = (glyph + deltas[k]) & 0xFFFF if glyph else 0
    return glyphs


def _postscript_name(data: bytes, name: int) -> str:
    """The font's PostScript name (name ID 6), as a PDF name may hold it."""
    count, strings = struct.unpack(">HH", data[name + 2 : name + 6])
    for i in range(count):
        platform, _encoding, _language, name_id, length, offset = struct.unpack(">6H", data[name + 6 + 12 * i : name + 18 + 12 * i])
        if name_id == 6:
            raw = data[name + strings + offset : name + strings + offset + length]
            text = raw.decode("utf-16-be" if platform in (0, 3) else "latin-1")
            return "".join(c for c in text if c.isascii() and c.isalnum() or c in "-_") or "Numbers"
    return "Numbers"


@lru_cache(maxsize=1)
def _number_font() -> _TrueType:
    """The font the page's numbers are written in (see ``labels.font``)."""
    data = getattr(font(_MEASURING_EM_PX), "font_bytes", None)
    if not data:
        raise RuntimeError("the numbers' font is not a TrueType font Pillow loaded from bytes; it can't be embedded")
    return _TrueType.parse(data)


class _Document:
    """A PDF being written: objects numbered from 1 in the order added, the catalog and page tree first."""

    def __init__(self) -> None:
        self._objects: list[bytes] = [b"", b""]  # 1 is the catalog and 2 the page tree, filled in last
        self._pages: list[int] = []

    def add(self, body: bytes) -> int:
        self._objects.append(body)
        return len(self._objects)

    def stream(self, entries: str, data: bytes) -> int:
        """A Flate-compressed stream; ``entries`` are its dictionary's own, each followed by a space."""
        packed = zlib.compress(data)
        return self.add(f"<< {entries}/Filter /FlateDecode /Length {len(packed)} >>\nstream\n".encode() + packed + b"\nendstream")

    def font(self, face: _TrueType) -> int:
        """``face`` embedded whole, as a simple font of the digits."""
        program = self.stream(f"/Length1 {len(face.data)} ", face.data)
        descriptor = self.add(
            (
                f"<< /Type /FontDescriptor /FontName /{face.name} /Flags 32 /FontBBox [{' '.join(face.em(v) for v in face.bbox)}] "
                f"/ItalicAngle 0 /Ascent {face.em(face.ascent)} /Descent {face.em(face.descent)} "
                f"/CapHeight {face.em(face.cap_height)} /StemV 80 /FontFile2 {program} 0 R >>"
            ).encode()
        )
        return self.add(
            (
                f"<< /Type /Font /Subtype /TrueType /BaseFont /{face.name} /FirstChar 48 /LastChar 57 "
                f"/Widths [{' '.join(face.em(v) for v in face.digit_advances)}] /FontDescriptor {descriptor} 0 R "
                "/Encoding /WinAnsiEncoding >>"
            ).encode()
        )

    def page(self, sheet_mm: tuple[float, float], content: str, resources: dict[str, dict[str, int]]) -> None:
        contents = self.stream("", content.encode("latin-1"))
        listed = " ".join(f"/{kind} << {' '.join(f'/{name} {ref} 0 R' for name, ref in refs.items())} >>" for kind, refs in resources.items())
        width, height = (v * _PT_PER_MM for v in sheet_mm)
        self._pages.append(
            self.add(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(width)} {_num(height)}] "
                f"/Resources << {listed} >> /Contents {contents} 0 R >>".encode()
            )
        )

    def to_bytes(self) -> bytes:
        self._objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        self._objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in self._pages)}] /Count {len(self._pages)} >>".encode()
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(self._objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(self._objects) + 1}\n0000000000 65535 f \n".encode()
        out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
        out += f"trailer\n<< /Size {len(self._objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return bytes(out)

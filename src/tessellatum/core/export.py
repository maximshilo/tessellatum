"""Save a generated coloring page and its legend as a PNG or a PDF, sized for printing on A4 (see ``print_size``)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from tessellatum.core.legend import render_legend
from tessellatum.core.print_size import A4, MM_PER_INCH, PRINT_DPI, PT_PER_INCH, print_scale

# The space between the page and the legend in a PNG, on paper.
LEGEND_GAP_MM = 6.0

_PT_PER_MM = PT_PER_INCH / MM_PER_INCH


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


def save_pdf(page: Image.Image, palette_rgb: Sequence[tuple[int, int, int]], path: Path) -> None:
    """Save as two A4 sheets, the coloring page and then its legend, both stored losslessly.

    The page prints as the print model places it, at its own resolution:
    scaled to fill the printable area inside the margins, centered, on a
    landscape sheet when it is wider than tall. The legend is drawn again for a
    portrait sheet of its own, across its printable width from the top margin,
    its swatches sized in millimeters (see ``legend``), at 300 dpi whatever the
    page's resolution: it is drawn from the palette, so a small picture's
    legend prints as crisply as a large one's. The file carries no date, so the
    same page always gives the same bytes.
    """
    scale = print_scale(page.size)
    page_sheet = _sheet_size_mm(scale.landscape)
    printed_w, printed_h = scale.printed_size_mm
    sheets = [
        _Placed(page_sheet, page, ((page_sheet[0] - printed_w) / 2, (page_sheet[1] - printed_h) / 2), (printed_w, printed_h))
    ]

    area_w, area_h = A4.printable_mm()
    palette_bgr = np.array([rgb[::-1] for rgb in palette_rgb], dtype=np.uint8).reshape(-1, 3)
    px_per_mm = PRINT_DPI / MM_PER_INCH
    legend = render_legend(palette_bgr, round(area_w * px_per_mm), px_per_mm)
    legend_w, legend_h = legend.width / px_per_mm, legend.height / px_per_mm
    fit = min(1.0, area_h / legend_h)  # a legend never runs past the sheet; 40 colors take a quarter of it
    sheets.append(_Placed(_sheet_size_mm(False), legend, (A4.margin_mm, A4.margin_mm), (legend_w * fit, legend_h * fit)))
    Path(path).write_bytes(_pdf(sheets))


def _sheet_size_mm(landscape: bool) -> tuple[float, float]:
    return (A4.height_mm, A4.width_mm) if landscape else (A4.width_mm, A4.height_mm)


@dataclass(frozen=True)
class _Placed:
    """An image on a sheet: the sheet's (width, height), and the image's top-left corner and (width, height), in mm."""

    sheet_mm: tuple[float, float]
    image: Image.Image
    at_mm: tuple[float, float]
    size_mm: tuple[float, float]


def _pdf(sheets: list[_Placed]) -> bytes:
    """A PDF with one sheet per ``_Placed``, its image Flate-compressed, in gray where it has no color."""
    # Object 1 is the catalog, 2 the page tree, and each sheet takes three more: its page, its content and its image.
    objects: list[bytes] = [b"", b""]
    kids = []
    for sheet in sheets:
        page_id, content_id, image_id = len(objects) + 1, len(objects) + 2, len(objects) + 3
        kids.append(page_id)
        sheet_w, sheet_h = (v * _PT_PER_MM for v in sheet.sheet_mm)
        x, y = sheet.at_mm[0] * _PT_PER_MM, (sheet.sheet_mm[1] - sheet.at_mm[1] - sheet.size_mm[1]) * _PT_PER_MM
        w, h = (v * _PT_PER_MM for v in sheet.size_mm)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(sheet_w)} {_num(sheet_h)}] "
            f"/Resources << /XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(_stream(b"", f"q {_num(w)} 0 0 {_num(h)} {_num(x)} {_num(y)} cm /Im0 Do Q".encode()))
        pixels = np.asarray(sheet.image.convert("RGB"))
        gray = bool((pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all())
        data = pixels[..., 0] if gray else pixels
        objects.append(
            _stream(
                f"/Type /XObject /Subtype /Image /Width {sheet.image.width} /Height {sheet.image.height} "
                f"/ColorSpace /{'DeviceGray' if gray else 'DeviceRGB'} /BitsPerComponent 8 /Filter /FlateDecode ".encode(),
                zlib.compress(np.ascontiguousarray(data).tobytes()),
            )
        )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _stream(entries: bytes, data: bytes) -> bytes:
    return b"<< " + entries + f"/Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def _num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")

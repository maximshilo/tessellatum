"""Save a generated coloring page and its legend as a PNG or a PDF, sized for printing on its paper (see
``print_size``)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from tessellatum.core.boundaries import region_outlines, smoothing_length_px
from tessellatum.core.ink_outline import ink_outline
from tessellatum.core.labels import baseline_bbox, font
from tessellatum.core.legend import (
    LABEL_FONT_RATIO,
    SWATCH_BORDER_MM,
    SWATCH_GAP_MM,
    SWATCH_MM,
    DEFAULT_LEGEND_DETAIL,
    cell_mm,
    number_fill,
    render_legend,
    swatch_texts,
)
from tessellatum.core.painting import Painting, Version, tint_rgb
from tessellatum.core.print_size import A4, MM_PER_INCH, PT_PER_INCH, PaperFormat, print_scale
from tessellatum.core.render import PageDrawing, number_origin

# The space between the page and the legend in a PNG, on paper.
LEGEND_GAP_MM = 6.0
# How wide a painted version's fills are stroked round their edge, in their own color, on paper. Two fills meet along
# the same line, but a viewer that smooths each one's edge on its own leaves a hairline of paper between them, which
# this covers. It moves an edge by half that, 0.05 mm, less than a pixel of a page at 300 dpi.
FILL_EDGE_MM = 0.1

_PT_PER_MM = PT_PER_INCH / MM_PER_INCH
# The characters the embedded font covers: printable ASCII, the numbers' digits and the legend's names and codes.
FIRST_CHAR, LAST_CHAR = 32, 126
# The size Pillow's measurements of a number's text box are taken at, in pixels an em: large enough that rounding them
# to whole pixels is a thousandth of the em.
_MEASURING_EM_PX = 1000
# The longest edge, in pixels, a page is drawn at by ``rasterize``: A2 at 600 dpi is 13559 px, and a page this size
# (188 Mpx at most) takes 1.9 GB more memory to draw, and 2 to about 11 s, by how much is on it, on the benchmark
# machine (T3b; the lion at Realistic on A2 drew 123 Mpx in 7.6 s).
MAX_RASTER_LONG_EDGE = 16384
# PDFium -- QtPdf's renderer, and Chrome's and Edge's -- rasterizes each path with AGG, which keeps at most 1024 blocks
# of 4096 coverage cells for it and silently drops the rest: a path past about 4.2 million cells stops where it got to,
# and a row whose cells were cut is filled across to the page's edge. A cell is a device pixel an edge of the path's
# outline crosses, so a path's cells grow with the length of its outline in device pixels, and with the resolution:
# the lines of the lion on A2 at 724 dpi, written as one path, came to 12.7 million, and a PNG of it lost every line
# below the top 42% (1.0.13, measured against PDFium in ``_outline_cells``). So every path on the sheet is kept to a
# quarter of the limit at the largest a PNG is drawn, ``MAX_RASTER_LONG_EDGE`` across: a viewer or a printer drawing
# the PDF four times as finely still draws it whole.
RASTER_CELL_LIMIT = 1024 * 4096
PATH_CELL_BUDGET = RASTER_CELL_LIMIT // 4
# Where a fill is cut into bands to keep it within the budget, how far each band reaches past the cut, on paper: the
# two overlap by twice this, so that a device pixel on the cut is covered whole by one band or the other rather than
# by half of each, which would leave a light seam, at 51 dpi and finer.
FILL_BAND_OVERLAP_MM = 0.25
# The Qt application ``rasterize`` starts where none runs (a script, the benchmark), kept while the module is.
_qt_application = None


def export_png(
    drawing: PageDrawing,
    palette_rgb: Sequence[tuple[int, int, int]],
    path: Path,
    dpi: float,
    version: Version = Version.PAGE,
    painting: Painting | None = None,
) -> tuple[int, int]:
    """Save ``version`` of the page laid out in ``drawing`` as a PNG at ``dpi`` on its paper: the page drawn there
    (see ``rasterize``) above its legend of ``palette_rgb`` drawn at the same resolution, the file recording it.
    Returns the page's pixels (width, height)."""
    page = rasterize(drawing, dpi, version, painting)
    palette_bgr = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
    legend = render_legend(palette_bgr, width=page.width, px_per_mm=dpi / MM_PER_INCH, detail=drawing.style.legend_detail)
    save_png(page, legend, path, drawing.style.paper_format)
    return page.size


def save_png(page: Image.Image, legend: Image.Image, path: Path, paper: PaperFormat = A4) -> None:
    """Stack the page and its legend into one PNG (a PNG has no pages).

    The file records the page's print resolution, so that it prints the page as
    large as the print model places it on ``paper``, the sheet it was made for.
    """
    scale = print_scale(page.size, paper)
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
    """Save as two sheets of vector art, of the paper the page was made for (``drawing.style.paper``): the coloring
    page, then its legend.

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
    ``painting``), painted with ``painting`` unless it is the page itself. The
    area each paint covers is filled along a path round it made of the page's
    lines (see ``boundaries.region_outlines``), so that two neighbors meet on
    the same line:

    - the completed version fills every region with its paint and prints the
      letters and the ink over it, solid, as above; no line and no number;
    - the tinted version fills them with their wash (``painting.tint_rgb``) and
      prints the page over it, drawn as above, in the Multiply blend mode: the
      page's grays multiplied by the wash, as on the tinted image.

    The legend gets a portrait sheet of its own, across the printable width
    from the top margin: its swatches, ``legend.SWATCH_MM`` squares in their
    exact colors, numbered as the page is, and beside each what the style's
    ``legend_detail`` asks for (its name, hex code and RGB values), as text.
    The file carries no date, so the same page always gives the same bytes.
    """
    document = _Document()
    numbers = document.font(_number_font())
    _page_sheet(document, drawing, numbers, version, painting)
    _legend_sheet(document, palette_rgb, numbers, drawing.style.paper_format, drawing.style.legend_detail)
    Path(path).write_bytes(document.to_bytes())


def rasterize(
    drawing: PageDrawing, dpi: float, version: Version = Version.PAGE, painting: Painting | None = None
) -> Image.Image:
    """The page as it prints, drawn at ``dpi`` on its paper: the vector art ``save_pdf`` prints for ``version``, the
    printed area alone (no margins), rendered by QtPdf. Its numbers are the drawing's, where it put them on paper, at
    any resolution (see ``pipeline.draw_page``). The image is the printed size at ``dpi``, to the pixel, at most
    ``MAX_RASTER_LONG_EDGE`` across (see ``max_raster_dpi``)."""
    size = raster_size(drawing, dpi)
    if max(size) > MAX_RASTER_LONG_EDGE:
        raise ValueError(
            f"{dpi:g} dpi is {size[0]} x {size[1]} px, more than {MAX_RASTER_LONG_EDGE} px across: at most "
            f"{max_raster_dpi(drawing):.0f} dpi"
        )
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QByteArray, QCoreApplication, QIODevice, QSize
    from PySide6.QtGui import QGuiApplication, QImage
    from PySide6.QtPdf import QPdfDocument

    global _qt_application
    if QCoreApplication.instance() is None:
        _qt_application = QGuiApplication([])  # QtPdf renders only in an application; kept for the next page
    document = _Document()
    _page_sheet(document, drawing, document.font(_number_font()), version, painting, bare=True)
    data = QByteArray(document.to_bytes())
    buffer = QBuffer(data)
    buffer.open(QIODevice.ReadOnly)
    pdf = QPdfDocument()
    pdf.load(buffer)
    if pdf.status() != QPdfDocument.Status.Ready:
        raise RuntimeError(f"QtPdf could not read the page: {pdf.error()}")
    image = pdf.render(0, QSize(*size)).convertToFormat(QImage.Format_RGBA8888_Premultiplied)
    pdf.close()
    width, height = image.width(), image.height()
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())[:, : width * 4]
    rgba = rgba.reshape(height, width, 4)
    # Rendered on no background: composited over white paper, which with premultiplied colors adds the paper left.
    rgb = rgba[..., :3] + (255 - rgba[..., 3:])
    return Image.fromarray(np.ascontiguousarray(rgb), "RGB")


def raster_size(drawing: PageDrawing, dpi: float) -> tuple[int, int]:
    """The pixels (width, height) ``rasterize`` draws ``drawing`` at, at ``dpi``: its printed size on its paper."""
    printed_w, printed_h = print_scale(drawing.size, drawing.style.paper_format).printed_size_mm
    return max(1, round(printed_w * dpi / MM_PER_INCH)), max(1, round(printed_h * dpi / MM_PER_INCH))


def max_raster_dpi(drawing: PageDrawing) -> float:
    """The most dots per inch ``rasterize`` draws ``drawing`` at: ``MAX_RASTER_LONG_EDGE`` across its printed size."""
    printed = print_scale(drawing.size, drawing.style.paper_format).printed_size_mm
    return (MAX_RASTER_LONG_EDGE - 0.5) * MM_PER_INCH / max(printed)


def _sheet_size_mm(landscape: bool, paper: PaperFormat = A4) -> tuple[float, float]:
    return (paper.height_mm, paper.width_mm) if landscape else (paper.width_mm, paper.height_mm)


def _page_sheet(
    document: _Document,
    drawing: PageDrawing,
    numbers: int,
    version: Version,
    painting: Painting | None,
    bare: bool = False,
) -> None:
    """The page's sheet, drawn in the page's pixels, y down, mapped onto where the print model puts the page; with
    ``bare``, a sheet the page's printed size, with no margins."""
    width, height = drawing.size
    paper = drawing.style.paper_format
    scale = print_scale(drawing.size, paper)
    printed_w, printed_h = scale.printed_size_mm
    sheet = (printed_w, printed_h) if bare else _sheet_size_mm(scale.landscape, paper)
    left, top = (sheet[0] - printed_w) / 2, (sheet[1] - printed_h) / 2
    pt_per_px = _PT_PER_MM / scale.px_per_mm
    if not isinstance(version, Version):
        raise ValueError(f"no such version of the page: {version!r}")
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
        ops += _fill_ops(painting, painting.palette_rgb, scale.px_per_mm, drawing.style.line_smoothing_mm, paper)
        ops += _picture_ink_ops(drawing, darken=False)
    elif version is Version.TINTED:
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
        ops += _fill_ops(
            painting, [tint_rgb(rgb) for rgb in painting.palette_rgb], scale.px_per_mm, drawing.style.line_smoothing_mm,
            paper,
        )
        ops.append("q /Mu gs /Pg Do Q")
    else:
        raise ValueError(f"no such version of the page: {version!r}")
    ops.append("Q")
    document.page(sheet, "\n".join(ops), resources)


def _page_ops(drawing: PageDrawing, px_per_mm: float) -> list[str]:
    """The page as it prints, in its pixels: its lines, letters, ink, leaders and numbers, with the resources /F1 (the
    numbers' font) and /Dk (the Darken blend mode). The lines are a path or several, each within the cell budget (see
    ``PATH_CELL_BUDGET``)."""
    style = drawing.style
    line_width = style.line_width_mm * px_per_mm
    ops = [f"1 J 1 j {_num(line_width)} w"]
    if drawing.strokes:
        ops.append(f"{_gray(style.line_gray)} G")
        for group in _stroke_groups(drawing.strokes, line_width, _cell_scale(drawing.size)):
            ops += [_path(group, offset=0.5), "S"]
    ops += _picture_ink_ops(drawing, darken=True)
    leaders = [label.leader for label in drawing.labels if label.leader is not None]
    if leaders:
        leader_width = style.leader_width_mm * px_per_mm
        dot = leader_width * style.leader_dot_ratio / 2
        ops += [f"q /Dk gs {_num(leader_width)} w {_gray(style.leader_gray)} G {_gray(style.leader_gray)} g"]
        ops += [_path([np.array(leader, dtype=np.float64)], offset=0.5) + " S" for leader in leaders]
        ops += [_disk(x + 0.5, y + 0.5, dot) for _, (x, y) in leaders]
        ops.append("Q")
    if drawing.labels:
        ops.append("BT")
        fill = None
        for label in drawing.labels:
            if style.label_fill(label) != fill:  # a number with a leader prints in the leaders' gray
                fill = style.label_fill(label)
                ops.append(f"{_gray(fill)} g")
            x, y = number_origin(label)
            ops.append(f"/F1 {_num(label.font_size)} Tf 1 0 0 -1 {_num(x)} {_num(y)} Tm {_text(label.text)} Tj")
        ops.append("ET")
    return ops


def _fill_ops(
    painting: Painting,
    colors: Sequence[tuple[int, int, int]],
    px_per_mm: float,
    smoothing_mm: float,
    paper: PaperFormat = A4,
) -> list[str]:
    """Every paint's area filled in its entry of ``colors``, its edge stroked ``FILL_EDGE_MM`` wide.

    The areas are outlined from the map of the paints rather than of the
    regions: where two regions of one paint touch, no line divides their paint,
    and a path round each would run along their boundary twice, which a viewer
    smoothing the edges draws as a seam. Elsewhere the two maps have the same
    boundaries and junctions, so the paths are the page's lines, smoothed
    ``smoothing_mm`` along them as the page's are (see ``render.PageStyle``).

    A paint whose area or edge is past the cell budget (see
    ``PATH_CELL_BUDGET``) is drawn in several paths that together print what
    one would. Its rings go in groups that keep each ring with every ring it
    could lie inside or hold, so each group fills by the even-odd rule what
    it did among the rest (see ``_nests``), down the page, each group within
    the budget. A nest past the budget on its own is filled in bands (see
    ``_fill_bands``) and its edge stroked in groups of rings.
    """
    region_color = np.asarray(painting.region_color)
    paints = np.where(painting.region_id_map >= 0, region_color[np.clip(painting.region_id_map, 0, None)], -1)
    edge_width = FILL_EDGE_MM * px_per_mm
    ops = [f"1 J 1 j {_num(edge_width)} w"]
    size = painting.region_id_map.shape[1::-1]
    smoothing_px = smoothing_length_px(size, smoothing_mm, paper)
    scale = _cell_scale(size)
    overlap = FILL_BAND_OVERLAP_MM * px_per_mm
    for paint, rings in sorted(region_outlines(paints, smoothing_px).items()):
        rgb = " ".join(_gray(v) for v in colors[paint])
        ops.append(f"{rgb} rg {rgb} RG")
        measures = _outline_measures(rings)
        filled, stroked = _cells(measures, scale), _cells(measures, scale, edge_width)
        if max(filled.sum(), stroked.sum()) <= PATH_CELL_BUDGET:
            ops += [_path(rings, offset=0.5), "B*"]
            continue
        nest = _nests(rings)
        nest_filled, nest_stroked = np.bincount(nest, filled), np.bincount(nest, stroked)
        counts = np.array([len(r) for r in rings])
        down = np.bincount(nest, np.add.reduceat(np.concatenate(rings)[:, 1], np.cumsum(counts) - counts))
        down /= np.bincount(nest, counts)
        groups: list[list[int]] = [[]]
        totals = np.zeros(2)
        for index in np.argsort(down, kind="stable"):
            cells = np.array([nest_filled[index], nest_stroked[index]])
            if cells.max() > PATH_CELL_BUDGET:  # too much on its own: in bands, and its edge in groups
                members = [i for i in range(len(rings)) if nest[i] == index]
                part = [rings[i] for i in members]
                for band in _fill_bands(part, scale, overlap, filled[members]):
                    ops += [_path(band, offset=0.5), "f*"]
                for group in _stroke_groups(part, edge_width, scale, stroked[members]):
                    ops += [_path(group, offset=0.5), "S"]
                continue
            if groups[-1] and (totals + cells).max() > PATH_CELL_BUDGET:
                groups.append([])
                totals[:] = 0
            groups[-1].append(index)
            totals += cells
        in_group = np.full(nest.max() + 1, -1)
        for number, group in enumerate(groups):
            in_group[group] = number
        for number in range(len(groups)):
            members = np.flatnonzero(in_group[nest] == number)
            if members.size:
                ops += [_path([rings[i] for i in members], offset=0.5), "B*"]
    return ops


def _nests(rings: list[np.ndarray]) -> np.ndarray:
    """Each ring's nest: rings joined wherever one's bounding box holds another's, by number from 0.

    A ring inside another has its box inside the other's, so a nest holds
    every ring each of its rings lies inside or holds: filled by the even-odd
    rule on its own, apart from the other nests, it fills what it did among
    them. Nests are what a set of rings can be cut into without clipping.
    """
    counts = np.array([len(r) for r in rings])
    points = np.concatenate(rings)
    starts = np.cumsum(counts) - counts
    low = np.stack([np.minimum.reduceat(points[:, 0], starts), np.minimum.reduceat(points[:, 1], starts)], axis=1)
    high = np.stack([np.maximum.reduceat(points[:, 0], starts), np.maximum.reduceat(points[:, 1], starts)], axis=1)
    pairs = []
    for at in range(0, len(rings), 512):  # which boxes hold each of these, a block at a time
        inner = slice(at, at + 512)
        held = (
            (low[None, :, 0] <= low[inner, None, 0]) & (low[None, :, 1] <= low[inner, None, 1])
            & (high[None, :, 0] >= high[inner, None, 0]) & (high[None, :, 1] >= high[inner, None, 1])
        )  # fmt: skip
        held_ring, holder = np.nonzero(held)
        pairs.append(np.stack([held_ring + at, holder], axis=1))
    pairs = np.concatenate(pairs)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    nest = np.arange(len(rings))
    while pairs.size:  # every pair joined: each ring takes the least number in reach, until none changes
        least = np.minimum(nest[pairs[:, 0]], nest[pairs[:, 1]])
        before = nest.copy()
        np.minimum.at(nest, pairs[:, 0], least)
        np.minimum.at(nest, pairs[:, 1], least)
        nest = nest[nest]
        if (nest == before).all():
            break
    return np.unique(nest, return_inverse=True)[1]


def _picture_ink_ops(drawing: PageDrawing, darken: bool) -> list[str]:
    """What the page prints of the picture itself, in its ink's gray: the letters, then the ink, solid, each a path
    or, past the cell budget, several (see ``_fill_bands``).

    ``darken`` draws the letters in the Darken blend mode (/Dk), the darker of them and what is under them, as the page
    prints them over its lines.
    """
    ops = []
    scale = _cell_scale(drawing.size)
    overlap = FILL_BAND_OVERLAP_MM * drawing.style.scale(drawing.size).px_per_mm
    if drawing.lettering:
        ops.append(f"q /Dk gs {_gray(drawing.ink_gray)} g" if darken else f"{_gray(drawing.ink_gray)} g")
        for band in _fill_bands(drawing.lettering, scale, overlap):
            ops += [_path(band, offset=0.5), "f*"]
        if darken:
            ops[-1] = "f* Q"
    if drawing.ink is not None and drawing.ink.any():
        ops.append(f"{_gray(drawing.ink_gray)} g")
        for band in _fill_bands(ink_outline(drawing.ink), scale, overlap):
            ops += [_path(band, offset=0.5), "f*"]
    return ops


def _cell_scale(size: tuple[int, int]) -> float:
    """Device pixels a pixel of a page of ``size`` (width, height) is drawn at by the largest PNG: the scale its paths'
    cells are counted at (see ``PATH_CELL_BUDGET``)."""
    return MAX_RASTER_LONG_EDGE / max(size)


def _outline_cells(paths: Sequence[np.ndarray], scale: float, width: float | None = None) -> np.ndarray:
    """About how many coverage cells PDFium makes for each of ``paths``, Nx2 points in a unit drawn ``scale`` device
    pixels across: filled (``width`` None), or stroked ``width`` units wide with round caps and joins.

    An edge crosses about |dx| + |dy| cells, and one more where it starts. A
    stroke's outline runs along both its sides, round a circle at its two
    caps, and round an arc at each turn, as wide as the stroke and as long
    as the turn. Against where PDFium cut real and made-up paths (T1a, 1.0.13),
    lines, zig-zags and the lion's lines from 0.1 to 18 device pixels wide,
    this comes to 1.00-1.18 times the cells a stroke had when it was cut,
    and twice those of a fill.
    """
    return _cells(_outline_measures(paths), scale, width)


def _outline_measures(paths: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Each of ``paths``' L1 length, the L1 length of the edge closing it, its segments and its turning in radians:
    what ``_cells`` counts its cells from, filled or stroked."""
    if not len(paths):
        return (np.zeros(0),) * 4
    counts = np.array([len(p) for p in paths])
    points = np.concatenate(paths).astype(np.float64)
    owner = np.repeat(np.arange(len(paths)), counts)
    steps = np.diff(points, axis=0)
    keep = (owner[1:] == owner[:-1]) & (steps != 0).any(axis=1)
    steps, owner = steps[keep], owner[:-1][keep]
    length = np.bincount(owner, np.abs(steps).sum(axis=1), len(paths))
    segments = np.bincount(owner, minlength=len(paths))
    ends = np.cumsum(counts)
    closing = np.abs(points[ends - 1] - points[ends - counts]).sum(axis=1)  # a ring not repeating its start
    before, after = steps[:-1], steps[1:]
    turn = np.abs(np.arctan2(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0], (before * after).sum(axis=1)))
    same = owner[1:] == owner[:-1]
    turning = np.bincount(owner[1:][same], turn[same], len(paths))
    return length, closing, segments, turning


def _cells(measures: tuple[np.ndarray, ...], scale: float, width: float | None = None) -> np.ndarray:
    """The cells of paths of ``measures`` (see ``_outline_measures``), as ``_outline_cells`` counts them."""
    length, closing, segments, turning = measures
    if width is None:
        return scale * (length + closing) + segments + 1
    return scale * (2 * length + width * (turning + 4)) + 2 * segments


def _stroke_groups(
    paths: list[np.ndarray], width: float, scale: float, cells: np.ndarray | None = None
) -> list[list[np.ndarray]]:
    """``paths`` stroked ``width`` wide at ``scale`` (see ``_outline_cells``), in groups of at most
    ``PATH_CELL_BUDGET`` cells: all of them in one if they fit, as they were.

    Past it, a group is the paths lying next down the page, by where their
    points are on average, in the order given, so that few lines in one group
    meet lines in another: where two do, the edges of their round ends are
    drawn twice and come out a shade darker. A path past the budget on its
    own is cut into pieces sharing their ends, which stroked with round ends
    cover what it does. ``cells`` are the paths' own, if already counted.
    """
    if cells is None:
        cells = _outline_cells(paths, scale, width)
    if cells.sum() <= PATH_CELL_BUDGET:
        return [paths]
    pieces = []
    for path, count in zip(paths, cells):
        parts = int(np.ceil(count / (PATH_CELL_BUDGET / 2)))
        if parts <= 1 or len(path) < 3:
            pieces.append(path)
            continue
        cuts = np.linspace(0, len(path) - 1, min(parts, len(path) - 1) + 1).round().astype(int)
        pieces += [path[a : b + 1] for a, b in zip(cuts[:-1], cuts[1:])]
    if len(pieces) != len(paths):
        paths, cells = pieces, _outline_cells(pieces, scale, width)
    counts = np.array([len(p) for p in paths])
    down = np.add.reduceat(np.concatenate(paths)[:, 1], np.cumsum(counts) - counts) / counts
    groups: list[list[int]] = [[]]
    total = 0.0
    for index in np.argsort(down, kind="stable"):
        if groups[-1] and total + cells[index] > PATH_CELL_BUDGET:
            groups.append([])
            total = 0.0
        groups[-1].append(index)
        total += cells[index]
    return [[paths[i] for i in sorted(group)] for group in groups]


def _fill_bands(
    rings: list[np.ndarray], scale: float, overlap: float, cells: np.ndarray | None = None
) -> list[list[np.ndarray]]:
    """``rings`` filled by the even-odd rule at ``scale`` (see ``_outline_cells``), as bands down the page of at most
    ``PATH_CELL_BUDGET`` cells each: all of them in one if they fit, as they were.

    A ring can't simply go to a band of its own: a hole is filled only
    together with the ring round it, and line art's ink is one ring with
    thousands of holes. So the rings are cut in two across the page, where
    as much of their outline lies above as below, and each part again until
    it fits. A ring crossing the cut is clipped to each side, the two
    reaching ``overlap`` past it: clipped ring by ring to a band, by the
    even-odd rule, the rings fill what they filled inside it. Where the bands
    overlap, an edge's anti-aliased pixels are drawn twice, a shade darker.
    ``cells`` are the rings' own, if already counted.
    """
    if cells is None:
        cells = _outline_cells(rings, scale)
    if cells.sum() <= PATH_CELL_BUDGET:
        return [rings]
    points = np.concatenate(rings)
    top, bottom = points[:, 1].min(), points[:, 1].max()
    if bottom - top <= 4 * overlap:
        return [rings]  # a band can't be cut finer: drawn as it is
    # Where half the outline lies above: the edges' midpoints, weighted by their length.
    counts = np.array([len(r) for r in rings])
    owner = np.repeat(np.arange(len(rings)), counts)
    same = owner[1:] == owner[:-1]
    middles = (points[1:, 1] + points[:-1, 1])[same] / 2
    weights = np.abs(np.diff(points, axis=0)[same]).sum(axis=1)
    order = np.argsort(middles)
    cumulative = np.cumsum(weights[order])
    cut = middles[order][np.searchsorted(cumulative, cumulative[-1] / 2)] if cumulative[-1] > 0 else (top + bottom) / 2
    cut = min(max(cut, top + 2 * overlap), bottom - 2 * overlap)
    # Each band takes every ring reaching into it, clipped to it, so that inside it they fill what they did.
    above: list[np.ndarray] = []
    below: list[np.ndarray] = []
    for ring in rings:
        lowest, highest = ring[:, 1].min(), ring[:, 1].max()
        if lowest < cut + overlap:
            above.append(ring if highest <= cut + overlap else _clip_ring(ring, cut + overlap, below=False))
        if highest > cut - overlap:
            below.append(ring if lowest >= cut - overlap else _clip_ring(ring, cut - overlap, below=True))
    above = [r for r in above if len(r) >= 3]
    below = [r for r in below if len(r) >= 3]
    return [band for part in (above, below) if part for band in _fill_bands(part, scale, overlap)]


def _clip_ring(ring: np.ndarray, y: float, below: bool) -> np.ndarray:
    """The closed ``ring`` clipped to the side of the line at ``y`` below it (y down) or above it (Sutherland-Hodgman):
    where the ring crosses the line, it runs along it. Filled by the even-odd rule with the rest of its rings, it fills
    what the ring did on that side."""
    ring = np.asarray(ring, dtype=np.float64)
    following = np.roll(ring, -1, axis=0)
    inside = ring[:, 1] >= y if below else ring[:, 1] <= y
    crosses = inside != np.roll(inside, -1)
    rise = following[:, 1] - ring[:, 1]
    t = np.divide(y - ring[:, 1], rise, out=np.zeros_like(rise), where=crosses)
    crossing = ring + t[:, None] * (following - ring)
    crossing[:, 1] = np.where(crosses, y, crossing[:, 1])
    return np.stack([ring, crossing], axis=1)[np.stack([inside, crosses], axis=1)]


def _legend_sheet(
    document: _Document,
    palette_rgb: Sequence[tuple[int, int, int]],
    numbers: int,
    paper: PaperFormat = A4,
    detail: str = DEFAULT_LEGEND_DETAIL,
) -> None:
    """The legend's portrait sheet of ``paper``, drawn in millimeters from the top-left margin, y down, with ``detail``
    beside each swatch."""
    area_w, area_h = paper.printable_mm()
    cell = SWATCH_MM + SWATCH_GAP_MM
    columns = max(1, int(area_w // cell_mm(detail)))
    rows = -(-len(palette_rgb) // columns)
    # A legend never runs past the sheet: 40 colors take a quarter of an A4 sheet numbered only, and 64 with their
    # details 243 mm of its 277.
    fit = min(1.0, area_h / (rows * cell + SWATCH_GAP_MM))
    texts = swatch_texts(palette_rgb, detail)
    scale = _PT_PER_MM * fit
    em = SWATCH_MM * LABEL_FONT_RATIO
    ops = [
        f"q {_num(scale, 8)} 0 0 {_num(-scale, 8)} {_num(paper.margin_mm * _PT_PER_MM)} "
        f"{_num((paper.height_mm - paper.margin_mm) * _PT_PER_MM)} cm"
    ]
    for index, rgb in enumerate(palette_rgb):
        row, column = divmod(index, columns)
        x, y = SWATCH_GAP_MM + column * cell_mm(detail), SWATCH_GAP_MM + row * cell
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
        for line in texts[index]:
            ops.append(
                f"{_gray(line.gray)} g BT /F1 {_num(line.size_mm)} Tf 1 0 0 -1 {_num(x + line.x_mm)} "
                f"{_num(y + line.baseline_mm)} Tm {_text(line.text)} Tj ET"
            )
    ops.append("Q")
    document.page(_sheet_size_mm(False, paper), "\n".join(ops), {"Font": {"F1": numbers}})


def _path(paths: list[np.ndarray], offset: float) -> str:
    """Path construction for ``paths``, Nx2 (x, y) points each, one subpath apiece, moved by ``offset`` in x and y.

    The page's lines, leaders and ink outline put pixel centers at integer
    coordinates; the sheet's image of the page puts them half a pixel in, so
    their ``offset`` is 0.5. Two decimals: a hundredth of a pixel is a
    hundredth of 0.1-0.4 mm, far finer than a printer's dot.
    """
    points = np.concatenate(paths) + offset
    # One format for the whole path, filled in at once: under half the time of formatting each point.
    template = "\n".join("%.2f %.2f m" + "\n%.2f %.2f l" * (len(path) - 1) for path in paths)
    return template % tuple(points.ravel().tolist())


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
    """``text`` as a PDF string: printable ASCII, which the embedded font's widths cover, with its parentheses and
    backslashes escaped."""
    if not all(FIRST_CHAR <= ord(c) <= LAST_CHAR for c in text):
        raise ValueError(f"the PDF writes printable ASCII only, got {text!r}")
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})"


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
    advances: tuple[int, ...]  # the advance widths of the characters FIRST_CHAR to LAST_CHAR, in font units

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
        for char in range(FIRST_CHAR, LAST_CHAR + 1):
            glyph = min(glyphs.get(char, 0), metrics - 1)  # glyphs past the last metric take its advance
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
        """``face`` embedded whole, as a simple font of printable ASCII: the numbers' digits and the legend's text."""
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
                f"<< /Type /Font /Subtype /TrueType /BaseFont /{face.name} /FirstChar {FIRST_CHAR} /LastChar {LAST_CHAR} "
                f"/Widths [{' '.join(face.em(v) for v in face.advances)}] /FontDescriptor {descriptor} 0 R "
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

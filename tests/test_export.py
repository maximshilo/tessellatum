"""Export: a PNG that prints at the page's size, and a vector PDF of two A4 sheets, the page placed by the print model."""

import dataclasses
import re
from collections import Counter

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFont

from tessellatum.core import export, text
from tessellatum.core.boundaries import smoothing_length_px, trace_boundaries
from tessellatum.core.ink_outline import ink_outline
from tessellatum.core.legend import LABEL_FONT_RATIO, SWATCH_BORDER_MM, SWATCH_GAP_MM, SWATCH_MM, render_legend
from tessellatum.core.painting import Painting, Version, completed, tint_rgb, tinted
from tessellatum.core.print_size import A4, print_scale
from tessellatum.core.regions import extract_regions
from tessellatum.core.render import (
    LABEL_GRAY,
    LINE_GRAY,
    PageDrawing,
    PageStyle,
    ink_coverage,
    number_origin,
    paper_under,
    render_page,
)

import pdf_reading
from pdf_reading import PT_PER_MM

PALETTE = [(200, 40, 40), (40, 160, 60), (230, 210, 120), (20, 20, 120)]
# A page shaped like A4's printable area, 4 px a millimeter: it fills the area exactly, 10 mm (40 px) in from the
# sheet's edges, so rendered at 4 px a millimeter its pixels lie on the sheet's.
ALIGNED = (760, 1108)


def _split(size: tuple[int, int]) -> np.ndarray:
    """A region map of two regions, divided down a crack in the middle."""
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    ids[:, width // 2 :] = 1
    return ids


def _drawn(size, ids, color, style=PageStyle(), **printing):
    """``ids`` rendered as ``render_page`` draws it, with a number in every region."""
    regions = extract_regions(ids, np.asarray(color, dtype=np.int32), printed=printing.get("ink"))
    return render_page(size, regions, ids, style, **printing)


def _full_page(bare: bool = False):
    """The aligned page with everything a page can print: lines, numbers, a leader, printed ink and lettering.

    ``bare`` leaves a patch of it in no region, as line art leaves a pocket no brush reaches.
    """
    ids, colors, printing = _full_page_parts(bare)
    return _drawn(ALIGNED, ids, colors, **printing)


def _full_page_parts(bare: bool = False):
    """The full page's region map, its regions' colors and what it prints (see ``_full_page``)."""
    width, height = ALIGNED
    ids = _split(ALIGNED)
    cv2.circle(ids, (200, 300), 120, 2, -1)
    cv2.rectangle(ids, (450, 600), (700, 900), 3, -1)
    ids[700:708, 100:108] = 4  # 2 mm across: too small for its number, which goes outside with a leader
    if bare:
        ids[100:160, 450:530] = -1
    ink = np.zeros((height, width), dtype=np.uint8)
    cv2.line(ink, (60, 900), (300, 1050), 1, 3)
    cv2.circle(ink, (580, 200), 40, 1, 2)
    ink[500:540, 600:610] = 1  # a hole in it
    ink[515:525, 603:607] = 0
    printing = dict(ink=ink.astype(bool), ink_gray=30, lettering=_letters_on_full_page())
    return ids, [0, 1, 2, 3, 0], printing


def _painted_full_page():
    """The full page with a bare patch, and what it is painted with: each region in its color of ``PALETTE``."""
    ids, colors, printing = _full_page_parts(bare=True)
    rendered = _drawn(ALIGNED, ids, colors, **printing)
    painting = Painting(ids, np.array(colors), PALETTE, rendered.picture_ink, rendered.drawing.ink_gray)
    return rendered, painting


def _letters_on_full_page():
    """A line of dark letters on a pale ground, rows 1000-1059 and columns 420-739 of the aligned page."""
    width, height = ALIGNED
    picture = np.full((height, width, 3), 225, dtype=np.uint8)
    cv2.putText(picture, "Paint 2026", (430, 1045), cv2.FONT_HERSHEY_DUPLEX, 1.6, (25, 25, 25), 3, cv2.LINE_AA)
    line = text.TextLine(quad=((420.0, 1000.0), (740.0, 1000.0), (740.0, 1060.0), (420.0, 1060.0)), score=1.0)
    return text.lettering(picture, [line])


def _save(tmp_path, drawing: PageDrawing, palette=PALETTE, name="page.pdf"):
    export.save_pdf(drawing, palette, tmp_path / name)
    return tmp_path / name, pdf_reading.sheets((tmp_path / name).read_bytes())


@pytest.mark.parametrize("size", [(1600, 1000), (900, 1400), (600, 450), ALIGNED])
def test_a_pdf_is_two_a4_sheets_the_page_placed_as_the_print_model_prints_it(tmp_path, size):
    # D-052 (Q36): the page fills the printable area inside the 10 mm margins, centered, on a landscape sheet when wider
    # than tall; the legend gets a portrait sheet of its own.
    _, (first, second) = _save(tmp_path, _drawn(size, _split(size), [0, 1]).drawing)

    scale = print_scale(size)
    sheet = (297.0, 210.0) if scale.landscape else (210.0, 297.0)
    printed = scale.printed_size_mm
    assert first["size_mm"] == pytest.approx(sheet, abs=1e-3)
    left, top_from_bottom, mm_per_px_x, mm_per_px_y = pdf_reading.page_transform(first["content"])
    assert (mm_per_px_x, mm_per_px_y) == pytest.approx((1 / scale.px_per_mm,) * 2, rel=1e-6)
    assert (left, sheet[1] - top_from_bottom) == pytest.approx(((sheet[0] - printed[0]) / 2, (sheet[1] - printed[1]) / 2), abs=1e-3)
    assert min(left, sheet[1] - top_from_bottom) == pytest.approx(A4.margin_mm, abs=1e-3)  # it reaches the margins on one side
    assert f"0 0 {size[0]} {size[1]} re W n" in first["content"]  # nothing is drawn off the page
    assert second["size_mm"] == pytest.approx((210.0, 297.0), abs=1e-3)


@pytest.mark.parametrize("width_mm", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("size", [ALIGNED, (300, 437)])  # 102 dpi; 40 dpi, where the raster page draws a line 1 px wide
def test_lines_are_paths_stroked_with_a_round_pen_of_the_asked_for_width_on_paper(tmp_path, width_mm, size):
    style = PageStyle(line_width_mm=width_mm)
    rendered = _drawn(size, _split(size), [0, 1], style)
    _, (first, _) = _save(tmp_path, rendered.drawing)

    content = first["content"]
    _, _, mm_per_px, _ = pdf_reading.page_transform(content)
    width_px = float(re.search(r"1 J 1 j (\S+) w", content).group(1))  # round caps and joins: the page's round pen
    assert width_px * mm_per_px == pytest.approx(width_mm, abs=1e-4)
    # Every line, every point of it: a subpath of its own, half a pixel in, since the page's pixel centers lie there.
    strokes = re.search(r"(?s)G\n(.*?)\nS\n", content).group(1).split("\n")
    expected = np.concatenate(rendered.strokes) + 0.5
    drawn = np.array([[float(v) for v in word.split()[:2]] for word in strokes])
    np.testing.assert_allclose(drawn, expected, atol=0.005)
    assert [word.split()[2] for word in strokes].count("m") == len(rendered.strokes)


@pytest.mark.parametrize("width_mm", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("size", [ALIGNED, (300, 437)])
def test_a_line_prints_as_wide_as_asked_whatever_the_page_s_resolution(tmp_path, width_mm, size):
    # T7.3's acceptance: a page prints from the vector file at A4 with lines of the asked-for width. Read back by
    # Qt's renderer at 1200 dpi, the ink across the line down the middle of the page is that wide, where the raster
    # page of the small one draws it a pixel wide: 0.63 mm.
    style = PageStyle(line_width_mm=width_mm)
    path, (first, _) = _save(tmp_path, _drawn(size, _split(size), [0, 1], style).drawing)
    left, _, mm_per_px, _ = pdf_reading.page_transform(first["content"])
    middle_mm = left + size[0] / 2 * mm_per_px
    px_per_mm = 1200 / 25.4

    window = pdf_reading.render(path, 0, px_per_mm, clip_mm=(middle_mm - 2, 100, 4, 3)).mean(axis=2)
    ink = (255 - window) / (255 - LINE_GRAY)
    across = ink.sum(axis=1) / px_per_mm
    assert across == pytest.approx(np.full_like(across, width_mm), abs=0.01)


def test_read_back_the_vector_page_is_the_raster_page(tmp_path):
    rendered = _full_page()
    drawing = rendered.drawing
    path, _ = _save(tmp_path, drawing)
    width, height = ALIGNED
    back = pdf_reading.render(path, 0, 4.0).mean(axis=2)[40 : 40 + height, 40 : 40 + width]
    page = np.asarray(rendered.image.convert("L")).astype(np.float64)

    lines = ink_coverage(ALIGNED, rendered.strokes, drawing.style.line_width_px(ALIGNED)) > 0
    numbers = np.zeros(lines.shape, dtype=bool)
    for label in rendered.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        numbers[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] = True
    leaders = np.asarray(rendered.leaders) < 255
    near = cv2.dilate((lines | numbers | leaders).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # The letters' edges are anti-aliased each its own way, the raster page's from the picture's pixels; they are
    # checked in the file below.
    letters = pdf_reading.inside_even_odd(drawing.lettering, (height, width)).astype(np.uint8)
    lettered = cv2.dilate(letters, np.ones((5, 5), np.uint8)).astype(bool)

    assert np.abs(back - page).mean() < 1.0
    # The printed ink is solid over its pixels, and bare paper is white, but where the ink's outline runs: there it is
    # smoothed off the pixels' staircase (see ``ink_outline``), and its pixels are as dark as it covers them, the ink
    # taking from one what it gives another.
    cross = np.ones((3, 3), np.uint8)
    solid = cv2.erode(drawing.ink.astype(np.uint8), cross).astype(bool)
    edge = cv2.dilate(drawing.ink.astype(np.uint8), cross).astype(bool) & ~solid & ~near & ~lettered
    np.testing.assert_array_equal(back[solid & ~near].round(), 30)
    np.testing.assert_array_equal(back[~near & ~drawing.ink & ~edge & ~lettered], 255)
    covered = (255 - back[edge]) / (255 - 30)
    assert covered.sum() == pytest.approx(drawing.ink[edge].sum(), rel=0.02)
    assert np.abs(back - page)[lines].mean() < 6  # the lines, anti-aliased each its own way
    others = cv2.dilate((lines | leaders).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~numbers
    for label in rendered.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        window = (slice(y0 - 3, y1 + 3), slice(x0 - 3, x1 + 3))
        ys, xs = np.nonzero((back[window] < 250) & ~others[window])
        top, bottom, left, right = ys.min() + y0 - 3, ys.max() + y0 - 2, xs.min() + x0 - 3, xs.max() + x0 - 2
        # Each number's ink lies in the box it was placed in, as high as the page writes it.
        assert x0 - 1 <= left and right <= x1 + 1 and y0 - 1 <= top and bottom <= y1 + 1
        assert (top, bottom) == pytest.approx((y0, y1), abs=1)


def test_the_printed_ink_is_its_outline_smoothed_filled(tmp_path):
    drawing = _full_page().drawing
    _, (first, _) = _save(tmp_path, drawing)
    content = first["content"]
    path = re.search(r"(?s)\n0\.1176 g\n(.*?)\nf\*\n", content).group(1)  # the ink's gray, 30, then the filled outlines
    rings = [
        np.array([[float(v) for v in word.split()[:2]] for word in ring.split("\n")]) - 0.5  # back to the page's pixels
        for ring in re.split(r"\n(?=\S+ \S+ m$)", path, flags=re.M)
    ]
    expected = ink_outline(drawing.ink)
    assert len(rings) == len(expected)
    for ring, want in zip(rings, expected):
        np.testing.assert_allclose(ring, want, atol=0.005 + 1e-9)  # to a hundredth of a pixel
    # Filled by the even-odd rule, every pixel's middle is ink where the page prints ink, and paper where it doesn't.
    width, height = ALIGNED
    np.testing.assert_array_equal(pdf_reading.inside_even_odd(rings, (height, width)), drawing.ink)


def test_read_back_at_300_dpi_the_ink_joins_and_parts_what_the_page_does(tmp_path):
    # T7.4's acceptance, on a page of everything that breaks or merges when an outline is smoothed carelessly: hatching
    # a pixel apart across and along the diagonal, strokes a pixel wide whose pixels meet only at their corners,
    # specks and pinholes. Read back by QtPdf at 300 dpi, each piece of ink and each piece of paper the page has is
    # one piece on paper, no two run together, and none is lost.
    size = (1300, 900)  # 4.7 px a millimeter, as the benchmark's comics page at its own size
    width, height = size
    ink = np.zeros((height, width), dtype=bool)
    ink[100:300:2, 100:400] = True
    for k in range(0, 300, 3):
        ink[np.arange(100, 300), np.arange(100, 300) + 400 + k] = True
    rng = np.random.default_rng(0)
    ink[400:800, 100:600] = rng.random((400, 500)) < 0.08  # specks
    ink[400:800, 700:1200] = rng.random((400, 500)) > 0.08  # pinholes
    drawing = PageDrawing(size, PageStyle(), [], [], ink, 30, None)
    path = tmp_path / "ink.pdf"
    export.save_pdf(drawing, PALETTE, path)
    px_per_mm = 300 / 25.4
    back = pdf_reading.render(path, 0, px_per_mm).mean(axis=2) < (255 + 30) / 2
    scale = print_scale(size)
    left, top = (A4.height_mm - scale.printed_size_mm[0]) / 2, (A4.width_mm - scale.printed_size_mm[1]) / 2
    columns = np.floor((left + (np.arange(width) + 0.5) / scale.px_per_mm) * px_per_mm).astype(int)
    rows = np.floor((top + (np.arange(height) + 0.5) / scale.px_per_mm) * px_per_mm).astype(int)
    at_centers = back[np.ix_(rows, columns)]
    # A 300 dpi dot is wider than the margins, so a stroke's rounded tip can print lighter than halfway at its last
    # pixel's middle; the pieces are compared where the printed page agrees with the page.
    assert (at_centers == ink).mean() > 0.999
    for printed, page, connectivity in ((back, ink, 8), (~back, ~ink, 4)):
        _, on_paper = cv2.connectedComponents(printed.astype(np.uint8), connectivity=connectivity)
        padded = np.pad(page, 1, constant_values=connectivity == 4)  # the paper round the page is one piece
        count, on_page = cv2.connectedComponents(padded.astype(np.uint8), connectivity=connectivity)
        agree = page & (at_centers == ink)
        pairs = np.unique(np.column_stack([on_page[1:-1, 1:-1][agree], on_paper[np.ix_(rows, columns)][agree]]), axis=0)
        assert len(np.unique(pairs[:, 0])) == count - 1  # none lost
        assert len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1]))  # none broken, none merged


def test_the_letters_are_their_outline_filled_darkening_what_is_under_them(tmp_path):
    # T7.5: the lettering is vector art like the rest of the sheet, not an image at the page's pixels.
    rendered = _full_page()
    drawing = rendered.drawing
    path, (first, _) = _save(tmp_path, drawing)
    content = first["content"]
    assert "XObject" not in first["resources"] and b"/Subtype /Image" not in path.read_bytes()
    dk = first["resources"]["ExtGState"]["Dk"]
    assert first["objects"][dk][0] == "<< /Type /ExtGState /BM /Darken >>"
    letters = re.search(r"(?s)\nq /Dk gs 0\.1176 g\n(.*?)\nf\* Q\n", content)  # in the ink's gray, 30
    rings = [
        np.array([[float(v) for v in word.split()[:2]] for word in ring.split("\n")]) - 0.5  # back to the page's pixels
        for ring in re.split(r"\n(?=\S+ \S+ m$)", letters.group(1), flags=re.M)
    ]
    assert len(rings) == len(drawing.lettering) > 6
    for ring, want in zip(rings, drawing.lettering):
        np.testing.assert_allclose(ring, want, atol=0.005 + 1e-9)
    assert letters.start() < content.index("\n0.1176 g\n")  # the printed ink goes over them, as on the page
    # Read back at four times the page's pixels, they cover what the raster page's letters do.
    back = pdf_reading.render(path, 0, 16.0).mean(axis=2)[160 : 160 + 4 * ALIGNED[1], 160 : 160 + 4 * ALIGNED[0]]
    window = (slice(4 * 990, 4 * 1070), slice(4 * 410, 4 * 750))
    printed = (255 - back[window]) / (255 - 30)
    letters_ink = _letters_on_full_page().ink.astype(np.float64) / 255
    assert printed.sum() / 16 == pytest.approx(letters_ink[990:1070, 410:750].sum(), rel=0.03)
    assert (back[window] == 255).mean() > 0.6  # their ground, bare paper


def test_the_numbers_are_text_in_the_page_s_font_embedded_where_the_page_writes_them(tmp_path):
    rendered = _full_page()
    _, (first, _) = _save(tmp_path, rendered.drawing)
    content = first["content"]
    shown = re.findall(r"/F1 (\d+) Tf 1 0 0 -1 (\S+) (\S+) Tm \((\d+)\) Tj", content)
    assert len(shown) == len(rendered.labels)
    for (size, x, y, text), label in zip(shown, rendered.labels):
        assert (int(size), text) == (label.font_size, label.text)
        assert (float(x), float(y)) == pytest.approx(number_origin(label), abs=1e-4)
    assert f"{LABEL_GRAY / 255:.4f}".rstrip("0") + " g BT" in content

    font = first["objects"][first["resources"]["Font"]["F1"]][0]
    assert "/Subtype /TrueType /BaseFont /Aileron-Regular /FirstChar 48 /LastChar 57" in font
    widths = [float(v) for v in re.search(r"/Widths \[(.*?)\]", font).group(1).split()]
    aileron = ImageFont.load_default(size=1000)
    assert widths == pytest.approx([aileron.getlength(str(d)) for d in range(10)], abs=1)
    descriptor = first["objects"][int(re.search(r"/FontDescriptor (\d+) 0 R", font).group(1))][0]
    embedded = first["objects"][int(re.search(r"/FontFile2 (\d+) 0 R", descriptor).group(1))][1]
    assert embedded == aileron.font_bytes  # Pillow's own Aileron, CC0, whole


def test_a_leader_runs_to_a_dot_inside_its_region(tmp_path):
    rendered = _full_page()
    (label,) = [label for label in rendered.labels if label.leader is not None]
    path, (first, _) = _save(tmp_path, rendered.drawing)
    (start, end) = label.leader
    assert f"{start[0] + 0.5:.2f} {start[1] + 0.5:.2f} m\n{end[0] + 0.5:.2f} {end[1] + 0.5:.2f} l S" in first["content"]
    back = pdf_reading.render(path, 0, 4.0).mean(axis=2)[40:, 40:]
    x, y = (int(round(v)) for v in end)
    assert back[y, x] == pytest.approx(LABEL_GRAY, abs=2)  # the dot, solid, at the point it numbers


def test_the_legend_is_vector_swatches_in_their_exact_colors_numbered_on_its_own_sheet(tmp_path):
    palette = [(i * 6, 100, 255 - i * 6) for i in range(30)] + [(250, 250, 250), (5, 5, 5)]
    path, (_, second) = _save(tmp_path, _drawn(ALIGNED, _split(ALIGNED), [0, 1]).drawing, palette)
    content = second["content"]

    fills = re.findall(r"(\S+) (\S+) (\S+) rg (\S+) (\S+) (\S+) (\S+) re f", content)
    assert len(fills) == len(palette)
    per_row = int(A4.printable_mm()[0] // (SWATCH_MM + SWATCH_GAP_MM))  # 12 across 190 mm
    for index, ((r, g, b, x, y, w, h), rgb) in enumerate(zip(fills, palette)):
        assert tuple(round(float(v) * 255) for v in (r, g, b)) == rgb
        row, column = divmod(index, per_row)
        assert (float(x), float(y)) == pytest.approx((SWATCH_GAP_MM + column * (SWATCH_MM + SWATCH_GAP_MM), SWATCH_GAP_MM + row * (SWATCH_MM + SWATCH_GAP_MM)))
        assert (float(w), float(h)) == (SWATCH_MM, SWATCH_MM)
    assert content.count(f"0 G {SWATCH_BORDER_MM} w") == len(palette)
    numbers = re.findall(r"(\S+) g BT /F1 (\S+) Tf 1 0 0 -1 \S+ \S+ Tm \((\d+)\) Tj ET", content)
    assert [text for _, _, text in numbers] == [str(i + 1) for i in range(len(palette))]
    assert all(float(size) == pytest.approx(SWATCH_MM * LABEL_FONT_RATIO) for _, size, _ in numbers)
    assert [fill for fill, _, _ in numbers][-2:] == ["0", "1"]  # black on the light swatch, white on the dark one
    assert re.match(rf"q {PT_PER_MM:.8f} 0 0 -{PT_PER_MM:.8f} 28.3465 813.5433 cm", content)  # mm from the top-left margin

    # Read back at 2 px a millimeter, each swatch's middle is its color, and the number in it is centered.
    sheet = pdf_reading.render(path, 1, 2.0)
    for index, rgb in enumerate(palette):
        row, column = divmod(index, per_row)
        x0 = round((A4.margin_mm + SWATCH_GAP_MM + column * (SWATCH_MM + SWATCH_GAP_MM)) * 2)
        y0 = round((A4.margin_mm + SWATCH_GAP_MM + row * (SWATCH_MM + SWATCH_GAP_MM)) * 2)
        swatch = sheet[y0 + 2 : y0 + 22, x0 + 2 : x0 + 22]
        assert tuple(swatch[1, 1]) == pytest.approx(rgb, abs=1)
        ys, xs = np.nonzero((np.abs(swatch - np.array(rgb)) > 40).any(axis=2))
        assert ((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2) == pytest.approx((9.5, 9.5), abs=1)


def test_the_same_page_gives_the_same_file(tmp_path):
    drawing = _full_page().drawing
    export.save_pdf(drawing, PALETTE, tmp_path / "a.pdf")
    export.save_pdf(drawing, PALETTE, tmp_path / "b.pdf")

    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()
    assert b"CreationDate" not in (tmp_path / "a.pdf").read_bytes()


def test_a_page_without_numbers_ink_or_lettering_still_makes_a_valid_file(tmp_path):
    ids = np.zeros((300, 400), dtype=np.int32)
    rendered = render_page((400, 300), [], ids)
    path, (first, second) = _save(tmp_path, rendered.drawing, [])
    assert "XObject" not in first["resources"] and "f*" not in first["content"] and "BT" not in first["content"]
    assert len(trace_boundaries(ids)) == 1  # the page edge, the one line, half of it off the paper
    # Read back, the frame is drawn round the page and nothing else is; the legend's sheet is blank paper.
    back = pdf_reading.render(path, 0, 4.0).mean(axis=2)
    width, height = print_scale((400, 300)).printed_size_mm  # on a landscape sheet
    left, top = round((297 - width) / 2 * 4), round((210 - height) / 2 * 4)
    assert back[top + 4 : top + round(height * 4) - 4, left + 4 : left + round(width * 4) - 4].min() == 255
    assert back[top + 40, left - 1 : left + 2].min() < 255  # the frame's half that is on the paper
    assert pdf_reading.render(path, 1, 1.0).min() == 255


def test_qt_s_pdf_reader_opens_it_as_two_a4_sheets_with_the_page_in_place(tmp_path):
    pytest.importorskip("PySide6.QtPdf")
    size = (1600, 1000)
    rendered = _drawn(size, _split(size), [0, 1])
    path, _ = _save(tmp_path, rendered.drawing)

    # Rendered at 1 px a millimeter, the page's frame lies where the print model puts it, to within a pixel.
    sheet = pdf_reading.render(path, 0, 1.0).mean(axis=2)
    assert sheet.shape == (210, 297)
    ys, xs = np.nonzero(sheet < 250)
    printed = print_scale(size).printed_size_mm
    left, top = (297 - printed[0]) / 2, (210 - printed[1]) / 2
    assert (xs.min(), ys.min()) == pytest.approx((left, top), abs=1)
    assert (xs.max() + 1, ys.max() + 1) == pytest.approx((left + printed[0], top + printed[1]), abs=1)
    assert pdf_reading.render(path, 1, 1.0).shape == (297, 210, 3)


def test_a_png_stacks_the_page_over_its_legend_and_prints_at_the_page_s_size(tmp_path):
    page = Image.fromarray(np.asarray(_drawn((1100, 825), _split((1100, 825)), [0, 1]).image))
    scale = print_scale(page.size)
    palette_bgr = np.array([rgb[::-1] for rgb in PALETTE], dtype=np.uint8)
    legend = render_legend(palette_bgr, page.width, scale.px_per_mm)
    export.save_png(page, legend, tmp_path / "page.png")

    with Image.open(tmp_path / "page.png") as saved:
        gap = round(scale.mm_to_px(export.LEGEND_GAP_MM))
        assert saved.size == (1100, 825 + gap + legend.height)
        assert saved.info["dpi"] == pytest.approx((scale.dpi, scale.dpi), abs=0.01)
        stacked = np.asarray(saved.convert("RGB"))
    np.testing.assert_array_equal(stacked[:825], np.asarray(page))
    np.testing.assert_array_equal(stacked[825 + gap :], np.asarray(legend))


def _read_back_page(path) -> np.ndarray:
    """The aligned page's sheet read back at 4 px a millimeter, where its pixels lie on the page's: HxWx3 float."""
    width, height = ALIGNED
    return pdf_reading.render(path, 0, 4.0)[40 : 40 + height, 40 : 40 + width]


def _away_from_edges(ids: np.ndarray, *masks: np.ndarray, reach: int = 2) -> np.ndarray:
    """The pixels further than ``reach`` from any boundary of ``ids`` and from ``masks``."""
    edges = np.zeros(ids.shape, dtype=bool)
    edges[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    edges[:, 1:] |= ids[:, :-1] != ids[:, 1:]
    edges[:-1, :] |= ids[:-1, :] != ids[1:, :]
    edges[1:, :] |= ids[:-1, :] != ids[1:, :]
    for mask in masks:
        edges |= mask
    kernel = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
    return ~cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)


def test_a_completed_pdf_fills_every_region_with_its_paint_and_prints_the_picture_s_ink_over_it(tmp_path):
    rendered, painting = _painted_full_page()
    drawing = rendered.drawing
    path = tmp_path / "completed.pdf"
    export.save_pdf(drawing, PALETTE, path, version=Version.COMPLETED, painting=painting)
    first, second = pdf_reading.sheets(path.read_bytes())

    content = first["content"]
    # One path a paint, filled by the even-odd rule and its edge stroked in its own color; then the letters and the ink,
    # solid in the ink's gray. No line, leader or number: the paint covers them.
    fills = re.findall(r"(\S+ \S+ \S+) rg \1 RG\n[^a-zA-Z]*?[\d.]+ [\d.]+ m", content)
    assert fills == [" ".join(export._gray(v) for v in rgb) for rgb in PALETTE]
    assert content.count("B*") == len(PALETTE) and content.count("\nf*\n") == 2 and "BT" not in content
    assert f"{export._num(export.FILL_EDGE_MM * 4.0)} w" in content  # 4 px a millimeter
    assert first["resources"] == {} and second["size_mm"] == pytest.approx((210.0, 297.0), abs=1e-3)

    back = _read_back_page(path)
    image = np.asarray(completed(painting)).astype(np.float64)
    letters = pdf_reading.inside_even_odd(drawing.lettering, painting.region_id_map.shape)
    inside = _away_from_edges(painting.region_id_map, drawing.ink, letters)
    np.testing.assert_allclose(back[inside], image[inside], atol=1)  # each region its paint
    bare = inside & (painting.region_id_map < 0)
    assert bare.sum() > 50 * 70 and (back[bare] == 255).all()  # the patch in no region, white
    solid = cv2.erode(drawing.ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    np.testing.assert_allclose(back[solid], 30, atol=1)
    assert np.abs(back - image).mean() < 1.0


def test_a_tinted_pdf_is_the_page_multiplied_into_a_wash_of_its_paint(tmp_path):
    rendered, painting = _painted_full_page()
    drawing = rendered.drawing
    path = tmp_path / "tinted.pdf"
    export.save_pdf(drawing, PALETTE, path, version=Version.TINTED, painting=painting)
    first, _ = pdf_reading.sheets(path.read_bytes())

    content = first["content"]
    fills = re.findall(r"(\S+ \S+ \S+) rg \1 RG\n", content)
    assert fills == [" ".join(export._gray(v) for v in tint_rgb(rgb)) for rgb in PALETTE]
    assert content.endswith("B*\nq /Mu gs /Pg Do Q\nQ")
    objects = first["objects"]
    assert objects[first["resources"]["ExtGState"]["Mu"]][0] == "<< /Type /ExtGState /BM /Multiply >>"
    # The page is drawn on its own, as the page's sheet draws it, an isolated group laid over the wash.
    form, page_content = objects[first["resources"]["XObject"]["Pg"]]
    assert "/Subtype /Form" in form and "/Group << /S /Transparency /I true /CS /DeviceRGB >>" in form
    page_path = tmp_path / "page.pdf"
    export.save_pdf(drawing, PALETTE, page_path)
    page_sheet = pdf_reading.sheets(page_path.read_bytes())[0]["content"]
    assert page_content.decode("latin-1") in page_sheet

    back = _read_back_page(path)
    image = np.asarray(tinted(rendered.image, painting)).astype(np.float64)
    page = _read_back_page(page_path)
    # Away from the regions' edges, the read-back page multiplied by its wash, where the page is bare the wash alone.
    inside = _away_from_edges(painting.region_id_map)
    wash = painting.fill([tint_rgb(rgb) for rgb in PALETTE]).astype(np.float64)
    np.testing.assert_allclose(back[inside], (page * wash / 255)[inside], atol=2)  # the renderer truncates
    # Where both pages are bare -- the raster page's numbers and letters are anti-aliased its own way -- the tinted image.
    paper = inside & (page == 255).all(axis=2) & (np.asarray(rendered.image) == 255).all(axis=2)
    assert paper.mean() > 0.5
    np.testing.assert_allclose(back[paper], image[paper], atol=1)
    assert np.abs(back - image).mean() < 1.0


@pytest.mark.parametrize("version", [Version.COMPLETED, Version.TINTED])
def test_a_painted_pdf_needs_what_the_page_is_painted_with(tmp_path, version):
    with pytest.raises(ValueError, match=f"{version.value.lower()} version"):
        export.save_pdf(_full_page().drawing, PALETTE, tmp_path / "page.pdf", version=version)


def test_a_version_there_is_not_is_refused(tmp_path):
    _, painting = _painted_full_page()
    with pytest.raises(ValueError, match="no such version"):
        export.save_pdf(_full_page().drawing, PALETTE, tmp_path / "page.pdf", version="Completed", painting=painting)


@pytest.mark.parametrize("version", [Version.COMPLETED, Version.TINTED])
def test_a_paint_s_path_runs_along_no_boundary_twice_where_two_regions_of_it_touch(tmp_path, version):
    # On the full page the 2 mm square, region 4, lies in region 0, and both take paint 0: their boundary divides no
    # paint, so the path round paint 0 doesn't run along it, once for each, which a viewer would draw as a seam.
    rendered, painting = _painted_full_page()
    path = tmp_path / "painted.pdf"
    export.save_pdf(rendered.drawing, PALETTE, path, version=version, painting=painting)
    content = pdf_reading.sheets(path.read_bytes())[0]["content"]

    paths = re.findall(r"(?s) RG\n(.*?)\nB\*", content)
    assert len(paths) == len(PALETTE)
    for fill in paths:
        steps = Counter()
        previous = None
        for word in fill.split("\n"):
            x, y, op = word.split()
            point = (x, y)
            if op == "l":
                steps[tuple(sorted((previous, point)))] += 1
            previous = point
        assert steps and max(steps.values()) == 1


@pytest.mark.parametrize("version", [Version.COMPLETED, Version.TINTED])
def test_a_paint_s_path_is_smoothed_as_the_page_s_lines_are(tmp_path, monkeypatch, version):
    rendered, painting = _painted_full_page()
    smoothing = []
    real = export.region_outlines
    def spy(paints, smoothing_px):
        smoothing.append(smoothing_px)
        return real(paints, smoothing_px)

    monkeypatch.setattr(export, "region_outlines", spy)
    size = rendered.drawing.size
    for smoothing_mm in (PageStyle().line_smoothing_mm, 1.5):
        drawing = dataclasses.replace(rendered.drawing, style=PageStyle(line_smoothing_mm=smoothing_mm))
        export.save_pdf(drawing, PALETTE, tmp_path / "painted.pdf", version=version, painting=painting)
    assert smoothing == [smoothing_length_px(size), smoothing_length_px(size, 1.5)]
    assert smoothing[1] > smoothing[0]

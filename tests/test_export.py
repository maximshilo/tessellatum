"""Export: a PNG that prints at the page's size, and a vector PDF of two A4 sheets, the page placed by the print model."""

import re

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFont

from tessellatum.core import export
from tessellatum.core.boundaries import trace_boundaries
from tessellatum.core.legend import LABEL_FONT_RATIO, SWATCH_BORDER_MM, SWATCH_GAP_MM, SWATCH_MM, render_legend
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


def _full_page():
    """The aligned page with everything a page can print: lines, numbers, a leader, printed ink and lettering."""
    width, height = ALIGNED
    ids = _split(ALIGNED)
    cv2.circle(ids, (200, 300), 120, 2, -1)
    cv2.rectangle(ids, (450, 600), (700, 900), 3, -1)
    ids[700:708, 100:108] = 4  # 2 mm across: too small for its number, which goes outside with a leader
    ink = np.zeros((height, width), dtype=np.uint8)
    cv2.line(ink, (60, 900), (300, 1050), 1, 3)
    cv2.circle(ink, (580, 200), 40, 1, 2)
    ink[500:540, 600:610] = 1  # a hole in it
    ink[515:525, 603:607] = 0
    area = np.zeros((height, width), dtype=bool)
    area[1000:1060, 420:740] = True
    lettering = np.zeros((height, width), dtype=np.uint8)
    columns = np.arange(width)
    lettering[area] = np.broadcast_to(np.where(np.sin(columns / 3.0) > 0.3, 230, 40), (height, width))[area]
    return _drawn(ALIGNED, ids, [0, 1, 2, 3, 0], ink=ink.astype(bool), ink_gray=30, lettering=lettering, lettering_area=area)


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
    # The renderer grows an image a pixel at its far edges where it lies on the pixel grid; the lettering's own pixels
    # are checked in the file below.
    lettered = cv2.dilate(drawing.lettering_area.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    assert np.abs(back - page).mean() < 1.0
    np.testing.assert_array_equal(back[drawing.ink & ~near].round(), 30)  # the printed ink, solid, to the pixel
    np.testing.assert_array_equal(back[~near & ~drawing.ink & ~lettered], 255)  # bare paper everywhere else
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


def test_the_printed_ink_is_the_outline_of_its_pixels_filled(tmp_path):
    drawing = _full_page().drawing
    _, (first, _) = _save(tmp_path, drawing)
    content = first["content"]
    rings = re.search(r"(?s)0\.1176 g\n(.*?)\nf\*\n", content).group(1)  # the ink's gray, 30, then the filled outlines
    width, height = ALIGNED
    # Filled by the even-odd rule, at the pixels' middles: a pixel is inside when an odd number of the outlines'
    # upright edges lie to its left in its row.
    parity = np.zeros((height, width + 1), dtype=np.int64)
    for ring in re.split(r"\n(?=\S+ \S+ m$)", rings, flags=re.M):
        points = np.array([[int(v) for v in word.split()[:2]] for word in ring.split("\n")])
        assert points.dtype.kind == "i" and np.array_equal(points[0], points[-1])  # pixel corners, and closed
        for (xa, ya), (xb, yb) in zip(points[:-1], points[1:]):
            assert xa == xb or ya == yb  # along the cracks between pixels
            if xa == xb:
                parity[min(ya, yb) : max(ya, yb), xa] += 1
    inside = np.cumsum(parity, axis=1)[:, :width] % 2 == 1
    np.testing.assert_array_equal(inside, drawing.ink)


def test_the_lettering_is_its_tones_at_the_page_s_pixels_darkening_what_is_under_it(tmp_path):
    drawing = _full_page().drawing
    path, (first, _) = _save(tmp_path, drawing)
    content = first["content"]
    image = first["resources"]["XObject"]["Lt"]
    info, pixels = first["objects"][image]
    w, h = (int(v) for v in re.search(r"/Width (\d+) /Height (\d+)", info).groups())
    assert "/ColorSpace /DeviceGray /BitsPerComponent 8" in info
    assert re.search(rf"q /Dk gs {w} 0 0 -{h} 420 1060 cm /Lt Do Q", content)  # its box, rows 1000-1059, columns 420-739
    dk = first["resources"]["ExtGState"]["Dk"]
    assert first["objects"][dk][0] == "<< /Type /ExtGState /BM /Darken >>"
    expected = paper_under(drawing.lettering[1000:1060, 420:740], drawing.ink_gray)
    np.testing.assert_array_equal(np.frombuffer(pixels, dtype=np.uint8).reshape(h, w), expected)
    assert content.index("/Lt Do") < content.index("f*")  # the printed ink goes over it, as on the page


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
    back = pdf_reading.render(path, 0, 1.0)
    assert back.min() >= 0  # the renderer read it


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

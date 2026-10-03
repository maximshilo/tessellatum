"""Export: a PNG that prints at the page's size, and a PDF of two A4 sheets, the page placed by the print model."""

import os
import re
import zlib

import numpy as np
import pytest
from PIL import Image

from tessellatum.core import export
from tessellatum.core.legend import render_legend
from tessellatum.core.print_size import A4, MM_PER_INCH, PT_PER_INCH, print_scale

PT_PER_MM = PT_PER_INCH / MM_PER_INCH
PALETTE = [(200, 40, 40), (40, 160, 60), (230, 210, 120)]


def _page(size: tuple[int, int]) -> Image.Image:
    """A gray page: a frame and a few gray strokes on paper, as the renderer draws one."""
    width, height = size
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    page[[0, -1], :] = page[:, [0, -1]] = 0x59
    page[height // 3, 10 : width - 10] = 0x8C
    page[10 : height - 10, width // 2] = 0x30
    return Image.fromarray(page)


def _objects(data: bytes) -> dict[int, tuple[str, bytes | None]]:
    """Each object of a PDF written by ``export``, found through its cross-reference table: (dictionary, stream)."""
    assert data.startswith(b"%PDF-1.4\n")
    xref_at = int(re.search(rb"startxref\n(\d+)\n%%EOF\n$", data).group(1))
    head = re.match(rb"xref\n0 (\d+)\n0000000000 65535 f \n", data[xref_at:])
    count = int(head.group(1))
    table = data[xref_at + head.end() :]
    objects = {}
    for number in range(1, count):
        entry = table[(number - 1) * 20 : number * 20]
        assert re.fullmatch(rb"\d{10} 00000 n \n", entry), entry
        at = int(entry[:10])
        prefix = f"{number} 0 obj\n".encode()
        assert data[at : at + len(prefix)] == prefix, number
        body = data[at + len(prefix) :]
        stream = None
        head = re.match(rb"<< [^\n]*/Length (\d+) >>\nstream\n", body)  # every dictionary is one line
        if head:
            stream = body[head.end() : head.end() + int(head.group(1))]
            assert body[head.end() + len(stream) :].startswith(b"\nendstream\nendobj\n")
            body = body[: head.end()]
        else:
            body = body[: body.index(b"\nendobj\n")]
        objects[number] = (body.decode("latin-1"), stream)
    assert re.search(rf"trailer\n<< /Size {count} /Root 1 0 R >>".encode(), data)
    return objects


def _sheets(data: bytes) -> list[dict]:
    """Each sheet: its size in mm, where its image lies (x, y from the top-left, width, height, in mm), and the image."""
    objects = _objects(data)
    assert objects[1][0] == "<< /Type /Catalog /Pages 2 0 R >>"
    kids = [int(k) for k in re.findall(r"(\d+) 0 R", re.search(r"/Kids \[(.*?)\]", objects[2][0]).group(1))]
    assert f"/Count {len(kids)}" in objects[2][0]
    sheets = []
    for kid in kids:
        page = objects[kid][0]
        sheet_w, sheet_h = (float(v) / PT_PER_MM for v in re.search(r"/MediaBox \[0 0 (\S+) (\S+)\]", page).groups())
        content = objects[int(re.search(r"/Contents (\d+) 0 R", page).group(1))][1].decode()
        w, h, x, y = (float(v) / PT_PER_MM for v in re.fullmatch(r"q (\S+) 0 0 (\S+) (\S+) (\S+) cm /Im0 Do Q", content).groups())
        info, stream = objects[int(re.search(r"/Im0 (\d+) 0 R", page).group(1))]
        width, height = (int(v) for v in re.search(r"/Width (\d+) /Height (\d+)", info).groups())
        gray = "/ColorSpace /DeviceGray" in info
        assert "/BitsPerComponent 8 /Filter /FlateDecode" in info and (gray or "/ColorSpace /DeviceRGB" in info)
        pixels = np.frombuffer(zlib.decompress(stream), dtype=np.uint8).reshape((height, width) if gray else (height, width, 3))
        sheets.append(
            {"size_mm": (sheet_w, sheet_h), "at_mm": (x, sheet_h - y - h), "image_mm": (w, h), "gray": gray, "pixels": pixels}
        )
    return sheets


@pytest.mark.parametrize("size", [(1600, 1000), (900, 1400), (600, 450)])
def test_a_pdf_is_two_a4_sheets_the_page_placed_as_the_print_model_prints_it(tmp_path, size):
    # D-052 (Q36): the page fills the printable area inside the 10 mm margins, centered, on a landscape sheet when wider
    # than tall; the legend gets a portrait sheet of its own.
    page = _page(size)
    export.save_pdf(page, PALETTE, tmp_path / "page.pdf")
    first, second = _sheets((tmp_path / "page.pdf").read_bytes())

    scale = print_scale(size)
    sheet = (297.0, 210.0) if scale.landscape else (210.0, 297.0)
    printed = scale.printed_size_mm
    assert first["size_mm"] == pytest.approx(sheet, abs=1e-3)
    assert first["image_mm"] == pytest.approx(printed, abs=1e-3)
    assert first["at_mm"] == pytest.approx(((sheet[0] - printed[0]) / 2, (sheet[1] - printed[1]) / 2), abs=1e-3)
    assert min(first["at_mm"]) == pytest.approx(A4.margin_mm, abs=1e-3)  # it reaches the margins on one side
    assert second["size_mm"] == pytest.approx((210.0, 297.0), abs=1e-3)
    assert second["at_mm"] == pytest.approx((A4.margin_mm, A4.margin_mm), abs=1e-3)


def test_the_page_is_stored_losslessly_in_gray_and_the_legend_in_color_at_the_page_s_resolution(tmp_path):
    # Pillow's own PDF writer saves both as JPEG, which grays the paper around every line and moves the swatches' colors.
    page = _page((1600, 1000))
    export.save_pdf(page, PALETTE, tmp_path / "page.pdf")
    first, second = _sheets((tmp_path / "page.pdf").read_bytes())

    assert first["gray"]
    np.testing.assert_array_equal(first["pixels"], np.asarray(page.convert("L")))
    scale = print_scale(page.size)
    palette_bgr = np.array([rgb[::-1] for rgb in PALETTE], dtype=np.uint8)
    legend = render_legend(palette_bgr, round(scale.mm_to_px(A4.printable_mm()[0])), scale.px_per_mm)
    assert not second["gray"]
    np.testing.assert_array_equal(second["pixels"], np.asarray(legend))
    # The legend prints at the page's resolution, across the printable width: its swatches come out their size in mm.
    assert second["image_mm"] == pytest.approx((legend.width / scale.px_per_mm, legend.height / scale.px_per_mm), abs=1e-3)
    assert second["image_mm"][0] == pytest.approx(A4.printable_mm()[0], abs=0.1)


def test_a_page_with_any_color_is_stored_in_color(tmp_path):
    page = np.asarray(_page((400, 300))).copy()
    page[150, 200] = (255, 250, 255)
    export.save_pdf(Image.fromarray(page), PALETTE, tmp_path / "page.pdf")
    first = _sheets((tmp_path / "page.pdf").read_bytes())[0]

    assert not first["gray"]
    np.testing.assert_array_equal(first["pixels"], page)


def test_the_same_page_gives_the_same_file(tmp_path):
    page = _page((800, 600))
    export.save_pdf(page, PALETTE, tmp_path / "a.pdf")
    export.save_pdf(page, PALETTE, tmp_path / "b.pdf")

    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()
    assert b"CreationDate" not in (tmp_path / "a.pdf").read_bytes()


def test_qt_s_pdf_reader_opens_it_as_two_a4_sheets_with_the_page_in_place(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtPdf")
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QImage
    from PySide6.QtPdf import QPdfDocument
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])  # noqa: F841 - Qt's renderer needs an application
    page = _page((1600, 1000))
    export.save_pdf(page, PALETTE, tmp_path / "page.pdf")
    document = QPdfDocument()
    assert document.load(str(tmp_path / "page.pdf")) == QPdfDocument.Error.None_
    assert document.pageCount() == 2
    sizes = [document.pagePointSize(i) for i in range(2)]
    assert (sizes[0].width(), sizes[0].height()) == pytest.approx((297 * PT_PER_MM, 210 * PT_PER_MM), abs=0.01)
    assert (sizes[1].width(), sizes[1].height()) == pytest.approx((210 * PT_PER_MM, 297 * PT_PER_MM), abs=0.01)

    # Rendered at 1 px a millimeter, the page's frame lies where the print model puts it, to within a pixel.
    image = document.render(0, QSize(297, 210)).convertToFormat(QImage.Format_RGBA8888)
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, : 297 * 4]
    alpha = rgba.reshape(210, 297, 4)[..., 3]
    ys, xs = np.nonzero(alpha > 0)
    printed = print_scale(page.size).printed_size_mm
    left, top = (297 - printed[0]) / 2, (210 - printed[1]) / 2
    assert (xs.min(), ys.min()) == pytest.approx((left, top), abs=1)
    assert (xs.max() + 1, ys.max() + 1) == pytest.approx((left + printed[0], top + printed[1]), abs=1)
    document.close()


def test_a_png_stacks_the_page_over_its_legend_and_prints_at_the_page_s_size(tmp_path):
    page = _page((1100, 825))
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

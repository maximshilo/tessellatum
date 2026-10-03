"""Reading back the PDFs ``export`` writes, for tests: their objects, sheets and content, and rendering them with QtPdf."""

from __future__ import annotations

import os
import re
import zlib

import numpy as np

from tessellatum.core.print_size import MM_PER_INCH, PT_PER_INCH

PT_PER_MM = PT_PER_INCH / MM_PER_INCH


def objects(data: bytes) -> dict[int, tuple[str, bytes | None]]:
    """Each object of a PDF written by ``export``, found through its cross-reference table: (dictionary, stream).

    A stream comes back decompressed.
    """
    assert data.startswith(b"%PDF-1.4\n")
    xref_at = int(re.search(rb"startxref\n(\d+)\n%%EOF\n$", data).group(1))
    head = re.match(rb"xref\n0 (\d+)\n0000000000 65535 f \n", data[xref_at:])
    count = int(head.group(1))
    table = data[xref_at + head.end() :]
    found = {}
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
            if b"/Filter /FlateDecode" in body:
                stream = zlib.decompress(stream)
        else:
            body = body[: body.index(b"\nendobj\n")]
        found[number] = (body.decode("latin-1"), stream)
    assert re.search(rf"trailer\n<< /Size {count} /Root 1 0 R >>".encode(), data)
    return found


def sheets(data: bytes) -> list[dict]:
    """Each sheet: its size in mm, its content stream as text, and its resources, by kind and name, as object numbers."""
    found = objects(data)
    assert found[1][0] == "<< /Type /Catalog /Pages 2 0 R >>"
    kids = [int(k) for k in re.findall(r"(\d+) 0 R", re.search(r"/Kids \[(.*?)\]", found[2][0]).group(1))]
    assert f"/Count {len(kids)}" in found[2][0]
    result = []
    for kid in kids:
        page = found[kid][0]
        sheet_w, sheet_h = (float(v) / PT_PER_MM for v in re.search(r"/MediaBox \[0 0 (\S+) (\S+)\]", page).groups())
        content = found[int(re.search(r"/Contents (\d+) 0 R", page).group(1))][1].decode("latin-1")
        resources = {
            kind: dict((name, int(ref)) for name, ref in re.findall(r"/(\w+) (\d+) 0 R", refs))
            for kind, refs in re.findall(r"/(\w+) << ((?:/\w+ \d+ 0 R ?)+) >>", re.search(r"/Resources << (.*) >> /Contents", page).group(1))
        }
        result.append({"size_mm": (sheet_w, sheet_h), "content": content, "resources": resources, "objects": found})
    return result


def page_transform(content: str) -> tuple[float, float, float, float]:
    """The page's placement from its sheet's first ``cm``: (left, top) in mm on the sheet, and mm per pixel in x and y."""
    a, b, c, d, e, f = (float(v) for v in re.match(r"q (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) cm", content).groups())
    assert b == c == 0 and a > 0 and d == -a
    return e / PT_PER_MM, f / PT_PER_MM, a / PT_PER_MM, -d / PT_PER_MM


def render(path, index: int, px_per_mm: float, clip_mm: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Sheet ``index`` of the PDF at ``path`` rendered by QtPdf at ``px_per_mm``, composited on white paper: HxWx3 float.

    ``clip_mm`` (left, top, width, height) renders only that part of the sheet.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRect, QSize
    from PySide6.QtGui import QImage
    from PySide6.QtPdf import QPdfDocument, QPdfDocumentRenderOptions
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])  # noqa: F841 - Qt's renderer needs an application
    document = QPdfDocument()
    assert document.load(str(path)) == QPdfDocument.Error.None_
    size_pt = document.pagePointSize(index)
    sheet = QSize(round(size_pt.width() / PT_PER_MM * px_per_mm), round(size_pt.height() / PT_PER_MM * px_per_mm))
    options = QPdfDocumentRenderOptions()
    if clip_mm is None:
        image = document.render(index, sheet)
    else:
        left, top, width, height = (round(v * px_per_mm) for v in clip_mm)
        options.setScaledSize(sheet)
        options.setScaledClipRect(QRect(left, top, width, height))
        image = document.render(index, QSize(width, height), options)
    document.close()
    image = image.convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())[:, : width * 4]
    rgba = rgba.reshape(height, width, 4).astype(np.float64)
    alpha = rgba[..., 3:] / 255  # QtPdf renders onto a transparent background
    return rgba[..., :3] * alpha + 255 * (1 - alpha)

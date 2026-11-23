"""A page laid out once to print and drawn at any resolution (T3b): the layout at 300 dpi on its paper, whatever the
picture's pixels, and every output -- the PDF, a PNG at any dpi -- drawn from it, its numbers where it put them."""

import dataclasses

import numpy as np
import pytest

from tessellatum.core import export, pipeline
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.print_size import A3, MM_PER_INCH, long_edge_at_dpi, print_scale
from tessellatum.core.render import PageStyle

import pdf_reading


def _blocks(size: tuple[int, int]) -> np.ndarray:
    """Four flat blocks dotted with small squares, ``size`` (width, height)."""
    w, h = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[: h // 2, : w // 2] = (200, 40, 40)
    image[: h // 2, w // 2 :] = (40, 180, 40)
    image[h // 2 :, : w // 2] = (40, 40, 200)
    image[h // 2 :, w // 2 :] = (210, 210, 40)
    step = max(8, w // 16)
    for y in range(step, h // 2 - step, step):
        for x in range(w // 2 + step, w - step, step):
            image[y : y + step // 3, x : x + step // 3] = (40, 40, 200)
    return image


def test_the_print_layout_is_300_dpi_on_its_paper_however_small_the_picture():
    small = _blocks((600, 400))
    assert pipeline.export_long_edge(small) == long_edge_at_dpi((600, 400)) == 3272
    assert pipeline.export_long_edge(small, A3) == long_edge_at_dpi((600, 400), paper=A3)


def test_a_small_picture_is_scaled_up_to_its_working_size():
    # D-009: a page is made where its brush is a few pixels across, the picture scaled up if it has fewer.
    small, params = _blocks((600, 400)), params_for_preset("Medium")
    working = pipeline.working_long_edge(small, params)
    assert working > 600
    page = pipeline.generate(small, params, working)
    assert max(page.page.size) == working
    assert pipeline.resize_to_long_edge(small, 900).shape == (400, 600, 3)  # the detectors' size: never scaled up
    assert pipeline.resize_to_long_edge(small, 900, upscale=True).shape == (600, 900, 3)


@pytest.fixture(scope="module")
def laid_out():
    """A page laid out to print, from a picture smaller than the layout."""
    image, params = _blocks((800, 560)), params_for_preset("Easy")
    page = pipeline.generate(image, params, pipeline.export_long_edge(image), style=PageStyle(), vector=True)
    assert max(page.page.size) == pipeline.export_long_edge(image)
    assert page.drawing.labels and all(label.exact for label in page.drawing.labels)
    return page


def _number_ink(drawing, dpi: float) -> np.ndarray:
    """How much ink the numbers alone put on the page drawn at ``dpi``: the page less the page without them."""
    with_numbers = np.asarray(export.rasterize(drawing, dpi).convert("L"), dtype=np.float64)
    bare = dataclasses.replace(drawing, labels=[label for label in drawing.labels if label.leader is not None])
    without = np.asarray(export.rasterize(bare, dpi).convert("L"), dtype=np.float64)
    return np.clip(without - with_numbers, 0, None)


def _center_of_mass(weights: np.ndarray) -> tuple[float, float]:
    ys, xs = np.indices(weights.shape)
    total = weights.sum()
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


def test_a_png_at_any_dpi_is_the_printed_size_to_the_pixel(laid_out):
    printed = print_scale(laid_out.page.size).printed_size_mm
    for dpi in (150, 300, 600):
        image = export.rasterize(laid_out.drawing, dpi)
        assert image.size == tuple(round(mm * dpi / MM_PER_INCH) for mm in printed)


def test_a_png_is_at_most_the_cap_across(laid_out):
    drawing = laid_out.drawing
    most = export.max_raster_dpi(drawing)
    assert max(export.raster_size(drawing, most)) == export.MAX_RASTER_LONG_EDGE
    assert max(export.raster_size(drawing, most + 0.1)) > export.MAX_RASTER_LONG_EDGE
    with pytest.raises(ValueError, match="at most"):
        export.rasterize(drawing, most + 1)
    # A2 at 600 dpi is under it.
    a2 = dataclasses.replace(drawing, style=PageStyle(paper="A2"))
    assert export.max_raster_dpi(a2) > 600


def test_the_numbers_are_at_the_same_places_on_paper_at_every_dpi(laid_out):
    drawing = laid_out.drawing
    px_per_mm = print_scale(drawing.size).px_per_mm
    placed = [label for label in drawing.labels if label.leader is None]
    assert placed
    centers = {}
    for dpi in (150, 300, 600):
        ink = _number_ink(drawing, dpi)
        out_per_mm = dpi / MM_PER_INCH
        found = []
        for label in placed:
            x0, y0, x1, y1 = (v / px_per_mm * out_per_mm for v in label.box)
            pad = out_per_mm * 0.3
            rows = slice(max(0, int(y0 - pad)), int(np.ceil(y1 + pad)))
            box = (rows, slice(max(0, int(x0 - pad)), int(np.ceil(x1 + pad))))
            cy, cx = _center_of_mass(ink[box])
            found.append(((box[1].start + cx + 0.5) / out_per_mm, (box[0].start + cy + 0.5) / out_per_mm))
        centers[dpi] = np.array(found)
    # In millimeters, to a fraction of a pixel at 150 dpi (0.17 mm).
    assert np.abs(centers[300] - centers[600]).max() < 0.05
    assert np.abs(centers[150] - centers[600]).max() < 0.08


def test_a_png_is_the_pdf_s_page_without_its_margins(laid_out, tmp_path):
    drawing = laid_out.drawing
    path = tmp_path / "page.pdf"
    export.save_pdf(drawing, laid_out.palette_rgb, path)
    dpi = 300
    scale = print_scale(drawing.size)
    sheet = pdf_reading.render(path, 0, dpi / MM_PER_INCH)
    left = (297 - scale.printed_size_mm[0]) / 2 * dpi / MM_PER_INCH
    top = (210 - scale.printed_size_mm[1]) / 2 * dpi / MM_PER_INCH
    png = np.asarray(export.rasterize(drawing, dpi), dtype=np.float64)
    crop = sheet[round(top) : round(top) + png.shape[0], round(left) : round(left) + png.shape[1]]
    assert crop.shape == png.shape
    assert np.abs(crop - png).mean() < 2.0

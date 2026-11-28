"""The paper a page is made for (T3a): A4, A3 or A2, its limits in millimeters and points applying on that sheet."""

import numpy as np
import pytest
from PIL import Image

from tessellatum.core import export, measures, pipeline
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.labels import min_font_size
from tessellatum.core.print_size import (
    A1, A2, A3, A4, LEGAL, LETTER, PAPERS, TABLOID, legend_paper, long_edge_at_dpi, print_scale,
)
from tessellatum.core.render import PageStyle

import pdf_reading


def test_the_three_sheets_and_their_printable_areas():
    assert list(PAPERS)[:3] == ["A4", "A3", "A2"]
    assert A3.printable_mm() == (277.0, 400.0) and A2.printable_mm(landscape=True) == (574.0, 400.0)
    # A page prints larger on a larger sheet, so a millimeter there is fewer of its pixels.
    size = (3000, 2000)
    per_mm = [print_scale(size, paper).px_per_mm for paper in (A4, A3, A2)]
    assert per_mm[0] > per_mm[1] > per_mm[2]
    assert print_scale(size, A3).printed_size_mm == pytest.approx((400.0, 400.0 * 2 / 3))
    assert long_edge_at_dpi(size, 300, A3) == round(400 * 300 / 25.4)


def test_a_style_s_sizes_apply_on_its_sheet():
    size = (2000, 1500)
    a4, a3 = PageStyle(), PageStyle(paper="A3")
    assert a4 == PageStyle(paper="A4") and a4.paper_format is A4 and a3.paper_format is A3
    assert a3.line_width_px(size) < a4.line_width_px(size)
    assert a3.line_width_px(size) == pytest.approx(print_scale(size, A3).mm_to_px(a3.line_width_mm))
    assert min_font_size(size, 3.0, exact=True, paper=A3) < min_font_size(size, 3.0, exact=True)
    with pytest.raises(KeyError):
        PageStyle(paper="A0").paper_format


def _picture() -> np.ndarray:
    """Flat blocks dotted with small squares, larger than any sheet's working size here."""
    image = np.zeros((1800, 2400, 3), dtype=np.uint8)
    image[:900, :1200] = (200, 40, 40)
    image[:900, 1200:] = (40, 180, 40)
    image[900:, :1200] = (40, 40, 200)
    image[900:, 1200:] = (210, 210, 40)
    for y in range(60, 840, 90):
        for x in range(1260, 2340, 90):
            image[y : y + 30, x : x + 30] = (40, 40, 200)
    return image


def test_a_larger_sheet_makes_the_page_at_more_pixels():
    image, params = _picture(), params_for_preset("Medium")
    edges = [pipeline.working_long_edge(image, params, paper=paper) for paper in (A4, A3, A2)]
    assert edges[0] < edges[1] < edges[2]  # past the picture's 2400 px on A2: scaled up (D-009)
    exports = [pipeline.export_long_edge(image, paper) for paper in (A4, A3, A2)]
    assert exports == [long_edge_at_dpi((2400, 1800), 300, paper) for paper in (A4, A3, A2)]
    assert edges[2] < exports[2]


def test_a_page_made_for_a3_is_finer_and_a4_is_as_before():
    image, params = _picture(), params_for_preset("Medium")
    a4 = pipeline.generate(image, params, 1200, collect_analysis=True)
    again = pipeline.generate(image, params, 1200, style=PageStyle(paper="A4"), collect_analysis=True)
    assert again.page.tobytes() == a4.page.tobytes()
    a3 = pipeline.generate(image, params, 1200, style=PageStyle(paper="A3"), collect_analysis=True)
    # The smallest region is a smaller share of the picture on the larger sheet.
    assert a3.analysis.min_region_area_px < a4.analysis.min_region_area_px
    assert a3.drawing.style.paper == "A3"


@pytest.mark.parametrize("paper, size, sheet", [
    ("A3", (1600, 1000), (420.0, 297.0)),
    ("A3", (1000, 1600), (297.0, 420.0)),
    ("A2", (1600, 1000), (594.0, 420.0)),
    ("A1", (1000, 1600), (594.0, 841.0)),
    ("Letter", (1000, 1600), (215.9, 279.4)),
    ("Legal", (1600, 1000), (355.6, 215.9)),
    ("Tabloid", (1000, 1600), (279.4, 431.8)),
])
def test_a_pdf_s_sheets_are_its_paper_the_legend_on_one_of_its_own(tmp_path, paper, size, sheet):
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, : size[0] // 2] = (200, 40, 40)
    image[:, size[0] // 2 :] = (40, 180, 40)
    page = pipeline.generate(image, params_for_preset("Easy"), max(size), style=PageStyle(paper=paper), vector=True)
    path = tmp_path / "page.pdf"
    export.save_pdf(page.drawing, page.palette_rgb, path)
    first, second = pdf_reading.sheets(path.read_bytes())
    legend_sheet = legend_paper(PAPERS[paper])  # A1's on A4 (T2)
    portrait = legend_sheet.width_mm, legend_sheet.height_mm
    assert first["size_mm"] == pytest.approx(sheet, abs=1e-3)
    assert second["size_mm"] == pytest.approx(portrait, abs=1e-3)
    # The page fills the printable area inside the 10 mm margins, centered.
    left, _top, mm_per_px, _ = pdf_reading.page_transform(first["content"])
    scale = print_scale(page.page.size, PAPERS[paper])
    assert 1 / mm_per_px == pytest.approx(scale.px_per_mm, rel=1e-6)
    assert left == pytest.approx((sheet[0] - scale.printed_size_mm[0]) / 2, abs=1e-3)


@pytest.mark.parametrize("paper, points", [
    (LETTER, (612, 792)), (LEGAL, (612, 1008)), (TABLOID, (792, 1224)), (A1, (1683.8, 2383.9)),
])
def test_the_us_sheets_and_a1_measure_their_size_in_points(paper, points):
    assert (paper.width_mm * 72 / 25.4, paper.height_mm * 72 / 25.4) == pytest.approx(points, abs=0.1)
    assert paper.margin_mm == 10.0 and paper.name in PAPERS


@pytest.mark.parametrize("name", ["Letter", "A1"])
def test_a_page_on_letter_or_a1_keeps_the_rules_on_its_sheet(name):
    # The smallest region is the same size on paper as on A4, and so are the brush and the smallest number.
    image, params = _picture(), params_for_preset("Medium")
    page = pipeline.generate(image, params, 1600, style=PageStyle(paper=name), collect_analysis=True)
    a4 = pipeline.generate(image, params, 1600, collect_analysis=True)
    scale, a4_scale = (print_scale(p.page.size, p.drawing.style.paper_format) for p in (page, a4))
    on_paper = page.analysis.min_region_area_px / scale.mm2_to_px(1)
    assert on_paper == pytest.approx(a4.analysis.min_region_area_px / a4_scale.mm2_to_px(1), rel=0.02)
    ids = page.model.region_id_map
    found = measures.measure_page(ids, page.model.labels, page.model.legend_size, scale.px_per_mm,
                                  params.min_width_mm, params.min_region_area_mm2, page.model.style.min_label_pt,
                                  scale.px_per_pt)
    assert found.min_label_pt is None or found.min_label_pt >= page.model.style.min_label_pt - 1e-6
    assert page.drawing.style.paper == name


def test_a_png_records_the_resolution_it_prints_at_on_its_paper(tmp_path):
    page, legend = Image.new("RGB", (3000, 2000), "white"), Image.new("RGB", (3000, 200), "white")
    export.save_png(page, legend, tmp_path / "a3.png", A3)
    with Image.open(tmp_path / "a3.png") as png:
        assert png.info["dpi"] == pytest.approx((print_scale((3000, 2000), A3).dpi,) * 2, abs=0.01)
    assert print_scale((3000, 2000), A3).dpi < print_scale((3000, 2000)).dpi

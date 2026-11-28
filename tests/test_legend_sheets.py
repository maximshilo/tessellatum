"""The legend's PDF sheets on any paper (paper-sizes T2): never printed below its own sizes, it goes on over more sheets
where it doesn't fit, and a poster's or a canvas's legend prints on an ordinary sheet."""

import re

import numpy as np
import pytest

from tessellatum.core import export
from tessellatum.core.legend import DETAIL_PT, NAME_RECIPE, NUMBER_ONLY, SWATCH_GAP_MM, SWATCH_MM
from tessellatum.core.print_size import A1, A2, A3, A4, LEGAL, LETTER, TABLOID, PaperFormat, legend_paper
from tessellatum.core.render import PageStyle, render_page

import pdf_reading
from pdf_reading import PT_PER_MM

MM_PER_PT = 25.4 / 72


def _legend(tmp_path, paper, count=64, detail=NAME_RECIPE, mixed_from=33):
    """The legend sheets of ``count`` colors on ``paper``, with names, three-line recipes, and the colors from
    ``mixed_from`` on mixed from the paints: 17 rows of four, the most 64 colors take, and two headings, 266.5 mm."""
    palette = [(i * 3 % 256, 100 + i % 50, 255 - i * 3 % 256) for i in range(count)]
    names = [f"color {i + 1}" for i in range(count)]
    recipes = [["3 Titanium white", "1 Ultramarine", "1 Cadmium yellow"]] * count
    mixes = [None] * mixed_from + [["2 Titanium white", "1 Burnt umber", "1 Ultramarine"]] * (count - mixed_from)
    document = export._Document()
    numbers = document.font(export._number_font())
    export._legend_sheet(document, palette, numbers, paper, detail, names, recipes, mixes)
    path = tmp_path / "legend.pdf"
    path.write_bytes(document.to_bytes())
    return pdf_reading.sheets(path.read_bytes())


def _text_sizes_mm(content: str) -> list[float]:
    return [float(size) for size in re.findall(r"/F1 (\S+) Tf", content)]


def _swatches(content: str) -> list[tuple[float, float]]:
    """Each swatch's top-left (x, y) in mm from the margin, as filled."""
    return [(float(x), float(y)) for x, y in re.findall(rf"rg (\S+) (\S+) {SWATCH_MM:g} {SWATCH_MM:g} re f", content)]


@pytest.mark.parametrize("paper", [LETTER, PaperFormat("Small", 120.0, 160.0, 10.0)])
def test_a_long_legend_goes_on_over_more_sheets_at_its_own_sizes(tmp_path, paper):
    sheets = _legend(tmp_path, paper)
    assert len(sheets) >= 2
    _, area_h = paper.printable_mm()
    numbers = []
    for sheet in sheets:
        content = sheet["content"]
        assert sheet["size_mm"] == pytest.approx((paper.width_mm, paper.height_mm), abs=1e-3)
        # Drawn in millimeters, not scaled down: its text 7 pt at least, its swatches SWATCH_MM, inside the margins.
        assert re.match(rf"q {PT_PER_MM:.8f} 0 0 -{PT_PER_MM:.8f} ", content)
        assert min(_text_sizes_mm(content)) >= DETAIL_PT * MM_PER_PT - 1e-4
        assert max(y for _, y in _swatches(content)) + SWATCH_MM <= area_h
        numbers += [int(n) for n in re.findall(r"\((\d+)\) Tj", content)]
    assert numbers == list(range(1, 65))  # every color once, in order
    later = " ".join(sheet["content"] for sheet in sheets[1:])
    assert "continued" in later and "continued" not in sheets[0]["content"]


def test_a_further_sheet_says_what_it_continues():
    document = export._Document()
    numbers = document.font(export._number_font())
    palette = [(i, i, i) for i in range(64)]
    export._legend_sheet(document, palette, numbers, PaperFormat("Small", 120.0, 160.0, 10.0), NUMBER_ONLY)
    sheets = pdf_reading.sheets(document.to_bytes())
    assert len(sheets) == 2  # 12 rows of 4, then 4 more
    assert "(Legend, continued)" in sheets[1]["content"] and "continued" not in sheets[0]["content"]
    assert export._continued("Mixed colors: mix each from your paints, in the parts given") == "Mixed colors, continued"


@pytest.mark.parametrize("paper", [A4, A3, A2, TABLOID, LEGAL])
def test_a_legend_that_fits_stays_on_one_sheet_as_before(tmp_path, paper):
    sheets = _legend(tmp_path, paper)
    assert len(sheets) == 1
    content = sheets[0]["content"]
    assert "continued" not in content
    # Rows 15 mm apart from the top margin: the sections one under the other, as before.
    ys = sorted({y for _, y in _swatches(content)})
    assert ys[0] == pytest.approx(5.0 + SWATCH_GAP_MM)  # under the first section's heading


@pytest.mark.parametrize("paper, legend", [
    (A4, A4), (A3, A3), (A2, A2), (LETTER, LETTER), (LEGAL, LEGAL), (TABLOID, TABLOID),
    (A1, A4),
    (PaperFormat("Canvas 40 × 50 cm", 400.0, 500.0, 10.0), A4),
    (PaperFormat("Canvas 16 × 20 in", 406.4, 508.0, 10.0, inches=True), LETTER),
    (PaperFormat("Small", 150.0, 200.0, 10.0), PaperFormat("Small", 150.0, 200.0, 10.0)),
])
def test_a_poster_s_or_a_canvas_s_legend_prints_on_an_ordinary_sheet(paper, legend):
    assert legend_paper(paper) == legend


def test_an_a1_page_s_pdf_has_its_legend_on_a4(tmp_path):
    ids = np.zeros((300, 400), dtype=np.int32)
    drawing = render_page((400, 300), [], ids, style=PageStyle(paper="A1")).drawing
    export.save_pdf(drawing, [(200, 40, 40)], tmp_path / "a1.pdf")
    page, legend = pdf_reading.sheets((tmp_path / "a1.pdf").read_bytes())
    assert page["size_mm"] == pytest.approx((841.0, 594.0), abs=1e-3)
    assert legend["size_mm"] == pytest.approx((210.0, 297.0), abs=1e-3)

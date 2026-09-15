"""The print-size model: placing a page on A4, unit conversions, and how the benchmark harness loads it."""

import subprocess
import sys
from pathlib import Path

import pytest

from tessellatum.core import print_size as ps

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bench_metrics as bm  # noqa: E402


def test_a4_printable_area_and_full_sheet_pixels():
    assert ps.A4.printable_mm() == (190.0, 277.0)
    assert ps.A4.printable_mm(landscape=True) == (277.0, 190.0)
    full_sheet_px = [round(mm * ps.PRINT_DPI / ps.MM_PER_INCH) for mm in (ps.A4.width_mm, ps.A4.height_mm)]
    assert full_sheet_px == [2480, 3508]


@pytest.mark.parametrize(
    ("size_px", "landscape", "printed_mm"),
    [
        ((1900, 2000), False, (190.0, 200.0)),  # portrait, limited by the width
        ((1000, 2770), False, (100.0, 277.0)),  # tall, limited by the height
        ((2770, 1000), True, (277.0, 100.0)),  # wide: landscape sheet
        ((1900, 1300), True, (277.0, 277 * 1300 / 1900)),  # landscape, limited by the width
        ((500, 500), False, (190.0, 190.0)),  # square: portrait sheet
    ],
)
def test_page_fills_the_printable_area_in_the_orientation_of_the_image(size_px, landscape, printed_mm):
    scale = ps.print_scale(size_px)

    assert scale.landscape == landscape
    assert scale.printed_size_mm == pytest.approx(printed_mm)


def test_preview_and_export_of_one_image_print_at_the_same_size():
    preview = ps.print_scale((825, 1100))
    export = ps.print_scale((1800, 2400))

    assert preview.printed_size_mm == pytest.approx(export.printed_size_mm)
    assert export.px_per_mm / preview.px_per_mm == pytest.approx(2400 / 1100)


def test_unit_conversions():
    scale = ps.print_scale((1900, 2000))  # 10 px per mm

    assert scale.px_per_mm == pytest.approx(10.0)
    assert scale.dpi == pytest.approx(254.0)
    assert scale.mm_to_px(3.0) == pytest.approx(30.0)
    assert scale.px_to_mm(30.0) == pytest.approx(3.0)
    assert scale.pt_to_px(72.0) == pytest.approx(254.0)  # 72 pt = 1 in = 25.4 mm
    assert scale.px_to_pt(254.0) == pytest.approx(72.0)
    assert scale.mm2_to_px(9.0) == pytest.approx(900.0)


def test_default_thresholds_at_print_resolution():
    scale = ps.PrintScale(size_px=(2244, 3272), landscape=False, px_per_mm=ps.PRINT_DPI / ps.MM_PER_INCH)

    assert scale.dpi == pytest.approx(300.0)
    assert scale.mm_to_px(ps.MIN_PAINTABLE_WIDTH_MM) == pytest.approx(35.43, abs=0.01)
    assert scale.pt_to_px(ps.MIN_LABEL_SIZE_PT) == pytest.approx(25.0)
    assert scale.mm_to_px(ps.OUTLINE_WIDTH_MM) == pytest.approx(3.54, abs=0.01)


def test_other_paper_formats():
    letter = ps.PaperFormat("Letter", width_mm=215.9, height_mm=279.4, margin_mm=6.35)

    assert ps.print_scale((1000, 1000), letter).printed_size_mm == pytest.approx((203.2, 203.2))


@pytest.mark.parametrize("size_px", [(0, 100), (100, -1)])
def test_rejects_empty_images(size_px):
    with pytest.raises(ValueError):
        ps.print_scale(size_px)


@pytest.mark.parametrize(
    ("width_mm", "height_mm", "margin_mm"),
    [(20.0, 30.0, 10.0), (297.0, 210.0, 10.0), (210.0, 297.0, -1.0)],
)
def test_rejects_paper_without_a_printable_area(width_mm, height_mm, margin_mm):
    with pytest.raises(ValueError):
        ps.PaperFormat("bad", width_mm=width_mm, height_mm=height_mm, margin_mm=margin_mm)


def test_harness_uses_the_model_file_of_its_own_checkout():
    assert Path(bm.print_size.__file__).resolve() == Path(ps.__file__).resolve()


def test_harness_loads_the_model_without_importing_the_package():
    code = (
        "import sys\n"
        "sys.modules['tessellatum'] = None  # any import of the package now fails\n"
        f"sys.path.insert(0, {str(BENCH_DIR)!r})\n"
        "import bench_metrics\n"
        "print(bench_metrics.print_size.print_scale((1900, 2000)).px_per_mm)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert float(result.stdout) == pytest.approx(10.0)

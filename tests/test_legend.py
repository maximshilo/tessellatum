"""The legend: numbered swatches sized on paper, so that it prints as large at any resolution."""

import numpy as np
import pytest

from tessellatum.core import legend as legend_module
from tessellatum.core.legend import render_legend

PALETTE_BGR = np.array([(30, 30, 30), (200, 220, 240), (60, 120, 200)], dtype=np.uint8)


def _first_swatch(image: np.ndarray) -> tuple[int, int, int, int]:
    """The first swatch's framed box: x0, y0, x1, y1 (exclusive), from its dark frame."""
    framed = (image < 10).all(axis=2)
    rows, cols = np.nonzero(framed[: image.shape[0], :])
    first_row = rows.min()
    x0 = cols[rows == first_row].min()
    x1 = x0
    while framed[first_row, x1]:
        x1 += 1
    y1 = first_row
    while framed[y1, x0]:
        y1 += 1
    return int(x0), int(first_row), int(x1), int(y1)


@pytest.mark.parametrize("px_per_mm", [3.97, 4.34, 7.39, 11.81])  # previews, an export at the source's size, 300 dpi
def test_a_swatch_and_its_gap_are_their_size_on_paper(px_per_mm):
    image = np.asarray(render_legend(PALETTE_BGR, 2000, px_per_mm))
    x0, y0, x1, y1 = _first_swatch(image)
    # The frame is drawn on the swatch's last row and column too, one pixel past its size.
    assert (x1 - x0 - 1) / px_per_mm == pytest.approx(legend_module.SWATCH_MM, abs=0.6 / px_per_mm)
    assert (y1 - y0 - 1) / px_per_mm == pytest.approx(legend_module.SWATCH_MM, abs=0.6 / px_per_mm)
    assert x0 / px_per_mm == pytest.approx(legend_module.SWATCH_GAP_MM, abs=0.6 / px_per_mm)
    assert image.shape[0] / px_per_mm == pytest.approx(legend_module.SWATCH_MM + 2 * legend_module.SWATCH_GAP_MM, abs=1.5 / px_per_mm)


def test_forty_colors_wrap_into_rows_that_fit_a_quarter_of_a_sheet():
    px_per_mm = 11.81  # 300 dpi
    palette = np.array([(i * 6, 100, 255 - i * 6) for i in range(40)], dtype=np.uint8)
    image = render_legend(palette, round(190 * px_per_mm), px_per_mm)
    # 12 swatches a row across 190 mm, so 4 rows: 4 x 15 mm + 3 mm.
    assert image.height / px_per_mm == pytest.approx(63, abs=0.5)


def test_each_number_is_written_in_white_on_a_dark_swatch_and_in_black_on_a_light_one():
    px_per_mm = 7.39
    image = np.asarray(render_legend(PALETTE_BGR, 2000, px_per_mm))
    x0, y0, x1, y1 = _first_swatch(image)
    step = x1 - x0 - 1 + round(legend_module.SWATCH_GAP_MM * px_per_mm)
    border = round(legend_module.SWATCH_BORDER_MM * px_per_mm)
    for index, bgr in enumerate(PALETTE_BGR):
        inside = image[y0 + border + 1 : y1 - border - 1, x0 + index * step + border + 1 : x1 + index * step - border - 1]
        fill = tuple(int(v) for v in bgr[::-1])
        assert tuple(inside[2, 2]) == fill  # the swatch is its color, exactly
        written = inside[(inside != fill).any(axis=2)]
        assert len(written) > 0
        if legend_module._luminance(fill) < 140:
            assert written.min(axis=1).max() == 255  # some of the number is pure white
        else:
            assert written.max(axis=1).min() == 0  # some of it pure black

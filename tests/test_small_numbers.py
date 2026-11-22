"""Numbers down to 1 pt (T2a): a page laid out for the PDF places them at any fraction of a pixel, as small as the
style asks; a page drawn in pixels keeps its floor."""

import dataclasses
import re

import numpy as np
import pytest

from tessellatum.core import pipeline
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.labels import (
    EXACT_SIZE_STEP,
    MIN_FONT_SIZE,
    LabelSpacing,
    _sizes,
    min_font_size,
    outline_bbox,
    place_labels,
    text_box_size,
)
from tessellatum.core.print_size import print_scale
from tessellatum.core.regions import extract_regions
from tessellatum.core.render import PAPER, PageStyle, number_origin, render_page

import pdf_reading
from test_export import ALIGNED, _save
from test_labels import LINE_PX, _page


@pytest.mark.parametrize("size", [(825, 1100), (1800, 2400), (2048, 1366), (600, 450), (4000, 3000)])
@pytest.mark.parametrize("points", [1.0, 1.5, 3.0, 7.25])
def test_a_vector_layout_has_no_pixel_floor_and_prints_exactly_its_points(size, points):
    scale = print_scale(size)
    exact = min_font_size(size, points, exact=True)
    assert scale.px_to_pt(exact) >= points  # the arithmetic the benchmark measures a number with
    assert exact == pytest.approx(scale.pt_to_px(points), abs=1e-9)
    # In pixels, the floor and whole pixels stay.
    raster = min_font_size(size, points)
    assert isinstance(raster, int) and raster >= MIN_FONT_SIZE and raster >= exact


def boxes_of(sizes):
    return [(w, h) for _, w, h in sizes]


def test_exact_sizes_shrink_in_fractions_of_a_pixel_trying_each_box_once():
    sizes = list(_sizes("27", 12, 3.3, exact=True))
    font_sizes = [size for size, _, _ in sizes]
    assert font_sizes[0] == 12 and boxes_of(sizes)[-1] == text_box_size("27", 3.3, exact=True)  # down to the smallest
    assert all(a > b for a, b in zip(font_sizes, font_sizes[1:]))
    boxes = boxes_of(sizes)
    assert len(set(boxes)) == len(boxes)  # each box once, at the largest size that has it
    assert all(text_box_size("27", size, exact=True) == box for size, box in zip(font_sizes, boxes))
    on_grid = [size for size in font_sizes if abs((12 - size) / EXACT_SIZE_STEP - round((12 - size) / EXACT_SIZE_STEP)) < 1e-9]
    assert on_grid == font_sizes or on_grid == font_sizes[:-1]  # the smallest, unless its box came before
    # Many more sizes than whole pixels would give, and every box the whole pixels give among them.
    whole = list(_sizes("27", 12, 4, exact=False))
    assert [size for size, _, _ in whole] == list(range(12, 3, -1))
    assert len(sizes) > len({(w, h) for _, w, h in whole})


def test_an_exact_box_holds_the_number_s_outline_rounded_out():
    for size in (1.41, 3.0, 4.23, 9.5, 17.0):
        x0, y0, x1, y1 = outline_bbox("38", size)
        width, height = text_box_size("38", size, exact=True)
        assert x1 - x0 <= width < x1 - x0 + 1 and y1 - y0 <= height < y1 - y0 + 1


def _small_region_page():
    """A region 7 px across in a large one: too small for a 10 px number, room for one of 4 px."""
    ids = np.zeros((120, 160), dtype=np.int32)
    ids[50:57, 70:77] = 1
    return ids


def test_a_number_fits_a_small_region_in_a_vector_layout_where_one_in_pixels_needs_a_leader():
    ids = _small_region_page()
    regions, free = _page(ids)
    spacing = dict(label_gap_px=LINE_PX, leader_width_px=LINE_PX, leader_dot_px=3 * LINE_PX, leader_reach_px=35.0)

    in_pixels = {label.region_id: label for label in place_labels(regions, ids, free, LabelSpacing(10, **spacing))}
    exact = {label.region_id: label for label in place_labels(regions, ids, free, LabelSpacing(4.2, exact=True, **spacing))}

    assert in_pixels[1].leader is not None and not in_pixels[1].exact
    small = exact[1]
    assert small.leader is None and small.exact and 4.2 <= small.font_size < 10
    x0, y0, x1, y1 = small.box
    assert (ids[y0:y1, x0:x1] == 1).all() and free[y0:y1, x0:x1].all()
    assert exact[0].exact and exact[0].font_size == in_pixels[0].font_size  # a roomy region's is as it was


def test_a_fractional_size_goes_through_placement_the_page_and_the_pdf(tmp_path):
    # ALIGNED is 4 px a millimeter: 1 pt is 1.41 px, and a region 1 mm across holds a 1 pt number.
    width, height = ALIGNED
    ids = np.zeros((height, width), dtype=np.int32)
    ids[400:404, 300:304] = 1
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))
    rendered = render_page(ALIGNED, regions, ids, PageStyle(min_label_pt=1.0), vector=True)

    small = next(label for label in rendered.labels if label.region_id == 1)
    assert small.exact and small.leader is None
    assert print_scale(ALIGNED).px_to_pt(small.font_size) >= 1.0 and small.font_size != int(small.font_size)
    # The number's outline lies in the middle of its box.
    x, y = number_origin(small)
    x0, y0, x1, y1 = outline_bbox(small.text, small.font_size)
    box = small.box
    assert (x + x0 + x + x1) / 2 == pytest.approx((box[0] + box[2]) / 2)
    assert (y + y0 + y + y1) / 2 == pytest.approx((box[1] + box[3]) / 2)
    assert box[0] <= x + x0 and x + x1 <= box[2] and box[1] <= y + y0 and y + y1 <= box[3]
    # The page's own image draws it there.
    page = np.asarray(rendered.image.convert("L"))
    assert (page[box[1] : box[3], box[0] : box[2]] < PAPER).any()

    # The PDF writes it at that size, at that origin, and it prints inside its box.
    path, (first, _) = _save(tmp_path, rendered.drawing)
    shown = re.findall(r"/F1 (\S+) Tf 1 0 0 -1 (\S+) (\S+) Tm \((\d+)\) Tj", first["content"])
    assert len(shown) == len(rendered.labels)
    for (size, sx, sy, text), label in zip(shown, rendered.labels):
        assert text == label.text and float(size) == pytest.approx(label.font_size, abs=1e-4)
        assert (float(sx), float(sy)) == pytest.approx(number_origin(label), abs=1e-4)
    # Printed, it is the only difference from the sheet without it, and it lies in its box.
    others = dataclasses.replace(rendered.drawing, labels=[label for label in rendered.labels if label is not small])
    bare_path, _ = _save(tmp_path, others, name="bare.pdf")
    zoom = 16  # px a millimeter: 4 a page pixel
    number = np.abs(pdf_reading.render(path, 0, float(zoom)) - pdf_reading.render(bare_path, 0, float(zoom))).max(axis=2)
    ys, xs = np.nonzero(number > 8)
    assert len(ys) > 0
    margin = 10 * zoom  # the page's top left on the sheet
    px = zoom / 4
    assert box[0] - 0.5 <= (xs.min() - margin) / px and (xs.max() + 1 - margin) / px <= box[2] + 0.5
    assert box[1] - 0.5 <= (ys.min() - margin) / px and (ys.max() + 1 - margin) / px <= box[3] + 0.5



def _blocks_with_specks() -> np.ndarray:
    """Four flat blocks, one of them dotted with small squares of another color, 1.8 mm across on paper."""
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    image[:450, :600] = (200, 40, 40)
    image[:450, 600:] = (40, 180, 40)
    image[450:, :600] = (40, 40, 200)
    image[450:, 600:] = (210, 210, 40)
    for y in range(40, 420, 60):
        for x in range(640, 1180, 60):
            image[y : y + 8, x : x + 8] = (40, 40, 200)
    return image


def test_a_page_laid_out_for_a_pdf_has_the_same_regions_and_numbers_no_larger():
    image = _blocks_with_specks()
    params = params_for_preset("Realistic")
    in_pixels = pipeline.generate(image, params, 1200, collect_analysis=True)
    for points in (3.0, 1.0):
        style = PageStyle(min_label_pt=points)
        raster = pipeline.generate(image, params, 1200, style=style, collect_analysis=True)
        vector = pipeline.generate(image, params, 1200, style=style, collect_analysis=True, vector=True)
        assert np.array_equal(vector.analysis.region_id_map, raster.analysis.region_id_map)
        assert np.array_equal(vector.analysis.region_id_map, in_pixels.analysis.region_id_map)
        assert vector.palette_rgb == raster.palette_rgb
        scale = print_scale(vector.page.size)
        sizes = [scale.px_to_pt(label.font_size) for label in vector.drawing.labels]
        assert all(label.exact for label in vector.drawing.labels) and min(sizes) >= points
        assert max(label.font_size for label in vector.drawing.labels) <= max(
            label.font_size for label in raster.drawing.labels
        )
        leaders = lambda page: sum(label.leader is not None for label in page.drawing.labels)  # noqa: E731
        assert leaders(vector) <= leaders(raster)
    # At 1 pt the specks' numbers go inside them, where the 10 px floor needs leaders.
    assert leaders(raster) > 0 and leaders(vector) < leaders(raster)
    assert min(sizes) < 3.0


def test_a_page_drawn_again_at_its_own_size_is_the_page_and_for_a_pdf_only_its_numbers_change(sample_image_bgr):
    params = params_for_preset("Medium")
    page = pipeline.generate(sample_image_bgr, params, 200)
    again = pipeline.draw_page(page, sample_image_bgr, 200)
    assert again.page.tobytes() == page.page.tobytes()
    assert [(l.box, l.font_size) for l in again.drawing.labels] == [(l.box, l.font_size) for l in page.drawing.labels]
    vector = pipeline.draw_page(page, sample_image_bgr, 200, vector=True)
    assert np.array_equal(vector.painting.region_id_map, page.painting.region_id_map)
    assert vector.drawing.strokes is page.model.strokes
    assert all(label.exact for label in vector.drawing.labels)

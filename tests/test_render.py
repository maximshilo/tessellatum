"""How the page is drawn: the width of a line, where its ink falls, and the tone of the ink."""

import dataclasses

import numpy as np
import pytest

from tessellatum.core.print_size import OUTLINE_WIDTH_MM, print_scale
from tessellatum.core.regions import extract_regions
from tessellatum.core.render import LINE_GRAY, PAPER, PageStyle, ink_coverage, render_page


def _split_page(size: tuple[int, int]) -> np.ndarray:
    """A page of two regions, divided down a crack in the middle."""
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    ids[:, width // 2 :] = 1
    return ids


def _ink_across(size: tuple[int, int], style: PageStyle) -> np.ndarray:
    """A row of the line layer across the middle boundary, as ink from 0 to 1, clear of the frame."""
    width, height = size
    outlines = np.asarray(render_page(size, [], _split_page(size), style).outlines)
    return (PAPER - outlines[height // 2].astype(np.float64))[4 : width - 4] / PAPER


@pytest.mark.parametrize("width_mm", [0.2, 0.3, 0.45, 0.8])
def test_a_line_lays_down_as_much_ink_as_the_paper_asks_for(width_mm):
    # Not only the nearest width the drawing grid can hold: a width in between
    # two of its steps is drawn as a blend of the two, so the ink is right.
    size = (1200, 1600)
    style = PageStyle(line_width_mm=width_mm)
    assert print_scale(size).mm_to_px(width_mm) > style.min_line_width_px  # the floor is not what is being measured

    ink = _ink_across(size, style)

    assert ink.sum() == pytest.approx(style.line_width_px(size), abs=0.02)
    assert print_scale(size).px_to_mm(ink.sum()) == pytest.approx(width_mm, abs=0.005)


def _ink_along(side: int, angle_deg: float, width_px: float) -> float:
    """Ink a straight line at ``angle_deg`` lays down, per unit of its length on the page.

    The line runs through the middle of a square page and far past both edges,
    so its round ends fall outside and what is left is the band alone. A line
    through the middle of a square of this side is this long on the page.
    """
    direction = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    middle = np.array([side / 2 - 0.5, side / 2 - 0.5])
    line = np.stack([middle - 2 * side * direction, middle + 2 * side * direction])
    ink = ink_coverage((side, side), [line], width_px).sum() / PAPER
    return ink / (side / max(abs(direction[0]), abs(direction[1])))


@pytest.mark.parametrize("width_px", [1.0, 1.3, 2.84])
def test_a_line_down_a_crack_lays_down_its_width_to_a_fraction_of_a_percent(width_px):
    for angle_deg in (0, 90):
        assert _ink_along(200, angle_deg, width_px) == pytest.approx(width_px, rel=0.01)


@pytest.mark.parametrize("angle_deg", [10, 20, 30, 45, 60, 80])
@pytest.mark.parametrize("width_px", [1.0, 1.3, 2.84])
def test_a_line_running_any_other_way_lays_down_its_width_to_within_a_tenth(angle_deg, width_px):
    # The pen is dragged along a centreline rasterized on the drawing grid, and
    # a staircase is not the line it stands for: where a line follows neither a
    # row nor a column, the band comes out a little wide, or a little thin at
    # the floor. On the shipped grid that runs from 0.883 of the asked-for ink
    # (45 degrees, 1 px) to 1.088 (near 20 degrees), against 0.999-1.004 down a
    # crack. It is the grid's coarseness rather than the pen's shape: against
    # the width measured exactly, a circle's band goes 1.061x at this grid to
    # 1.035x, 1.023x and 1.014x as the grid doubles.
    assert _ink_along(200, angle_deg, width_px) == pytest.approx(width_px, rel=0.12)


def test_a_line_is_centred_on_the_crack_it_is_drawn_on():
    size = (600, 800)

    ink = _ink_across(size, PageStyle())

    inked = np.flatnonzero(ink)
    assert inked.size >= 2
    np.testing.assert_allclose(ink[inked], ink[inked][::-1])  # the same either side of the crack


def test_a_preview_and_an_export_of_one_image_print_the_same_line():
    # The width is a size on paper, so it follows the page's shape, not its
    # pixel count (see print_size).
    style = PageStyle()
    preview, export = (825, 1100), (1800, 2400)

    widths_mm = [print_scale(size).px_to_mm(_ink_across(size, style).sum()) for size in (preview, export)]

    assert widths_mm[0] == pytest.approx(OUTLINE_WIDTH_MM, abs=0.005)
    assert widths_mm[1] == pytest.approx(OUTLINE_WIDTH_MM, abs=0.005)
    assert style.line_width_px(export) > 2 * style.line_width_px(preview) * 0.9  # wider in pixels, as the page is


def test_a_line_is_never_drawn_thinner_than_the_floor():
    # The pipeline never upscales, so a small source prints at a low
    # resolution -- 60 dpi here -- where 0.3 mm is less than a pixel.
    size = (600, 450)
    style = PageStyle()
    assert print_scale(size).mm_to_px(style.line_width_mm) < style.min_line_width_px

    ink = _ink_across(size, style)

    assert ink.sum() == pytest.approx(style.min_line_width_px, abs=0.02)


def test_a_line_that_crosses_part_of_a_pixel_inks_part_of_it():
    # Anti-aliasing: what keeps a smooth line looking smooth once it is drawn.
    size = (300, 300)
    ids = np.zeros((300, 300), dtype=np.int32)
    rows, columns = np.mgrid[:300, :300]
    ids[rows > columns] = 1  # a boundary at 45 degrees, which no pixel edge follows

    outlines = np.asarray(render_page(size, [], ids, PageStyle()).outlines)

    inked = outlines != PAPER
    partly = inked & (outlines != 0)
    assert partly.sum() > 0.5 * inked.sum()  # most of a thin line is a shade, not solid


def test_a_line_thinner_than_the_drawing_grid_can_hold_is_drawn_as_thin_as_it_can():
    # Nothing in the app asks for this -- the floor keeps lines at a pixel --
    # but a style may, and it must draw a line rather than fail.
    size = (600, 800)
    style = PageStyle(line_width_mm=0.001, min_line_width_px=0.0)

    ink = _ink_across(size, style)

    assert ink.sum() == pytest.approx(0.5, abs=0.02)  # two pixels of the grid, which is four to a pixel


def test_the_ink_prints_in_the_style_s_grays():
    size = (600, 800)
    ids = _split_page(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))
    style = dataclasses.replace(PageStyle(), line_width_mm=1.5, line_gray=0x59, label_gray=0x8C)

    rendered = render_page(size, regions, ids, style)
    page = np.asarray(rendered.image)
    outlines = np.asarray(rendered.outlines)

    assert set(np.unique(page[outlines == 0])) == {style.line_gray}  # solid line: the line's own gray
    assert rendered.labels
    numbered = np.zeros(outlines.shape, dtype=bool)
    for label in rendered.labels:
        x0, y0, x1, y1 = label.box
        numbered[int(y0) - 1 : int(y1) + 2, int(x0) - 1 : int(x1) + 2] = True
    assert page[(outlines == PAPER) & ~numbered].min() == PAPER  # bare paper stays white

    x0, y0, x1, y1 = rendered.labels[0].box
    number = page[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1, 0]
    assert number.min() == style.label_gray  # the number is drawn no darker than its gray


def test_the_grays_change_the_tone_and_nothing_else():
    size = (600, 800)
    ids = _split_page(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))

    plain = render_page(size, regions, ids, PageStyle())
    black = render_page(size, regions, ids, PageStyle(line_gray=0, label_gray=0))

    assert plain.labels == black.labels
    np.testing.assert_array_equal(np.asarray(plain.outlines), np.asarray(black.outlines))
    assert np.asarray(black.image).mean() < np.asarray(plain.image).mean()


def test_a_wider_line_draws_the_same_lines_with_more_ink_and_the_numbers_stay_clear_of_it():
    size = (600, 800)
    ids = _split_page(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))

    plain = render_page(size, regions, ids, PageStyle())
    bold = render_page(size, regions, ids, PageStyle(line_width_mm=1.0))

    assert len(plain.strokes) == len(bold.strokes)
    for one, other in zip(plain.strokes, bold.strokes):
        np.testing.assert_array_equal(one, other)
    assert np.asarray(bold.outlines).mean() < np.asarray(plain.outlines).mean()
    for label in bold.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert (np.asarray(bold.outlines)[y0:y1, x0:x1] == PAPER).all()


def _square_too_small_for_its_number(size: tuple[int, int]) -> np.ndarray:
    """A page of background with a square in its middle too small to hold a number."""
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    ids[height // 2 : height // 2 + 6, width // 2 : width // 2 + 6] = 1
    return ids


def test_a_leader_is_drawn_in_the_numbers_gray_and_ends_in_a_dot_inside_the_region():
    size = (300, 400)
    ids = _square_too_small_for_its_number(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))
    style = PageStyle()

    rendered = render_page(size, regions, ids, style)

    square = next(label for label in rendered.labels if label.region_id == 1)
    (_end, (x, y)) = square.leader
    leaders = np.asarray(rendered.leaders)
    page = np.asarray(rendered.image)[:, :, 0]
    assert leaders[int(y), int(x)] == 0 and ids[int(y), int(x)] == 1  # the dot is solid, inside the square
    assert page[int(y), int(x)] == style.label_gray  # and printed in the numbers' gray, not the lines'
    # The dot is wider than the line: across it there is more solid ink than across the line anywhere else.
    dot_width = (leaders[int(y)] == 0).sum()
    assert dot_width >= 2 * style.line_width_px(size)
    # It is the leader's layer, not the lines': the lines are what they would be without it.
    bare = render_page(size, [], ids, style)
    np.testing.assert_array_equal(np.asarray(rendered.outlines), np.asarray(bare.outlines))
    assert (np.asarray(bare.leaders) == PAPER).all()


def test_the_leader_layer_is_bare_paper_when_every_number_fits_in_its_region():
    size = (600, 800)
    ids = _split_page(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))

    rendered = render_page(size, regions, ids)

    assert all(label.leader is None for label in rendered.labels)
    assert (np.asarray(rendered.leaders) == PAPER).all()


def test_the_default_grays_are_gray_and_the_default_width_is_the_print_model_s():
    style = PageStyle()

    assert style.line_width_mm == OUTLINE_WIDTH_MM
    assert 0 < style.line_gray < PAPER and style.line_gray == LINE_GRAY
    assert style.line_gray < style.label_gray < PAPER  # a number is lighter than a line, never darker

"""How the page is drawn: the width of a line, where its ink falls, and the tone of the ink."""

import dataclasses

import cv2
import numpy as np
import pytest

from tessellatum.core import labels as labels_module
from tessellatum.core import boundaries
from tessellatum.core import render as render_module
from tessellatum.core.print_size import OUTLINE_WIDTH_MM, print_scale
from tessellatum.core.regions import extract_regions
from tessellatum.core.color import bgr_to_lab
from tessellatum.core.render import LABEL_GRAY, LINE_GRAY, PAPER, PageStyle, ink_coverage, render_page
from tessellatum.core.text import Lettering


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


def _lightness(gray: int) -> float:
    return float(bgr_to_lab(np.array([gray, gray, gray], dtype=np.uint8))[0])


def test_the_tones_the_app_offers_step_both_grays_alike_and_keep_the_numbers_lighter():
    # D-052 (Q36): Light / Medium / Dark, Medium being D-030's grays, the others 15 L* either side of it.
    assert list(render_module.TONES) == ["Light", "Medium", "Dark"]
    assert render_module.TONES["Medium"] == (LINE_GRAY, LABEL_GRAY) and render_module.DEFAULT_TONE == "Medium"
    medium_gap = _lightness(LABEL_GRAY) - _lightness(LINE_GRAY)
    lines = [_lightness(line) for line, _ in render_module.TONES.values()]
    for line, label in render_module.TONES.values():
        assert line < label < PAPER
        assert _lightness(label) - _lightness(line) == pytest.approx(medium_gap, abs=1.0)
    assert lines[0] - lines[1] == pytest.approx(15, abs=1.0) and lines[1] - lines[2] == pytest.approx(15, abs=1.0)


def test_the_app_s_settings_give_the_default_style_at_their_defaults():
    assert PageStyle.from_settings() == PageStyle()
    assert PageStyle.from_settings(0.45, "Dark") == PageStyle(line_width_mm=0.45, line_gray=0x36, label_gray=0x66)
    low, high = render_module.LINE_WIDTH_MM_RANGE
    assert low < OUTLINE_WIDTH_MM < high
    steps = (high - low) / render_module.LINE_WIDTH_MM_STEP
    assert steps == pytest.approx(round(steps))  # the range is a whole number of steps
    assert ((OUTLINE_WIDTH_MM - low) / render_module.LINE_WIDTH_MM_STEP) == pytest.approx(4)  # the default is one


def test_line_art_s_ink_prints_solid_in_its_own_tone_and_carries_no_line_and_no_number():
    size = (160, 120)
    ids = np.zeros((120, 160), dtype=np.int32)
    ids[:, 80:] = 1
    ink = np.zeros(ids.shape, dtype=bool)
    ink[55:65, :] = True  # a band of ink across both fills, 10 px high
    ids[ink] = -1
    ids[65:, :80], ids[65:, 80:] = 2, 3
    regions = extract_regions(ids, np.array([0, 1, 2, 3], dtype=np.int32))

    rendered = render_page(size, regions, ids, ink=ink, ink_gray=30)

    page = np.asarray(rendered.image.convert("L"))
    outlines = np.asarray(rendered.outlines)
    assert (page[ink] == 30).all()  # solid, in the tone asked for
    assert (outlines[ink] == 0).all()  # and in the line layer, solid ink
    for line in rendered.strokes:  # no line runs along it: the one between the fills stops at it
        points = np.asarray(line)
        assert not (((points[:-1, 1] == points[1:, 1]) & np.isin(points[:-1, 1], (54.5, 64.5))).any())
    for label in rendered.labels:  # and no number goes on it
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert not ink[y0:y1, x0:x1].any()
    assert len(rendered.labels) == 4
    # The fills' own boundary is still drawn, down the page's middle above and below the band.
    assert any(np.allclose(np.asarray(line)[:, 0], 79.5) for line in rendered.strokes)


def test_without_ink_the_page_is_drawn_as_before():
    size = (60, 40)
    ids = _split_page(size)
    plain = render_page(size, [], ids)
    no_ink = render_page(size, [], ids, ink=np.zeros(ids.shape, dtype=bool), ink_gray=0)

    assert np.array_equal(np.asarray(plain.image), np.asarray(no_ink.image))
    assert np.array_equal(np.asarray(plain.outlines), np.asarray(no_ink.outlines))


def test_a_number_written_outside_its_region_goes_beside_the_ink_not_on_it():
    # A region too small for its number, ringed by ink 6 px thick, inside a wide white one: the nearest room outside
    # it is on the ring, and the number must go past it, onto the paper.
    size = (400, 300)
    ids = np.zeros((300, 400), dtype=np.int32)
    ink = np.zeros(ids.shape, dtype=bool)
    ink[142:158, 192:208] = True
    ids[ink] = -1
    ids[148:152, 198:202] = 1  # 4 x 4: no number fits in it
    ink[148:152, 198:202] = False
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))

    rendered = render_page(size, regions, ids, ink=ink, ink_gray=0)

    small = next(label for label in rendered.labels if label.region_id == 1)
    assert small.leader is not None
    x0, y0, x1, y1 = (int(v) for v in small.box)
    assert not ink[y0:y1, x0:x1].any()


def _lines_of(rendered, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The ink a page's own lines put down (0 to 255 solid), and the paper they leave, in the default style."""
    ink = ink_coverage(size, rendered.strokes, PageStyle().line_width_px(size)).astype(np.int64)
    return ink, np.rint(PAPER - ink * ((PAPER - LINE_GRAY) / PAPER))


def _letters(area: np.ndarray, ink: np.ndarray, outline=()) -> Lettering:
    """Letters as ``text.lettering`` gives them: printed ``ink`` inside the lines' boxes, ``area``."""
    return Lettering(area=area, ink=ink, outline=list(outline), dark=area & (ink >= 128))


def _lettered_page(size: tuple[int, int] = (120, 80)):
    """A page split down the middle, a band of solid ink across it, and a line of letters over the crack and band."""
    width, height = size
    ids = _split_page(size)
    ids[:30, : width // 2] = 2  # a third region, whose boundary crosses the lettering where it is too faint to print
    ink = np.zeros((height, width), dtype=bool)
    ink[36:40, :] = True  # printed ink across the page, through the line of lettering and beside it
    area = np.zeros((height, width), dtype=bool)
    area[20:50, 30:90] = True
    lettering = np.zeros((height, width), dtype=np.uint8)
    lettering[20:50, 30:90] = np.linspace(0, 255, 60).round().astype(np.uint8)[None, :]  # from paper to solid
    return ids, ink, area, lettering


def test_the_letters_print_as_they_cover_the_page_under_the_lines_in_place_of_any_other_ink():
    size = (120, 80)
    ids, ink, area, lettering = _lettered_page(size)
    printed = ink | (area & (lettering >= 128))

    ring = np.array([[30.0, 20.0], [89.0, 20.0], [89.0, 49.0], [30.0, 20.0]])
    rendered = render_page(size, [], ids, ink=printed, ink_gray=40, lettering=_letters(area, lettering, [ring]))
    plain = render_page(size, [], ids, ink=printed, ink_gray=40)

    page = np.asarray(rendered.image.convert("L")).astype(np.float64)
    lines_ink, lines = _lines_of(rendered, size)
    tone = np.rint(PAPER - lettering * ((PAPER - 40) / PAPER))
    np.testing.assert_array_equal(page[area], np.minimum(lines, tone)[area])
    assert (page[area] < tone[area]).sum() > 20  # a line runs through it, over the lettering
    # Outside it, the page is as it was: the band solid, the lines as drawn.
    np.testing.assert_array_equal(page[~area], np.asarray(plain.image.convert("L"))[~area])
    # The line layer holds the lettering's ink there, and the lines'.
    outlines = np.asarray(rendered.outlines).astype(np.int64)
    np.testing.assert_array_equal((PAPER - outlines)[area], np.maximum(lines_ink, lettering.astype(np.int64))[area])
    np.testing.assert_array_equal(outlines[~area], np.asarray(plain.outlines)[~area])
    # The band's ink inside the line is the lettering's now: paper where the lettering is, as the picture shows.
    assert page[37, 31] > 240 and page[37, 88] < 45
    # The PDF draws the letters' outline (see ``export``), and none without letters.
    (drawn,) = rendered.drawing.lettering
    np.testing.assert_array_equal(drawn, ring)
    assert plain.drawing.lettering is None


def test_printed_ink_in_no_region_stays_solid_inside_the_lettering():
    # Line art's bold ink, and the seam down a line two regions share, keep the regions apart: inside a line of text,
    # where the letters take the place of the ink lying in a region, they still print solid, whatever the letters say
    # there. Otherwise a region's paint could run into them.
    size = (120, 80)
    ids, ink, area, lettering = _lettered_page(size)
    ids[36:40, :] = -1  # the band of ink is in no region now: a wall
    printed = ink | (area & (lettering >= 128))

    rendered = render_page(size, [], ids, ink=printed, ink_gray=40, lettering=_letters(area, lettering))

    page = np.asarray(rendered.image.convert("L"))
    outlines = np.asarray(rendered.outlines)
    assert (page[ink] == 40).all() and (outlines[ink] == 0).all()  # solid, where the lettering is paper too
    assert lettering[36:40, 30:40].max() < 128  # (which it is, at the line's left end)
    inside = area & ~ink
    tone = np.rint(PAPER - lettering * ((PAPER - 40) / PAPER))
    _, lines = _lines_of(rendered, size)
    np.testing.assert_array_equal(page[inside], np.minimum(lines, tone)[inside])


def test_no_number_goes_on_the_lettering():
    # One wide region with a line of lettering across its middle, where its number would go.
    size = (200, 120)
    ids = np.zeros((120, 200), dtype=np.int32)
    area = np.zeros(ids.shape, dtype=bool)
    area[45:75, 40:160] = True
    lettering = np.zeros(ids.shape, dtype=np.uint8)
    lettering[50:70, 50:150:4] = 200  # strokes
    lettering[50:70, 51:151:4] = 60  # their anti-aliased edges, too faint to count as printed
    printed = area & (lettering >= 128)
    regions = extract_regions(ids, np.array([0], dtype=np.int32))  # its label point at its middle, on the lettering
    bare = render_page(size, regions, ids)
    (alone,) = bare.labels
    x0, y0, x1, y1 = (int(v) for v in alone.box)
    assert (lettering[y0:y1, x0:x1] > 0).any()  # without the lettering, the number sits on it

    rendered = render_page(size, regions, ids, ink=printed, ink_gray=0, lettering=_letters(area, lettering))

    (label,) = rendered.labels
    x0, y0, x1, y1 = (int(v) for v in label.box)
    assert not (lettering[y0:y1, x0:x1] > 0).any()


def test_an_empty_line_of_lettering_changes_nothing():
    size = (60, 40)
    ids = _split_page(size)
    plain = render_page(size, [], ids)
    empty = render_page(size, [], ids, lettering=_letters(np.zeros((40, 60), dtype=bool), np.zeros((40, 60), np.uint8)))
    assert np.array_equal(np.asarray(plain.image), np.asarray(empty.image))
    assert np.array_equal(np.asarray(plain.outlines), np.asarray(empty.outlines))
    assert empty.drawing.lettering is None


@pytest.mark.parametrize("size", [(1100, 825), (2048, 1367)])
def test_numbers_keep_half_a_millimeter_from_the_lines_of_text(size, monkeypatch):
    # The gap is set on paper (D-051), so a preview and an export keep the same distance.
    assert labels_module.TEXT_GAP_MM == 0.5
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    regions = extract_regions(ids, np.array([0], dtype=np.int32))
    (alone,) = render_page(size, regions, ids).labels
    x, y = (int(v) for v in alone.box[:2])
    area = np.zeros(ids.shape, dtype=bool)
    area[y - 20 : y + 20, x - 150 : x + 150] = True  # a line of text where the number goes
    seen = []
    real = render_module.place_labels

    def spy(regions, region_id_map, free, spacing, clearable=None, text=None, **kwargs):
        seen.append((spacing, text))
        return real(regions, region_id_map, free, spacing, clearable, text=text, **kwargs)

    monkeypatch.setattr(render_module, "place_labels", spy)

    rendered = render_page(size, regions, ids, lettering=_letters(area, np.zeros(ids.shape, dtype=np.uint8)))
    render_page(size, regions, ids)

    (spacing, text), (_, no_text) = seen
    gap = print_scale(size).mm_to_px(0.5)
    assert spacing.text_gap_px == pytest.approx(gap) and text is area and no_text is None
    (label,) = rendered.labels
    away = cv2.distanceTransform((~area).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    x0, y0, x1, y1 = (int(v) for v in label.box)
    assert away[y0:y1, x0:x1].min() > gap


def test_the_line_smoothing_and_the_numbers_follow_the_style(monkeypatch):
    size = (200, 150)
    ids = _split_page(size)
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))
    spacings = []
    real = render_module.place_labels

    def spy(regions, region_id_map, free, spacing, clearable=None, text=None, **kwargs):
        spacings.append(spacing)
        return real(regions, region_id_map, free, spacing, clearable, text, **kwargs)

    monkeypatch.setattr(render_module, "place_labels", spy)
    smoothing = []
    real_trace = render_module.trace_boundaries
    def trace_spy(ids, smoothing_px=None, **kwargs):
        smoothing.append(smoothing_px)
        return real_trace(ids, smoothing_px, **kwargs)

    monkeypatch.setattr(render_module, "trace_boundaries", trace_spy)
    render_page(size, regions, ids)
    style = PageStyle(line_smoothing_mm=1.5, min_label_pt=9.0, leader_reach_mm=3.0, text_gap_mm=1.0)
    render_page(size, regions, ids, style)
    scale = print_scale(size)
    assert smoothing == [boundaries.smoothing_length_px(size), max(2.0, scale.mm_to_px(1.5))]
    default, chosen = spacings
    assert default.leader_reach_px == scale.mm_to_px(8.0) and default.text_gap_px == scale.mm_to_px(0.5)
    assert chosen.leader_reach_px == scale.mm_to_px(3.0) and chosen.text_gap_px == scale.mm_to_px(1.0)
    assert chosen.min_font_size == labels_module.min_font_size(size, 9.0) >= default.min_font_size
    assert scale.px_to_pt(labels_module.min_font_size((2000, 1500), 9.0)) >= 9.0

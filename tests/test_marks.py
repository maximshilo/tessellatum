"""Detail marks: the thin dark marks in a face that are printed instead of painted (T4.3, D-044)."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import faces, ink, marks, pipeline
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM, print_scale

SAMPLES = Path(__file__).resolve().parent / "sample_images"

# A 1100 x 825 page prints landscape at 4.34 px/mm: the brush is 13.03 px, so the element is 15 px across (the
# smallest odd width over the brush), the smoothing's sigma 1.30 px, and 2 mm is 8.68 px.
SIZE = (1100, 825)
SCALE = print_scale(SIZE)
PAPER = 190  # L* 76.89
# Gray levels this much darker than PAPER, in L*: 11.44, 12.57, 19.81 and 29.83.
UNDER_THRESHOLD, OVER_THRESHOLD, DARK, DARKER = 159, 156, 137, 112


def lightness_of(gray: int) -> float:
    return float(cv2.cvtColor(np.full((1, 1, 3), gray / 255, dtype=np.float32), cv2.COLOR_BGR2Lab)[0, 0, 0])


def page(*shapes, paper=PAPER):
    """A gray page with ``(gray, x0, y0, x1, y1)`` rectangles drawn on it (end exclusive)."""
    image = np.full((SIZE[1], SIZE[0], 3), paper, dtype=np.uint8)
    for gray, x0, y0, x1, y1 in shapes:
        image[y0:y1, x0:x1] = gray
    return image


def find(image, where=None, ids=None, palette=(PAPER,)):
    """The marks on ``image`` with one region painted ``palette[0]`` everywhere, unless ``ids`` says otherwise."""
    height, width = image.shape[:2]
    where = np.ones((height, width), dtype=bool) if where is None else where
    ids = np.zeros((height, width), dtype=np.int32) if ids is None else ids
    palette_bgr = np.array([(gray,) * 3 for gray in palette], dtype=np.uint8)
    colors = np.arange(len(palette), dtype=np.int32)
    return marks.detail_marks(image, where, ids, colors, palette_bgr, SCALE)


def test_the_page_is_what_the_tests_assume():
    assert lightness_of(PAPER) - lightness_of(OVER_THRESHOLD) == pytest.approx(12.57, abs=0.01)
    assert lightness_of(PAPER) - lightness_of(UNDER_THRESHOLD) == pytest.approx(11.44, abs=0.01)
    assert SCALE.mm_to_px(MIN_PAINTABLE_WIDTH_MM) == pytest.approx(13.03, abs=0.01)
    assert SCALE.mm_to_px(marks.MIN_LENGTH_MM) == pytest.approx(8.68, abs=0.01)


# A line 8 px (1.8 mm) wide and 52 px (12 mm) long, 20 L* darker than the paper. The smoothing leaves the middle of it
# at nearly the full contrast; its blurred edge and ends are lighter.
LINE = (DARK, 300, 400, 352, 408)
LINE_MIDDLE = (slice(402, 406), slice(303, 349))


def test_a_thin_dark_line_is_printed_where_it_lies():
    found = find(page(LINE))
    assert found[LINE_MIDDLE].all()
    line = np.zeros(found.shape, dtype=bool)
    line[400:408, 300:352] = True
    near = cv2.dilate(line.view(np.uint8), np.ones((5, 5), np.uint8)).view(bool)  # its blurred edge, two pixels round it
    assert not found[~near].any()


def test_marks_are_looked_for_only_where_asked():
    image = page(LINE)
    where = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    assert not find(image, where).any()
    where[:, 330:] = True  # 22 px of the line
    found = find(image, where)
    assert found.any() and not found[:, :330].any()


def test_a_mark_as_wide_as_the_brush_is_printed_and_a_wider_one_is_not():
    # A band as wide as the brush (13 px, 2.99 mm): the 15 px element fits in no part of it, so the closing lifts it to the
    # paper. At 20 px (4.6 mm) the element fits, and the band is a dark area, which the regions keep if it is large enough.
    brush_wide = find(page((DARKER, 300, 300, 313, 400)))
    assert brush_wide[330:370, 303:310].all()
    assert not find(page((DARKER, 300, 300, 320, 400))).any()


def test_a_mark_must_be_at_least_the_contrast_darker():
    # 8 px wide (1.8 mm): the smoothing leaves its middle at the full contrast.
    assert find(page((OVER_THRESHOLD, 300, 300, 308, 380)))[320:360, 303:305].all()
    assert not find(page((UNDER_THRESHOLD, 300, 300, 308, 380))).any()


def test_a_mark_is_printed_only_where_its_paint_is_no_darker_than_it():
    image = page(LINE)
    mark = lightness_of(DARK)
    darker_paint = next(g for g in range(DARK, 0, -1) if lightness_of(g) < mark - 2)
    lighter_paint = next(g for g in range(DARK, 256) if lightness_of(g) > mark + 2)
    assert not find(image, palette=(darker_paint,)).any()  # a region painted as dark shows the mark already
    assert find(image, palette=(lighter_paint,))[LINE_MIDDLE].all()


def test_specks_are_left_out_and_a_mark_counts_its_longer_side():
    # A 5 px dot (1.2 mm) is a speck; a 12 px dash (2.8 mm), across or down, is a mark.
    assert not find(page((DARKER, 300, 300, 305, 305))).any()
    assert find(page((DARKER, 300, 300, 312, 305)))[301:304, 302:310].all()
    assert find(page((DARKER, 300, 300, 305, 312)))[302:310, 301:304].all()


def test_pixels_in_no_region_are_never_marks():
    image = page(LINE)
    ids = np.zeros((SIZE[1], SIZE[0]), dtype=np.int32)
    ids[:, 330:] = -1
    found = find(image, ids=ids)
    assert found[402:406, 303:330].all() and not found[:, 330:].any()


def test_nothing_is_looked_for_without_where():
    found = find(page(LINE), np.zeros((SIZE[1], SIZE[0]), dtype=bool))
    assert found.shape == (SIZE[1], SIZE[0]) and not found.any()


def whole_picture_marks(image, where, ids, colors, palette_bgr, scale):
    """``detail_marks`` computed over the whole picture, as the windowed search must match."""
    sigma = scale.mm_to_px(marks.SMOOTHING_MM)
    blur_reach = math.ceil(4 * sigma)
    reach = math.floor((scale.mm_to_px(MIN_PAINTABLE_WIDTH_MM) - 1) / 2) + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    lightness = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 0]
    lightness = cv2.GaussianBlur(lightness, (2 * blur_reach + 1,) * 2, sigma)
    contrast = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, element)
    paint = cv2.cvtColor(palette_bgr.astype(np.float32).reshape(1, -1, 3) / 255, cv2.COLOR_BGR2Lab)[0, :, 0]
    found = where & (ids >= 0) & (contrast >= marks.MIN_CONTRAST) & (lightness <= paint[colors[np.clip(ids, 0, None)]])
    _count, components, stats, _ = cv2.connectedComponentsWithStats(found.astype(np.uint8), connectivity=8)
    keep = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= scale.mm_to_px(marks.MIN_LENGTH_MM)
    keep[0] = False
    return keep[components]


@pytest.mark.parametrize(
    "boxes",
    [
        [(400, 300, 200, 150)],  # in the middle
        [(0, 0, 150, 120), (1000, 700, 100, 125)],  # at two corners
        [(10, 500, 1080, 60), (520, 20, 40, 780)],  # across the page both ways
    ],
)
def test_the_window_judges_every_pixel_as_the_whole_picture_would(boxes):
    rng = np.random.default_rng(7)
    image = np.full((SIZE[1], SIZE[0], 3), PAPER, dtype=np.uint8)
    # Dark strokes of every width up to past the brush's, everywhere, many of them across the boxes' edges.
    for _ in range(400):
        x, y = int(rng.integers(0, SIZE[0])), int(rng.integers(0, SIZE[1]))
        dx, dy = (int(v) for v in rng.integers(-40, 41, size=2))
        cv2.line(image, (x, y), (x + dx, y + dy), (int(rng.integers(60, 180)),) * 3, int(rng.integers(1, 18)))
    image = np.clip(image.astype(np.int16) + rng.integers(-6, 7, size=image.shape), 0, 255).astype(np.uint8)
    ids = (np.arange(SIZE[0])[None, :] // 90 + np.arange(SIZE[1])[:, None] // 70 * 13).astype(np.int32)
    ids[600:610, 200:900] = -1
    colors = (np.arange(ids.max() + 1) % 4).astype(np.int32)
    palette_bgr = np.array([(200,) * 3, (170,) * 3, (140,) * 3, (100,) * 3], dtype=np.uint8)
    where = faces.mask([faces.Face(box=b, score=1.0, detector="yunet") for b in boxes], SIZE)

    found = marks.detail_marks(image, where, ids, colors, palette_bgr, SCALE)
    expected = whole_picture_marks(image, where, ids, colors, palette_bgr, SCALE)
    assert expected.sum() > 1000  # there are marks to find, in every box
    np.testing.assert_array_equal(found, expected)


def test_a_photographed_face_prints_its_marks_and_keeps_its_regions(monkeypatch):
    # Q27: inside the faces found on a photograph, thin dark marks are printed, in their own tone; the regions stay.
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-cats-face.jpg")
    params = params_for_preset("Medium")
    result = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    analysis = result.analysis
    printed = analysis.printed_ink
    assert printed.sum() > 200 and not printed[~analysis.detail].any()
    resized = pipeline.resize_to_long_edge(image, pipeline.PREVIEW_LONG_EDGE)
    assert analysis.ink_gray == ink.ink_gray(resized, printed) and 0 < analysis.ink_gray < 128
    page = np.asarray(result.page.convert("L"))
    assert (page[printed] == analysis.ink_gray).all()
    assert (analysis.outlines[printed] == 0).all()  # the marks are the page's ink
    for label in analysis.labels:  # no number on a mark
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert not printed[y0:y1, x0:x1].any()

    pipeline.clear_cache()
    monkeypatch.setattr(marks, "detail_marks", lambda image, where, *args: np.zeros(where.shape, dtype=bool))
    plain = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    np.testing.assert_array_equal(plain.region_id_map, analysis.region_id_map)
    assert len(plain.regions) == len(analysis.regions)
    assert not plain.printed_ink.any() and plain.ink_gray == 0
    pipeline.clear_cache()


def test_marks_are_looked_for_only_in_the_faces_of_a_picture_drawn_from_its_colors(monkeypatch):
    def refuse(*args):
        raise AssertionError("looked for marks")

    monkeypatch.setattr(marks, "detail_marks", refuse)
    for name in ("scene.png", "m-cartoon-bold-lines-girl.png"):  # no faces found; line art, whose face is its own ink
        pipeline.clear_cache()
        image = pipeline.load_image_bgr(SAMPLES / name)
        pipeline.generate(image, params_for_preset("Medium"), pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    pipeline.clear_cache()

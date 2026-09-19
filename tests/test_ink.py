"""Ink lines: the dark lines line art is drawn with, and whether a picture is line art at all."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import ink
from tessellatum.core.print_size import print_scale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402

# An A4 landscape page at 4 px/mm, about what a preview prints at.
PAGE = (1108, 760)
PX_PER_MM = 4.0
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Flat fills (BGR), all lighter than L* 49, so that a black line on any of them is a deep one.
FILLS = [(90, 150, 230), (60, 160, 60), (200, 120, 40), (180, 180, 250), (90, 90, 220), (120, 200, 230)]


def _page(color=WHITE) -> np.ndarray:
    return np.full((PAGE[1], PAGE[0], 3), color, np.uint8)


def _drawing(lines: bool = True, texture: float = 0.0) -> np.ndarray:
    """Six flat fills, each outlined in black 2 mm wide and crossed by a black line 0.5 mm wide.

    ``texture`` adds a smooth random texture of that standard deviation to each
    color channel, as a photograph's surfaces have.
    """
    image = _page()
    for i, color in enumerate(FILLS):
        x, y = 60 + (i % 3) * 340, 60 + (i // 3) * 340
        image[y : y + 280, x : x + 300] = color
        if lines:
            cv2.rectangle(image, (x, y), (x + 299, y + 279), BLACK, 8)
            cv2.line(image, (x + 40, y + 140), (x + 260, y + 140), BLACK, 2)
    if texture:
        rng = np.random.default_rng(0)
        noise = [cv2.GaussianBlur(rng.normal(0, 1, image.shape[:2]).astype(np.float32), (0, 0), 6) for _ in range(3)]
        field = np.stack(noise, axis=-1)
        image = np.clip(image + field * (texture / field.std()), 0, 255).astype(np.uint8)
    return image


def _gray_with_lightness(lstar: float) -> int:
    """The sRGB gray whose OpenCV lightness is nearest ``lstar`` (CIE L*)."""
    grays = np.arange(256, dtype=np.uint8).reshape(1, -1, 1).repeat(3, axis=2)
    lightness = cv2.cvtColor(grays, cv2.COLOR_BGR2Lab)[0, :, 0] * (100 / 255)
    return int(np.argmin(np.abs(lightness - lstar)))


def test_the_test_page_prints_at_four_pixels_a_millimeter():
    assert print_scale(PAGE).px_per_mm == PX_PER_MM


def test_the_agreed_values():
    # D-035, Q17: lines up to the harness's own 5 mm, breaks up to 0.5 mm; line art has flat fills and deep lines.
    assert ink.MAX_LINE_WIDTH_MM == bm.INK_MAX_WIDTH_MM == 5.0
    assert ink.GAP_MM == 0.5
    assert (ink.MAX_FLATNESS, ink.DEEP_LINE_DEPTH, ink.MIN_DEEP_LINE_SHARE) == (3.4, 40.0, 0.016)
    # D-035: flatness is the best of three k-means runs, which holds a textured picture's within 0.3 of itself over seeds.
    assert ink._KMEANS_ATTEMPTS == 3


def test_lightness_and_depth_are_in_cie_lstar():
    gray = _gray_with_lightness(40)
    lstar = float(cv2.cvtColor(np.full((1, 1, 3), gray, np.uint8), cv2.COLOR_BGR2Lab)[0, 0, 0]) * 100 / 255
    image = _page()
    image[100:400, 200:204] = gray  # a gray line 1 mm wide on white paper

    lightness, depth = ink._lightness_and_depth(image, PX_PER_MM * ink.MAX_LINE_WIDTH_MM)

    assert lightness[200, 202] == pytest.approx(lstar, abs=1e-4)
    assert depth[200, 202] == pytest.approx(100 - lstar, abs=1e-4)
    assert lightness[200, 100] == pytest.approx(100) and depth[200, 100] == 0


def test_line_art_is_flat_fills_and_deep_lines():
    decision = ink.line_art(_drawing())

    assert decision.is_line_art
    assert decision.flatness < 0.1
    # Nothing black is 5 mm wide, so every black pixel is on a line, and on these light fills every line is deep.
    lightness = cv2.cvtColor(np.array([FILLS], np.uint8), cv2.COLOR_BGR2Lab)[0, :, 0].astype(float) * 100 / 255
    assert lightness.min() > ink.DEEP_LINE_DEPTH
    assert decision.deep_line_share == pytest.approx((_drawing() == 0).all(axis=2).mean())


def test_lines_on_dark_fills_are_not_deep():
    # Black lines between fills no lighter than L* 35: lines, but not dark against light, so the picture isn't line art
    # however flat it is. A line is only as deep as the lighter of what lies on its two sides.
    dark = [(70, 40, 40), (40, 70, 40), (40, 40, 90), (60, 60, 60), (80, 50, 70), (50, 80, 80)]
    lightness = cv2.cvtColor(np.array([dark], np.uint8), cv2.COLOR_BGR2Lab)[0, :, 0].astype(float) * 100 / 255
    assert lightness.max() < 35
    image = _page(dark[0])
    for i, color in enumerate(dark):
        x, y = 60 + (i % 3) * 340, 60 + (i // 3) * 340
        image[y : y + 280, x : x + 300] = color
        cv2.rectangle(image, (x, y), (x + 299, y + 279), BLACK, 8)

    decision = ink.line_art(image)

    assert decision.flatness < 0.1
    # Only the page's corners are deep: off the page counts as white paper, and no 5 mm disk reaches into a corner.
    assert decision.deep_line_share < ink.MIN_DEEP_LINE_SHARE / 100
    assert not decision.is_line_art
    assert ink.ink_lines(image).any()  # still lines: black is well over 15 L* darker than these fills


def test_flat_fills_without_lines_are_not_line_art():
    decision = ink.line_art(_drawing(lines=False))

    assert not decision.is_line_art
    assert decision.flatness < 0.1
    assert decision.deep_line_share < ink.MIN_DEEP_LINE_SHARE / 10


def test_lines_on_textured_fills_are_not_line_art():
    # The dark windows in a sunlit wall: deep lines, but no flat fills around them.
    decision = ink.line_art(_drawing(texture=30))

    assert not decision.is_line_art
    assert decision.deep_line_share > 3 * ink.MIN_DEEP_LINE_SHARE
    assert decision.flatness > 2 * ink.MAX_FLATNESS


def test_print_texture_is_evened_out_before_flatness_is_measured():
    # Grain a pixel across, as a printed page's dots and a scan's noise are, blurred away by 0.25 mm; the photograph's
    # texture of the test above is a few millimeters across, and stays.
    rng = np.random.default_rng(1)
    grainy = np.clip(_drawing() + rng.normal(0, 12, _drawing().shape), 0, 255).astype(np.uint8)

    decision = ink.line_art(grainy)

    assert decision.is_line_art
    assert decision.flatness < ink.MAX_FLATNESS / 2


def test_flatness_is_the_median_so_a_textured_patch_does_not_make_a_picture_unflat():
    # One of the six fills photograph-textured, as a comic's photo panel would be.
    image = _drawing().astype(np.float32)
    rng = np.random.default_rng(0)
    field = np.stack([cv2.GaussianBlur(rng.normal(0, 1, image.shape[:2]).astype(np.float32), (0, 0), 6) for _ in range(3)], -1)
    image[60:340, 60:360] += field[60:340, 60:360] * (30 / field.std())
    image = np.clip(image, 0, 255).astype(np.uint8)

    decision = ink.line_art(image)

    assert decision.is_line_art
    assert decision.flatness < 0.1


def test_flatness_leaves_out_the_pixels_within_half_a_millimeter_of_a_line(monkeypatch):
    seen = {}
    measure = ink._flatness

    def spy(image, off_lines, blur_px):
        seen.update(off_lines=off_lines, blur_px=blur_px)
        return measure(image, off_lines, blur_px)

    monkeypatch.setattr(ink, "_flatness", spy)
    image = _drawing()

    ink.line_art(image)

    black = (image == 0).all(axis=2)
    distance = cv2.distanceTransform((~black).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    assert (seen["off_lines"] == (distance > PX_PER_MM * ink.LINE_MARGIN_MM)).all()
    assert seen["blur_px"] == PX_PER_MM * ink.FLAT_BLUR_MM


def test_flatness_samples_the_pixels_off_the_lines_whatever_pattern_they_make():
    # Off the lines only every sixth column, as between the strokes of a hatching; those columns are textured. A regular
    # grid of samples in step with them would see only the lines' columns, and nothing.
    image = _page((200, 200, 200))
    rng = np.random.default_rng(0)
    image[:, 3::6] = rng.integers(0, 256, size=image[:, 3::6].shape)
    off_lines = np.zeros(image.shape[:2], bool)
    off_lines[:, 3::6] = True

    assert ink._flatness(image, off_lines, 0.0) > 10


def test_ink_is_found_only_on_line_art():
    for picture, expected in ((_drawing(), True), (_drawing(lines=False), False), (_drawing(texture=30), False)):
        decision, found = ink.find_ink(picture)
        assert decision == ink.line_art(picture)
        assert decision.is_line_art == expected
        assert found.shape == picture.shape[:2] and found.dtype == bool
        assert found.any() == expected


def test_the_decision_is_made_on_the_picture_it_is_given():
    drawing = _drawing()
    photo = _drawing(texture=30)
    small = cv2.resize(drawing, (PAGE[0] // 2, PAGE[1] // 2), interpolation=cv2.INTER_AREA)

    decision, found = ink.find_ink(drawing, photo)
    assert not decision.is_line_art and not found.any()

    decision, found = ink.find_ink(small, drawing)
    assert decision == ink.line_art(drawing)
    assert (found == ink.ink_lines(small)).all()


def test_on_a_black_and_white_drawing_the_ink_lines_are_what_the_harness_calls_ink():
    """Black on white, the two find the same pixels: the parts no disk 5 mm wide fits into, the page's edge a boundary."""
    image = _page()
    image[100:108, 50:600] = BLACK  # a line 2 mm wide
    image[150:170, 50:600] = BLACK  # 5 mm: a disk as wide fits only exactly
    image[200:230, 50:600] = BLACK  # 7.5 mm: a fill, except where its ends' corners are too narrow
    cv2.circle(image, (800, 300), 60, BLACK, -1)  # a black disk 30 mm across
    cv2.circle(image, (800, 300), 30, WHITE, -1)  # with a hole: a ring 7.5 mm wide
    image[400:700, 700:720] = BLACK  # a bar 5 mm wide, bending into
    image[680:700, 700:1000] = BLACK
    image[:12, 300:700] = BLACK  # along the page's edge, 3 mm wide
    image[740:, :200] = BLACK  # into a corner, 5 mm wide
    image[:, 1100:] = BLACK  # the whole right edge, 2 mm wide

    lightness, depth = ink._lightness_and_depth(image, PX_PER_MM * ink.MAX_LINE_WIDTH_MM)
    reference = bm.source_ink(image, np.array([WHITE], np.uint8), np.array([BLACK], np.uint8), PX_PER_MM * bm.INK_MAX_WIDTH_MM)

    assert (ink._ink(lightness, depth) == reference).all()
    assert (ink.ink_lines(image) == reference).all()  # no breaks to close
    assert reference[100:108, 50:600].all() and not reference[215, 100:550].any()


def test_a_slight_darkening_in_a_dark_area_is_not_a_line():
    # Near black, a stroke can be darker than halfway to black without being 15 L* darker than what lies around it.
    very_dark, darker = _gray_with_lightness(12), _gray_with_lightness(4)
    image = _page()
    image[100:400, 100:500] = very_dark
    image[246:250, 150:450] = darker  # 1 mm wide, 8 L* darker than the fill around it

    lightness, depth = ink._lightness_and_depth(image, PX_PER_MM * ink.MAX_LINE_WIDTH_MM)
    assert 2 * lightness[248, 300] < lightness[248, 300] + depth[248, 300]  # darker than halfway
    assert depth[248, 300] < ink.MIN_LINE_DEPTH
    assert not ink.ink_lines(image)[200:300, 150:450].any()


def test_a_line_along_a_dark_fill_is_held_to_the_fill():
    navy = (103, 54, 50)  # the bold-line girl's navy: L* 25
    image = _page()
    image[100:400, 100:500] = navy
    image[100:400, 496:504] = BLACK  # between the navy and white paper
    image[246:254, 100:496] = BLACK  # across the navy, which lies on both sides of it

    found = ink.ink_lines(image)

    assert found[100:400, 496:504].all()
    assert found[246:254, 110:490].all()
    assert not found[110:240, 110:490].any() and not found[260:390, 110:490].any()


def test_an_anti_aliased_edge_is_ink_where_it_is_darker_than_halfway_to_black():
    darker, lighter = _gray_with_lightness(45), _gray_with_lightness(55)
    image = _page()
    image[100:400, 200:204] = BLACK
    image[100:400, 204] = darker  # an edge pixel mostly covered by the line
    image[100:400, 199] = lighter  # one mostly not

    found = ink.ink_lines(image)

    assert found[110:390, 200:205].all()
    assert not found[110:390, 199].any()


@pytest.mark.parametrize("direction", ["row", "column", "diagonal", "antidiagonal"])
def test_a_break_up_to_half_a_millimeter_closes_where_the_line_fades(direction):
    faint = (_gray_with_lightness(80),) * 3  # 20 L* darker than the paper, much lighter than halfway to black
    # 0.5 mm is 2 px: two pixels along a row or column, one step (1.4 px) along a diagonal.
    closes, too_long = (2, 3) if direction in ("row", "column") else (1, 2)

    def line_with_break(length: int, fill) -> tuple[np.ndarray, list]:
        image = _page()
        points = []
        for k in range(200, 400):
            y, x = {"row": (300, k), "column": (k, 300), "diagonal": (k, k), "antidiagonal": (k, 700 - k)}[direction]
            points.append((y, x))
        broken = points[100 : 100 + length]
        for i, (y, x) in enumerate(points):
            image[y, x] = fill if (y, x) in broken else BLACK
        return image, broken

    for length, fill, closed in ((closes, faint, True), (too_long, faint, False), (closes, WHITE, False)):
        image, broken = line_with_break(length, fill)
        found = ink.ink_lines(image)
        assert all(found[p] for p in broken) == closed, (length, fill)


@pytest.mark.parametrize("gap_px", [1.99, 2.9, 3.4, 5.0])
def test_closing_fills_exactly_the_breaks_up_to_its_length_in_every_direction(gap_px):
    at = {
        "row": lambda k, t: (30 + t, k),
        "column": lambda k, t: (k, 30 + t),
        "diagonal": lambda k, t: (k, k + t),
        "antidiagonal": lambda k, t: (k, 59 - k - t),
    }
    for direction, point in at.items():
        steps = math.floor(gap_px / (math.sqrt(2) if "diagonal" in direction else 1))
        for thickness in (1, 2):
            for length in (1, 2, 3, 4):
                mask = np.zeros((60, 60), bool)
                for k in range(60):  # from edge to edge, which closing must not extend
                    for t in range(thickness):
                        y, x = point(k, t)
                        if 0 <= x < 60:
                            mask[y, x] = True
                broken = [point(k, t) for k in range(28, 28 + length) for t in range(thickness)]
                for p in broken:
                    mask[p] = False

                closed = ink._close_breaks(mask, gap_px)

                expected = mask.copy()
                if length <= max(1, steps):
                    for p in broken:
                        expected[p] = True
                assert (closed == expected).all(), (direction, thickness, length)


def test_closing_leaves_an_empty_page_empty():
    assert not ink._close_breaks(np.zeros((40, 50), bool), 3.4).any()


def test_a_line_is_judged_by_how_wide_it_prints():
    # A black bar 4 mm wide and a black square 8 mm wide, drawn at 4 and at 8 px/mm.
    for scale in (1, 2):
        image = np.full((760 * scale, 1108 * scale, 3), 255, np.uint8)
        image[100 * scale : 400 * scale, 100 * scale : 116 * scale] = BLACK
        image[100 * scale : 132 * scale, 300 * scale : 332 * scale] = BLACK
        found = ink.ink_lines(image)
        assert found[100 * scale : 400 * scale, 100 * scale : 116 * scale].all()
        assert not found[106 * scale : 126 * scale, 306 * scale : 326 * scale].any()


def test_finding_ink_is_deterministic():
    picture = _drawing(texture=10)
    first, second = ink.find_ink(picture), ink.find_ink(picture.copy())
    assert first[0] == second[0]
    assert (first[1] == second[1]).all()


def test_deciding_on_the_picture_itself_measures_it_once(monkeypatch):
    drawing = _drawing()
    calls = []
    measure = ink._lightness_and_depth
    monkeypatch.setattr(ink, "_lightness_and_depth", lambda *args: calls.append(1) or measure(*args))

    decision, lines = ink.find_ink(drawing)

    assert len(calls) == 1
    monkeypatch.setattr(ink, "_lightness_and_depth", measure)
    assert decision == ink.line_art(drawing) and (lines == ink.ink_lines(drawing)).all()


def test_the_ink_prints_in_the_median_gray_of_its_own_pixels():
    image = _page()
    image[100:110, :] = (30, 30, 30)
    image[200:204, :] = (90, 90, 90)
    image[300:302, :] = (0, 0, 0)
    lines = np.zeros(image.shape[:2], dtype=bool)
    lines[100:110, :] = lines[200:204, :] = lines[300:302, :] = True

    assert ink.ink_gray(image, lines) == 30  # ten rows of 30 against four of 90 and two of 0
    assert ink.ink_gray(image, np.zeros_like(lines)) == 0  # no ink: black


def test_near_grows_a_mask_by_the_disk_of_pixels_within_its_radius():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    assert ink.near(mask, 0.0).sum() == 1
    assert ink.near(mask, 1.0).sum() == 5  # the pixel and its four neighbors
    assert ink.near(mask, 1.5).sum() == 9  # and the diagonals, sqrt(2) away
    assert ink.near(mask, 2.0).sum() == 13

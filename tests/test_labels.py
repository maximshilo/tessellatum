"""Where each region's number goes: inside it and clear of the lines, smaller if it must, or outside it with a leader."""

import dataclasses
import itertools
import math

import cv2
import numpy as np
import pytest

from tessellatum.core.boundaries import trace_boundaries
from tessellatum.core.labels import (
    FONT_SIZE_RADIUS_RATIO,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    LabelSpacing,
    _LeaderRoom,
    _pixels_along,
    min_font_size,
    place_labels,
    text_box_size,
)
from tessellatum.core.print_size import MIN_LABEL_SIZE_PT, print_scale
from tessellatum.core.regions import extract_regions
from tessellatum.core.render import PAPER, ink_coverage, render_page

LINE_PX = 1.3  # a 0.3 mm line on a 3:4 preview
SPACING = LabelSpacing(
    min_font_size=10, label_gap_px=LINE_PX, leader_width_px=LINE_PX, leader_dot_px=3 * LINE_PX, leader_reach_px=35.0
)


def _page(ids: np.ndarray, colors=None):
    """The page's regions, and where its lines leave the paper bare."""
    colors = np.arange(ids.max() + 1, dtype=np.int32) if colors is None else np.asarray(colors, dtype=np.int32)
    regions = extract_regions(ids, colors)
    free = ink_coverage(ids.shape[::-1], trace_boundaries(ids), LINE_PX) == 0
    return regions, free


def _pixels(box) -> tuple[slice, slice]:
    x0, y0, x1, y1 = (int(v) for v in box)
    return slice(y0, y1), slice(x0, x1)


def _preferred(region) -> int:
    return max(SPACING.min_font_size, int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))


@pytest.mark.parametrize("size", [(825, 1100), (1100, 825), (1100, 1100), (1800, 2400), (2048, 1366), (600, 450), (64, 48), (4000, 3000)])
def test_the_smallest_number_prints_at_least_six_points_and_is_never_under_ten_pixels(size):
    scale = print_scale(size)

    smallest = min_font_size(size)

    assert smallest >= MIN_FONT_SIZE
    assert smallest / scale.px_per_pt >= MIN_LABEL_SIZE_PT  # the arithmetic the benchmark measures a number with
    assert smallest == MIN_FONT_SIZE or (smallest - 1) / scale.px_per_pt < MIN_LABEL_SIZE_PT  # and no larger than that


def test_the_preview_keeps_its_ten_pixel_floor_and_an_export_gets_six_points():
    assert min_font_size((825, 1100)) == 10  # 6 pt is 8.4 px here
    assert min_font_size((1800, 2400)) == 21  # 6 pt is 20.05 px here: the whole pixel above it


def test_a_roomy_region_keeps_its_number_at_its_middle_at_the_size_it_prefers():
    ids = np.zeros((200, 300), dtype=np.int32)
    ids[40:160, 60:240] = 1
    regions, free = _page(ids)
    inner = regions[1]

    label = next(label for label in place_labels(regions, ids, free, SPACING) if label.region_id == 1)

    width, height = text_box_size("2", _preferred(inner))
    x, y = inner.interior_point
    assert (label.text, label.font_size, label.leader) == ("2", _preferred(inner), None)
    assert label.box == (x - width // 2, y - height // 2, x - width // 2 + width, y - height // 2 + height)


def test_a_number_moves_off_a_line_through_the_middle_and_keeps_as_far_from_the_lines_as_it_can():
    ids = np.ones((60, 200), dtype=np.int32)
    ids[:, :1] = 0  # a sliver down the left edge, so the page has a line at all
    regions, free = _page(ids)
    wide = next(region for region in regions if region.region_id == 1)
    x, _y = wide.interior_point
    free[:, x - 2 : x + 3] = False  # something drawn down the middle of the region

    label = next(label for label in place_labels(regions, ids, free, SPACING) if label.region_id == 1)

    assert label.font_size == _preferred(wide) and label.leader is None
    rows, columns = _pixels(label.box)
    assert free[rows, columns].all() and (ids[rows, columns] == 1).all()
    # Well clear of what runs down the middle, and of the page's edges above and below.
    assert columns.stop <= x - 2 - 10 or columns.start >= x + 3 + 10
    assert rows.start >= 10 and rows.stop <= 50


def test_a_number_that_does_not_fit_at_the_size_it_prefers_is_made_smaller_but_never_under_the_smallest():
    ids = np.zeros((150, 150), dtype=np.int32)
    ids[20:130, 20:130] = 1
    regions, free = _page(ids)
    big = next(region for region in regions if region.region_id == 1)
    room = np.zeros_like(free)
    room[60:70, 50:63] = True  # a window 13 x 10 is all the room the region has
    free &= room | (ids == 0)

    label = next(label for label in place_labels(regions, ids, free, SPACING) if label.region_id == 1)

    assert SPACING.min_font_size <= label.font_size < _preferred(big)
    assert label.leader is None
    assert free[_pixels(label.box)].all() and room[_pixels(label.box)].all()
    for larger in range(label.font_size + 1, _preferred(big) + 1):  # the largest that fits: every larger one is too big
        width, height = text_box_size("2", larger)
        assert width > 13 or height > 10


def _small_squares(count: int, side: int, spacing: int) -> np.ndarray:
    """A 200 x 300 page of background with ``count`` squares ``side`` px wide in a row across its middle."""
    ids = np.zeros((200, 300), dtype=np.int32)
    for i in range(count):
        left = 150 - (count * (side + spacing)) // 2 + i * (side + spacing)
        ids[100 : 100 + side, left : left + side] = i + 1
    return ids


def test_a_region_too_small_for_its_number_gets_it_written_beside_it_with_a_leader_pointing_in():
    ids = _small_squares(1, side=8, spacing=0)
    regions, free = _page(ids, colors=[0, 11])  # the square is color 12: a two-digit number
    square = next(region for region in regions if region.region_id == 1)

    label = next(label for label in place_labels(regions, ids, free, SPACING) if label.region_id == 1)

    assert (label.text, label.font_size) == ("12", SPACING.min_font_size)
    rows, columns = _pixels(label.box)
    assert free[rows, columns].all() and (ids[rows, columns] == 0).all()  # in the neighbor, clear of its lines
    end, anchor = label.leader
    assert anchor == tuple(float(v) for v in square.interior_point)  # it points at the middle of the square
    # It stops short of the number, by half its own width and a pixel of anti-aliasing...
    x0, y0, x1, y1 = label.box
    gap = math.hypot(max(x0 - 0.5 - end[0], 0, end[0] - (x1 - 0.5)), max(y0 - 0.5 - end[1], 0, end[1] - (y1 - 0.5)))
    assert gap == pytest.approx(SPACING.leader_width_px / 2 + 1)
    # ...and runs through the square and the region the number is in, nowhere else, within reach. It can end
    # on the square's own line: the number is then just across it.
    path = _pixels_along(anchor, end)
    assert set(ids[path].tolist()) <= {0, 1}
    assert math.dist(anchor, end) <= SPACING.leader_reach_px + gap


def test_numbers_written_beside_their_regions_keep_clear_of_each_other_and_of_every_leader():
    ids = _small_squares(6, side=8, spacing=6)
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, SPACING)

    with_leaders = [label for label in labels if label.leader is not None]
    assert len(with_leaders) == 6 and labels[0].leader is None  # the background's own number goes first
    for one, other in itertools.combinations(labels, 2):
        a, b = one.box, other.box
        apart = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])
        assert apart >= math.ceil(SPACING.label_gap_px)  # in whole pixels
    for label in with_leaders:
        path = _pixels_along(label.leader[1], label.leader[0])
        for other in labels:
            if other is not label:
                rows, columns = _pixels(other.box)
                assert not ((path[0] >= rows.start) & (path[0] < rows.stop) & (path[1] >= columns.start) & (path[1] < columns.stop)).any()


def test_a_leader_crosses_another_region_only_when_there_is_no_other_way_out():
    ids = np.zeros((200, 200), dtype=np.int32)
    ids[90:110, 90:110] = 2  # a ring too thin to hold a number...
    ids[96:104, 96:104] = 1  # ...around a square too small to hold one
    regions, free = _page(ids)

    labels = {label.region_id: label for label in place_labels(regions, ids, free, SPACING)}

    end, anchor = labels[1].leader
    assert 2 in set(ids[_pixels_along(anchor, end)].tolist())
    assert (ids[_pixels(labels[1].box)] == 0).all()


def test_with_no_room_anywhere_a_number_still_goes_at_the_middle_of_its_region():
    ids = _small_squares(1, side=30, spacing=0)
    regions, free = _page(ids)
    nowhere = np.zeros_like(free)

    labels = place_labels(regions, ids, nowhere, SPACING)

    assert [label.region_id for label in labels] == [0, 1]
    assert all(label.leader is None and label.font_size == SPACING.min_font_size for label in labels)
    x, y = regions[1].interior_point
    x0, y0, x1, y1 = labels[1].box
    assert x0 <= x < x1 and y0 <= y < y1


def test_every_region_is_numbered_clear_of_the_lines_and_of_the_other_numbers():
    # Patches of five colors, some of them roomy and some not, with squares too small for a number among them.
    rng = np.random.default_rng(3)
    colors = np.repeat(np.repeat(rng.integers(0, 5, size=(10, 12)), 20, axis=0), 20, axis=1)
    for top, left in rng.integers(10, 180, size=(12, 2)):
        colors[top : top + 6, left : left + 6] = 5
    # One id per connected patch, as the region stage leaves them.
    patches = np.zeros(colors.shape, dtype=np.int32)
    next_id = 0
    for color in range(6):
        count, components = cv2.connectedComponents((colors == color).astype(np.uint8), connectivity=8)
        patches[components > 0] = components[components > 0] + next_id - 1
        next_id += count - 1
    regions, free = _page(patches, colors=np.arange(next_id) % 12)

    labels = place_labels(regions, patches, free, SPACING)

    assert sorted(label.region_id for label in labels) == sorted(region.region_id for region in regions)
    assert any(label.leader is not None for label in labels)
    for label in labels:
        assert free[_pixels(label.box)].all()
        assert label.font_size >= SPACING.min_font_size
    for one, other in itertools.combinations(labels, 2):
        a, b = one.box, other.box
        assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    assert place_labels(regions, patches, free, SPACING) == labels  # and the same page every time


def test_a_number_stays_inside_its_own_region_whatever_the_ink_leaves_free():
    # On a real page a line inks both pixels beside every crack, so a box free
    # of ink is inside one region anyway; place_labels promises it regardless.
    ids = _small_squares(1, side=6, spacing=0)
    regions, _free = _page(ids)
    no_lines = np.ones(ids.shape, dtype=bool)

    square = next(label for label in place_labels(regions, ids, no_lines, SPACING) if label.region_id == 1)

    # A 10 px number is 8 px tall: its box centred on the 6 px square would reach into the background.
    assert square.leader is not None
    assert (ids[_pixels(square.box)] == 0).all()


def test_a_leader_takes_the_long_way_round_rather_than_cross_another_region():
    ids = np.full((200, 300), 2, dtype=np.int32)
    ids[128:, :] = 0  # open room below
    ids[94:128, 140:158] = 3  # a wall too thin to hold a number, down to the open room...
    ids[106:128, 146:152] = 0  # ...with a corridor through it
    ids[100:106, 146:152] = 1  # a square too small for its number, walled in but for the corridor
    regions, free = _page(ids)

    square = next(label for label in place_labels(regions, ids, free, SPACING) if label.region_id == 1)

    # Across the wall is nearer, but the leader would cross it; down the corridor it stays in two regions.
    end, anchor = square.leader
    assert set(ids[_pixels_along(anchor, end)].tolist()) == {0, 1}
    assert (ids[_pixels(square.box)] == 0).all()
    assert math.dist(anchor, end) > 20


def _slivers(seed: int) -> np.ndarray:
    """A page of eight small squares dropped on top of each other, leaving slivers with no room for a number."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((120, 160), dtype=np.int32)
    for i, (top, left) in enumerate(rng.integers(30, 90, size=(8, 2))):
        ids[top : top + 6, left : left + 6] = i + 1
    return ids


def _leader_ink(size: tuple[int, int], labels) -> np.ndarray:
    """Where the leaders' lines and dots put any ink, drawn as the renderer draws them."""
    ink = np.zeros(size[::-1], dtype=bool)
    for label in labels:
        if label.leader is not None:
            points = np.array(label.leader, dtype=np.float64)
            ink |= ink_coverage(size, [points], SPACING.leader_width_px) > 0
            ink |= ink_coverage(size, [points[1:].repeat(2, axis=0)], SPACING.leader_dot_px) > 0
    return ink


@pytest.mark.parametrize("seed", range(40))
def test_no_leader_s_line_or_dot_reaches_a_number_its_own_included(seed):
    ids = _slivers(seed)
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, SPACING)

    leader_ink = _leader_ink(ids.shape[::-1], labels)
    for label in labels:
        rows, columns = _pixels(label.box)
        if free[rows, columns].all():  # a number with no room anywhere goes on the lines regardless
            assert not leader_ink[rows, columns].any()


def test_a_leader_once_placed_holds_its_path_against_the_numbers_after_it():
    ids = _small_squares(1, side=6, spacing=0)
    regions, free = _page(ids)
    square = next(region for region in regions if region.region_id == 1)
    room = _LeaderRoom(ids, free, [], [square.interior_point], SPACING)

    label = room.place(square)

    end, anchor = label.leader
    assert room.leaders[_pixels_along(anchor, end)].all() and room.boxes[_pixels(label.box)].all()


def _clear_of_everything(rendered) -> None:
    """Every number's box holds no ink of a line or a leader, and overlaps no other number."""
    lines, leaders = np.asarray(rendered.outlines), np.asarray(rendered.leaders)
    for label in rendered.labels:
        rows, columns = _pixels(label.box)
        assert (lines[rows, columns] == PAPER).all() and (leaders[rows, columns] == PAPER).all(), label
    for one, other in itertools.combinations(rendered.labels, 2):
        a, b = one.box, other.box
        assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (one, other)


@pytest.mark.parametrize("size", [(1100, 825), (2400, 1800)])
def test_a_ring_and_the_square_it_walls_in_both_get_their_numbers_clear_of_everything(size):
    # Neither has room for its number, and the square's leader has to cross the ring: at print scale, where a
    # leader reaches 8 mm, the ring's number must not be crowded out by the square's.
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    ids[height // 2 - 7 : height // 2 + 7, width // 2 - 7 : width // 2 + 7] = 2
    ids[height // 2 - 4 : height // 2 + 4, width // 2 - 4 : width // 2 + 4] = 1
    regions = extract_regions(ids, np.array([0, 1, 11], dtype=np.int32))  # the ring's number has two digits

    rendered = render_page(size, regions, ids)

    numbers = {label.region_id: label for label in rendered.labels}
    assert numbers[1].leader is not None and numbers[2].leader is not None
    _clear_of_everything(rendered)


@pytest.mark.parametrize("size", [(1100, 825), (2400, 1800)])
def test_a_region_two_pixels_wide_gets_a_leader_that_stops_short_of_its_number(size):
    # Its middle is on its edge, so the nearest room is under a pixel away: too near for a leader, let alone a dot.
    width, height = size
    ids = np.zeros((height, width), dtype=np.int32)
    ids[height // 2 : height // 2 + 2, width // 2 - 40 : width // 2 + 40] = 1
    regions = extract_regions(ids, np.array([0, 1], dtype=np.int32))

    rendered = render_page(size, regions, ids)

    strip = next(label for label in rendered.labels if label.region_id == 1)
    end, anchor = strip.leader
    assert math.dist(end, anchor) > 0
    _clear_of_everything(rendered)


def _scattered(seed: int) -> np.ndarray:
    """A page of 6-13 rectangles 3-9 px across dropped on top of each other: regions too small for a number, touching."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((120, 160), dtype=np.int32)
    for i in range(int(rng.integers(6, 14))):
        top, left = rng.integers(30, 90, size=2)
        height, width = rng.integers(3, 10, size=2)
        ids[top : top + height, left : left + width] = i + 1
    return ids


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_a_number_written_outside_its_region_stays_wholly_outside_it_whatever_the_ink_leaves_free(seed):
    # With lines drawn, a box free of ink cannot reach into its own region anyway; place_labels promises it regardless.
    ids = _scattered(seed)
    regions, _free = _page(ids)

    labels = place_labels(regions, ids, np.ones(ids.shape, dtype=bool), SPACING)

    for label in labels:
        if label.leader is not None:
            assert not (ids[_pixels(label.box)] == label.region_id).any()


@pytest.mark.parametrize("seed", [12, 15, 18])
def test_no_leader_s_line_runs_through_another_number(seed):
    ids = _scattered(seed)
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, SPACING)

    for label in labels:
        if label.leader is None:
            continue
        line = ink_coverage(ids.shape[::-1], [np.array(label.leader, dtype=np.float64)], SPACING.leader_width_px) > 0
        for other in labels:
            if other is not label:
                assert not line[_pixels(other.box)].any()


@pytest.mark.parametrize("seed", [0, 3, 5])
def test_with_a_dot_smaller_than_its_line_no_number_goes_too_near_for_a_leader_to_reach_it(seed):
    # The dot keeps every number farther from the point a leader starts from than the leader stops short of it; a
    # style with a small dot does not, and a number that near is passed over, not given a leader running backwards.
    ids = _scattered(seed)
    regions, free = _page(ids)
    small_dot = dataclasses.replace(SPACING, leader_dot_px=0.5)

    labels = place_labels(regions, ids, free, small_dot)

    for label in labels:
        if label.leader is not None:
            (x, y) = label.leader[1]
            x0, y0, x1, y1 = label.box
            near = (min(max(x, x0 - 0.5), x1 - 0.5), min(max(y, y0 - 0.5), y1 - 0.5))
            assert math.dist((x, y), near) > small_dot.leader_width_px / 2 + 1

"""A region pinched so thin that it reads as two regions on paper gets its number on each side."""

import dataclasses
import math
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tessellatum.core import pipeline, settings
from tessellatum.core.boundaries import trace_boundaries
from tessellatum.core.labels import SPLIT_NECK_MM, Label, LabelSpacing, _farthest, _LeaderRoom, find_sides, place_labels
from tessellatum.core.regions import extract_regions
from tessellatum.core.render import PAPER, PageStyle, ink_coverage

LINE_PX = 1.3
SPACING = LabelSpacing(
    min_font_size=10, label_gap_px=LINE_PX, leader_width_px=LINE_PX, leader_dot_px=3 * LINE_PX, leader_reach_px=35.0
)


def _sides_spacing(neck_px: float, min_side_px: float = 200.0) -> LabelSpacing:
    return dataclasses.replace(SPACING, split_neck_px=neck_px, min_side_px=min_side_px)


def _dumbbell(neck: int, radius: int = 30, right_radius: int | None = None, gap: int = 24) -> np.ndarray:
    """Region 1: two disks side by side joined by a bar ``neck`` px tall (none if 0), in region 0."""
    right_radius = radius if right_radius is None else right_radius
    height = 2 * max(radius, right_radius) + 40
    width = 2 * radius + gap + 2 * right_radius + 40
    ids = np.zeros((height, width), dtype=np.int32)
    middle = height // 2
    left_x, right_x = 20 + radius, 20 + 2 * radius + gap + right_radius
    shape = np.zeros_like(ids, dtype=np.uint8)
    cv2.circle(shape, (left_x, middle), radius, 1, -1)
    cv2.circle(shape, (right_x, middle), right_radius, 1, -1)
    if neck:
        shape[middle - neck // 2 : middle - neck // 2 + neck, left_x : right_x] = 1
    ids[shape > 0] = 1
    return ids


def _page(ids: np.ndarray):
    regions = extract_regions(ids, np.arange(ids.max() + 1, dtype=np.int32))
    free = ink_coverage(ids.shape[::-1], trace_boundaries(ids), LINE_PX) == 0
    return regions, free


def _numbers_of(labels, region_id: int):
    return [label for label in labels if label.region_id == region_id]


def _side_of(ids: np.ndarray, label) -> str:
    """Which disk of a dumbbell a number is on, or points into: 'left' or 'right' of the page's middle."""
    if label.leader is not None:
        x = label.leader[1][0]
    else:
        x = (label.box[0] + label.box[2]) / 2
    return "left" if x < ids.shape[1] / 2 else "right"


def test_a_neck_narrower_than_the_setting_gives_each_side_its_number():
    ids = _dumbbell(neck=6)
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=10))

    numbers = _numbers_of(labels, 1)
    assert [label.side for label in numbers] == [0, 1]
    assert {_side_of(ids, label) for label in numbers} == {"left", "right"}
    assert all(label.text == "2" and not label.cramped for label in numbers)


def test_a_neck_at_least_as_wide_as_the_setting_holds_the_region_together():
    ids = _dumbbell(neck=16)
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=10))

    assert [label.side for label in _numbers_of(labels, 1)] == [0]


def test_without_the_setting_every_region_is_numbered_once():
    ids = _dumbbell(neck=6)
    regions, free = _page(ids)

    assert [label.side for label in _numbers_of(place_labels(regions, ids, free, SPACING), 1)] == [0]


def test_the_region_s_own_numbers_go_exactly_where_they_went_without_sides():
    ids = _dumbbell(neck=4)
    regions, free = _page(ids)

    plain = place_labels(regions, ids, free, SPACING)
    with_sides = place_labels(regions, ids, free, _sides_spacing(neck_px=10))

    assert [label for label in with_sides if label.side == 0] == plain
    assert len(with_sides) == len(plain) + 1


def test_a_neck_the_lines_close_parts_the_sides_whatever_the_setting():
    ids = _dumbbell(neck=20)
    regions, free = _page(ids)
    middle_x = ids.shape[1] // 2
    free[:, middle_x - 1 : middle_x + 2] = False  # a line drawn right across the wide neck

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=0.0))

    assert [label.side for label in _numbers_of(labels, 1)] == [0, 1]


def test_a_side_smaller_than_the_smallest_region_does_not_count():
    ids = _dumbbell(neck=3, radius=30, right_radius=7)  # the right side is about 180 px, the line under it included
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=5, min_side_px=200))
    assert [label.side for label in _numbers_of(labels, 1)] == [0]

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=5, min_side_px=60))
    assert [label.side for label in _numbers_of(labels, 1)] == [0, 1]


def test_a_blob_narrower_than_the_setting_everywhere_is_no_side_of_its_own():
    ids = _dumbbell(neck=3, radius=30, right_radius=5)  # its paper is narrower than 12 px across
    _regions, free = _page(ids)

    assert find_sides(ids, free, neck_px=12, min_area_px=30) == {}
    assert 1 in find_sides(ids, free, neck_px=5, min_area_px=30)


def test_a_side_with_no_room_inside_gets_its_number_with_a_leader():
    ids = _dumbbell(neck=3, radius=30, right_radius=4)  # too small for a 10 px number, large enough to count
    regions, free = _page(ids)

    labels = place_labels(regions, ids, free, _sides_spacing(neck_px=4, min_side_px=30))

    numbers = _numbers_of(labels, 1)
    assert [label.side for label in numbers] == [0, 1]
    side = numbers[1]
    assert side.leader is not None and _side_of(ids, side) == "right"
    rows = slice(int(side.box[1]), int(side.box[3]))
    columns = slice(int(side.box[0]), int(side.box[2]))
    assert (ids[rows, columns] != 1).all()  # written outside its region, as a region's own would be


def test_a_side_with_no_room_anywhere_stays_unnumbered_rather_than_written_over_the_lines():
    ids = _dumbbell(neck=3, radius=30, right_radius=4)
    regions, free = _page(ids)
    nowhere = free & (ids == 1)  # no paper outside the region: no room for a number with a leader

    labels = place_labels(regions, ids, nowhere, _sides_spacing(neck_px=4, min_side_px=30))

    assert not any(label.side for label in labels)


def test_find_sides_maps_each_side_and_its_middle():
    ids = _dumbbell(neck=4)
    _regions, free = _page(ids)

    sides = find_sides(ids, free, neck_px=10, min_area_px=200)

    assert list(sides) == [1]
    found = sides[1]
    assert len(found.points) == 2
    left, right = sorted(found.points)
    assert left[0] < ids.shape[1] / 2 < right[0]
    assert found.side_at(left) != found.side_at(right)
    # All of the region's paper is on one side or the other.
    rows = slice(found.y0, found.y0 + found.map.shape[0])
    columns = slice(found.x0, found.x0 + found.map.shape[1])
    assert ((found.map > 0) == ((ids[rows, columns] == 1) & free[rows, columns])).all()


@pytest.mark.parametrize("neck", range(6, 17))
def test_a_neck_splits_the_region_where_its_paper_is_narrower_than_the_setting(neck):
    """A neck splits the region if its paper, between the lines, is fewer pixels across than the setting, 10 px."""
    ids = _dumbbell(neck=neck)
    _regions, free = _page(ids)
    middle_x = ids.shape[1] // 2
    paper_across = int((free[:, middle_x] & (ids[:, middle_x] == 1)).sum())

    assert (1 in find_sides(ids, free, neck_px=10, min_area_px=200)) is (paper_across < 10)


def test_the_setting_is_offered_with_its_default():
    setting = settings.setting("split_neck_mm")
    assert setting.owner == settings.STYLE and setting.group == settings.LINES
    assert setting.default == SPLIT_NECK_MM == PageStyle().split_neck_mm
    assert setting.minimum == 0.0


def test_a_side_s_leader_points_where_its_dot_keeps_clear_of_the_numbers_placed_before():
    ids = _dumbbell(neck=3, radius=30, right_radius=14)
    _regions, free = _page(ids)
    found = find_sides(ids, free, neck_px=4, min_area_px=30)[1]
    right = max(range(len(found.points)), key=lambda k: found.points[k][0])
    x, y = found.points[right]
    number = Label(0, "1", 10, (x - 6, y + 1, x + 6, y + 9))  # a number placed just below the side's middle
    room = _LeaderRoom(ids, free, [number], [], SPACING)

    on_side, x0, y0 = found.side(right)
    point = room._side_anchor(on_side, x0, y0)

    assert point is not None and point != (x, y)
    assert on_side[point[1] - y0, point[0] - x0]
    nearest_x = min(max(point[0], number.box[0]), number.box[2])
    nearest_y = min(max(point[1], number.box[1]), number.box[3])
    assert math.hypot(point[0] - nearest_x, point[1] - nearest_y) > room.dot_reach


def test_on_an_export_a_region_s_own_number_takes_its_place_before_a_side_s():
    """An export's number that finds no room goes where its preview put it; a side's number there gives way."""
    size = (60, 40)
    own = Label(1, "2", 10, (10, 10, 22, 18), cramped=True)
    side = Label(1, "2", 10, (30, 12, 42, 20), side=1)
    other = Label(2, "3", 10, (45, 25, 57, 33))
    rendered = SimpleNamespace(
        labels=[own, side, other], outlines=np.full(size[::-1], PAPER, np.uint8), leader_ink={}
    )
    preview = [replace(own, box=(30, 12, 42, 20), cramped=False)]

    labels = pipeline._with_room(rendered, preview, 1.0, 1.0, size)

    assert labels is not None
    assert [label.side for label in labels if label.region_id == 1] == [0]
    assert labels[0].box[1] == 12 and (labels[0].box[0] + labels[0].box[2]) / 2 == 36 and not labels[0].cramped
    assert other in labels


def test_a_region_s_detail_ink_past_its_box_on_the_region_map_is_part_of_its_sides():
    """Line art gives detail ink back to its region: its paper can run past the region map's box for it."""
    ids = _dumbbell(neck=3)
    ids[2:5, 2:60] = 2  # a strip too thin for its number at its middle, so every region's box is worked out
    regions, free = _page(ids)
    top = int(np.flatnonzero((ids == 1).any(axis=1))[0])
    row = top + 4  # detail ink across the region, under the line over its top rows: on the map, in no region
    region_map, clearable = ids.copy(), np.full_like(ids, -1)
    over = ids[top : row + 1] == 1
    region_map[top : row + 1][over] = -1
    clearable[row][ids[row] == 1] = 1
    free[top:row][ids[top:row] == 1] = False
    free[row : row + 3] |= ids[row : row + 3] == 1

    labels = place_labels(regions, region_map, free, _sides_spacing(neck_px=10), clearable=clearable)

    assert [label.side for label in _numbers_of(labels, 1)] == [0, 1]
    assert _numbers_of(labels, 2)


def test_points_as_far_from_the_lines_tie_to_the_first_however_their_last_bit_falls():
    """A precise distance transform can give one distance a bit apart on two calls: the point must not move with it."""
    clearance = np.zeros((3, 4), np.float32)
    clearance[0, 3] = np.float32(math.sqrt(37))
    clearance[2, 1] = np.nextafter(clearance[0, 3], np.float32(10))  # the same distance, its last bit up
    clearance[1, 0] = np.float32(7.0)  # farther, but not in ``where``
    where = np.ones_like(clearance, dtype=bool)
    where[1, 0] = False

    assert _farthest(clearance, where) == (0, 3)

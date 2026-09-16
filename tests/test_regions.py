import cv2
import numpy as np
import pytest

from tessellatum.core.regions import absorb_thin_parts, build_regions, extract_regions


def test_small_regions_get_merged_away():
    # A 20x20 field of color 0, with a single 2x2 speckle of color 1 in the middle.
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[9:11, 9:11] = 1

    region_id_map, region_color = build_regions(labels, num_colors=2, min_area_px=10)
    regions = extract_regions(region_id_map, region_color)

    # The speckle (area 4 < min_area_px 10) should have been merged into its
    # only neighbor, leaving a single region covering the whole image.
    assert len(regions) == 1
    assert regions[0].area == 400


def test_large_regions_survive_merging():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[:, 10:] = 1  # two equal halves, both well above threshold

    region_id_map, region_color = build_regions(labels, num_colors=2, min_area_px=10)
    regions = extract_regions(region_id_map, region_color)

    assert len(regions) == 2
    assert {r.area for r in regions} == {200, 200}


def test_regions_have_valid_interior_points():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[:, 15:] = 1

    region_id_map, region_color = build_regions(labels, num_colors=2, min_area_px=5)
    regions = extract_regions(region_id_map, region_color)

    for region in regions:
        x, y = region.interior_point
        assert 0 <= x < 30
        assert 0 <= y < 30
        assert region.interior_radius > 0


def _same_color_neighbor_pixels(region_id_map: np.ndarray, region_color: np.ndarray) -> int:
    """Pairs of 8-adjacent pixels in different regions of one color: 0 on a well-formed page."""
    ids = region_id_map.astype(np.int64)
    pairs = (
        (ids[:, :-1], ids[:, 1:]),
        (ids[:-1, :], ids[1:, :]),
        (ids[:-1, :-1], ids[1:, 1:]),
        (ids[:-1, 1:], ids[1:, :-1]),
    )
    return sum(
        int(np.count_nonzero(region_color[a[differ]] == region_color[b[differ]]))
        for a, b in pairs
        for differ in [(a != b) & (a >= 0) & (b >= 0)]
    )


def test_same_colored_neighbors_become_one_region():
    # Two halves of color 0 split by a column of color 1 too small to keep.
    # The column merges into the left half (a tie on shared boundary, broken
    # by region id), which then touches the right half in its own color.
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[:, 10] = 1

    region_id_map, region_color = build_regions(labels, num_colors=2, min_area_px=25)
    regions = extract_regions(region_id_map, region_color)

    assert len(regions) == 1
    assert regions[0].area == 400
    assert _same_color_neighbor_pixels(region_id_map, region_color) == 0


def test_same_colored_neighbors_touching_only_diagonally_become_one_region():
    # `P` (color 2) merges into the big `A` region on its left, and its top
    # pixel then meets the `B` block, of A's color, at a corner only. Regions
    # are 8-connected, so that corner is a boundary between two same-colored
    # regions just as a shared edge would be.
    a = b = 0
    x, p = 1, 2
    labels = np.array(
        [
            [x, x, x, x, b, b, b],
            [a, a, a, p, x, b, b],
            [a, a, a, p, x, x, x],
            [a, a, a, p, x, x, x],
            [a, a, a, a, a, a, a],
        ],
        dtype=np.int32,
    )

    region_id_map, region_color = build_regions(labels, num_colors=3, min_area_px=4)
    regions = extract_regions(region_id_map, region_color)

    assert [(r.color_index, r.area) for r in regions] == [(0, 24), (1, 11)]
    assert np.array_equal(region_id_map == regions[0].region_id, labels != x)
    assert _same_color_neighbor_pixels(region_id_map, region_color) == 0


@pytest.mark.parametrize("seed, num_colors, min_area_px", [(0, 5, 30), (1, 12, 60), (2, 6, 100), (3, 8, 4)])
@pytest.mark.parametrize("min_width_px", [0.0, 5.0])
def test_no_two_neighboring_regions_share_a_color(seed, num_colors, min_area_px, min_width_px):
    labels = _blobby_labels(seed, num_colors)

    region_id_map, region_color = build_regions(labels, num_colors, min_area_px, min_width_px)

    assert _same_color_neighbor_pixels(region_id_map, region_color) == 0
    # Joining regions that touch keeps every region one 8-connected shape.
    for rid in np.unique(region_id_map):
        mask = (region_id_map == rid).astype(np.uint8)
        assert cv2.connectedComponents(mask, connectivity=8)[0] == 2  # background + the region


def _blobby_labels(seed: int, num_colors: int, shape: tuple[int, int] = (90, 120)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 2.0)
    edges = np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])
    return np.digitize(field, edges).astype(np.int32)


def _painted(region_id_map: np.ndarray, region_color: np.ndarray) -> np.ndarray:
    """Each pixel's color index, -1 outside every region."""
    inside = region_id_map >= 0
    return np.where(inside, region_color[np.where(inside, region_id_map, 0)], -1)


def _brush_reach(region_id_map: np.ndarray) -> np.ndarray:
    """Per pixel, the distance to the nearest pixel of another region, counting off the page as one.

    The slow, obvious version: every pixel against every other pixel. A brush
    of radius r fits on the pixels whose reach is more than r.
    """
    h, w = region_id_map.shape
    ys, xs = np.mgrid[0:h, 0:w]
    reach = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            rid = region_id_map[y, x]
            if rid < 0:
                continue
            others = region_id_map != rid
            off_page = min(x + 1, y + 1, w - x, h - y)  # the nearest pixel outside the page
            nearest = np.hypot(ys - y, xs - x)[others].min() if others.any() else np.inf
            reach[y, x] = min(nearest, off_page)
    return reach


def _disk(radius: float) -> np.ndarray:
    span = np.arange(-int(radius), int(radius) + 1)
    return (np.hypot(*np.meshgrid(span, span)) <= radius).astype(np.uint8)


def test_a_region_narrower_than_the_brush_is_split_between_its_neighbors():
    # A 3 px stripe between two wide blocks, with a brush 5 px across.
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 19:22] = 1
    labels[:, 22:] = 2

    region_id_map, region_color = build_regions(labels, 3, min_area_px=4, min_width_px=5.0)

    assert sorted(region_color[np.unique(region_id_map)]) == [0, 2]  # the stripe's color is gone
    left = region_id_map == region_id_map[0, 0]
    right = region_id_map == region_id_map[0, -1]
    assert (left | right).all() and not (left & right).any()
    # Both blocks keep everything they had, and share out the stripe.
    assert left[:, :19].all() and right[:, 22:].all()
    assert 0 < left[:, 19:22].sum() < left[:, 19:22].size


def test_a_region_as_wide_as_the_brush_survives():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 17:24] = 1
    labels[:, 24:] = 2

    region_id_map, region_color = build_regions(labels, 3, min_area_px=4, min_width_px=5.0)

    stripe = region_id_map[20, 20]
    assert region_color[stripe] == 1
    assert (region_id_map[:, 18:23] == stripe).all()  # it may lose its outermost pixel on each side


def test_the_thin_tail_of_a_region_is_handed_over_where_its_own_brush_stops_reaching():
    # A 20 px block of color 1 with a 3 px tail reaching 12 px into color 0.
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[19:22, 30:42] = 1

    region_id_map, region_color = build_regions(labels, 2, min_area_px=4, min_width_px=7.0)

    block = region_id_map[20, 20]
    assert region_color[block] == 1
    tail = region_id_map[20, 30:42] == block
    assert tail[0] and not tail[-1]  # the base stays, the far end goes to the neighbor
    assert (tail[:-1] >= tail[1:]).all()  # and it is handed over from one point on, not in patches


def test_nothing_moves_when_no_brush_fits_on_the_page():
    labels = np.zeros((4, 60), dtype=np.int32)
    labels[:, 20:40] = 1

    without_ids, without_color = build_regions(labels, 2, min_area_px=4)
    ids, color = build_regions(labels, 2, min_area_px=4, min_width_px=30.0)

    np.testing.assert_array_equal(ids, without_ids)
    np.testing.assert_array_equal(color, without_color)


@pytest.mark.parametrize("seed, num_colors, min_area_px, min_width_px", [(0, 5, 30, 5.0), (3, 8, 4, 7.0)])
def test_every_region_left_has_room_for_the_brush(seed, num_colors, min_area_px, min_width_px):
    labels = _blobby_labels(seed, num_colors, shape=(60, 80))

    region_id_map, _region_color = build_regions(labels, num_colors, min_area_px, min_width_px)

    reach = _brush_reach(region_id_map)
    for rid in np.unique(region_id_map[region_id_map >= 0]):
        assert reach[region_id_map == rid].max() > min_width_px / 2, f"no brush fits inside region {rid}"


@pytest.mark.parametrize("seed, min_width_px", [(1, 5.0), (3, 7.0)])
def test_what_a_regions_own_brush_covers_keeps_its_color(seed, min_width_px):
    labels = _blobby_labels(seed, 6, shape=(60, 80))
    region_id_map, region_color = build_regions(labels, 6, 30)

    widened = absorb_thin_parts(region_id_map, region_color, labels, min_width_px)

    # Where a region's own brush fits, it sweeps a disk that no other
    # region's core comes within a brush radius of, so every pixel under it
    # keeps its color and only the thin parts move.
    fits = _brush_reach(region_id_map) > min_width_px / 2
    swept = np.zeros(labels.shape, dtype=bool)
    for rid in np.unique(region_id_map[fits]):
        mine = region_id_map == rid
        swept |= mine & cv2.dilate((mine & fits).astype(np.uint8), _disk(min_width_px / 2)).astype(bool)
    assert swept.mean() > 0.2
    np.testing.assert_array_equal(widened[swept], _painted(region_id_map, region_color)[swept])
    assert not np.array_equal(widened, _painted(region_id_map, region_color))

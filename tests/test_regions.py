import cv2
import numpy as np
import pytest

from tessellatum.core import ink as ink_module
from tessellatum.core import kernels
from tessellatum.core import regions as regions_module
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


def test_nearest_seed_within_goes_round_what_it_cannot_pass():
    # A wall down column 4, open only in the bottom row. Seed 7 top left, seed 9 bottom right.
    passable = np.ones((5, 9), dtype=bool)
    passable[:4, 4] = False
    seeds = np.full((5, 9), -1, dtype=np.int32)
    seeds[0, 0], seeds[4, 8] = 7, 9

    nearest = kernels.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), 5, 9).reshape(5, 9)

    # (0, 5) lies 5 steps from seed 7 in a straight line and about 5 from seed 9, but the way to 7 goes round the wall.
    assert nearest[0, 5] == 9
    assert nearest[0, 3] == 7
    assert (nearest[~passable] == -1).all()
    enclosed = passable.copy()
    enclosed[:, 4] = False  # the wall now runs the whole height: nothing on the right reaches seed 7
    split = kernels.nearest_seed_within(seeds.reshape(-1), enclosed.reshape(-1), 5, 9).reshape(5, 9)
    assert (split[:, :4] == 7).all() and (split[:, 5:] == 9).all()
    alone = enclosed.copy()
    alone[:, 8] = False  # and seed 9 is now walled off itself: its side reaches nothing
    assert (kernels.nearest_seed_within(seeds.reshape(-1), alone.reshape(-1), 5, 9).reshape(5, 9)[:, 5:8] == -1).all()


def _thin_arm_beside_the_ink() -> np.ndarray:
    """Fill 0 left of a line of ink (label 3) one pixel wide; right of it, a thin arm of fill 1 beside fill 2.

    The arm is 3 px wide, rows 0-24, and joins fill 1's block below. At the
    ink, fill 0's paintable core lies nearer the arm than fill 2's does.
    """
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, 20] = 3
    labels[:, 21:] = 2
    labels[:25, 21:24] = 1
    labels[25:, 21:] = 1
    return labels


def test_a_thin_part_is_never_given_to_a_core_across_the_ink():
    labels = _thin_arm_beside_the_ink()
    ids, colors = regions_module._regions_from_labels(labels, 3, 1)

    widened = regions_module.absorb_thin_parts(ids, colors, labels, 8.0)

    assert (widened[:, 20] == 3).all()  # the ink stays ink
    arm = widened[:18, 21:24]
    assert not (arm == 0).any()  # nothing across the ink reaches it
    assert (arm == 2).all()  # the fill beside it, on its own side, does
    # With the ink a fill of color 0 instead, the arm's first column goes to fill 0, which is nearer.
    unfenced = labels.copy()
    unfenced[:, 20] = 0
    ids, colors = regions_module._regions_from_labels(unfenced, 3, 1)
    assert (regions_module.absorb_thin_parts(ids, colors, unfenced, 8.0)[:18, 21] == 0).all()


def test_a_thin_part_with_no_core_on_its_side_of_the_ink_keeps_its_color():
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, 20] = labels[:, 25] = 3  # two lines of ink, 4 px apart
    labels[:, 21:25] = 1  # a strip of fill 1 between them, narrower than the brush
    labels[:, 26:] = 2
    ids, colors = regions_module._regions_from_labels(labels, 3, 1)

    widened = regions_module.absorb_thin_parts(ids, colors, labels, 8.0)

    assert (widened[:, 21:25] == 1).all()


BLACK_BGR = (0, 0, 0)
FILL_BGR = (200, 120, 40)


def _patches_between_ink_lines():
    """A fill (label 0) crossed by ink (label 2) in two horizontal lines 4 px apart, with patches of label 1 between them.

    Between the lines, from the left: a small black patch (A), a small dark
    gray one (B), a large black one (C). Below the lower line, a small black
    patch (D) that touches the ink only along its top. Label 1 is the same
    palette color for all four: what they are is in the image.
    """
    labels = np.zeros((40, 120), dtype=np.int32)
    image = np.full((40, 120, 3), FILL_BGR, dtype=np.uint8)
    labels[10:12, :] = labels[16:18, :] = 2
    image[10:12, :] = image[16:18, :] = BLACK_BGR
    patches = {"A": (slice(12, 16), slice(5, 16)), "B": (slice(12, 16), slice(25, 36)), "C": (slice(12, 16), slice(45, 75))}
    patches["D"] = (slice(18, 22), slice(90, 101))
    for name, where in patches.items():
        labels[where] = 1
        image[where] = (70, 70, 70) if name == "B" else BLACK_BGR
    return labels, image, patches


def test_a_small_patch_in_the_ink_s_color_mostly_edged_by_the_ink_joins_it():
    labels, image, patches = _patches_between_ink_lines()

    joined = regions_module.join_ink(labels, 2, image, BLACK_BGR, min_area_px=60)

    assert (joined[patches["A"]] == 2).all()  # 44 px, black, ink above and below: the ink running wide
    assert (joined[patches["B"]] == 1).all()  # dark gray is not the ink's color (L* 29 against 0)
    assert (joined[patches["C"]] == 1).all()  # 120 px: large enough to stay a region to paint
    assert (joined[patches["D"]] == 1).all()  # black, but mostly edged by the fill
    assert (joined[labels != 1] == labels[labels != 1]).all()
    assert (labels[patches["A"]] == 1).all()  # a new map: the one given is the stage cache's


def test_a_patch_s_color_is_its_own_away_from_the_ink_s_edge():
    labels, image, patches = _patches_between_ink_lines()
    rows, columns = patches["A"]
    image[rows, columns] = (68, 42, 24)  # dark navy: 17 from black in CIEDE2000
    image[rows.start, columns] = image[rows.stop - 1, columns] = BLACK_BGR  # its rows beside the ink anti-aliased to black
    off_edge = ~ink_module.near(labels == 2, 1.0)  # not the ink, nor the pixels right beside it

    assert (regions_module.join_ink(labels, 2, image, BLACK_BGR, 60)[rows, columns] == 2).all()  # half black, half navy
    assert (regions_module.join_ink(labels, 2, image, BLACK_BGR, 60, own=off_edge)[rows, columns] == 1).all()


def test_a_shape_the_ink_encloses_alone_keeps_its_number_if_a_brush_fits_whatever_its_size():
    ids = np.full((60, 120), -1, dtype=np.int32)  # the ink
    ids[5:25, 5:55] = 0  # a fill beside another: not enclosed alone
    ids[5:25, 55:110] = 1
    ids[35:45, 10:20] = 2  # enclosed alone, 10 x 10: a brush 8 px wide fits
    ids[35:39, 40:60] = 3  # enclosed alone, 4 px wide: no brush fits
    ids[35:45, 80:90] = 4  # enclosed alone, 10 x 10, black
    ids[25:28, 30:40] = 5  # black and 3 px wide, but beside region 0: not enclosed alone
    ids[40:50, 100:110] = 6  # black, and touching a pixel of region 1 only at a corner, diagonally: not alone either
    ids[39, 99] = 1
    image = np.full((60, 120, 3), FILL_BGR, dtype=np.uint8)
    image[ids == -1] = image[ids == 4] = image[ids == 5] = image[ids == 6] = BLACK_BGR
    colors = np.array([0, 1, 0, 0, 1, 1, 1], dtype=np.int32)

    settled, inked = regions_module.settle_enclosed(ids, colors, image, BLACK_BGR, 8.0)

    assert (settled[ids == 2] == 2).all()  # kept, 100 px, however small the difficulty's floor
    assert (settled[ids == 3] == -1).all() and not inked[ids == 3].any()  # bare paper: no brush fits
    assert (settled[ids == 4] == -1).all() and inked[ids == 4].all()  # the ink's own color: printed with it
    assert inked.sum() == 100
    assert (settled[ids <= 1] == ids[ids <= 1]).all()  # the regions with a neighbor are left as they are
    assert (settled[ids == 5] == 5).all() and not inked[ids == 5].any()  # however thin or dark
    assert (settled[ids == 6] == 6).all() and not inked[ids == 6].any()
    assert (ids[ids == 3] == 3).all()  # a new map


def test_a_patch_edged_by_the_ink_and_a_fill_alike_joins_the_ink():
    # A black patch 3 x 3 under a band of ink, with a stub of ink along its left side: its ring of 16 pixels is 8 ink
    # (the 5 above, the 3 on the left) and 8 fill (the 4 on the right, the 4 below).
    labels = np.zeros((20, 20), dtype=np.int32)
    image = np.full((20, 20, 3), FILL_BGR, dtype=np.uint8)
    labels[:4, :] = labels[4:7, 7] = 2
    labels[4:7, 8:11] = 1
    image[labels > 0] = BLACK_BGR
    ring = [labels[y, x] for y in range(3, 8) for x in range(7, 12) if not (4 <= y < 7 and 8 <= x < 11)]
    assert ring.count(2) == ring.count(0) == 8  # a tie

    assert (regions_module.join_ink(labels, 2, image, BLACK_BGR, 60)[4:7, 8:11] == 2).all()


def test_the_search_weighs_a_diagonal_step_as_the_longer_one():
    # From (4, 0), seed 1 lies 4 straight steps away (20) and seed 2 three diagonal ones (21): 4 against 4.24 px.
    seeds = np.full((8, 8), -1, dtype=np.int32)
    seeds[0, 0], seeds[1, 3] = 1, 2
    passable = np.ones((8, 8), dtype=bool)

    nearest = kernels.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), 8, 8).reshape(8, 8)

    assert nearest[4, 0] == 1

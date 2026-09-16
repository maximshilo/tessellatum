import cv2
import numpy as np
import pytest

from tessellatum.core.regions import build_regions, extract_regions


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
def test_no_two_neighboring_regions_share_a_color(seed, num_colors, min_area_px):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random((90, 120)).astype(np.float32), (0, 0), 2.0)
    edges = np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])
    labels = np.digitize(field, edges).astype(np.int32)

    region_id_map, region_color = build_regions(labels, num_colors, min_area_px)

    assert _same_color_neighbor_pixels(region_id_map, region_color) == 0
    # Joining regions that touch keeps every region one 8-connected shape.
    for rid in np.unique(region_id_map):
        mask = (region_id_map == rid).astype(np.uint8)
        assert cv2.connectedComponents(mask, connectivity=8)[0] == 2  # background + the region

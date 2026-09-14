import numpy as np

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

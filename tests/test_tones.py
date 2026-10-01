"""Tones: a face's regions in the palette colors nearest them, the faintest steps between them joined (T4.4, D-045)."""

from pathlib import Path

import numpy as np
import pytest

from tessellatum.core import faces, kernels, marks, pipeline, tones
from tessellatum.core.color import bgr_to_lab, ciede2000
from tessellatum.core.difficulty import params_for_preset

SAMPLES = Path(__file__).resolve().parent / "sample_images"

HEIGHT = 20


def gray_bgr(*grays) -> np.ndarray:
    return np.array([(gray,) * 3 for gray in grays], dtype=np.uint8)


def distance(pixel: int, paint: int) -> float:
    """CIEDE2000 between two grays, as the palette's margin is measured."""
    return float(ciede2000(bgr_to_lab(gray_bgr(pixel))[0], bgr_to_lab(gray_bgr(paint))[0]))


def strips(*parts):
    """Regions side by side, left to right, from ``(region id, width, pixel gray)``: the region map and the picture.

    Pixel grays are middles of the color cube's cells (8 k + 4), so the cells read them exactly.
    """
    ids = np.concatenate([np.full((HEIGHT, width), region, dtype=np.int32) for region, width, _gray in parts], axis=1)
    image = np.concatenate(
        [np.full((HEIGHT, width, 3), gray, dtype=np.uint8) for _region, width, gray in parts], axis=1
    )
    return ids, image


def everywhere(ids: np.ndarray) -> np.ndarray:
    return np.ones(ids.shape, dtype=bool)


def over(ids: np.ndarray, *regions) -> np.ndarray:
    return np.isin(ids, regions)


def test_the_step_a_region_is_kept_for():
    assert tones.MIN_STEP_DE00 == 1.0
    # The grays the tests below stand on: two levels from 132 is a fainter step than that, three a clearer one.
    assert distance(132, 134) == pytest.approx(0.731, abs=0.001)
    assert distance(132, 135) == pytest.approx(1.093, abs=0.001)


def test_a_region_takes_the_palette_color_nearest_its_pixels():
    # Region 0 is painted the mid gray, but its pixels are nearly white; its neighbor is dark, and stays apart.
    ids, image = strips((0, 30, 196), (1, 30, 20))
    colors = np.array([0, 1], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, gray_bgr(60, 20, 200))
    assert new_colors.tolist() == [2, 1]
    np.testing.assert_array_equal(new_ids, ids)


def test_a_region_keeps_its_color_when_another_is_no_nearer():
    # 132 lies exactly as far from a second copy of its own color.
    ids, image = strips((0, 30, 132), (1, 30, 20))
    colors = np.array([2, 1], dtype=np.int32)
    _ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, gray_bgr(132, 20, 132))
    assert new_colors.tolist() == [2, 1]


@pytest.mark.parametrize("columns, settled", [(15, False), (16, True)])
def test_only_regions_mostly_inside_are_settled(columns, settled):
    # Region 0 is 30 columns wide: with half of it inside it is left alone, with a column more it is settled.
    ids, image = strips((0, 30, 196), (1, 30, 20))
    colors = np.array([0, 1], dtype=np.int32)
    where = np.zeros(ids.shape, dtype=bool)
    where[:, :columns] = True
    _ids, new_colors = tones.settle_tones(image, where, ids, colors, gray_bgr(60, 20, 200))
    assert new_colors.tolist() == ([2, 1] if settled else [0, 1])


@pytest.mark.parametrize("neighbor_inside", [True, False])
def test_a_region_given_its_neighbor_s_color_joins_it(neighbor_inside):
    # Region 1's pixels are region 0's color: it takes it, and the two are one region under the lower id.
    ids, image = strips((0, 20, 196), (1, 20, 196), (2, 20, 20))
    colors = np.array([2, 0, 1], dtype=np.int32)
    where = everywhere(ids) if neighbor_inside else over(ids, 1)
    new_ids, new_colors = tones.settle_tones(image, where, ids, colors, gray_bgr(60, 20, 200))
    assert new_colors[0] == 2 and new_colors[2] == 1
    assert (new_ids[:, :40] == 0).all() and (new_ids[:, 40:] == 2).all()


@pytest.mark.parametrize("paint, joins", [(134, True), (135, False)])
def test_a_faint_step_joins_a_neighbor_and_a_clear_one_stays(paint, joins):
    # Region 0 is painted what it is, 132. Its neighbor outside is painted 134, 0.73 away, or 135, 1.09 away.
    ids, image = strips((0, 30, 132), (1, 30, 228))
    colors = np.array([0, 1], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 0), ids, colors, gray_bgr(132, paint))
    assert new_colors.tolist() == ([1, 1] if joins else [0, 1])
    assert (new_ids == 0).all() == joins


@pytest.mark.parametrize("lighter, joins", [(103, True), (101, False)])
def test_the_step_is_the_mean_over_a_region_s_pixels(lighter, joins):
    # 2400 pixels at 132 painted 132, beside paint at 135: 1.093 further for each. Some of them at 140 are nearer 135
    # than 132, which brings the mean down: a thousandth under the step with 103 of them, a thousandth over with 101.
    ids, image = strips((0, 120, 132), (1, 30, 228))
    image.reshape(-1, 3)[np.flatnonzero(ids == 0)[:lighter]] = 140
    step = ((2400 - lighter) * distance(132, 135) + lighter * (distance(140, 135) - distance(140, 132))) / 2400
    assert (step < tones.MIN_STEP_DE00) == joins and abs(step - tones.MIN_STEP_DE00) < 0.001
    colors = np.array([0, 1], dtype=np.int32)
    _ids, new_colors = tones.settle_tones(image, over(ids, 0), ids, colors, gray_bgr(132, 135))
    assert new_colors.tolist() == ([1, 1] if joins else [0, 1])


def test_a_region_joins_the_neighbor_it_loses_least_to():
    # 132 between paints of 134 (0.731 away) and 130 (0.742 away), both a faint step: it takes the nearer.
    assert distance(132, 134) < distance(132, 130) < tones.MIN_STEP_DE00
    ids, image = strips((0, 20, 228), (1, 20, 132), (2, 20, 28))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(130, 132, 134)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1), ids, colors, palette)
    assert new_colors.tolist() == [0, 2, 2]
    assert (new_ids[:, :20] == 0).all() and (new_ids[:, 20:] == 1).all()


def test_the_faintest_step_on_the_page_goes_first():
    # Region 1 (132, painted 132) and region 2 (124, painted 130) are a faint step from each other, 0.74 either way.
    # Fainter still is region 1's step to region 0 outside, painted 133 (0.37): it goes first, and region 2, a clear
    # step from 133, is left as it is. In any other order the two would have joined.
    ids, image = strips((0, 20, 228), (1, 20, 132), (2, 20, 124))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(133, 132, 130)
    own = distance(124, 130)
    assert distance(132, 133) < distance(132, 130) < distance(124, 132) - own < tones.MIN_STEP_DE00
    assert distance(124, 133) - own > tones.MIN_STEP_DE00
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1, 2), ids, colors, palette)
    assert new_colors.tolist() == [0, 0, 2]
    assert (new_ids[:, :40] == 0).all() and (new_ids[:, 40:] == 2).all()


def test_regions_that_have_joined_count_as_one():
    # Region 0 (148, painted 151) is a faint step from region 1's paint, 144, and joins it first (0.36). Region 1's own
    # pixels, 124, are a faint step from region 2's paint, 104 (0.65) -- but with region 0's they are far from it.
    ids, image = strips((0, 30, 148), (1, 10, 124), (2, 20, 116))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(151, 144, 104)
    first = distance(148, 144) - distance(148, 151)
    alone = distance(124, 104) - distance(124, 144)
    together = (30 * (distance(148, 104) - distance(148, 144)) + 10 * alone) / 40
    assert first < alone < tones.MIN_STEP_DE00 < together
    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, palette)
    assert new_colors.tolist() == [1, 1, 2]
    assert (new_ids[:, :40] == 0).all() and (new_ids[:, 40:] == 2).all()


def test_joined_regions_are_judged_by_all_their_pixels_and_touch_what_either_touched():
    # Regions 1 and 2 are both 132; region 1, painted dark, takes region 2's paint and joins it. Only region 1 touches
    # region 0 outside, painted 134: the two together are a faint step from it (0.73 a pixel), and join it.
    ids, image = strips((0, 20, 228), (1, 20, 132), (2, 20, 132))
    colors = np.array([0, 1, 2], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1, 2), ids, colors, gray_bgr(134, 60, 132))
    assert new_colors.tolist() == [0, 0, 0]
    assert (new_ids == 0).all()


def test_joined_regions_keep_the_neighbors_of_both():
    # Regions 1 and 2 join as above. Region 0 touches only region 1; it is a faint step from their paint (0.31), and
    # they a clear one from its paint (1.45): it joins them.
    ids, image = strips((0, 8, 132), (0, 12, 140), (1, 20, 132), (2, 20, 132))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(136, 60, 132)
    mine = (8 * distance(132, 136) + 12 * distance(140, 136)) / 20
    theirs = 12 * distance(140, 132) / 20
    assert 0.2 < theirs - mine < 0.4 and distance(132, 136) > tones.MIN_STEP_DE00
    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, palette)
    assert new_colors.tolist() == [2, 2, 2]
    assert (new_ids == 0).all()


def test_a_region_ends_in_the_color_of_the_last_region_it_joined_through():
    # Region 0 joins region 1, as above; the two, painted 130, are a faint step from region 2's paint, 135 (0.35), and
    # join it, region 2 being a clear step from theirs. Region 0 ends in region 2's color too.
    ids, image = strips((0, 20, 132), (1, 20, 132), (2, 20, 140))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(60, 130, 135)
    assert distance(132, 135) - distance(132, 130) < tones.MIN_STEP_DE00 < distance(140, 130) - distance(140, 135)
    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, palette)
    assert new_colors.tolist() == [2, 2, 2]
    assert (new_ids == 0).all()


def test_a_region_joined_to_one_outside_is_a_neighbor_outside():
    # Region 1 (132, painted 132) joins region 0 outside, painted 134 (0.73). Region 2 (140, painted 136) touches only
    # region 1, a clear step from its old paint and a faint one from its new: it joins region 0's color through it.
    ids, image = strips((0, 20, 228), (1, 20, 132), (2, 20, 140))
    colors = np.array([0, 1, 2], dtype=np.int32)
    palette = gray_bgr(134, 132, 136)
    own = distance(140, 136)
    assert distance(140, 134) - own < tones.MIN_STEP_DE00 < distance(140, 132) - own
    assert distance(132, 134) < tones.MIN_STEP_DE00 < distance(132, 136)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1, 2), ids, colors, palette)
    assert new_colors.tolist() == [0, 0, 0]
    assert (new_ids == 0).all()


def test_no_two_neighbors_are_left_sharing_a_color():
    # Region 1 joins region 0 outside; region 2, outside too and the same color, touches region 1 but not region 0.
    ids, image = strips((0, 20, 228), (1, 20, 132), (2, 20, 28))
    colors = np.array([1, 0, 1], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1), ids, colors, gray_bgr(132, 134))
    assert new_colors.tolist() == [1, 1, 1]
    assert (new_ids == 0).all()


def test_neighbors_touch_at_a_corner():
    # Region 1 meets region 0 only at a corner, and joins it.
    ids = np.full((20, 40), 2, dtype=np.int32)
    ids[:10, :20] = 0
    ids[10:, 20:] = 1
    image = np.full((20, 40, 3), 28, dtype=np.uint8)
    image[ids == 0] = 228
    image[ids == 1] = 132
    colors = np.array([1, 0, 2], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, over(ids, 1), ids, colors, gray_bgr(132, 134, 28))
    assert new_colors.tolist() == [1, 1, 2]
    assert (new_ids[ids == 1] == 0).all()


def test_pixels_in_no_region_take_no_part():
    # A band in no region between the two: they are not neighbors, and the band stays as it is.
    ids, image = strips((0, 20, 228), (-1, 3, 132), (1, 20, 132))
    colors = np.array([1, 0], dtype=np.int32)
    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, gray_bgr(132, 134))
    assert new_colors.tolist() == [1, 0]
    np.testing.assert_array_equal(new_ids, ids)


def test_a_pixel_stands_for_the_middle_of_its_cell():
    # 135 is 138's as much as 132's, but it is read as 132, the middle of its cell of the cube: painted 132, it stays.
    ids, image = strips((0, 30, 135), (1, 30, 228))
    colors = np.array([0, 1], dtype=np.int32)
    assert distance(135, 138) - distance(135, 132) < tones.MIN_STEP_DE00 < distance(132, 138)
    _ids, new_colors = tones.settle_tones(image, over(ids, 0), ids, colors, gray_bgr(132, 138))
    assert new_colors.tolist() == [0, 1]


def test_a_pixel_s_color_is_read_channel_by_channel():
    # An orange-blue whose three channels all differ; the palette holds it and the colors it turns into with two
    # channels swapped. The region is painted one of those, and takes its own.
    ids, image = strips((0, 30, 0), (1, 30, 20))
    image[:, :30] = (228, 124, 28)
    palette = np.array([(28, 124, 228), (20, 20, 20), (228, 28, 124), (124, 228, 28), (228, 124, 28)], dtype=np.uint8)
    for painted in (0, 2, 3):
        colors = np.array([painted, 1], dtype=np.int32)
        _ids, new_colors = tones.settle_tones(image, over(ids, 0), ids, colors, palette)
        assert new_colors.tolist() == [4, 1]


def test_nothing_is_settled_without_where_and_what_is_passed_in_is_left_alone():
    ids, image = strips((0, 30, 196), (1, 30, 196))
    colors = np.array([0, 1], dtype=np.int32)
    palette = gray_bgr(60, 200)
    kept_ids, kept_colors = ids.copy(), colors.copy()
    new_ids, new_colors = tones.settle_tones(image, np.zeros(ids.shape, dtype=bool), ids, colors, palette)
    np.testing.assert_array_equal(new_ids, ids)
    assert new_colors.tolist() == [0, 1] and new_ids is not ids and new_colors is not colors

    new_ids, new_colors = tones.settle_tones(image, everywhere(ids), ids, colors, palette)
    assert new_colors.tolist() == [1, 1] and (new_ids == 0).all()
    np.testing.assert_array_equal(ids, kept_ids)
    np.testing.assert_array_equal(colors, kept_colors)


def test_arrays_that_would_send_the_kernels_astray_are_refused():
    # The kernels read and write unchecked: a picture or a mask of another size, a region without a color, or a color
    # off the palette must not reach them.
    ids, image = strips((0, 30, 196), (1, 30, 20))
    colors = np.array([0, 1], dtype=np.int32)
    palette = gray_bgr(60, 20, 200)
    where = everywhere(ids)
    with pytest.raises(ValueError, match="size"):
        tones.settle_tones(image[:, :-1], where, ids, colors, palette)
    with pytest.raises(ValueError, match="size"):
        tones.settle_tones(image[:, :, :2], where, ids, colors, palette)
    with pytest.raises(ValueError, match="size"):
        tones.settle_tones(image, where[:-1], ids, colors, palette)
    with pytest.raises(ValueError, match="region 1 is on the map"):
        tones.settle_tones(image, where, ids, colors[:1], palette)
    with pytest.raises(ValueError, match="palette has 3"):
        tones.settle_tones(image, where, ids, np.array([0, 3], dtype=np.int32), palette)
    with pytest.raises(ValueError, match="palette has 3"):
        tones.settle_tones(image, where, ids, np.array([-1, 1], dtype=np.int32), palette)
    # Refused even where nothing would be settled: the check is of what is handed over, not of what is reached.
    with pytest.raises(ValueError, match="palette has 3"):
        tones.settle_tones(image, np.zeros(ids.shape, dtype=bool), ids, np.array([0, 3], dtype=np.int32), palette)


def test_the_census_counts_pixels_by_cell_and_finds_every_neighbor():
    rng = np.random.default_rng(7)
    height, width = 24, 31
    ids = rng.integers(-1, 6, size=(height // 3, width // 3 + 1)).repeat(3, axis=0).repeat(3, axis=1)[:height, :width]
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    region_color = np.array([3, 1, 0, 2, 1, 3], dtype=np.int32)
    slot = np.array([0, -1, 1, -1, 2, -1], dtype=np.int32)  # regions 0, 2 and 4 are asked about
    cells, held, touching, beside = kernels.tone_census(
        ids.reshape(-1), image.reshape(-1, 3), slot, region_color, width, 0, 0, width, height, 5, 3, 4
    )

    blue, green, red = (image[..., channel].astype(np.int64) >> 3 for channel in range(3))
    code = (blue << 10) | (green << 5) | red
    assert len(set(cells.tolist())) == len(cells)
    for row, region in enumerate((0, 2, 4)):
        assert held[row].sum() == (ids == region).sum()
        for column, cell in enumerate(cells.tolist()):
            assert held[row, column] == ((ids == region) & (code == cell)).sum()
    expected_touching = np.zeros((3, 3), dtype=bool)
    expected_beside = np.zeros((3, 4), dtype=bool)
    for y in range(height):
        for x in range(width):
            for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < height and 0 <= xx < width):
                    continue
                for r, t in ((ids[y, x], ids[yy, xx]), (ids[yy, xx], ids[y, x])):
                    if r < 0 or t < 0 or r == t or slot[r] < 0:
                        continue
                    if slot[t] >= 0:
                        expected_touching[slot[r], slot[t]] = True
                    else:
                        expected_beside[slot[r], region_color[t]] = True
    assert expected_touching.any() and expected_beside.any()
    np.testing.assert_array_equal(touching, expected_touching)
    np.testing.assert_array_equal(beside, expected_beside)


@pytest.mark.parametrize("first, second", [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, 0))])
def test_the_census_finds_a_neighbor_in_every_direction_and_none_in_no_region(first, second):
    # Two pixels of a 2 x 2 map, the rest in no region: side by side, one above the other, and on either diagonal.
    ids = np.full((2, 2), -1, dtype=np.int32)
    ids[first], ids[second] = 0, 1
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    region_color = np.array([0, 1], dtype=np.int32)

    def census(slot):
        slot = np.array(slot, dtype=np.int32)
        return kernels.tone_census(ids.reshape(-1), image.reshape(-1, 3), slot, region_color, 2, 0, 0, 2, 2, 5, 2, 2)

    _cells, held, touching, beside = census([0, 1])  # both asked about: each touches the other, and nothing else
    assert held.tolist() == [[1], [1]]
    assert touching.tolist() == [[False, True], [True, False]] and not beside.any()
    _cells, held, touching, beside = census([0, -1])  # only the first: it is beside the second's color
    assert held.tolist() == [[1], [0]] and not touching.any()
    assert beside.tolist() == [[False, True], [False, False]]
    _cells, held, touching, beside = census([-1, 0])  # only the second: it is beside the first's
    assert held.tolist() == [[1], [0]] and not touching.any()
    assert beside.tolist() == [[True, False], [False, False]]


def test_the_census_reads_only_its_box():
    # Region 0 has region 1 to its right and region 2 below. The box ends above region 2: it is not found.
    ids = np.zeros((6, 8), dtype=np.int32)
    ids[:, 5:] = 1
    ids[4:, :5] = 2
    image = np.full((6, 8, 3), 100, dtype=np.uint8)
    slot = np.array([0, -1, -1], dtype=np.int32)
    region_color = np.array([0, 1, 2], dtype=np.int32)
    flat = ids.reshape(-1), image.reshape(-1, 3), slot, region_color
    cells, held, _touching, beside = kernels.tone_census(*flat, 8, 0, 0, 6, 4, 5, 1, 3)
    assert cells.tolist() == [(12 << 10) | (12 << 5) | 12] and held.tolist() == [[20]]
    assert beside.tolist() == [[False, True, False]]
    _cells, held, _touching, beside = kernels.tone_census(*flat, 8, 0, 0, 5, 5, 5, 1, 3)  # and ends left of region 1
    assert held.tolist() == [[20]] and beside.tolist() == [[False, False, True]]
    _cells, held, _touching, beside = kernels.tone_census(*flat, 8, 1, 1, 6, 5, 5, 1, 3)  # and starts a pixel in
    assert held.tolist() == [[12]] and beside.tolist() == [[False, True, True]]


def test_regions_inside_counts_each_region_s_pixels_in_the_mask():
    ids = np.array([[0, 0, 1, 1], [0, -1, 1, 2], [3, 3, 3, 2]], dtype=np.int32)
    inside = np.array([[1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    bounds, areas, areas_inside = kernels.regions_inside(ids.reshape(-1), inside.reshape(-1), 3, 4, 5)
    expected_bounds, expected_areas = kernels.region_bounds(ids.reshape(-1), 3, 4, 5)
    np.testing.assert_array_equal(bounds, expected_bounds)
    np.testing.assert_array_equal(areas, expected_areas)
    assert areas.tolist() == [3, 3, 2, 3, 0] and areas_inside.tolist() == [2, 2, 1, 1, 0]


def _painting(analysis) -> np.ndarray:
    return analysis.palette_bgr[analysis.region_color[analysis.region_id_map]]


def _mean_distance(image: np.ndarray, painting: np.ndarray, where: np.ndarray) -> float:
    return float(ciede2000(bgr_to_lab(image[where]), bgr_to_lab(painting[where])).mean())


def test_a_photographed_face_is_painted_in_fewer_tones_no_further_from_it(monkeypatch):
    # Q28: inside the faces found, fewer regions, and the painting no further from the picture there.
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-lion.jpg")
    params = params_for_preset("Hard")
    seen, settled, marked = [], [], []
    real, real_marks = tones.settle_tones, marks.detail_marks

    def spy(picture, where, ids, colors, palette):
        seen.append(where)
        settled.append(real(picture, where, ids, colors, palette))
        return settled[-1]

    def marks_spy(picture, where, ids, colors, palette, scale):
        marked.append((ids, colors))
        return real_marks(picture, where, ids, colors, palette, scale)

    monkeypatch.setattr(tones, "settle_tones", spy)
    monkeypatch.setattr(marks, "detail_marks", marks_spy)
    analysis = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    assert len(seen) == 1
    # The faces found, not all the detail: the subject's regions may be smaller too, but only a face's are settled.
    size = analysis.region_id_map.shape[::-1]
    in_faces = faces.mask(analysis.faces, size)
    np.testing.assert_array_equal(seen[0], in_faces)
    assert (analysis.detail & ~in_faces).any()
    # The marks printed in the face are judged against its settled paint (see ``marks``).
    assert len(marked) == 1 and marked[0][0] is settled[0][0] and marked[0][1] is settled[0][1]
    monkeypatch.setattr(marks, "detail_marks", real_marks)

    pipeline.clear_cache()
    monkeypatch.setattr(tones, "settle_tones", lambda picture, where, ids, colors, palette: (ids, colors))
    plain = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    pipeline.clear_cache()

    assert len(analysis.regions) <= len(plain.regions) - 5
    # Every region is one or more of the regions there were: none shrank, none changed shape but by joining.
    pairs = np.unique(np.stack([plain.region_id_map.ravel(), analysis.region_id_map.ravel()]), axis=1)
    assert len(np.unique(pairs[0])) == pairs.shape[1]
    # Outside the faces nothing is repainted but a region that a face's region joined.
    before, after = _painting(plain), _painting(analysis)
    changed = (before != after).any(axis=2)
    assert changed.any()
    ids = plain.region_id_map
    mostly_inside = 2 * np.bincount(ids[in_faces], minlength=ids.max() + 1) > np.bincount(ids.ravel())
    assert not changed[~mostly_inside[ids]].any()
    # The face is no further from the picture, and no two neighbors share a color.
    resized = pipeline.resize_to_long_edge(image, pipeline.PREVIEW_LONG_EDGE)
    assert _mean_distance(resized, after, in_faces) < _mean_distance(resized, before, in_faces)
    new_ids = analysis.region_id_map
    for first, second in (
        (new_ids[:, :-1], new_ids[:, 1:]),
        (new_ids[:-1], new_ids[1:]),
        (new_ids[:-1, :-1], new_ids[1:, 1:]),
        (new_ids[:-1, 1:], new_ids[1:, :-1]),
    ):
        differ = first != second
        assert (analysis.region_color[first[differ]] != analysis.region_color[second[differ]]).all()


def test_tones_are_settled_only_in_the_faces_of_a_picture_drawn_from_its_colors(monkeypatch):
    def refuse(*args):
        raise AssertionError("settled tones")

    monkeypatch.setattr(tones, "settle_tones", refuse)
    for name in ("scene.png", "m-cartoon-bold-lines-girl.png"):  # no faces found; line art, whose face is its own ink
        pipeline.clear_cache()
        image = pipeline.load_image_bgr(SAMPLES / name)
        pipeline.generate(image, params_for_preset("Medium"), pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    pipeline.clear_cache()

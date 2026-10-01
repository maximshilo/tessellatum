"""Texture: the regions' edges settled by a vote of the page around each pixel, held to the picture's own (T5.1, D-046)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import kernels, pipeline, texture
from tessellatum.core.color import MIN_PALETTE_DE00
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.regions import build_regions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_images"

HEIGHT, WIDTH, EDGE = 60, 60, 30
SIGMA = 4.0  # a 12 px brush: the weights reach 12 px, and sum to 10.009 along a row

# Grays by their lightness in OpenCV's 8-bit Lab, where a step is 100/255 of an L*: 100 -> 108, 139 -> 148, 140 -> 149.
DARK, LIGHT = 100, 140
# 119 -> 128, midway between 100 and 139: a picture that suits both paints alike, so the share alone decides.
EVEN_PAINTS, EVEN = (100, 139), 119


def gray_bgr(*grays) -> np.ndarray:
    return np.array([(gray,) * 3 for gray in grays], dtype=np.uint8)


def halves(edge: int = EDGE) -> np.ndarray:
    """Color 0 left of column ``edge``, color 1 from it on."""
    labels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    labels[:, edge:] = 1
    return labels


def flat(gray: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), gray, dtype=np.uint8)


def shown(labels: np.ndarray, *grays) -> np.ndarray:
    """A picture in which every pixel is the gray given for its color."""
    return gray_bgr(*grays)[labels]


def vote(picture: np.ndarray, labels: np.ndarray, paints=EVEN_PAINTS, sigma: float = SIGMA) -> np.ndarray:
    return texture.settle_edges(picture, labels, gray_bgr(*paints), sigma)


def test_the_vote_reaches_a_third_of_the_brush_and_weighs_a_palette_step():
    assert texture.REACH == pytest.approx(1 / 3)
    assert texture.COLOR_STEP == 10.0 == MIN_PALETTE_DE00
    # The lightnesses the tests below stand on.
    lab = cv2.cvtColor(gray_bgr(100, 119, 122, 123, 127, 128, 139, 140).reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    assert lab[:, 0].tolist() == [108, 128, 131, 132, 136, 137, 148, 149]
    assert (lab[:, 1:] == 128).all()


def test_run_ends_gives_each_pixel_the_end_of_its_run():
    # First: every vote below walks its rows by these, and would never end on a run that ends where it starts.
    labels = np.array([[3, 3, 3, 1, 1, 3], [0, 0, 0, 0, 0, 0], [5, 4, 4, -1, -1, 2]], dtype=np.int32)
    ends = np.empty(18, dtype=np.int32)
    kernels.run_ends(labels.reshape(-1), 3, 6, ends)
    assert ends.reshape(3, 6).tolist() == [[3, 3, 3, 5, 5, 6], [6, 6, 6, 6, 6, 6], [1, 3, 3, 5, 5, 6]]


def test_a_straight_edge_stays_where_it_is_out_to_the_page_s_edge():
    labels = halves()
    np.testing.assert_array_equal(vote(flat(EVEN), labels), labels)
    # Along rows as along columns.
    np.testing.assert_array_equal(vote(flat(EVEN), labels.T.copy()), labels.T)


def test_a_bump_and_a_notch_are_flattened_into_bulges_a_brush_can_follow():
    labels = halves()
    labels[18:23, 26:30] = 1  # a bump of color 1 into color 0, 5 rows by 4 columns
    labels[38:43, 30:34] = 0  # and a notch the other way
    voted = vote(flat(EVEN), labels)
    # Half as deep and more than twice as long, with about the pixels it had: 18 of 20.
    bump = (voted[:, :EDGE] == 1).sum(axis=1)  # how far color 1 reaches past the edge, row by row
    notch = (voted[:, EDGE:] == 0).sum(axis=1)
    assert bump[12:29].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    assert notch[32:49].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    assert bump.sum() == notch.sum() == 18
    # Each is one run from the edge, and the rest of the edge is where it was.
    assert (voted[:, :EDGE - 2] == 0).all() and (voted[:, EDGE + 2:] == 1).all()
    assert (voted[14:27, EDGE:] == 1).all() and (voted[34:47, :EDGE] == 0).all()
    # A third less of the page is out of a 12 px brush's reach than before.
    assert bm.sliver_share(voted, 12.0) < 0.7 * bm.sliver_share(labels, 12.0)


def test_a_corner_is_rounded_and_a_side_is_not():
    labels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    labels[10:50, 10:50] = 1
    voted = vote(flat(EVEN), labels)
    assert voted[10, 10] == 0 and voted[49, 49] == 0 and voted[10, 49] == 0 and voted[49, 10] == 0
    assert (voted[10, 20:40] == 1).all() and (voted[20:40, 10] == 1).all()  # the sides keep their outermost pixels
    assert (voted[9, :] == 0).all() and (voted[:, 9] == 0).all()  # and nothing grows past them
    assert (voted != labels).sum() == 4 * 12  # twelve pixels a corner
    np.testing.assert_array_equal(voted, voted.T)
    np.testing.assert_array_equal(voted, voted[::-1, ::-1])


def test_a_speck_is_voted_away_and_a_patch_the_brush_fits_in_keeps_its_middle():
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    for radius, stays in ((3, False), (12, True)):
        labels = ((rows - 30) ** 2 + (columns - 30) ** 2 <= radius**2).astype(np.int32)
        voted = vote(flat(EVEN), labels)
        assert (voted[30, 30] == 1) == stays
        assert voted.any() == stays
        assert not (voted & ~labels.astype(bool)).any()  # a round patch only ever shrinks


def test_the_picture_s_own_edges_hold():
    # The same bump, notch and corners, on a picture that shows them: every pixel is the gray of its own paint.
    labels = halves()
    labels[18:23, 26:30] = 1
    labels[38:43, 30:34] = 0
    labels[0:6, 0:6] = 1
    picture = shown(labels, 0, 255)
    np.testing.assert_array_equal(vote(picture, labels, paints=(0, 255)), labels)


@pytest.mark.parametrize("turned", [False, True])
@pytest.mark.parametrize("picture_edge, settled_edge", [(34, 34), (27, 27), (42, 42), (46, 42), (12, 18)])
def test_an_edge_moves_onto_the_picture_s_as_far_as_the_vote_reaches(picture_edge, settled_edge, turned):
    # The regions' edge is at column 30, the picture's a few columns off. The pixels between take the color the picture
    # shows them in, if it is painted within 12 columns of them: three sigmas. Across rows as across columns: an edge
    # along a row is one where only rows differ.
    turn = (lambda array: np.ascontiguousarray(np.swapaxes(array, 0, 1))) if turned else (lambda array: array)
    picture = turn(shown(halves(picture_edge), 0, 255))
    np.testing.assert_array_equal(vote(picture, turn(halves()), paints=(0, 255)), turn(halves(settled_edge)))


@pytest.mark.parametrize("gray, columns_taken", [(122, 0), (123, 1), (127, 1), (128, 2)])
def test_how_much_nearer_a_pixel_must_be_to_the_other_color(gray, columns_taken):
    # Beside a straight edge a pixel's own color holds 1.222 times the other's share (the weights of 13 columns of 25
    # against 12: (10.009 + 1) / (10.009 - 1)), and one column further in, 1.831 times. It goes over when the other
    # color is nearer to it by more than 10 ln 1.222 = 2.005, or 10 ln 1.831 = 6.050, in L*a*b*. Between paints of
    # lightness 108 and 149 (of 255), a pixel of 131 is 1.96 nearer the lighter, of 132 2.75, of 136 5.88, of 137 6.67.
    weights = np.exp(-0.5 * (np.arange(-12, 13) / SIGMA) ** 2)
    assert weights.sum() == pytest.approx(10.009, abs=0.001)
    assert 10 * np.log((weights.sum() + 1) / (weights.sum() - 1)) == pytest.approx(2.005, abs=0.001)
    assert 10 * np.log(weights[:14].sum() / weights[14:].sum()) == pytest.approx(6.050, abs=0.001)
    voted = vote(flat(gray), halves(), paints=(DARK, LIGHT))
    np.testing.assert_array_equal(voted, halves(EDGE - columns_taken))


def test_the_distance_is_in_all_three_of_l_a_and_b():
    # Two paints of one lightness, a red and a green: a reddish pixel beside the edge goes to the red.
    paints = np.array([(60, 170, 60), (60, 60, 230)], dtype=np.uint8)  # BGR: green, red
    labels = halves()
    picture = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    picture[:] = (60, 110, 190)  # nearer the red, which is color 1
    voted = texture.settle_edges(picture, labels, paints, SIGMA)
    assert (voted[:, EDGE - 1] == 1).all() and (voted[:, EDGE:] == 1).all()
    picture[:] = (60, 150, 100)  # nearer the green
    voted = texture.settle_edges(picture, labels, paints, SIGMA)
    assert (voted[:, EDGE] == 0).all() and (voted[:, : EDGE] == 0).all()


def test_a_pixel_takes_only_a_color_painted_within_reach_of_it():
    # Black, gray and white side by side, from columns 0, 30 and 50, on a picture that is white all over. White is
    # every pixel's own color, but a pixel can only take it where it is painted within 12 columns: the black ones never
    # do, and take the gray, the nearer of the two paints in their reach, where enough of it is.
    labels = halves()
    labels[:, 50:] = 2
    voted = vote(flat(255), labels, paints=(0, 128, 255))
    assert (voted[:, :18] == 0).all()  # out of the gray's reach
    assert (voted[:, :38] != 2).all()  # out of the white's
    expected = halves(20)
    expected[:, 41:] = 2
    np.testing.assert_array_equal(voted, expected)


def test_pixels_without_a_color_keep_none_and_count_for_nobody():
    labels = halves()
    labels[:, 28:32] = -1
    labels[18:23, 24:28] = 1  # a bump, across the gap from its own color
    voted = vote(flat(EVEN), labels)
    assert (voted[:, 28:32] == -1).all()
    expected = halves()
    expected[:, 28:32] = -1
    np.testing.assert_array_equal(voted, expected)


def test_what_is_passed_in_is_left_alone_and_a_new_map_comes_back():
    labels = halves()
    labels[18:23, 26:30] = 1
    picture = flat(EVEN)
    kept_labels, kept_picture = labels.copy(), picture.copy()
    voted = vote(picture, labels)
    assert voted is not labels and voted.dtype == np.int32
    np.testing.assert_array_equal(labels, kept_labels)
    np.testing.assert_array_equal(picture, kept_picture)
    one_color = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    again = vote(picture, one_color)
    assert again is not one_color
    np.testing.assert_array_equal(again, one_color)


def slow_vote(picture: np.ndarray, labels: np.ndarray, paints: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """The vote written out pixel by pixel from its description: the map, and how clearly each pixel's best color won."""
    height, width = labels.shape
    radius = int(np.ceil(3 * sigma))
    lab = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB).astype(np.float64)
    paint_lab = cv2.cvtColor(paints.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    lab[..., 0] *= 100 / 255
    paint_lab[:, 0] *= 100 / 255
    out = labels.copy()
    margin = np.full(labels.shape, np.inf)
    for y in range(height):
        for x in range(width):
            if labels[y, x] < 0:
                continue
            scores = np.zeros(len(paints))
            for yy in range(max(y - radius, 0), min(y + radius + 1, height)):
                for xx in range(max(x - radius, 0), min(x + radius + 1, width)):
                    if labels[yy, xx] >= 0:
                        scores[labels[yy, xx]] += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))
            scores *= np.exp(-np.linalg.norm(lab[y, x] - paint_lab, axis=1) / texture.COLOR_STEP)
            order = np.argsort(-scores, kind="stable")
            out[y, x] = order[0]
            margin[y, x] = (scores[order[0]] - scores[order[1]]) / scores[order[0]]
    return out, margin


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_vote_is_the_one_its_description_gives_pixel_by_pixel(seed):
    rng = np.random.default_rng(seed)
    height, width, sigma = 26, 33, 1.7
    paints = rng.integers(0, 256, size=(4, 3)).astype(np.uint8)
    blocks = rng.integers(0, 4, size=(height // 3 + 1, width // 3 + 1))
    labels = np.kron(blocks, np.ones((3, 3), dtype=np.int64))[:height, :width].astype(np.int32)
    labels[rng.random(labels.shape) < 0.1] = rng.integers(0, 4)  # specks
    labels[5:8, 20:23] = -1
    picture = paints[np.maximum(labels, 0)].astype(np.int16) + rng.integers(-60, 61, size=(height, width, 3))
    picture = np.clip(picture, 0, 255).astype(np.uint8)
    expected, margin = slow_vote(picture, labels, paints, sigma)
    voted = texture.settle_edges(picture, labels, paints, sigma)
    clear = margin > 1e-9  # elsewhere the two add the same weights up in another order
    assert clear.mean() > 0.99
    np.testing.assert_array_equal(voted[clear], expected[clear])
    assert (voted != labels).mean() > 0.02  # and it is a vote that changes something


def _kernel_vote(labels, pixels_lab, colors_lab, cumulative, row_weight, radius, near=None):
    height, width = labels.shape
    flat_labels = np.ascontiguousarray(labels, dtype=np.int32).reshape(-1)
    ends = np.empty(flat_labels.size, dtype=np.int32)
    kernels.run_ends(flat_labels, height, width, ends)
    out = np.empty(flat_labels.size, dtype=np.int32)
    near = np.ones(flat_labels.size, dtype=bool) if near is None else near.reshape(-1)
    kernels.vote_rows(
        flat_labels, ends, near, pixels_lab.reshape(-1, 3), colors_lab, np.asarray(cumulative, dtype=np.float64),
        np.asarray(row_weight, dtype=np.float64), radius, 10.0, 0, height, height, width, out,
    )
    return out.reshape(height, width)


@pytest.mark.parametrize("row, winner", [([1, 2, 2], 2), ([0, 2, 2], 2), ([1, 2, 0], 0), ([0, 2, 1], 0), ([1, 2, 1], 1)])
def test_on_a_tie_a_pixel_keeps_its_color_and_between_two_others_takes_the_lowest(row, winner):
    # One row, the middle pixel voting, with weights that leave its own column out: its neighbors' colors hold one
    # share each. All three paints are the pixel's own color, so the shares alone decide.
    labels = np.array([row], dtype=np.int32)
    lab = np.full((1, 3, 3), 128, dtype=np.uint8)
    colors = np.full((3, 3), 128.0)
    near = np.array([[False, True, False]])
    voted = _kernel_vote(labels, lab, colors, cumulative=[0.0, 1.0, 1.0, 2.0], row_weight=[0.0, 1.0, 0.0], radius=1, near=near)
    assert voted[0].tolist() == [row[0], winner, row[2]]


def test_only_pixels_near_vote_and_each_votes_on_the_page_as_it_was():
    # A row of 0 0 1 1 with no weight on a pixel's own column: every pixel would take its neighbors' color, and does
    # where it is asked to, from the row as it was and not as the pixel before it left it.
    labels = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32)
    lab = np.full((1, 6, 3), 128, dtype=np.uint8)
    colors = np.full((2, 3), 128.0)
    weights = dict(cumulative=[0.0, 1.0, 1.0, 2.0], row_weight=[0.0, 1.0, 0.0], radius=1)
    assert _kernel_vote(labels, lab, colors, **weights)[0].tolist() == [1, 0, 1, 0, 1, 0]
    near = np.array([[False, False, True, True, False, False]])
    assert _kernel_vote(labels, lab, colors, near=near, **weights)[0].tolist() == [0, 1, 1, 0, 0, 1]


def test_the_rows_can_be_voted_in_stripes(monkeypatch):
    rng = np.random.default_rng(3)
    labels = np.kron(rng.integers(0, 3, size=(30, 20)), np.ones((4, 4), dtype=np.int64)).astype(np.int32)
    picture = rng.integers(0, 256, size=labels.shape + (3,)).astype(np.uint8)
    paints = gray_bgr(30, 120, 220)
    monkeypatch.setenv("TESSELLATUM_THREADS", "1")
    alone = texture.settle_edges(picture, labels, paints, 2.5)
    monkeypatch.setenv("TESSELLATUM_THREADS", "4")
    np.testing.assert_array_equal(texture.settle_edges(picture, labels, paints, 2.5), alone)
    assert (alone != labels).any()


def test_a_page_that_the_vote_leaves_alone_comes_back_as_it_was():
    ids, colors = build_regions(halves(), 2, 50, 12.0)
    new_ids, new_colors = texture.smooth_regions(flat(EVEN), ids, colors, gray_bgr(*EVEN_PAINTS), 50, 12.0)
    assert new_ids is ids and new_colors is colors


def test_the_regions_are_rebuilt_from_the_vote(monkeypatch):
    labels = halves()
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    labels[(rows - 45) ** 2 + (columns - 12) ** 2 <= 5**2] = 2  # a round patch, a region of its own, which the vote shrinks
    ids, colors = build_regions(labels, 3, 60, 0.0)
    assert colors.tolist() == [0, 1, 2] and (ids == 2).sum() == 81
    kept_ids, kept_colors = ids.copy(), colors.copy()
    calls = []
    real = texture.settle_edges

    def spy(picture, voted_labels, palette, sigma):
        calls.append((voted_labels.copy(), sigma))
        return real(picture, voted_labels, palette, sigma)

    monkeypatch.setattr(texture, "settle_edges", spy)
    picture = flat(EVEN)
    paints = gray_bgr(100, 139, 100)
    new_ids, new_colors = texture.smooth_regions(picture, ids, colors, paints, 60, 12.0)
    # The vote is of the regions' colors, a third of the brush to a sigma.
    assert len(calls) == 1 and calls[0][1] == pytest.approx(4.0)
    np.testing.assert_array_equal(calls[0][0], labels)
    # The patch is left 21 of its 81 pixels, under the 60 a region needs, and the region stage merges it into the
    # region around it.
    assert (real(picture, labels, paints, 4.0) == 2).sum() == 21
    assert new_colors.tolist() == [0, 1]
    np.testing.assert_array_equal(new_ids, halves())
    np.testing.assert_array_equal(ids, kept_ids)
    np.testing.assert_array_equal(colors, kept_colors)


def test_the_rebuild_is_the_region_stage_with_the_limits_the_regions_were_built_with(monkeypatch):
    labels = halves()
    labels[18:23, 26:30] = 1
    ids, colors = build_regions(labels, 2, 50, 12.0)
    detail = np.zeros(labels.shape, dtype=bool)
    detail[:20] = True
    calls = []

    def spy(*args):
        calls.append(args)
        return build_regions(*args)

    monkeypatch.setattr(texture, "build_regions", spy)
    paints = gray_bgr(*EVEN_PAINTS)
    new_ids, new_colors = texture.smooth_regions(flat(EVEN), ids, colors, paints, 50, 12.0, detail)
    assert len(calls) == 1
    voted, num_colors, min_area_px, min_width_px, where = calls[0]
    np.testing.assert_array_equal(voted, texture.settle_edges(flat(EVEN), colors[ids], paints, 4.0))
    assert (num_colors, min_area_px, min_width_px) == (2, 50, 12.0) and where is detail
    expected_ids, expected_colors = build_regions(voted, 2, 50, 12.0, detail)
    np.testing.assert_array_equal(new_ids, expected_ids)
    np.testing.assert_array_equal(new_colors, expected_colors)


def test_a_wider_brush_flattens_a_bump_further():
    # A bump 8 columns deep and 9 rows high. A 12 px brush's vote leaves it 6 deep over 19 rows, a 24 px brush's 3
    # deep over 33.
    labels = halves()
    labels[26:35, 22:30] = 1
    ids, colors = build_regions(labels, 2, 20, 0.0)
    paints = gray_bgr(*EVEN_PAINTS)
    shapes = []
    for brush in (12.0, 24.0):
        new_ids, new_colors = texture.smooth_regions(flat(EVEN), ids, colors, paints, 20, brush)
        bump = new_colors[new_ids][:, :EDGE] == 1
        shapes.append((int(bump.sum(axis=1).max()), int(bump.any(axis=1).sum())))
    assert shapes == [(6, 19), (3, 33)]


def test_pixels_in_no_region_stay_in_none():
    labels = halves()
    labels[18:23, 26:30] = 1
    labels[:, 40:44] = 2  # off a palette of two: no region
    ids, colors = build_regions(labels, 2, 20, 0.0)
    assert (ids[:, 40:44] == -1).all()
    new_ids, new_colors = texture.smooth_regions(flat(EVEN), ids, colors, gray_bgr(*EVEN_PAINTS), 20, 12.0)
    assert (new_ids[:, 40:44] == -1).all() and (new_ids[:, :40] >= 0).all() and (new_ids[:, 44:] >= 0).all()
    assert (new_colors[new_ids[:, :30]] == 0).all()  # the bump is gone


def test_nothing_is_voted_without_a_brush_or_without_regions():
    ids, colors = build_regions(halves(), 2, 20, 0.0)
    paints = gray_bgr(*EVEN_PAINTS)
    assert texture.smooth_regions(flat(EVEN), ids, colors, paints, 20, 0.0)[0] is ids
    empty = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    none = np.zeros(0, dtype=np.int32)
    new_ids, new_colors = texture.smooth_regions(flat(EVEN), empty, none, paints, 20, 12.0)
    assert new_ids is empty and new_colors is none


def test_arrays_that_would_send_the_kernel_astray_are_refused():
    # The kernel reads unchecked: a picture of another size, a region without a color, or a color off the palette must
    # not reach it.
    ids, colors = build_regions(halves(), 2, 20, 0.0)
    picture, paints = flat(EVEN), gray_bgr(*EVEN_PAINTS)
    with pytest.raises(ValueError, match="size"):
        texture.smooth_regions(picture[:, :-1], ids, colors, paints, 20, 12.0)
    with pytest.raises(ValueError, match="size"):
        texture.smooth_regions(picture[:, :, :2], ids, colors, paints, 20, 12.0)
    with pytest.raises(ValueError, match="region 1 is on the map"):
        texture.smooth_regions(picture, ids, colors[:1], paints, 20, 12.0)
    with pytest.raises(ValueError, match="palette has 2"):
        texture.smooth_regions(picture, ids, np.array([0, 2], dtype=np.int32), paints, 20, 12.0)
    with pytest.raises(ValueError, match="palette has 2"):
        texture.smooth_regions(picture, ids, np.array([-1, 1], dtype=np.int32), paints, 20, 12.0)


def test_the_vote_refuses_what_would_send_its_kernel_astray():
    # settle_edges is called on its own too: a picture of another size, a label past the palette or a sigma of 0 must
    # not reach the kernel, which reads where they send it.
    labels = halves()
    paints = gray_bgr(*EVEN_PAINTS)
    with pytest.raises(ValueError, match="size"):
        texture.settle_edges(flat(EVEN)[:, :-1], labels, paints, SIGMA)
    with pytest.raises(ValueError, match="size"):
        texture.settle_edges(flat(EVEN)[:, :, :2], labels, paints, SIGMA)
    with pytest.raises(ValueError, match="size"):
        texture.settle_edges(flat(EVEN)[0], labels[0], paints, SIGMA)
    with pytest.raises(ValueError, match="palette has 2"):
        texture.settle_edges(flat(EVEN), np.where(labels == 1, 2, 0).astype(np.int32), paints, SIGMA)
    for sigma in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma"):
            texture.settle_edges(flat(EVEN), labels, paints, sigma)
    # Pixels without a color are allowed, and a picture of floats is read as 8-bit.
    labels[:, :3] = -1
    expected = vote(flat(EVEN), labels)
    np.testing.assert_array_equal(texture.settle_edges(flat(EVEN).astype(np.float32), labels, paints, SIGMA), expected)


def test_warming_up_compiles_the_vote_as_the_pipeline_calls_it():
    kernels.warm_up()
    compiled = len(kernels.run_ends.signatures), len(kernels.vote_rows.signatures)
    labels = halves()
    labels[18:23, 26:30] = 1
    vote(flat(EVEN), labels)
    assert (len(kernels.run_ends.signatures), len(kernels.vote_rows.signatures)) == compiled == (1, 1)


def _painting(analysis) -> np.ndarray:
    return analysis.palette_bgr[analysis.region_color[analysis.region_id_map]]


def _mean_distance(picture: np.ndarray, painting: np.ndarray) -> float:
    return float(bm.ciede2000(bm.bgr_to_lab(picture), bm.bgr_to_lab(painting)).mean())


def test_a_photograph_s_edges_are_settled_and_its_page_is_easier_to_paint_and_no_further_from_it(monkeypatch):
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-lion.jpg")
    params = params_for_preset("Hard")
    calls, results = [], []
    real = pipeline.smooth_regions

    def spy(*args):
        calls.append(args)
        results.append(real(*args))
        return results[-1]

    monkeypatch.setattr(pipeline, "smooth_regions", spy)
    analysis = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    assert len(calls) == 1
    picture, ids, colors, palette, min_area_px, min_width_px, detail = calls[0]
    resized = pipeline.resize_to_long_edge(image, pipeline.PREVIEW_LONG_EDGE)
    np.testing.assert_array_equal(picture, resized)  # the picture itself, not the smoothed one the colors came from
    assert min_area_px == analysis.min_region_area_px and min_width_px == analysis.min_paintable_width_px
    np.testing.assert_array_equal(detail, analysis.detail)  # the face keeps its smaller regions through the rebuild
    assert len(palette) >= analysis.legend_size

    pipeline.clear_cache()
    monkeypatch.setattr(pipeline, "smooth_regions", lambda picture, ids, colors, *rest: (ids, colors))
    plain = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    pipeline.clear_cache()

    # A third less of the page is out of a 3 mm brush's reach, the regions are rounder, and the painting is nearer.
    width = analysis.min_paintable_width_px
    assert bm.sliver_share(analysis.region_id_map, width) < 0.75 * bm.sliver_share(plain.region_id_map, width)
    assert bm.compactness_stats(analysis.region_id_map)["compactness_median"] > bm.compactness_stats(plain.region_id_map)["compactness_median"]
    assert _mean_distance(resized, _painting(analysis)) < 0.99 * _mean_distance(resized, _painting(plain))
    # The region stage's promises hold after it: no region too small, no two neighbors of one color.
    assert bm.count_undersized(analysis.region_id_map, analysis.min_region_area_px, analysis.detail) == 0
    new_ids = analysis.region_id_map
    for first, second in (
        (new_ids[:, :-1], new_ids[:, 1:]),
        (new_ids[:-1], new_ids[1:]),
        (new_ids[:-1, :-1], new_ids[1:, 1:]),
        (new_ids[:-1, 1:], new_ids[1:, :-1]),
    ):
        differ = first != second
        assert (analysis.region_color[first[differ]] != analysis.region_color[second[differ]]).all()


def test_the_tones_and_the_marks_of_a_face_are_settled_on_the_smoothed_regions(monkeypatch):
    from tessellatum.core import tones

    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-lion.jpg")
    smoothed, toned = [], []
    real, real_tones = pipeline.smooth_regions, tones.settle_tones

    def spy(*args):
        smoothed.append(real(*args))
        return smoothed[-1]

    def tones_spy(picture, where, ids, colors, palette):
        toned.append((ids, colors))
        return real_tones(picture, where, ids, colors, palette)

    monkeypatch.setattr(pipeline, "smooth_regions", spy)
    monkeypatch.setattr(tones, "settle_tones", tones_spy)
    pipeline.generate(image, params_for_preset("Medium"), pipeline.PREVIEW_LONG_EDGE)
    assert len(smoothed) == 1 and len(toned) == 1
    assert toned[0][0] is smoothed[0][0] and toned[0][1] is smoothed[0][1]
    pipeline.clear_cache()


def test_line_art_s_edges_are_its_ink_and_are_not_voted_on(monkeypatch):
    def refuse(*args):
        raise AssertionError("smoothed line art")

    monkeypatch.setattr(pipeline, "smooth_regions", refuse)
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "m-cartoon-bold-lines-girl.png")
    pipeline.generate(image, params_for_preset("Medium"), pipeline.PREVIEW_LONG_EDGE)
    pipeline.clear_cache()

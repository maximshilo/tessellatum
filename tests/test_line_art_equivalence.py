"""T3.5: the line-art region steps, compiled, must give exactly what v0.1.31's did (``reference_line_art``).

Each step is run by both on the same input -- the reference's own output of the step before -- so that a difference
points at the step that made it. The inputs are real line-art pages, and constructed ones: blobs of color crossed by
thin lines, bold ink, hatching and one-pixel diagonal lines, small enough to try many.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

import reference_line_art as ref
from tessellatum.core import difficulty, ink, kernels, pipeline, regions
from tessellatum.core.print_size import print_scale
from tessellatum.core.quantize import quantize
from tessellatum.core.render import PageStyle

SAMPLE_IMAGES = Path(__file__).resolve().parent / "sample_images"


def _same(actual, expected, what):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected), what
        for i, (a, e) in enumerate(zip(actual, expected)):
            _same(a, e, f"{what}[{i}]")
    elif expected is None:
        assert actual is None, what
    else:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected), err_msg=what)
        assert np.asarray(actual).dtype == np.asarray(expected).dtype, what


def _both(step, *args):
    """``step`` run by both implementations on the same arguments; asserts they agree, and returns the reference's."""
    expected = getattr(ref, step)(*[a.copy() if isinstance(a, np.ndarray) else a for a in args])
    actual = getattr(regions, step)(*[a.copy() if isinstance(a, np.ndarray) else a for a in args])
    _same(actual, expected, step)
    return expected


def _run_steps(labels, palette_bgr, image_bgr, own, ink_gray, min_area_px, min_width_px, thin_px, line_width_px):
    """The pipeline's line-art region stage, each step checked; returns what the steps did, to check they did something."""
    k = len(palette_bgr)
    ink_bgr = (ink_gray,) * 3
    labels = _both("join_ink", labels, k, image_bgr, ink_bgr, min_area_px, own)
    region_labels, corners = _both("look_through_hatching", labels, k, thin_px, min_width_px)
    printed = (labels >= k) | corners
    ids, colors = _both("build_regions", region_labels, k, min_area_px, min_width_px)
    ids, inked = _both("settle_enclosed", ids, colors, image_bgr, ink_bgr, min_width_px, own)
    printed = printed | inked
    ids = _both("paint_over_thin_ink", ids, colors, printed, thin_px, image_bgr, palette_bgr, own)
    before_pockets = ids
    ids = _both("leave_pockets", ids, colors, printed, min_width_px)
    split = ref._split_all(ids, colors, printed, min_width_px / 2)[1].size - colors.size  # white areas it takes apart
    ids, colors, corners = _both("split_areas", ids, colors, printed, min_width_px, min_area_px, k)
    printed = printed | corners
    _both("detail_ink", ids, printed, thin_px / 2, line_width_px)
    present = np.flatnonzero(np.bincount(ids[ids >= 0], minlength=colors.size))
    smallest = present[np.argsort(np.bincount(ids[ids >= 0], minlength=colors.size)[present], kind="stable")][:4]
    _both("merge_cramped", ids, colors, printed, smallest.tolist(), min_width_px)
    return {
        "looked through": int(((labels >= k) & (region_labels < k)).sum()),
        "corners": int(corners.sum()),
        "pockets": int(((before_pockets >= 0) & (ids < 0)).sum()),
        "split": int(split),
    }


# --- real pages -----------------------------------------------------------------------------------------------------


def _page_inputs(name: str, preset: str, long_edge: int):
    """What the pipeline's line-art region stage starts from on a sample image (see ``pipeline.generate``)."""
    image = pipeline.load_image_bgr(SAMPLE_IMAGES / name)
    params = difficulty.finest_params() if preset == "Max" else difficulty.params_for_preset(preset)
    resized = pipeline.resize_to_long_edge(image, long_edge)
    h, w = resized.shape[:2]
    pipeline.clear_cache()
    line_art, ink_lines = pipeline.detect_ink(image, resized, long_edge)
    assert line_art.is_line_art and ink_lines.any()
    halo_px = max(1.0, print_scale((w, h)).mm_to_px(ink.HALO_MM))
    labels, palette_bgr = quantize(resized, params.num_colors, params.blur_sigma, ink=ink_lines, halo_px=halo_px)
    min_area_px, min_width_px = pipeline._paintable_limits(params, (w, h))
    ink_gray = ink.ink_gray(resized, labels >= len(palette_bgr))
    own = ~ink.near(ink_lines, halo_px)
    thin_px = print_scale((w, h)).mm_to_px(ink.THIN_INK_MM)
    line_width_px = PageStyle().line_width_px((w, h))
    return labels, palette_bgr, resized, own, ink_gray, min_area_px, min_width_px, thin_px, line_width_px


@pytest.mark.parametrize(
    "name, preset, long_edge",
    [
        ("m-cartoon-bold-lines-girl.png", "Hard", 700),
        ("m-cartoon-bold-lines-reaper.png", "Easy", 900),
        ("m-cartoon-complex.jpg", "Max", 1100),
        ("m-cartoon-complex.jpg", "Medium", 650),
        ("m-comics-upside-downs-writing-pig.jpg", "Easy", 1100),
        ("m-comics-upside-downs-writing-pig.jpg", "Max", 750),
    ],
)
def test_every_line_art_step_matches_v0_1_31_on_a_real_page(name, preset, long_edge):
    did = _run_steps(*_page_inputs(name, preset, long_edge))

    assert did["looked through"] and did["split"]  # the pages exercise what the steps are for


# --- constructed pages ----------------------------------------------------------------------------------------------


def _constructed(seed: int, shape: tuple[int, int], num_colors: int):
    """Blobs of color with line art drawn over them: thin and bold strokes, hatching, one-pixel diagonal lines."""
    rng = np.random.default_rng(seed)
    h, w = shape
    field = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), max(2.0, min(shape) / 12))
    edges = np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])
    labels = np.digitize(field, edges).astype(np.int32)
    drawn = np.zeros(shape, dtype=np.uint8)
    for _ in range(int(rng.integers(3, 9))):  # lines 1-3 px wide
        (x0, x1), (y0, y1) = rng.integers(0, w, 2), rng.integers(0, h, 2)
        cv2.line(drawn, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(rng.integers(1, 4)))
    for _ in range(int(rng.integers(0, 4))):  # bold ink
        cv2.circle(drawn, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 9)), 1, -1)
    for _ in range(int(rng.integers(1, 4))):  # a hatched patch, strokes 1-2 px wide and 2-5 px apart
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        size, step, width = int(rng.integers(10, 40)), int(rng.integers(3, 7)), int(rng.integers(1, 3))
        for offset in range(-size, size, step):
            cv2.line(drawn, (x0 + offset, y0), (x0 + offset + size, y0 + size), 1, width)
    for _ in range(int(rng.integers(0, 3))):  # one pixel wide, diagonal
        x0, y0, n = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(10, 60))
        steps = np.arange(n)
        ys, xs = y0 + steps, x0 + steps * (1 if rng.random() < 0.5 else -1)
        keep = (ys < h) & (xs >= 0) & (xs < w)
        drawn[ys[keep], xs[keep]] = 1
    ink_mask = drawn.astype(bool)
    labels[ink_mask] = num_colors
    palette_bgr = rng.integers(40, 256, (num_colors, 3)).astype(np.uint8)
    image = np.where(ink_mask[..., None], 20, palette_bgr[np.minimum(labels, num_colors - 1)]).astype(np.int16)
    image = np.clip(image + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)
    own = ~ink.near(ink_mask, 1.0)
    return labels, palette_bgr, image, own


CONSTRUCTED = [
    # seed, shape, colors, min_area_px, min_width_px, thin_px
    (0, (90, 120), 4, 60, 6.0, 4.0),
    (1, (120, 90), 6, 40, 5.0, 3.5),
    (2, (150, 200), 5, 120, 7.0, 5.0),
    (3, (64, 64), 3, 20, 4.0, 3.0),
    (4, (200, 160), 8, 200, 9.0, 6.0),
    (5, (100, 140), 4, 30, 5.0, 4.5),
    (6, (180, 240), 6, 90, 6.5, 4.0),
    (7, (80, 200), 5, 50, 5.5, 3.0),
]


@pytest.mark.parametrize("seed, shape, num_colors, min_area_px, min_width_px, thin_px", CONSTRUCTED)
def test_every_line_art_step_matches_v0_1_31_on_a_constructed_page(seed, shape, num_colors, min_area_px, min_width_px, thin_px):
    labels, palette_bgr, image, own = _constructed(seed, shape, num_colors)

    _run_steps(labels, palette_bgr, image, own, 20, min_area_px, min_width_px, thin_px, 1.3)


def test_the_constructed_pages_exercise_the_steps():
    # Guard for the comparison above: pages on which no step did anything would test nothing.
    totals = {}
    for seed, shape, num_colors, min_area_px, min_width_px, thin_px in CONSTRUCTED:
        labels, palette_bgr, image, own = _constructed(seed, shape, num_colors)
        for key, value in _run_steps(labels, palette_bgr, image, own, 20, min_area_px, min_width_px, thin_px, 1.3).items():
            totals[key] = totals.get(key, 0) + value
    assert all(totals.values()), totals


# --- the helpers, on speckled maps -----------------------------------------------------------------------------------


def _speckled(seed: int):
    """A small region map full of what smooth pages rarely have: specks, and regions meeting at a corner only.

    Returns the region map (8-connected runs of one color, -1 on the ink), the regions' colors, the ink printed (all of
    what is in no region, and some of what is), and the number of colors.
    """
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(12, 40)), int(rng.integers(12, 40))
    num_colors = int(rng.integers(2, 6))
    field = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), float(rng.uniform(0.5, 3.0)))
    labels = np.digitize(field, np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])).astype(np.int32)
    ink = rng.random((h, w)) < rng.uniform(0.05, 0.45)
    labels[ink] = num_colors
    ids = np.empty((h, w), dtype=np.int32)
    colors, _areas = kernels.label_components(labels.reshape(-1), h, w, num_colors, ids.reshape(-1))
    printed = ink | ((ids >= 0) & (rng.random((h, w)) < 0.15))
    return ids, colors, printed, num_colors


@pytest.mark.parametrize("seed", range(40))
def test_the_region_helpers_match_v0_1_31_on_speckled_maps(seed):
    rng = np.random.default_rng(1000 + seed)
    ids, colors, printed, num_colors = _speckled(seed)
    count = int(colors.size)
    radius = float(rng.uniform(1.0, 3.0))
    min_area_px = int(rng.integers(2, 20))
    some = rng.random(count) < 0.4

    _both("_has_neighbor", ids, count)
    _both("_seams_round", ids, printed, some)
    _both("_without_unpaintable", ids, colors, printed, 2 * radius, min_area_px)
    _both("_painted", ids, colors, num_colors)
    shuffled = np.where(ids >= 0, rng.permutation(count).astype(np.int32)[np.maximum(ids, 0)] % max(1, count // 2), -1)
    _both("_joined", ids, shuffled.astype(np.int32), count)
    _both("_rebuilt_by_white", ids, colors, num_colors, min_area_px, printed)
    _both("_merged_bits", ids, colors, printed, radius, min_area_px, num_colors)
    _both("_split_all", ids, colors, printed, radius)
    _both("_split_all", ids, colors, printed, radius, some)
    _both("_connected_to_own", ids, rng.random(ids.shape) < 0.5, count)
    _both("_ink_is_main_neighbor", ids, some)
    _both("leave_pockets", ids, colors, printed, 2 * radius)
    _both("split_areas", ids, colors, printed, 2 * radius, min_area_px, num_colors)


@pytest.mark.parametrize("seed", range(20))
def test_the_claims_and_their_walls_match_v0_1_31_on_speckled_maps(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(8, 40)), int(rng.integers(8, 40))
    white = rng.random((h, w)) < rng.uniform(0.3, 0.8)
    count, piece = cv2.connectedComponents(white.view(np.uint8), connectivity=4)
    piece = piece.astype(np.int32) - 1
    paintable = rng.random(count - 1) < 0.3
    passable = white | (rng.random((h, w)) < 0.5)

    claim = _both("_claims", piece, paintable, passable)
    _both("_claim_walls", claim, ~white)


def test_a_region_touching_another_only_down_and_to_the_left_has_a_neighbor():
    ids = np.full((3, 3), -1, dtype=np.int32)
    ids[0, 2] = 0
    ids[1, 1] = 1  # the two meet at one corner, the second below and left of the first

    assert regions._has_neighbor(ids, 2).all()


# --- the claim search ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_nearest_seed_within_matches_v0_1_31(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(1, 70)), int(rng.integers(1, 70))
    passable = rng.random((h, w)) < rng.uniform(0.3, 1.0)
    if seed % 2:
        passable = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0) > 0.45
    seeds = np.where(rng.random((h, w)) < rng.uniform(0.001, 0.2), rng.integers(0, 9, (h, w)), -1).astype(np.int32)
    seeds[rng.random((h, w)) < 0.05] = 3  # ties between seeds of one label, and of others

    actual = kernels.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), h, w)
    expected = ref.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), h, w)

    np.testing.assert_array_equal(actual, expected)


def test_nearest_seed_within_matches_v0_1_31_along_a_winding_path():
    # A spiral takes many passes: the rows a pass skips must be the ones it would not have changed.
    size = 41
    passable = np.zeros((size, size), dtype=bool)
    top, left, bottom, right = 0, 0, size - 1, size - 1
    while top <= bottom and left <= right:
        passable[top, left : right + 1] = True
        passable[top : bottom + 1, right] = True
        passable[bottom, left : right + 1] = True
        passable[top + 2 : bottom + 1, left] = True
        top, left, bottom, right = top + 2, left + 2, bottom - 2, right - 2
        if top <= bottom:
            passable[top, left - 1] = True
    seeds = np.full((size, size), -1, dtype=np.int32)
    seeds[size // 2, size // 2] = 1
    seeds[0, 0] = 2

    actual = kernels.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), size, size)
    expected = ref.nearest_seed_within(seeds.reshape(-1), passable.reshape(-1), size, size)

    np.testing.assert_array_equal(actual, expected)
    assert (expected == 1).sum() > 50 and (expected == 2).sum() > 50

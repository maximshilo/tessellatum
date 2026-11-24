"""Mixing recipes: paint mixed as paint mixes, and each legend color's recipe from the painter's paints."""

import itertools
import math

import numpy as np

from tessellatum.core import mixing, spectral_data
from tessellatum.core.legend import (
    DETAIL_PT,
    NAME_HEX_RGB,
    NAME_RECIPE,
    SWATCH_GAP_MM,
    SWATCH_MM,
    TEXT_COLUMN_MM,
    swatch_texts,
    text_width_mm,
)

MM_PER_PT = 25.4 / 72

BOX = [
    ("Titanium white", (255, 255, 255)),
    ("Ivory black", (20, 20, 22)),
    ("Cadmium yellow", (254, 236, 0)),
    ("Cadmium red", (200, 20, 30)),
    ("Ultramarine", (25, 0, 89)),
    ("Phthalo blue", (0, 33, 133)),
    ("Burnt umber", (123, 72, 0)),
    ("Yellow ochre", (200, 140, 60)),
]
BOX_RGB = [rgb for _name, rgb in BOX]


def test_the_spectral_data_is_spectral_js_with_its_license():
    assert "MIT License" in spectral_data.__doc__ and "Ronald van Wijnen" in spectral_data.__doc__
    assert sorted(spectral_data.BASE_SPECTRA) == sorted("WCMYRGB")
    assert all(len(curve) == 38 for curve in spectral_data.BASE_SPECTRA.values())
    assert [len(row) for row in spectral_data.CIE_CMF] == [38, 38, 38]


def test_every_color_is_seen_again_as_itself():
    levels = np.linspace(0, 255, 18).round()
    grid = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), -1).reshape(-1, 3)
    assert (mixing.seen_rgb(mixing.reflectance(grid)) == grid).all()


def test_blue_and_yellow_make_green_and_white_lightens_without_graying():
    r, g, b = mixing.mixed_rgb(np.array([(0, 33, 133), (252, 210, 0)]), [1, 1])
    assert g > r + 40 and g > b + 40  # green, where RGB's average is a gray-olive (126, 122, 67)
    pink = mixing.mixed_rgb(np.array([(255, 255, 255), (200, 20, 30)]), [1, 1])
    assert pink[0] > 200 and pink[1] < pink[0] - 100  # still clearly red, only lighter
    assert mixing.mixed_rgb(np.array([(90, 107, 47), (90, 107, 47)]), [3, 1]) == (90, 107, 47)


def test_parts_are_small_whole_numbers_in_lowest_terms():
    for ratios in (mixing._PAIR_RATIOS, mixing._TRIPLE_RATIOS):
        sums = ratios.sum(axis=1)
        assert (ratios >= 1).all() and (sums <= mixing.MOST_PARTS).all()
        assert all(math.gcd(*map(int, parts)) == 1 for parts in ratios)
        assert list(sums) == sorted(sums)  # fewest parts first


def test_a_paint_owned_is_its_own_recipe_and_a_mix_of_two_is_found_again():
    ochre, = mixing.recipes([(200, 140, 60)], BOX_RGB)
    assert ochre.parts == ((7, 1),) and ochre.de00 == 0 and ochre.mixable
    tint = mixing.mixed_rgb(np.array([BOX_RGB[0], BOX_RGB[4]]), [3, 1])  # 3 white, 1 ultramarine
    found, = mixing.recipes([tint], BOX_RGB)
    assert found.parts == ((0, 3), (4, 1)) and found.de00 < 1e-6 and found.rgb == tint
    assert mixing.recipe_lines(found, BOX) == ["3 Titanium white", "1 Ultramarine"]


def test_a_paint_is_added_only_when_it_brings_the_mix_clearly_nearer():
    rng = np.random.default_rng(1)
    targets = [tuple(int(v) for v in t) for t in rng.integers(0, 256, (12, 3))]
    for target, recipe in zip(targets, mixing.recipes(targets, BOX_RGB)):
        assert 1 <= len(recipe.parts) <= 3 and sum(n for _, n in recipe.parts) <= mixing.MOST_PARTS
        assert [n for _, n in recipe.parts] == sorted((n for _, n in recipe.parts), reverse=True)
        fewer = mixing.recipes([target], [BOX_RGB[p] for p, _ in recipe.parts[:1]])[0]
        if len(recipe.parts) > 1:
            assert recipe.de00 < fewer.de00 - mixing.SIMPLER_BY_DE00 + 1e-9
        mixed = mixing.mixed_rgb(np.array([BOX_RGB[p] for p, _ in recipe.parts]), [n for _, n in recipe.parts])
        assert mixed == recipe.rgb


def test_a_color_the_paints_cant_reach_says_how_far_off_its_closest_mix_is():
    earth = [("Burnt umber", (123, 72, 0)), ("Yellow ochre", (200, 140, 60))]
    violet, = mixing.recipes([(150, 40, 220)], [rgb for _, rgb in earth])
    assert not violet.mixable
    lines = mixing.recipe_lines(violet, earth)
    assert lines[-1] == f"closest mix, {violet.de00:.0f} dE00 off"
    unnamed, = mixing.recipes([(200, 140, 60)], [(200, 140, 60)])
    assert mixing.recipe_lines(unnamed, [("", (200, 140, 60))]) == ["1 #C88C3C"]
    assert mixing.recipes([(1, 2, 3)], []) == [] and mixing.recipes([], BOX_RGB) == []


def test_many_paints_are_shortlisted_and_still_give_close_recipes():
    rng = np.random.default_rng(2)
    paints = [tuple(int(v) for v in c) for c in rng.integers(0, 256, (150, 3))]
    targets = [tuple(int(v) for v in t) for t in rng.integers(0, 256, (6, 3))]
    found = mixing.recipes(targets, paints)
    assert len(found) == 6 and all(r.mixable for r in found)
    assert all(p < 150 for r in found for p, _ in r.parts)


def test_the_legend_writes_each_recipe_under_its_name_and_it_fits_the_cell():
    palette = [(150, 40, 220), (200, 140, 60)]
    lines = [["3 Titanium white", "2 A paint with a very long name indeed", "1 Ultramarine", "closest mix, 7 dE00 off"],
             ["1 Yellow ochre"]]
    texts = swatch_texts(palette, NAME_RECIPE, names=["violet", "ochre"], recipes=lines)
    assert [t.text for t in texts[1]] == ["ochre", "1 Yellow ochre"]
    assert [t.text for t in texts[0]][0] == "violet" and len(texts[0]) == 5
    detail_mm = DETAIL_PT * MM_PER_PT
    assert texts[0][2].text.endswith("...") and text_width_mm(texts[0][2].text, detail_mm) <= TEXT_COLUMN_MM
    top = texts[0][0].baseline_mm - texts[0][0].size_mm
    assert top > -SWATCH_GAP_MM / 2 and texts[0][-1].baseline_mm < SWATCH_MM + SWATCH_GAP_MM / 2
    # With no recipes, the codes instead.
    assert swatch_texts(palette, NAME_RECIPE, names=["violet", "ochre"]) == swatch_texts(
        palette, NAME_HEX_RGB, names=["violet", "ochre"]
    )


def test_recipes_are_the_same_every_time():
    targets = [tuple(int(v) for v in t) for t in itertools.islice(itertools.product((40, 140, 240), repeat=3), 9)]
    assert mixing.recipes(targets, BOX_RGB) == mixing.recipes(targets, BOX_RGB)

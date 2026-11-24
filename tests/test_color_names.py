"""Color names on the legend: a different everyday name for every color, the nearest one, from a CC0 list."""

from importlib import resources

import numpy as np
import pytest

from tessellatum.core import color_names as names_module
from tessellatum.core.color import bgr_to_lab, ciede2000
from tessellatum.core.color_names import color_names, hex_code, name_colors
from tessellatum.core.legend import (
    DETAIL_PT,
    NAME_LINES,
    NAME_PT,
    TEXT_COLUMN_MM,
    _wrap,
    detail_line,
    text_width_mm,
)

MM_PER_PT = 25.4 / 72


def test_the_list_is_the_published_cc0_file():
    raw = resources.files("tessellatum").joinpath("resources/xkcd_color_names.txt").read_text(encoding="utf-8")
    assert raw.splitlines()[0] == "# License: https://creativecommons.org/publicdomain/zero/1.0/"
    assert len([line for line in raw.splitlines() if line and not line.startswith("#")]) == 949


def test_the_tidied_list_leaves_out_what_a_legend_shouldn_t_print():
    names, lab = color_names()
    assert 750 < len(names) == len(set(names)) == len(lab)
    words = {word for name in names for word in name.split()}
    assert not words & names_module._LEFT_OUT_WORDS
    assert not set(names) & names_module._LEFT_OUT_NAMES
    assert not any("/" in name or "grey" in name for name in names)
    assert not any(" " not in name and name.endswith("ish") for name in names)
    assert all(name.isascii() and name == name.strip() for name in names)
    assert {"gray", "dark gray", "black", "white", "dusty rose", "burnt sienna", "olive green"} <= set(names)


@pytest.mark.parametrize(
    "rgb, name",
    [
        ((0, 0, 0), "black"),
        ((255, 255, 255), "white"),
        ((0x92, 0x95, 0x91), "gray"),  # each name's own color names itself
        ((0xC0, 0x73, 0x7A), "dusty rose"),
        ((0xB0, 0x4E, 0x0F), "burnt sienna"),
        ((0xE5, 0x00, 0x00), "red"),
    ],
)
def test_a_color_is_named_by_the_nearest_name(rgb, name):
    assert name_colors([rgb]) == [name]


def test_two_look_alike_colors_get_two_names_the_closer_one_keeping_the_nearest():
    first, second = name_colors([(0xE5, 0x02, 0x02), (0xE5, 0x00, 0x00)])
    assert second == "red" and first != "red"  # red's own color is closer to red
    names, table = color_names()
    lab = bgr_to_lab(np.array([[0x02, 0x02, 0xE5]], dtype=np.uint8))
    distance = ciede2000(lab, table)
    assert first == names[int(np.argsort(distance)[1])]  # the next nearest


def test_every_legend_s_names_are_different_and_near_their_colors():
    rng = np.random.default_rng(7)
    for _ in range(20):
        palette = [tuple(int(v) for v in c) for c in rng.integers(0, 256, (64, 3))]
        named = name_colors(palette)
        assert len(set(named)) == len(named) == 64
        names, table = color_names()
        lab = bgr_to_lab(np.asarray(palette, dtype=np.uint8)[:, ::-1])
        picked = table[[names.index(n) for n in named]]
        assert np.median(ciede2000(lab, picked)) < 8


def test_no_colors_no_names():
    assert name_colors([]) == []


def test_hex_codes_and_the_values_line_are_exact():
    assert hex_code((90, 107, 47)) == "#5A6B2F"
    assert hex_code((0, 0, 0)) == "#000000"
    assert detail_line((90, 107, 47)) == "#5A6B2F  RGB 90 107 47"


def test_every_name_fits_its_lines_and_the_longest_values_line_fits_its_column():
    names, _ = color_names()
    for name in names:
        lines = _wrap(name, NAME_PT * MM_PER_PT, TEXT_COLUMN_MM, NAME_LINES)
        assert len(lines) <= NAME_LINES and " ".join(lines) == name
        assert all(text_width_mm(line, NAME_PT * MM_PER_PT) <= TEXT_COLUMN_MM for line in lines)
    assert text_width_mm(detail_line((255, 255, 255)), DETAIL_PT * MM_PER_PT) <= TEXT_COLUMN_MM


def test_a_long_name_wraps_between_words_never_inside_one():
    assert _wrap("british racing green", NAME_PT * MM_PER_PT, 20.0, 2) == ["british racing", "green"]
    assert _wrap("british racing green", NAME_PT * MM_PER_PT, 5.0, 2) == ["british", "racing green"]

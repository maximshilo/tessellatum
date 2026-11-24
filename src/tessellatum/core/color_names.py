"""A name for each legend color: the nearest of a few hundred everyday color names, by CIEDE2000.

The names are the xkcd color survey's (Randall Munroe, 2010): the 949 names over 200,000 people most often gave to
colors shown to them, each with the color that name was given to on average. They are in
``resources/xkcd_color_names.txt``, byte for byte as published at https://xkcd.com/color/rgb.txt under CC0 1.0 (public
domain dedication; the license line is the file's first). They are friendlier than a systematic list ("dusty rose"
rather than "moderate purplish pink") and dense enough that a palette's colors each find a close one.

A page is for anyone, so the list is tidied here, not in the file: names a legend shouldn't print are left out (crude
ones, brand names, survey jokes, misspellings, slashed and run-together forms of names it also has), and "grey" is spelled
"gray", as the app writes it. What is left out is listed below, so the tidying can be read and changed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Sequence

import numpy as np

from tessellatum.core.color import bgr_to_lab, ciede2000

# Words whose names are left out wherever they appear: crude, unkind or a judgment rather than a color.
_LEFT_OUT_WORDS = frozenset(
    "shit poo poop puke vomit barf booger snot piss diarrhea sick sickly ugly nasty gross icky dirty weird boring nice "
    "bland bruise".split()
)
# Whole names left out: brands and characters, survey jokes and informal spellings, misspellings of a name the list also
# has spelled right, and names the list also has with a space ("darkgreen", "dark green").
_LEFT_OUT_NAMES = frozenset(
    {
        # brands and characters
        "windows blue", "barbie pink", "barney", "barney purple", "kermit green", "tiffany blue",
        # jokes, informal forms and bare adjectives
        "blue blue", "light light green", "light light blue", "burple", "blurple", "purpley", "purply", "purpleish",
        "pinky", "dark", "pale", "light urple", "blue with a hint of purple", "really light blue",
        # misspellings of a name also on the list
        "liliac", "light lavendar", "toupe", "terracota", "burnt siena", "perrywinkle", "forrest green", "ocre",
        "kelley green", "manilla", "egg shell", "aqua marine", "robin egg blue",
        # run together
        "darkgreen", "darkblue", "lightblue", "lightgreen", "greenblue", "bluegreen", "yellowgreen", "orangered",
        "greyblue", "bluegrey",
    }
)
# Informal "-y" forms of a color word ("greeny blue", "orangey red"): the list has the "-ish" ones too.
_INFORMAL = re.compile(r"\b(purpleish|purpley|purply|pinky|bluey|greeny|browny|reddy|orangey|yellowy)\b")


@lru_cache(maxsize=1)
def color_names() -> tuple[tuple[str, ...], np.ndarray]:
    """The names a legend may use and their colors as CIE Lab (Nx3), the most common name first where two match."""
    text = resources.files("tessellatum").joinpath("resources/xkcd_color_names.txt").read_text(encoding="utf-8")
    entries: dict[str, tuple[int, int, int]] = {}
    # The file lists the least common names first: read from the end, so that of two names that tidy to the same one,
    # the more common keeps its color.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, hex_code = (part.strip() for part in line.split("\t")[:2])
        if name in _LEFT_OUT_NAMES or _INFORMAL.search(name) or _LEFT_OUT_WORDS & set(re.split(r"[ /]", name)):
            continue
        if "/" in name:
            continue  # "green/yellow": a slashed name, which the list also has as "green yellow" or "yellowish green"
        if " " not in name and name.endswith("ish"):
            continue  # "reddish", "grayish": half a name
        name = name.replace("grey", "gray")
        if name in entries:
            continue
        entries[name] = tuple(int(hex_code[i : i + 2], 16) for i in (1, 3, 5))
    names = tuple(entries)
    rgb = np.array([entries[n] for n in names], dtype=np.uint8)
    return names, bgr_to_lab(rgb[:, ::-1])


def name_colors(palette_rgb: Sequence[tuple[int, int, int]]) -> list[str]:
    """A different name for each of ``palette_rgb``: the nearest by CIEDE2000, unless a closer color took it.

    Every color and name are paired off closest first: the pair nearest of all takes its name, then the nearest pair of
    what is left, and so on. Two look-alike colors so get the two names nearest them, not one name twice, and the one
    closer to a name keeps it.
    """
    if len(palette_rgb) == 0:
        return []
    names, table = color_names()
    rgb = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
    distance = ciede2000(bgr_to_lab(rgb[:, ::-1])[:, None, :], table[None, :, :])
    # Ties go to the earlier color and the more common name: a stable sort of the pairs by distance.
    order = np.argsort(distance, axis=None, kind="stable")
    chosen: list[str | None] = [None] * len(rgb)
    taken = np.zeros(len(names), dtype=bool)
    left = len(rgb)
    for flat in order:
        color, name = divmod(int(flat), len(names))
        if chosen[color] is None and not taken[name]:
            chosen[color] = names[name]
            taken[name] = True
            left -= 1
            if left == 0:
                break
    return [name if name is not None else "" for name in chosen]


def hex_code(rgb: tuple[int, int, int]) -> str:
    """``rgb`` as a hex color code, ``#5A6B2F``."""
    return "#" + "".join(f"{int(v):02X}" for v in rgb)

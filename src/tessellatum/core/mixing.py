"""Mixing recipes: how to mix each legend color from the painter's own paints, in a few small whole parts.

Paint mixes subtractively: blue and yellow make green, not the gray their RGB average is. Each paint's color is
taken to a reflectance curve (spectral.js's sum of seven primary curves, the one sRGB color it is built from coming
back exactly), the paints are mixed by Kubelka-Munk theory -- their absorption over scattering, K/S, averaged with
each paint's share weighted by its luminance, as a light paint tints less than a dark one stains -- and the mix's
curve is seen as sRGB under D65 again. It is a model of paint, not a measurement of the painter's own: real pigments
of the same color mix differently, so a recipe is a place to start, its parts as a painter would measure them.

A recipe is one, two or three paints in parts that add up to at most ``MOST_PARTS``: the one whose mix is nearest the
color by CIEDE2000, a second or third paint only when it brings the mix ``SIMPLER_BY_DE00`` nearer. A color no recipe
comes within ``MIXABLE_DE00`` of is not mixable from the paints; its nearest recipe is given all the same.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tessellatum.core.color import bgr_to_lab, ciede2000
from tessellatum.core.color_names import hex_code
from tessellatum.core.spectral_data import BASE_SPECTRA, CIE_CMF, XYZ_TO_RGB

# A mix further than this from its color, in CIEDE2000, is too far to call it the color's recipe: about the gap two
# legend colors keep, at the most (the color margin is 4 at every level).
MIXABLE_DE00 = 5.0
# How much nearer, in CIEDE2000, another paint must bring a mix to be worth adding to it.
SIMPLER_BY_DE00 = 1.0
# The most parts a recipe adds up to: "3 + 1" is a recipe, "37 + 12" a measurement.
MOST_PARTS = 10
# The paints weighed for any recipe when there are more: those nearest the colors, and the lightest and darkest.
MOST_PAINTS_WEIGHED = 64
# The paints weighed for a color's three-paint recipes: those its best one- and two-paint recipes use.
TRIPLE_PAINTS = 8

_BASE = np.array([BASE_SPECTRA[key] for key in "WCMYRGB"])  # 7 x 38
_CMF = np.array(CIE_CMF)  # 3 x 38
_XYZ_TO_RGB = np.array(XYZ_TO_RGB)
_TINY = np.finfo(float).eps


def _ratios(count: int) -> np.ndarray:
    """Every way of ``count`` paints in whole parts, each at least 1, adding up to ``MOST_PARTS`` at most, in lowest
    terms (2 + 2 is 1 + 1): fewest parts first."""
    found = [
        parts
        for parts in itertools.product(range(1, MOST_PARTS + 1), repeat=count)
        if sum(parts) <= MOST_PARTS and math.gcd(*parts) == 1
    ]
    found.sort(key=lambda parts: (sum(parts), parts))
    return np.array(found, dtype=np.float64)


_PAIR_RATIOS = _ratios(2)
_TRIPLE_RATIOS = _ratios(3)


@dataclass(frozen=True)
class Recipe:
    """A color's recipe: ``parts`` of the paints, (the paint's index, how many parts) each, most parts first; ``rgb``,
    what it mixes to; and ``de00``, how far that is from the color."""

    parts: tuple[tuple[int, int], ...]
    rgb: tuple[int, int, int]
    de00: float

    @property
    def mixable(self) -> bool:
        return self.de00 <= MIXABLE_DE00


def reflectance(rgb: np.ndarray) -> np.ndarray:
    """The reflectance curve (..., 38) of each 8-bit sRGB color (..., 3), as spectral.js builds it: its white, then its
    cyan, magenta and yellow, then its red, green and blue, in linear light, each weighting that primary's curve."""
    linear = _linear(np.asarray(rgb, dtype=np.float64) / 255)
    white = linear.min(axis=-1)
    r, g, b = np.moveaxis(linear - white[..., None], -1, 0)
    weights = np.stack(
        [
            white,
            np.minimum(g, b),
            np.minimum(r, b),
            np.minimum(r, g),
            np.maximum(0, np.minimum(r - b, r - g)),
            np.maximum(0, np.minimum(g - b, g - r)),
            np.maximum(0, np.minimum(b - g, b - r)),
        ],
        axis=-1,
    )
    return np.maximum(_TINY, weights @ _BASE)


def seen_rgb(curve: np.ndarray) -> np.ndarray:
    """The 8-bit sRGB color (..., 3) each reflectance curve (..., 38) is seen as under D65."""
    linear = np.clip((curve @ _CMF.T) @ _XYZ_TO_RGB.T, 0, 1)
    encoded = np.where(linear > 0.0031308, 1.055 * linear ** (1 / 2.4) - 0.055, 12.92 * linear)
    return np.clip(np.round(encoded * 255), 0, 255).astype(np.uint8)


def mix(curves: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """The reflectance curve (..., 38) of ``curves`` (n x 38, one paint each) mixed in ``parts`` (..., n): their
    K/S averaged, each paint's share its parts squared times its luminance, and turned back into reflectance."""
    luminance = np.maximum(_TINY, curves @ _CMF[1])
    share = np.asarray(parts, dtype=np.float64) ** 2 * luminance
    ks = (share @ _ks(curves)) / share.sum(axis=-1, keepdims=True)
    return 1 + ks - np.sqrt(ks**2 + 2 * ks)


def mixed_rgb(paints_rgb: np.ndarray, parts: Sequence[float]) -> tuple[int, int, int]:
    """The 8-bit sRGB color ``paints_rgb`` (n x 3) make mixed in ``parts``."""
    rgb = seen_rgb(mix(reflectance(np.asarray(paints_rgb).reshape(-1, 3)), np.asarray(parts, dtype=np.float64)))
    return tuple(int(v) for v in rgb)  # type: ignore[return-value]


def recipes(
    palette_rgb: Sequence[tuple[int, int, int]], paints_rgb: Sequence[tuple[int, int, int]]
) -> list[Recipe]:
    """Each of ``palette_rgb``'s recipe from ``paints_rgb`` (see the module's docstring); none without paints."""
    if len(paints_rgb) == 0:
        return []
    paints = np.asarray(paints_rgb, dtype=np.uint8).reshape(-1, 3)
    targets = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
    if len(targets) == 0:
        return []
    paint_lab = _lab(paints)
    target_lab = _lab(targets)
    curves = reflectance(paints)
    weighed = _shortlist(paint_lab, target_lab)

    # Every pair of the paints weighed, in every ratio, mixed once for all the colors.
    pairs = np.array(list(itertools.combinations(weighed, 2)), dtype=np.int64).reshape(-1, 2)
    pair_rgb = np.empty((len(pairs), len(_PAIR_RATIOS), 3), dtype=np.uint8)
    for i, pair in enumerate(pairs):
        pair_rgb[i] = seen_rgb(mix(curves[pair], _PAIR_RATIOS))
    pair_lab = _lab(pair_rgb.reshape(-1, 3)).reshape(len(pairs), len(_PAIR_RATIOS), 3)

    found = []
    for target, lab in zip(targets, target_lab):
        single = ciede2000(lab[None, :], paint_lab)
        one = int(np.argmin(single))
        best = Recipe(((one, 1),), tuple(int(v) for v in paints[one]), float(single[one]))
        if len(pairs) == 0:
            found.append(best)
            continue
        two = ciede2000(lab[None, None, :], pair_lab)  # pairs x ratios
        p, r = np.unravel_index(int(np.argmin(two)), two.shape)
        if two[p, r] < best.de00 - SIMPLER_BY_DE00:
            best = _recipe(pairs[p], _PAIR_RATIOS[r], pair_rgb[p, r], two[p, r])
        triple = _best_triple(lab, curves, single, two, pairs)
        if triple is not None and triple.de00 < best.de00 - SIMPLER_BY_DE00:
            best = triple
        found.append(best)
    return found


def recipe_lines(recipe: Recipe, paints: Sequence[tuple[str, tuple[int, int, int]]]) -> list[str]:
    """A recipe as a legend writes it, from the ``paints`` (name, RGB each) it was found among: "3 Titanium white" a
    line, a paint with no name called by its hex code; and, if it is not mixable, how far off its mix is."""
    lines = []
    for paint, parts in recipe.parts:
        name, rgb = paints[paint]
        lines.append(f"{parts} {name or hex_code(rgb)}")
    if not recipe.mixable:
        lines.append(f"closest mix, {recipe.de00:.0f} dE00 off")
    return lines


def _best_triple(
    lab: np.ndarray, curves: np.ndarray, single: np.ndarray, two: np.ndarray, pairs: np.ndarray
) -> Recipe | None:
    """The best three-paint recipe for the color at ``lab`` among the ``TRIPLE_PAINTS`` paints its best one- and
    two-paint recipes use."""
    best_pairs = np.argsort(two.min(axis=1), kind="stable")
    chosen: list[int] = []
    for paint in itertools.chain(np.argsort(single, kind="stable")[:2], pairs[best_pairs].reshape(-1)):
        if int(paint) not in chosen:
            chosen.append(int(paint))
        if len(chosen) == TRIPLE_PAINTS:
            break
    if len(chosen) < 3:
        return None
    triples = np.array(list(itertools.combinations(sorted(chosen), 3)), dtype=np.int64)
    rgb = np.stack([seen_rgb(mix(curves[t], _TRIPLE_RATIOS)) for t in triples])  # triples x ratios x 3
    distance = ciede2000(lab[None, None, :], _lab(rgb.reshape(-1, 3)).reshape(rgb.shape))
    t, r = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return _recipe(triples[t], _TRIPLE_RATIOS[r], rgb[t, r], distance[t, r])


def _recipe(paints: np.ndarray, parts: np.ndarray, rgb: np.ndarray, de00: float) -> Recipe:
    order = sorted(zip(paints.tolist(), parts.astype(int).tolist()), key=lambda pp: (-pp[1], pp[0]))
    return Recipe(tuple((int(p), int(n)) for p, n in order), tuple(int(v) for v in rgb), float(de00))


def _shortlist(paint_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
    """The paints weighed: all of them, or ``MOST_PAINTS_WEIGHED`` -- the lightest, the darkest, and those nearest
    any of the colors."""
    if len(paint_lab) <= MOST_PAINTS_WEIGHED:
        return np.arange(len(paint_lab))
    nearest = ciede2000(target_lab[:, None, :], paint_lab[None, :, :]).min(axis=0)
    picked = [int(np.argmax(paint_lab[:, 0])), int(np.argmin(paint_lab[:, 0]))]
    for paint in np.argsort(nearest, kind="stable"):
        if len(picked) == MOST_PAINTS_WEIGHED:
            break
        if int(paint) not in picked:
            picked.append(int(paint))
    return np.array(sorted(set(picked)), dtype=np.int64)


def _ks(curves: np.ndarray) -> np.ndarray:
    return (1 - curves) ** 2 / (2 * curves)


def _linear(encoded: np.ndarray) -> np.ndarray:
    return np.where(encoded > 0.04045, ((encoded + 0.055) / 1.055) ** 2.4, encoded / 12.92)


def _lab(rgb: np.ndarray) -> np.ndarray:
    return bgr_to_lab(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1]))

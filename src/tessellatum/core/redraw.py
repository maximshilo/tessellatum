"""Drawing a page at a larger size than it was made at: its regions, its ink and its gaps, from its own geometry.

A page is made once, at the preview's size, and an export is that page drawn at print size (see
``pipeline.draw_page``), so that an export always has the regions, numbers and legend its preview shows. Drawing it
larger is more than scaling its pixels: the page's lines are its region map's boundaries smoothed off their
staircase (see ``boundaries``), and a map merely scaled up would keep the staircase a scaled pixel's step off them,
leaving paper between two regions that no line closes. So here:

- each region is filled inside its outline, smoothed as the lines are, scaled (``scaled_regions``): the pixels whose
  middles lie inside it (``fill``); and two regions touch, along a side or at a corner, only where they did on the
  page drawn from;
- the printed ink is filled inside its own smoothed outline (``filled``), but never cuts a region's paintable white
  in two where the page drawn from kept it whole (``whole_white``);
- and where the regions and the lines still part by a pixel, the pixel is found (``gaps``) and the page draws a dot of
  line on it.
"""

from __future__ import annotations

import numpy as np

from tessellatum.core import kernels
from tessellatum.core.boundaries import region_outlines
from tessellatum.core.ink_outline import ink_outline
from tessellatum.core.regions import white_areas

# Neighbors a pixel's side apart and a corner apart, as (rows, columns) steps.
_SIDES = ((0, 1), (1, 0))
_CORNERS = ((1, 1), (1, -1))


def nearest_index(source: int, target: int) -> np.ndarray:
    """For each of ``target`` pixels along an axis, the one of ``source`` its middle falls in."""
    return np.minimum(((np.arange(target) + 0.5) * source / target).astype(np.intp), source - 1)


def scaled_up(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``array`` (HxW) at ``size`` (width, height): each pixel the value of the one its middle falls in."""
    height, width = array.shape[:2]
    return array[np.ix_(nearest_index(height, size[1]), nearest_index(width, size[0]))]


def scaled_points(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Nx2 (x, y) points, pixel middles at whole numbers, on a page ``sx`` x ``sy`` times as large."""
    return (np.asarray(points, dtype=np.float64) + 0.5) * (sx, sy) - 0.5


def fill(groups: list[list[np.ndarray]], values, out: np.ndarray, sx: float, sy: float) -> None:
    """Fill each group of rings with its value on ``out`` (HxW int32), in order, by the even-odd rule: every pixel
    whose middle lies inside.

    The rings are Nx2 float (x, y), pixel middles at whole numbers, on a page ``sx`` x ``sy`` times smaller than
    ``out``. Two groups side by side along one line fill each pixel once, so a page's regions filled inside their
    outlines (see ``boundaries.region_outlines``) leave none twice and none out; and a mask's own outline (see
    ``ink_outline.traced_rings``) filled at its size is the mask.
    """
    rings = [ring for group in groups for ring in group]
    if not rings:
        return
    points = scaled_points(np.concatenate(rings), sx, sy)
    ring_starts = np.concatenate([[0], np.cumsum([len(ring) for ring in rings])]).astype(np.int64)
    group_starts = np.concatenate([[0], np.cumsum([len(group) for group in groups])]).astype(np.int64)
    height, width = out.shape
    kernels.fill_rings(
        np.ascontiguousarray(points[:, 0]), np.ascontiguousarray(points[:, 1]), ring_starts, group_starts,
        np.asarray(values, dtype=np.int32), height, width, out,
    )


def filled(rings: list[np.ndarray], size: tuple[int, int], sx: float, sy: float) -> np.ndarray:
    """HxW bool at ``size``: the pixels inside ``rings``, scaled ``sx`` x ``sy``, by the even-odd rule (see ``fill``)."""
    width, height = size
    canvas = np.zeros((height, width), dtype=np.int32)
    fill([rings], [1], canvas, sx, sy)
    return canvas.astype(bool)


def _pairs(shape: tuple[int, int], step: tuple[int, int]) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """The (one, other) slices of a ``shape`` map that put each pixel beside its neighbor ``step`` (rows, columns)
    away."""
    height, width = shape
    dy, dx = step
    if dx >= 0:
        return (slice(0, height - dy), slice(0, width - dx)), (slice(dy, height), slice(dx, width))
    return (slice(0, height - dy), slice(-dx, width)), (slice(dy, height), slice(0, width + dx))


def _codes(ids: np.ndarray, base: int, steps) -> np.ndarray:
    """The sorted codes, ``a * base + b`` with a < b, of the pairs of ``ids`` (int64) that are neighbors one of
    ``steps`` apart."""
    found = []
    for step in steps:
        one, other = _pairs(ids.shape, step)
        a, b = ids[one], ids[other]
        differ = a != b
        found.append(np.minimum(a[differ], b[differ]) * base + np.maximum(a[differ], b[differ]))
    return np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)


def contacts(region_id_map: np.ndarray, corners: bool = False) -> set[tuple[int, int]]:
    """The pairs of regions, (a, b) with a < b, that are neighbors on ``region_id_map``: sharing a pixel's side, or
    with ``corners`` a side or a corner. No region (-1) counts as one more."""
    ids = region_id_map.astype(np.int64) + 1
    base = int(ids.max()) + 1 if ids.size else 1
    codes = _codes(ids, base, _SIDES + (_CORNERS if corners else ()))
    return {(int(a) - 1, int(b) - 1) for a, b in zip((codes // base).tolist(), (codes % base).tolist())}


def _strays(ids: np.ndarray, base: int, sides: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """HxW bool: the pixels of ``ids`` (int64) in a pair of neighbors a side apart whose code (see ``_codes``) isn't in
    ``sides``, or a corner apart whose code isn't in ``corners``."""
    stray = np.zeros(ids.shape, dtype=bool)
    for steps, allowed in ((_SIDES, sides), (_CORNERS, corners)):
        for step in steps:
            one, other = _pairs(ids.shape, step)
            a, b = ids[one], ids[other]
            rows, columns = np.nonzero(a != b)
            first, second = a[rows, columns], b[rows, columns]
            new = ~np.isin(np.minimum(first, second) * base + np.maximum(first, second), allowed)
            stray[one][rows[new], columns[new]] = True
            stray[other][rows[new], columns[new]] = True
    return stray


def scaled_regions(region_id_map: np.ndarray, size: tuple[int, int], smoothing_px: float) -> np.ndarray:
    """``region_id_map`` drawn at ``size`` (width, height), each region filled inside its outline smoothed by
    ``smoothing_px`` as the page's lines are (see ``boundaries.region_outlines`` and ``fill``).

    A pixel no outline covers takes the region its middle falls in on the map (``scaled_up``), and pixels in no region
    there stay in none. Two regions touch only where they do on ``region_id_map``: a side apart only if they were a
    side apart there, a corner apart only if they were neighbors there at all, so that no line drawn from it is
    missing between them and no two regions of a color come to meet. Pixels where they would go back to the region
    their middle falls in, as does every pixel of a region the smoothing left none of.
    """
    width, height = size
    h1, w1 = region_id_map.shape
    nearest = scaled_up(region_id_map, size)
    drawn = np.full((height, width), -1, dtype=np.int32)
    outlines = region_outlines(region_id_map, smoothing_px)
    fill(list(outlines.values()), list(outlines.keys()), drawn, width / w1, height / h1)
    drawn = np.where(nearest < 0, -1, np.where(drawn < 0, nearest, drawn)).astype(np.int32)

    ids = region_id_map.astype(np.int64) + 1
    base = int(ids.max()) + 1 if ids.size else 1
    sides = _codes(ids, base, _SIDES)
    corners = np.union1d(sides, _codes(ids, base, _CORNERS))
    regions = np.unique(ids[ids > 0])
    # Every round turns pixels back to the map scaled up, whose neighbors were neighbors on ``region_id_map``: a pair
    # that wasn't has a pixel that isn't the map's yet, so the rounds end.
    while True:
        lost = regions[np.bincount(drawn.ravel() + 1, minlength=base)[regions] == 0] - 1
        if lost.size:
            gone = np.isin(nearest, lost)
            drawn[gone] = nearest[gone]
        stray = _strays(drawn.astype(np.int64) + 1, base, sides, corners)
        if not stray.any():
            return drawn
        drawn[stray] = nearest[stray]


def gaps(region_id_map: np.ndarray, white: np.ndarray) -> np.ndarray:
    """HxW bool: one pixel of each pair of 4-neighbors that are both ``white`` (nothing printed on them) and in
    different regions (no region counting as one more): where two regions' paint would run together."""
    found = np.zeros(region_id_map.shape, dtype=bool)
    height, width = region_id_map.shape
    for dy, dx in _SIDES:
        one, other = (slice(0, height - dy), slice(0, width - dx)), (slice(dy, height), slice(dx, width))
        found[one] |= white[one] & white[other] & (region_id_map[one] != region_id_map[other])
    return found


def whole_white(
    region_id_map: np.ndarray, smooth: np.ndarray, blocky: np.ndarray, radius: float
) -> np.ndarray:
    """``smooth``, the printed ink filled inside its smoothed outline, but ``blocky`` -- the ink scaled up pixel for
    pixel, whose white is the page drawn from's -- in every region whose white a brush of ``radius`` fits in falls into
    more pieces with ``smooth`` than with ``blocky``: where the smoothing pinches a narrow neck of paper shut."""
    with_smooth = white_areas(region_id_map, smooth, radius)
    with_blocky = white_areas(region_id_map, blocky, radius)
    cut = np.flatnonzero(with_smooth > with_blocky)
    if not cut.size:
        return smooth
    inside = np.isin(region_id_map, cut)
    return np.where(inside, blocky, smooth)


def scaled_ink(printed: np.ndarray, region_id_map: np.ndarray, radius: float) -> np.ndarray:
    """The printed ink ``printed`` (HxW bool, of the page drawn from) on ``region_id_map``, its page drawn larger.

    Filled inside its own smoothed outline (see ``ink_outline``), and all of it where it lies in no region -- line
    art's bold ink -- so that no paper opens between the ink and the regions round it; but scaled up pixel for pixel
    wherever the smoothing would cut a region's paintable white (a brush of ``radius``) in two (``whole_white``).
    """
    height, width = region_id_map.shape
    blocky = scaled_up(printed, (width, height))
    if not printed.any():
        return blocky
    h1, w1 = printed.shape
    smooth = filled(ink_outline(printed), (width, height), width / w1, height / h1) | (blocky & (region_id_map < 0))
    return whole_white(region_id_map, smooth, blocky, radius)

"""Tones: a face painted in a few large tones rather than blotches.

A face gets smaller regions than the rest of the page (see
``regions.build_regions``), and what they buy is its features. Its skin or
fur pays for them twice over: a region there can be painted a color that is
not the nearest to it -- it takes the color of the part of it a brush fits
in, and keeps that color whatever merges into it -- and a region can stand
apart from its neighbor for a step in tone too faint to be worth a number.
So, inside the faces the pipeline finds (see ``faces``):

- every region takes the palette color nearest its own pixels, which the
  painting only gains from;
- then a region joins a neighbor while painting it in that neighbor's color
  would put its pixels, on average, less than ``MIN_STEP_DE00`` further from
  the picture -- the faintest step first, and a region that would be no worse
  off, or better, in a neighbor's color before any other.

What is left is the tones that carry the face. The features keep theirs: an
eye or a lip is far from the color beside it.

A region's distance from a color is summed over its pixels in CIEDE2000, the
measure the palette keeps its margin in (see ``color``).
"""

from __future__ import annotations

import numpy as np

from tessellatum.core import kernels
from tessellatum.core.color import bgr_to_lab, ciede2000

# The faintest step in tone a region is kept for: how much further from the picture its pixels would be in its
# neighbor's color than in its own, in CIEDE2000, averaged over them.
MIN_STEP_DE00 = 1.0

# A region's pixels are counted by color in cells of the sRGB cube this many bits a channel across -- 32 steps of 8 --
# and each cell's pixels stand at its middle: at most 4 of 255 a channel from where they are, which evens out over a
# region.
_CELL_BITS = 5


def settle_tones(
    image_bgr: np.ndarray,
    where: np.ndarray,
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    palette_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The page's regions and their colors with the tones inside ``where`` (HxW bool) settled.

    ``image_bgr`` is the page's picture, ``region_id_map`` and
    ``region_color`` its regions (see ``regions.build_regions``) and
    ``palette_bgr`` their colors. The regions with more than half their
    pixels in ``where`` are the ones settled; the others keep their colors,
    though one can gain a region that joins it.

    Each of those regions first takes the palette color its pixels are
    nearest, summed over them; on a tie it keeps the color it has. Then,
    while some region would be less than ``MIN_STEP_DE00`` a pixel further
    from the picture in a neighbor's color, the one with the least to lose
    joins that neighbor and takes its color. Two that have joined count as
    one from then on, by all their pixels. A region that joins one lying
    mostly outside ``where`` has that region's color and is looked at no
    more. Neighbors are 8-connected, as everywhere in the region stage. On a
    tie the region with the lowest id goes first, to another of the regions
    being settled before one that isn't, by lowest id among the first and by
    lowest color among the rest.

    No two neighbors are left sharing a color anywhere on the page: regions
    that come to touch one of their own color become one, under the lowest of
    their ids. No region shrinks or changes shape otherwise, so every part of
    the page a brush could paint before it still can, and no region is
    smaller than it was.

    Returns new arrays: (region_id_map, region_color), as ``build_regions``
    returns them. Pixels in no region (-1) stay in none.
    """
    ids = np.array(region_id_map, dtype=np.int32)  # a copy: the regions are joined in it
    colors = np.array(region_color, dtype=np.int32)
    height, width = ids.shape
    where = np.asarray(where, dtype=bool)
    if colors.size == 0 or not where.any():
        return ids, colors

    bounds, pixels, inside = kernels.regions_inside(
        ids.reshape(-1), np.ascontiguousarray(where).reshape(-1), height, width, colors.size
    )
    settled = np.flatnonzero(2 * inside > pixels)  # the regions mostly in ``where``, by id
    if settled.size == 0:
        return ids, colors

    # Only the box around those regions is read, a pixel wider, so that their neighbors show.
    x0, y0 = max(int(bounds[settled, 0].min()) - 1, 0), max(int(bounds[settled, 1].min()) - 1, 0)
    x1, y1 = min(int(bounds[settled, 2].max()) + 2, width), min(int(bounds[settled, 3].max()) + 2, height)
    slot = np.full(colors.size, -1, dtype=np.int32)  # a region's place among the settled ones
    slot[settled] = np.arange(settled.size, dtype=np.int32)
    cells, held, touching, beside = kernels.tone_census(
        ids.reshape(-1),
        np.ascontiguousarray(image_bgr, dtype=np.uint8).reshape(-1, 3),
        slot,
        colors,
        width,
        x0,
        y0,
        x1,
        y1,
        _CELL_BITS,
        int(settled.size),
        len(palette_bgr),
    )
    error = held.astype(np.float64) @ _cell_distances(cells, palette_bgr)
    settled_colors = _settle(error, pixels[settled].astype(np.float64), colors[settled], touching, beside)

    if np.array_equal(settled_colors, colors[settled]):
        return ids, colors
    colors[settled] = settled_colors
    # Joining is left to the one rule the region stage has for it: neighbors of one color are one region.
    kernels.merge_same_color_neighbors(ids.reshape(-1), height, width, colors, pixels, True)
    return ids, colors


def _cell_distances(cells: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
    """cells x K float64: CIEDE2000 from the middle of each cell of the color cube to each palette color.

    ``cells`` are cell codes as ``kernels.tone_census`` gives them: a cell's
    three channels' top ``_CELL_BITS`` bits, blue highest.
    """
    shift = 8 - _CELL_BITS
    low = (1 << _CELL_BITS) - 1
    corners = np.stack([cells >> (2 * _CELL_BITS), (cells >> _CELL_BITS) & low, cells & low], axis=1) << shift
    middles_lab = bgr_to_lab((corners + (1 << shift) // 2).astype(np.uint8))
    palette_lab = bgr_to_lab(np.asarray(palette_bgr, dtype=np.uint8).reshape(-1, 3))
    return ciede2000(middles_lab[:, None, :], palette_lab[None, :, :])


def _settle(
    error: np.ndarray, pixels: np.ndarray, colors: np.ndarray, touching: np.ndarray, beside: np.ndarray
) -> np.ndarray:
    """The color each region ends with (see ``settle_tones``); regions that join end with one color.

    ``error`` (NxK) is each region's summed distance to each palette color,
    ``pixels`` (N) its size, ``colors`` (N) its color now, ``touching`` (NxN)
    and ``beside`` (NxK) its neighbors among the regions and the colors of
    its other neighbors (see ``kernels.tone_census``). Nothing passed in is
    changed.
    """
    count = len(colors)
    error, pixels, touching, beside = error.copy(), pixels.copy(), touching.copy(), beside.copy()
    rows = np.arange(count)
    nearest = error.argmin(axis=1)
    closer = error[rows, nearest] < error[rows, colors]  # on a tie a region keeps the color it has
    colors = np.where(closer, nearest, colors)

    leader = rows.copy()  # the region each has joined, itself until it does
    open_ = np.ones(count, dtype=bool)  # still looked at: not joined to another, nor to a region outside
    for _ in rows:  # every join closes a region, so there are at most as many as regions
        own = error[rows, colors]
        # What a pixel of each region would lose in each neighbor's color: the other settled regions', then the rest's.
        to_region = np.where(touching & open_[:, None] & open_[None, :], error[:, colors] - own[:, None], np.inf)
        to_color = np.where(beside & open_[:, None], error - own[:, None], np.inf)
        step = np.concatenate([to_region, to_color], axis=1) / pixels[:, None]
        best = int(np.argmin(step))
        mover, target = divmod(best, step.shape[1])
        if not step[mover, target] < MIN_STEP_DE00:
            break
        open_[mover] = False
        others = touching[mover].copy()
        touching[mover, :] = touching[:, mover] = False
        if target < count:  # into another settled region: the two are one from here on
            leader[mover] = target
            error[target] += error[mover]
            pixels[target] += pixels[mover]
            touching[target] |= others
            touching[:, target] |= others
            touching[target, target] = False
            beside[target] |= beside[mover]
        else:  # into a region outside: it has that color now, and its neighbors a neighbor of it outside
            colors[mover] = target - count
            beside[others, colors[mover]] = True

    for region in rows:  # a chain of joins ends at the region that has the color
        end = region
        while leader[end] != end:
            end = leader[end]
        colors[region] = colors[end]
    return colors.astype(np.int32)

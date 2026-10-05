"""Texture: fur, foliage and stone as patches with edges a brush can follow.

Where a picture is textured its colors alternate faster than a brush is wide.
The region stage makes paintable regions of that all the same -- pieces too
small are merged by size, parts too thin are handed to the region whose paint
reaches them first (see ``regions.build_regions``) -- but the edge it leaves
between two regions wanders with the last pixels of each: bumps and notches
a round brush cannot paint into. Once no part of a region is too thin, that
is what a page's slivers are made of, and they grow with every region the
page has.

So the edges are settled once more, by a vote. Every pixel within reach of an
edge takes the color for which

    share of the page around it  x  exp(-distance to the pixel's own color / COLOR_STEP)

is largest:

- a color's *share* is Gaussian-weighted, with ``REACH`` of the brush's width
  to one sigma, so two thirds of the weight lies under the brush itself: the
  page about as far as a brush at the pixel reaches. On its own it would be a
  majority vote, which rounds an edge off wherever it bends more sharply than
  the brush and leaves a straight one where it is;
- the *distance* is between the pixel's color in the picture and the palette
  color, in L*a*b*. It is what keeps an edge on the picture's own: where the
  picture steps from one color to the other the pixels on each side suit
  their own color far better, and the edge stays on the step; in fur, where
  they suit both nearly alike, the share decides and the edge runs smooth.

A pixel can only take a color painted within reach of it, so an edge moves
at most that far and no color appears where there was none. The regions are
then rebuilt from the result by the region stage itself, which keeps every
promise it makes: nothing smaller than the difficulty allows, nothing a brush
cannot paint, no two neighbors of one color.
"""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.color import MIN_PALETTE_DE00
from tessellatum.core.regions import build_regions

# One sigma of the vote's weights, as a share of the brush's width: 1 mm for the 3 mm brush. Within the brush's own
# radius, a sigma and a half, lies 68% of the weight.
REACH = 1 / 3

# The weights stop at this many sigmas, where they are down to 1% of the middle's.
_CUTOFF_SIGMAS = 3.0

# How far a color may be from a pixel's own, in L*a*b*, before it counts e times less in the pixel's vote: the margin
# the palette keeps between any two of its colors (see ``color``), so a color a whole step of the palette nearer the
# pixel outweighs one with up to e times its share.
COLOR_STEP = MIN_PALETTE_DE00


def smooth_regions(
    image_bgr: np.ndarray,
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    palette_bgr: np.ndarray,
    min_area_px: int,
    min_width_px: float,
    detail: np.ndarray | None = None,
    settling: float = 1.0,
    color_step: float = COLOR_STEP,
    detail_weight: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """The page's regions with their edges settled by the vote, rebuilt (see the module's docstring).

    ``image_bgr`` is the page's picture, ``region_id_map`` and
    ``region_color`` its regions as ``regions.build_regions`` returns them,
    and ``palette_bgr`` their colors. ``min_width_px`` is the brush, which
    sets how far the vote reaches -- ``REACH`` of it to one sigma, times
    ``settling``; 0 settles nothing. ``color_step`` is the vote's
    ``COLOR_STEP``. ``min_width_px``, ``min_area_px``, ``detail`` and
    ``detail_weight`` are what the regions were built with, and are rebuilt
    with.

    Returns (region_id_map, region_color) as ``build_regions`` does: the
    arrays passed in if the vote changes nothing, else new ones. Pixels in no
    region (-1) take no part and stay in none.

    Raises ValueError if the picture is not the region map's size, if a
    region on the map has no entry in ``region_color``, if a region's color is
    not one of ``palette_bgr``, or if ``settling`` is negative or
    ``color_step`` not positive.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    colors = np.asarray(region_color, dtype=np.int32)
    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    # The kernel reads where these arrays send it, unchecked: they are checked here.
    if image.shape != ids.shape + (3,):
        raise ValueError(f"picture {image.shape} must be the region map's size, {ids.shape}")
    if ids.size and int(ids.max()) >= colors.size:
        raise ValueError(f"region {int(ids.max())} is on the map, which has colors for {colors.size} regions")
    if colors.size and not 0 <= int(colors.min()) <= int(colors.max()) < len(palette_bgr):
        low, high = int(colors.min()), int(colors.max())
        raise ValueError(f"region colors run from {low} to {high}, the palette has {len(palette_bgr)}")
    if settling < 0 or not color_step > 0:
        raise ValueError(f"the vote needs a reach of at least 0 and a color step above 0, not {settling}, {color_step}")
    if colors.size == 0 or min_width_px <= 0 or settling == 0:
        return region_id_map, region_color

    inside = ids >= 0
    labels = np.where(inside, colors[np.where(inside, ids, 0)], -1).astype(np.int32)
    voted = settle_edges(image, labels, palette_bgr, REACH * min_width_px * settling, color_step)
    if np.array_equal(voted, labels):
        return region_id_map, region_color
    # Off the palette is how the region stage is told a pixel is in no region.
    return build_regions(
        np.where(inside, voted, len(palette_bgr)), len(palette_bgr), min_area_px, min_width_px, detail, detail_weight
    )


def settle_edges(
    image_bgr: np.ndarray, labels: np.ndarray, palette_bgr: np.ndarray, sigma_px: float, color_step: float = COLOR_STEP
) -> np.ndarray:
    """``labels`` (HxW int32: each pixel's palette color, -1 for none) after one vote with weights ``sigma_px`` wide.

    Every pixel takes the color with the largest Gaussian-weighted share of
    the pixels within three sigmas of it, rows and columns, times
    ``exp(-d / color_step)``, where ``d`` is the L*a*b* distance from the
    pixel's color in ``image_bgr`` to that color of ``palette_bgr``. Pixels
    off the page and pixels without a color count for nobody. On a tie a
    pixel keeps its color if it is among the best, else takes the lowest.
    Every pixel votes on the page as it was, so the result does not depend on
    the order they are taken in. Returns a new array.

    Raises ValueError if the picture is not the map's size, if a label is
    past the palette, or if ``sigma_px`` or ``color_step`` is not positive.
    """
    labels = np.asarray(labels)
    image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    palette = np.ascontiguousarray(palette_bgr, dtype=np.uint8).reshape(-1, 3)
    # The kernel reads where these arrays send it, unchecked: they are checked here.
    if labels.ndim != 2 or image_bgr.shape != labels.shape + (3,):
        raise ValueError(f"picture {image_bgr.shape} must be the map's size, {labels.shape}")
    if labels.size and int(labels.max()) >= len(palette):
        raise ValueError(f"label {int(labels.max())} is on the map, the palette has {len(palette)}")
    if not sigma_px > 0:
        raise ValueError(f"sigma must be positive, not {sigma_px}")
    if not color_step > 0:
        raise ValueError(f"the color step must be positive, not {color_step}")
    height, width = labels.shape
    flat = np.ascontiguousarray(labels, dtype=np.int32).reshape(-1)
    radius = max(1, int(np.ceil(_CUTOFF_SIGMAS * sigma_px)))
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_px) ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])

    # Only a pixel with two colors in its window can change: those within ``radius``, along rows or columns, of a
    # pixel that differs from the one to its right or the one below.
    changes = np.zeros((height, width), dtype=np.uint8)
    across = labels[:, 1:] != labels[:, :-1]
    down = labels[1:, :] != labels[:-1, :]
    changes[:, 1:] |= across
    changes[:, :-1] |= across
    changes[1:, :] |= down
    changes[:-1, :] |= down
    if not changes.any():
        return flat.reshape(height, width).copy()
    window = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    near = np.ascontiguousarray(cv2.dilate(changes, window)).view(bool).reshape(-1)

    ends = np.empty(height * width, dtype=np.int32)
    kernels.run_ends(flat, height, width, ends)
    pixels_lab = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)).reshape(-1, 3)
    colors_lab = cv2.cvtColor(palette.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    out = np.empty(height * width, dtype=np.int32)

    def vote(y_start: int, y_stop: int) -> None:
        kernels.vote_rows(
            flat, ends, near, pixels_lab, colors_lab, cumulative, weights, radius, float(color_step), y_start, y_stop,
            height, width, out,
        )

    parallel.for_each_stripe(vote, height)
    return out.reshape(height, width)

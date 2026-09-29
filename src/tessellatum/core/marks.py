"""Detail marks: the thin dark marks in a face that are too small to paint, printed instead.

A face is recognized by a few small dark marks -- the pupils, the line of the
eyelids and of the lips, a nose's rim, the dots whiskers grow from -- and
they are exactly what the region stage erases: narrower than the brush, they
cannot be regions of their own, so they merge into the skin or fur around
them and the finished painting loses them. So inside the faces the pipeline
finds (see ``faces``), such marks are printed on the page, solid, as line
art's ink is: part of the picture, not something to paint.

A mark is dark against what lies around it, and thin: the lightness around a
pixel is the image's morphological closing by a disk just wider than the
brush (see ``print_size``), which fills in the dark parts no wider than the
brush with the lighter colors beside them. A pixel is part of a mark where it
is at least ``MIN_CONTRAST`` darker than that, after the fine grain of the
picture -- film grain, a painting's crackle, single hairs -- is smoothed away,
and no lighter than the paint the page puts over it: a dark mark in a region
painted as dark already shows. Marks shorter than ``MIN_LENGTH_MM`` are specks
of texture and are left out.

Sizes are on the printed page, so a mark is judged by how it prints, whatever
the page's pixel count.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM, PrintScale

# How much darker than the lightness around it, in CIE L*, a pixel must be to be part of a mark.
MIN_CONTRAST = 12.0
# The shortest mark printed: its extent across or down, whichever is longer.
MIN_LENGTH_MM = 2.0
# The picture's lightness is smoothed by a Gaussian this wide (its sigma) before marks are looked for.
SMOOTHING_MM = 0.3


def detail_marks(
    image_bgr: np.ndarray,
    where: np.ndarray,
    region_id_map: np.ndarray,
    region_color: np.ndarray,
    palette_bgr: np.ndarray,
    scale: PrintScale,
) -> np.ndarray:
    """HxW bool: the thin dark marks of ``image_bgr``, a page's picture, inside ``where`` (HxW bool), to print.

    ``region_id_map``, ``region_color`` and ``palette_bgr`` are the page's
    regions and their colors: a mark is only where a region's paint is no
    darker than it. Pixels in no region (-1) are never part of one. ``scale``
    is the page's print scale (see ``print_size.print_scale``).

    Only the box around ``where`` is looked at, with a margin wide enough that
    every pixel in it is judged exactly as it would be over the whole picture.
    """
    marks = np.zeros(where.shape, dtype=bool)
    rows, columns = np.flatnonzero(where.any(axis=1)), np.flatnonzero(where.any(axis=0))
    if rows.size == 0:
        return marks
    sigma = scale.mm_to_px(SMOOTHING_MM)
    blur_reach = math.ceil(4 * sigma)
    # A round element the smallest odd number of pixels across wider than the brush: it fits in no mark as wide as the
    # brush or narrower.
    reach = math.floor((scale.mm_to_px(MIN_PAINTABLE_WIDTH_MM) - 1) / 2) + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    # A pixel's closing reads the smoothed lightness twice the element's reach away, and that the picture a blur's reach
    # further.
    margin = blur_reach + 2 * reach
    height, width = where.shape
    y0, y1 = max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, height)
    x0, x1 = max(int(columns[0]) - margin, 0), min(int(columns[-1]) + 1 + margin, width)

    window = np.ascontiguousarray(image_bgr[y0:y1, x0:x1], dtype=np.float32) / np.float32(255)
    lightness = cv2.cvtColor(window, cv2.COLOR_BGR2Lab)[:, :, 0]
    lightness = cv2.GaussianBlur(lightness, (2 * blur_reach + 1,) * 2, sigma)
    # Off the page counts for nothing: a mark at the page's edge is judged by what the page shows.
    contrast = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, disk)

    ids = region_id_map[y0:y1, x0:x1]
    palette_lightness = cv2.cvtColor(
        np.asarray(palette_bgr, dtype=np.float32).reshape(1, -1, 3) / np.float32(255), cv2.COLOR_BGR2Lab
    )[0, :, 0]
    paint = palette_lightness[np.asarray(region_color)[np.clip(ids, 0, None)]]
    found = where[y0:y1, x0:x1] & (ids >= 0) & (contrast >= MIN_CONTRAST) & (lightness <= paint)

    _count, components, stats, _centroids = cv2.connectedComponentsWithStats(found.view(np.uint8), connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    long_enough = extent >= scale.mm_to_px(MIN_LENGTH_MM)
    long_enough[0] = False  # the background
    marks[y0:y1, x0:x1] = long_enough[components]
    return marks

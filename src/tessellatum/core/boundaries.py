"""The lines of a coloring page: one per boundary between two regions.

A page's lines are the *cracks* between pixels of different regions, not the
outlines of the regions themselves. Tracing them from the region map gives
every boundary a single line that both of its regions share, instead of one
outline per region, which draws every boundary twice.

Coordinates are the page's, with pixel centers at integer positions, so a
crack falls on half-integers: the corner between the first four pixels is
``(0.5, 0.5)``.
"""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels

# How far a line may be moved to drop a corner from it, in pixels. The staircase
# of a pixel crack is not a line anyone drew: a step of one pixel is smaller than
# the brush, the paper's grain and the printer's dot, and keeping it only makes
# the page look ragged. T2.4 replaces this with real smoothing.
SIMPLIFY_EPSILON_PX = 1.2


def crack_edges(region_id_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The crack graph of ``region_id_map``, on the grid of pixel corners.

    Corner ``c = i * (width + 1) + j`` is the point ``(j - 0.5, i - 0.5)``.
    ``right[c]`` and ``down[c]`` say whether a crack edge joins it to the
    corner on its right or below it, which is the case when the two pixels
    that edge separates belong to different regions. Off the page counts as a
    region of its own, so the page edge is a boundary like any other and every
    region comes out enclosed; pixels in no region (id -1) count as one more.

    ``degree[c]`` is how many of the four edges around a corner exist. It is
    never 1: going round the corner, the four pixels either all match (0), or
    change back and forth an even number of times (2 or 4), or hold three
    different regions, which always makes 3 or 4.

    Returns ``(right, down, degree, num_edges)``, the first three flat arrays
    over the ``(height + 1) * (width + 1)`` corners.
    """
    ids = np.asarray(region_id_map)
    h, w = ids.shape
    right = np.zeros((h + 1, w + 1), dtype=bool)
    down = np.zeros((h + 1, w + 1), dtype=bool)
    # Column w of `right` and row h of `down` stay False: those edges would
    # leave the grid. kernels._crack_edge relies on that when it looks left
    # and up from a corner.
    np.not_equal(ids[:-1, :], ids[1:, :], out=right[1:h, :w])
    right[0, :w] = True  # the page edge above the first row of pixels
    right[h, :w] = True
    np.not_equal(ids[:, :-1], ids[:, 1:], out=down[:h, 1:w])
    down[:h, 0] = True
    down[:h, w] = True

    degree = right.astype(np.int8) + down
    degree[:, 1:] += right[:, :w]  # the edge left of a corner is the one right of its neighbor
    degree[1:, :] += down[:h, :]
    return right.reshape(-1), down.reshape(-1), degree.reshape(-1), int(right.sum() + down.sum())


def trace_boundaries(region_id_map: np.ndarray, simplify_px: float = SIMPLIFY_EPSILON_PX) -> list[np.ndarray]:
    """The page's lines: one polyline per boundary between two regions.

    Each line runs along the crack between two regions from one junction --
    where three regions meet, or two meet the page edge -- to the next, so
    the two regions share exactly the same geometry and no boundary is drawn
    twice or left out. A boundary that meets no junction comes back as a
    closed line, repeating its first point at the end.

    ``simplify_px`` drops the corners of the pixel staircase with
    Douglas-Peucker, which never moves a line further than that and keeps
    every junction where it is, so the lines still meet (see ``_simplify``).

    Returns Nx2 float64 ``(x, y)`` arrays in page coordinates.
    """
    ids = np.asarray(region_id_map)
    if ids.size == 0:
        return []
    height, width = ids.shape
    stride = width + 1
    right, down, degree, num_edges = crack_edges(ids)
    if num_edges == 0:
        return []
    corners, starts = kernels.trace_boundary_paths(right, down, degree, stride, num_edges)

    # Every corner of every path at once: the per-path work is then a slice.
    points = np.empty((corners.size, 2), dtype=np.float32)
    rows, columns = np.divmod(corners, stride)
    points[:, 0] = columns - 0.5
    points[:, 1] = rows - 0.5

    return [_simplify(points[begin:end], simplify_px) for begin, end in zip(starts[:-1], starts[1:])]


def _simplify(path: np.ndarray, epsilon: float) -> np.ndarray:
    """``path`` with the corners of the pixel staircase dropped, both ends kept.

    Douglas-Peucker keeps the first and last point of an open line, which is
    what holds the lines that meet at a junction together. A closed line has
    the same point at both ends; ``approxPolyDP`` measures that line from its
    first point instead of from a chord, and returns it without the repeat,
    which has to be put back.

    A closed line short enough to be simplified into a line with no inside is
    kept as it is: a region has to stay a shape, not become a stroke.
    """
    simplified = cv2.approxPolyDP(path, epsilon=epsilon, closed=False).reshape(-1, 2)
    if np.array_equal(path[0], path[-1]):
        simplified = np.vstack([simplified, simplified[:1]])
        if len(np.unique(simplified, axis=0)) < 3:
            simplified = path
    return simplified.astype(np.float64)

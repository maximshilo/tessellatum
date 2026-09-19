"""The lines of a coloring page: one per boundary between two regions.

A page's lines are the *cracks* between pixels of different regions, not the
outlines of the regions themselves. Tracing them from the region map gives
every boundary a single line that both of its regions share, instead of one
outline per region, which draws every boundary twice.

Coordinates are the page's, with pixel centers at integer positions, so a
crack falls on half-integers: the corner between the first four pixels is
``(0.5, 0.5)``. Smoothing then takes the line off its staircase, by up to
``MAX_SHIFT_PX``.
"""

from __future__ import annotations

import numpy as np

from tessellatum.core import kernels
from tessellatum.core.print_size import print_scale

# How far the smoothing may move a line from the crack it is drawn on, in pixels.
# One pixel is the finest step the region map can take, so a wiggle that moving a
# line this far can straighten is the grid's own staircase rather than a shape the
# image asked for. It is also what keeps the page closed: the two pixels a crack
# separates are one pixel deep each, so a line held this close still runs inside
# them, and neither region's paint can run into the other.
MAX_SHIFT_PX = 1.0

# How much the smoothing blurs a line along its length, as the standard deviation of
# the Gaussian it amounts to. The staircase of a pixel crack is not a line anyone drew,
# and keeping it only makes the page look ragged. A wiggle is the grid's own, rather
# than a shape the image asked for, when it is either a single pixel step -- those run
# 2 to 3 px along the line -- or too small for the printed page to show, which is half
# a millimeter: the brush, the paper's grain and the printer's dot are all coarser than
# that. So a line is blurred by whichever of the two is longer (``smoothing_length_px``):
# 2 px at preview size, where half a millimeter is about that anyway, and 4 to 5 px on an
# export, whose finer grid can hold a wiggle the paper still cannot show.
SMOOTHING_MIN_PX = 2.0
SMOOTHING_MM = 0.5

# What the printed ink counts as while boundaries are traced: a region apart from every real one (ids >= 0) and from
# the pixels in no region (-1), such as bare paper the ink encloses.
_INK_ID = -2

# The smallest area a closed line may be left enclosing. A region a pixel or two
# across is smaller than the corridor, so smoothing its outline would pull it shut
# into a stroke with nothing inside to paint; such a line is kept as it was traced.
_MIN_LOOP_AREA_PX = 1.0

# One pass of the smoothing is the kernel [step / 2, 1 - step, step / 2], which leaves
# a wiggle alternating from point to point multiplied by 1 - 2 * step: half a step is
# as far as a pass can go, and it wipes that wiggle out rather than turning it back on
# itself. A pass adds `step` to the variance, in units of the point spacing, which is
# 1 px along a crack path, so a blur of `s` px takes 2 * s**2 passes.
_SMOOTHING_STEP = 0.5


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


def smoothing_length_px(size: tuple[int, int]) -> float:
    """How far to blur a line along its length, on a page of ``size`` (width, height) pixels.

    The longer of a pixel step and what the printed page can show (see
    ``SMOOTHING_MIN_PX`` and ``SMOOTHING_MM``); the printed size of a page
    depends only on its shape, so both previews and exports of one image are
    smoothed to the same thing on paper.
    """
    return max(SMOOTHING_MIN_PX, print_scale(size).mm_to_px(SMOOTHING_MM))


def trace_boundaries(
    region_id_map: np.ndarray,
    smoothing_px: float | None = None,
    max_shift_px: float = MAX_SHIFT_PX,
    ink: np.ndarray | None = None,
) -> list[np.ndarray]:
    """The page's lines: one smooth polyline per boundary between two regions.

    Each line runs along the crack between two regions from one junction --
    where three regions meet, or two meet the page edge -- to the next, so
    the two regions share exactly the same geometry and no boundary is drawn
    twice or left out. A boundary that meets no junction comes back as a
    closed line, repeating its first point at the end.

    ``ink`` (HxW bool) is line art's printed ink, which is a line already: no
    line is drawn along it, and a boundary between two regions ends where it
    meets it, as at any junction.

    The crack itself is a staircase of single pixel steps, which
    ``smooth_boundaries`` takes out without letting a line move further than
    ``max_shift_px`` from it. ``smoothing_px`` defaults to what the page's
    size asks for (``smoothing_length_px``); 0 gives the staircase as traced.

    Returns Nx2 float64 ``(x, y)`` arrays in page coordinates.
    """
    ids = np.asarray(region_id_map)
    if ids.size == 0:
        return []
    height, width = ids.shape
    stride = width + 1
    if ink is not None:
        ids = np.where(ink, _INK_ID, ids)  # a region of its own, so that the boundaries along it are paths of their own
    right, down, degree, num_edges = crack_edges(ids)
    if num_edges == 0:
        return []
    corners, starts = kernels.trace_boundary_paths(right, down, degree, stride, num_edges)

    # Every corner of every path at once: the per-path work is then a slice.
    points = np.empty((corners.size, 2), dtype=np.float64)
    rows, columns = np.divmod(corners, stride)
    points[:, 0] = columns - 0.5
    points[:, 1] = rows - 0.5

    paths = [points[begin:end] for begin, end in zip(starts[:-1], starts[1:])]
    if ink is not None:
        along_ink = _along_ink(corners[starts[:-1]], corners[starts[:-1] + 1], stride, ink)
        paths = [path for path, skip in zip(paths, along_ink) if not skip]
    if smoothing_px is None:
        smoothing_px = smoothing_length_px((width, height))
    return smooth_boundaries(paths, smoothing_px, max_shift_px, size=(width, height))


def _along_ink(first: np.ndarray, second: np.ndarray, stride: int, ink: np.ndarray) -> np.ndarray:
    """For each path, given its first two corners, whether it runs along the ink.

    A path separates the same two regions all the way from one junction to
    the next, so its first crack edge says which two they are. The edge
    between corners ``(i, j)`` and ``(i, j + 1)`` separates the pixels
    ``(i - 1, j)`` and ``(i, j)``; the one between ``(i, j)`` and ``(i + 1, j)``
    the pixels ``(i, j - 1)`` and ``(i, j)``. Off the page is not ink.
    """
    padded = np.pad(np.asarray(ink, dtype=bool), 1)  # padded[i + 1, j + 1] is pixel (i, j)
    row0, column0 = np.divmod(first, stride)
    row1, column1 = np.divmod(second, stride)
    across_rows = row0 == row1  # the edge runs along a row of corners, between two rows of pixels
    row = np.minimum(row0, row1)
    column = np.minimum(column0, column1)
    one = np.where(across_rows, padded[row, column + 1], padded[row + 1, column])  # above, or left of, the edge
    other = padded[row + 1, column + 1]
    return one | other


def smooth_boundaries(
    paths: list[np.ndarray],
    smoothing_px: float,
    max_shift_px: float = MAX_SHIFT_PX,
    size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """``paths`` smoothed along their length, each point held near where it started.

    Smoothing a path is moving every point a step of the way towards the
    middle of its two neighbors, over and over: a Gaussian blur of
    ``smoothing_px`` along the line, and the same all the way round a closed
    path, which is the one whose first and last point are the same.

    What holds the page together while that happens:

    - **A path's ends stay put.** They are its junctions, where the other
      boundaries that meet there end too, so the lines go on meeting and the
      page keeps no gap for paint to run through.
    - **No point moves further than ``max_shift_px``** from the crack corner
      it started at, and none is ever dropped. So a line stays inside the two
      pixels whose boundary it draws, the page stays closed, and a wiggle
      deeper than a pixel -- which the region map really does have -- stays on
      the page instead of being smoothed away.
    - **The page edge stays put too**, when ``size`` (width, height) says
      where it is. The paper's edge is straight, and the page's corners are
      corners, not staircase steps to round off.
    - **A closed line keeps its inside.** One round a region a pixel or two
      across would be pulled shut, leaving nothing to paint, so it is kept as
      it was traced (``_MIN_LOOP_AREA_PX``).

    Returns new Nx2 float64 arrays; ``smoothing_px=0`` returns ``paths``.
    """
    passes = int(round(smoothing_px**2 / _SMOOTHING_STEP))
    if not paths or passes < 1:
        return paths
    points, previous, following, movable = _path_neighbors(paths, size)
    start = points.copy()
    for _ in range(passes):
        middle = 0.5 * (points[previous] + points[following])
        points = np.where(movable, points + _SMOOTHING_STEP * (middle - points), points)
        shift = points - start
        # sqrt of the sum of squares rather than hypot: both are exact to the last
        # bit whatever computes them, and numpy's hypot and the C library's are not
        # the same to the last bit, which is enough to move a point.
        distance = np.sqrt(shift[:, 0] ** 2 + shift[:, 1] ** 2)
        too_far = distance > max_shift_px
        if too_far.any():
            points[too_far] = start[too_far] + shift[too_far] * (max_shift_px / distance[too_far])[:, None]
    smoothed = _split_paths(points, paths)
    return [
        crack if _is_closed(crack) and _enclosed_area(smooth) < _MIN_LOOP_AREA_PX else smooth
        for crack, smooth in zip(paths, smoothed)
    ]


def _path_neighbors(
    paths: list[np.ndarray], size: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every path's points in one array, plus each point's two neighbors and whether it may move.

    A closed path is held without the repeat of its first point, so that its
    points form a ring; an open path's first and last point are its junctions,
    which stay where they are and are their own neighbors.
    """
    closed = np.array([_is_closed(path) for path in paths])
    counts = np.array([len(path) - 1 if shut else len(path) for path, shut in zip(paths, closed)])
    points = np.concatenate([path[:-1] if shut else path for path, shut in zip(paths, closed)])
    first = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    count = np.repeat(counts, counts)
    ring = np.repeat(closed, counts)
    at = np.arange(len(points)) - first
    previous = first + np.where(ring, (at - 1) % count, np.maximum(at - 1, 0))
    following = first + np.where(ring, (at + 1) % count, np.minimum(at + 1, count - 1))
    movable = ring | ((at > 0) & (at < count - 1))
    if size is not None:
        width, height = size
        x, y = points[:, 0], points[:, 1]
        movable &= (x > -0.5) & (y > -0.5) & (x < width - 0.5) & (y < height - 0.5)
    return points, previous, following, movable[:, None]


def _split_paths(points: np.ndarray, paths: list[np.ndarray]) -> list[np.ndarray]:
    """``points`` cut back into one array per path, with a closed path's repeat put back."""
    smoothed = []
    at = 0
    for path in paths:
        shut = _is_closed(path)
        piece = points[at : at + len(path) - shut]
        at += len(piece)
        smoothed.append(np.vstack([piece, piece[:1]]) if shut else piece)
    return smoothed


def _is_closed(path: np.ndarray) -> bool:
    """Does ``path`` come back to its first point, having met no junction at all?"""
    return bool(len(path) > 2 and np.array_equal(path[0], path[-1]))


def _enclosed_area(loop: np.ndarray) -> float:
    """How much a closed line encloses, by the shoelace formula."""
    x, y = loop[:-1, 0], loop[:-1, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2

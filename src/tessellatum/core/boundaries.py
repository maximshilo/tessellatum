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

# How far the smoothing may move a line from the crack it is drawn on, in pixels.
# One pixel is the finest step the region map can take, so a wiggle that moving a
# line this far can straighten is the grid's own staircase rather than a shape the
# image asked for. It is also what keeps the page closed: the line stays inside the
# two pixels whose boundary it is, so neither region's paint can run into the other.
MAX_SHIFT_PX = 1.0

# How much the smoothing blurs a line along its length, as the standard deviation of
# the Gaussian it amounts to. The staircase of a pixel crack is not a line anyone
# drew -- a step of one pixel is smaller than the brush, the paper's grain and the
# printer's dot, and keeping it only makes the page look ragged. Its wiggles are 2 to
# 3 px long, which 2 px of smoothing flattens; more changes nothing, because the
# corridor above binds first.
SMOOTHING_PX = 2.0

# The smallest area a closed line may be left enclosing. A region a pixel or two
# across is smaller than the corridor, so smoothing its outline would pull it shut
# into a stroke with nothing inside to paint; such a line is kept as it was traced.
_MIN_LOOP_AREA_PX = 1.0

# One pass of the smoothing is the kernel [step / 2, 1 - step, step / 2], whose
# variance is `step` in units of the point spacing, which is 1 px along a crack path.
# So n passes blur by sqrt(n * step), and half a pixel per pass -- the most a pass can
# take without amplifying the finest wiggle of all -- makes that 2 * SMOOTHING_PX**2.
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


def trace_boundaries(
    region_id_map: np.ndarray, smoothing_px: float = SMOOTHING_PX, max_shift_px: float = MAX_SHIFT_PX
) -> list[np.ndarray]:
    """The page's lines: one smooth polyline per boundary between two regions.

    Each line runs along the crack between two regions from one junction --
    where three regions meet, or two meet the page edge -- to the next, so
    the two regions share exactly the same geometry and no boundary is drawn
    twice or left out. A boundary that meets no junction comes back as a
    closed line, repeating its first point at the end.

    The crack itself is a staircase of single pixel steps, which
    ``smooth_boundaries`` takes out without letting a line move further than
    ``max_shift_px`` from it. ``smoothing_px=0`` gives the staircase as traced.

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
    points = np.empty((corners.size, 2), dtype=np.float64)
    rows, columns = np.divmod(corners, stride)
    points[:, 0] = columns - 0.5
    points[:, 1] = rows - 0.5

    paths = [points[begin:end] for begin, end in zip(starts[:-1], starts[1:])]
    return smooth_boundaries(paths, smoothing_px, max_shift_px, size=(width, height))


def smooth_boundaries(
    paths: list[np.ndarray],
    smoothing_px: float = SMOOTHING_PX,
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

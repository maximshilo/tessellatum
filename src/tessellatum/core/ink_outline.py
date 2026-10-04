"""The printed ink's outline, for drawing it as vector art: traced round its pixels, then smoothed off their staircase
without changing what the ink joins or keeps apart.

The page prints its ink -- line art's own, the thin dark marks in a face --
pixel by pixel, and the PDF draws that ink again as a filled outline (see
``export``). (The letters of signs and captions have an outline of their own,
traced between the pixels: see ``text.lettering``.) Traced exactly along the cracks between its
pixels, the outline prints their staircase: 0.21-0.23 mm steps on the
benchmark's two scans. Smoothed along its length, as the page's lines are (see
``boundaries``), the staircase goes; but each edge of a thin stroke then moves
on its own, so a scan's hatching thickens into blobs and strokes a pixel wide
break. So the outline is smoothed under rules that make both impossible:

- **Every point stays in a box round the pixel corner it was traced from**,
  short of the four pixel centers round that corner by a margin. So every
  pixel center stays on its own side of the outline -- read at the page's own
  pixels, the smooth ink is the page's ink -- and no two edges can cross: an
  edge only ever lies in the boxes of two neighboring corners, and those hold
  no pixel center and no other edge but the ones meeting it there.
- **Two ink pixels meeting only at a corner stay one piece**, as the page
  shows them: the outline turns towards the paper there, and each of its two
  visits to the corner keeps to its own paper pixel's side of the line
  between the two ink centers. The joint opens into a waist instead of
  pinching to a point.
- **A stroke or gap a pixel wide keeps its width**: round a thin pixel --
  ink with paper on two opposite sides, or paper with ink on two -- the
  margin is ``THIN_MARGIN_MM``, so such strokes and gaps stay twice that wide
  on paper. Elsewhere it is ``MARGIN_MM``. On a page finer than about 225 dpi
  a pixel is too narrow for that: there the margins stop at
  ``MAX_MARGIN_PX``, and such strokes and gaps keep most of a pixel instead.

Within those boxes the outline is smoothed along its length by a Gaussian as
long as the page's lines' (``boundaries.smoothing_length_px``), every ring is
offset back to the area of its pixels -- a closed curve shrinks as it is
smoothed, and a speck would shrink to nothing -- and then simplified within
``TOLERANCE_MM``, which the margins allow for. Rings round a few pixels, specks
and pinholes, get a point halfway along each side first, so that they come out
round.
"""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels
from tessellatum.core.boundaries import SMOOTHING_MIN_PX, SMOOTHING_MM
from tessellatum.core.print_size import print_scale

# How far a pixel center stays from the outline where the ink or the paper is a pixel thin, on paper: strokes and gaps
# a pixel wide stay at least twice as wide. Half of that and they printed merged or broken at 300 dpi (Q40, D-055).
THIN_MARGIN_MM = 0.04
# How far every other pixel center stays from it.
MARGIN_MM = 0.01
# How far the simplified outline may lie from the smoothed one: a quarter of a 1200 dpi dot. The margins hold it.
TOLERANCE_MM = 0.005
# The largest margin, in pixels, however fine the page: a point keeps at least a tenth of a pixel to move in each way.
# It holds on pages finer than about 225 dpi, where the thin margin would be more. At 300 dpi a pinhole whose four
# sides are ink touching only at its corners shrinks to about a third of its pixel, and prints filled (T7.4's review).
MAX_MARGIN_PX = 0.4

# Rings of this many corners or fewer -- a pixel or a few -- get a point halfway along each side, so they come out
# round rather than as a diamond of their four corners.
_SMALL_RING = 12
# Rounds of offsetting a ring back to its pixels' area, each one clipped to the boxes again.
_AREA_STEPS = 4
# One pass of the smoothing moves each point half way to the middle of its neighbors (see ``boundaries``).
_STEP = 0.5
# At a diagonal joint, how much of the room its box gives the waist and the paper beyond it may take: what is left
# keeps the half-planes and the box from pinning a point to a single spot.
_ROOM = 0.95
# How many pixels of paper round the ink the work is done with: enough for every margin to see what it looks at.
_AROUND = 3


def traced_rings(mask: np.ndarray) -> list[np.ndarray]:
    """The outline of ``mask``'s pixels, exactly along the cracks round them.

    Closed rings, the ink on their right as seen on the page, each repeating
    its first point at the end: filled by the even-odd rule (or the nonzero
    one -- the rings never cross) they cover exactly the pixels of ``mask``.
    In page coordinates, pixel centers at whole numbers.
    """
    points, _, starts = _trace(mask)
    return [np.vstack([points[a:b], points[a : a + 1]]) for a, b in zip(starts[:-1], starts[1:])]


def ink_outline(mask: np.ndarray, px_per_mm: float | None = None) -> list[np.ndarray]:
    """The outline of the ink ``mask`` (HxW bool) prints, smoothed off its pixels' staircase (see the module).

    Closed rings, as ``traced_rings`` gives them, filled by the even-odd rule.
    ``px_per_mm`` is how many of the page's pixels print a millimeter wide;
    by default, as many as the print model prints a page of the mask's size at.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    height, width = mask.shape
    if px_per_mm is None:
        px_per_mm = print_scale((width, height)).px_per_mm
    tolerance = TOLERANCE_MM * px_per_mm
    thin = min(MAX_MARGIN_PX, THIN_MARGIN_MM * px_per_mm + tolerance)
    base = min(MAX_MARGIN_PX, MARGIN_MM * px_per_mm + tolerance)
    smoothing_px = max(SMOOTHING_MIN_PX, SMOOTHING_MM * px_per_mm)

    # Only the ink and the paper a few pixels round it decide anything -- a pixel's margin looks at the pixels beside
    # it, a corner's at the corners beside it -- so the work is done in that window: a scan's whole page, a few
    # millimeters round a face's marks.
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    y0, y1 = max(0, rows[0] - _AROUND), min(height, rows[-1] + 1 + _AROUND)
    x0, x1 = max(0, columns[0] - _AROUND), min(width, columns[-1] + 1 + _AROUND)
    mask = mask[y0:y1, x0:x1]
    page_edge = (x0 == 0, y0 == 0, x1 == width, y1 == height)  # which of the window's sides are the page's

    corners, joints, starts = _trace(mask)
    counts = np.diff(starts)
    ring_of = np.repeat(np.arange(counts.size), counts)
    following = starts[:-1][ring_of] + (np.arange(corners.shape[0]) - starts[:-1][ring_of] + 1) % counts[ring_of]
    target = _ring_areas(corners, following, ring_of, counts.size)

    # Small rings get a point halfway along each side, placed right after the corner the side starts at.
    split = (counts <= _SMALL_RING)[ring_of]
    taken = 1 + split.astype(np.int64)  # how many points each corner becomes
    corner_at = np.cumsum(taken) - taken
    halfway_at = corner_at[split] + 1
    origin = np.empty((corners.shape[0] + int(split.sum()), 2))
    origin[corner_at] = corners
    origin[halfway_at] = 0.5 * (corners[split] + corners[following[split]])
    new_counts = counts * (1 + (counts <= _SMALL_RING))
    new_ring_of = np.repeat(np.arange(counts.size), new_counts)
    new_first = np.repeat(np.cumsum(new_counts) - new_counts, new_counts)
    at = np.arange(origin.shape[0]) - new_first
    previous = new_first + (at - 1) % new_counts[new_ring_of]
    after = new_first + (at + 1) % new_counts[new_ring_of]

    lo, hi = _boxes(mask, origin, corner_at, halfway_at, corners, joints, split, following, thin, base, page_edge)
    holds = _joint_holds(origin, corner_at, halfway_at, corners, joints, split, following, lo, hi, thin)

    def hold(points: np.ndarray) -> None:
        np.clip(points, lo, hi, out=points)
        for index, joint, normal, least in holds:
            off = np.einsum("ij,ij->i", points[index] - joint, normal)
            short = off < least
            if short.any():
                k = index[short]
                points[k] = np.clip(points[k] + (least[short] - off[short])[:, None] * normal[short], lo[k], hi[k])

    points = origin.copy()
    for _ in range(int(round(smoothing_px**2 / _STEP))):
        points += _STEP * (0.5 * (points[previous] + points[after]) - points)
        hold(points)
    for _ in range(_AREA_STEPS):
        area = _ring_areas(points, after, new_ring_of, counts.size)
        length = np.bincount(new_ring_of, np.hypot(*(points[after] - points).T), counts.size)
        offset = (target - area) / np.maximum(length, 1e-9)
        tangent = points[after] - points[previous]
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
        # Away from the ink, which lies on the right: growing a ring's ink, or shrinking the paper a hole ring holds.
        points += offset[new_ring_of][:, None] * np.column_stack([tangent[:, 1], -tangent[:, 0]])
        hold(points)

    rings = []
    for begin, end in zip(np.cumsum(new_counts) - new_counts, np.cumsum(new_counts)):
        ring = points[begin:end]
        kept = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), tolerance, True).reshape(-1, 2)
        if len(kept) >= 3:
            ring = kept.astype(np.float64)
        rings.append(np.vstack([ring, ring[:1]]) + (x0, y0))
    return rings


def thin_pixels(mask: np.ndarray) -> np.ndarray:
    """The pixels a pixel thick: ink with paper on two opposite sides, or paper with ink on two opposite sides.

    Opposite across, down, or along either diagonal; off the page is paper.
    """
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    middle = padded[1:-1, 1:-1]
    pairs = [
        (padded[:-2, 1:-1], padded[2:, 1:-1]),
        (padded[1:-1, :-2], padded[1:-1, 2:]),
        (padded[:-2, :-2], padded[2:, 2:]),
        (padded[:-2, 2:], padded[2:, :-2]),
    ]
    thin_ink = np.zeros_like(middle)
    thin_paper = np.zeros_like(middle)
    for one, other in pairs:
        thin_ink |= ~one & ~other
        thin_paper |= one & other
    return (middle & thin_ink) | (~middle & thin_paper)


def _trace(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``kernels.trace_ink_rings`` on ``mask``: every ring's corners as points, their joints, and where each ring starts."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    corners, joints, starts = kernels.trace_ink_rings(np.pad(mask, 1).reshape(-1), height, width)
    rows, columns = np.divmod(corners, width + 1)
    return np.column_stack([columns - 0.5, rows - 0.5]), joints, starts


def _ring_areas(points: np.ndarray, after: np.ndarray, ring_of: np.ndarray, count: int) -> np.ndarray:
    """Each ring's signed area, by the shoelace formula: positive round ink, negative round a hole in it."""
    cross = points[:, 0] * points[after, 1] - points[after, 0] * points[:, 1]
    return np.bincount(ring_of, cross, count) / 2


def _corner_margins(mask: np.ndarray, thin: float, base: float) -> np.ndarray:
    """Each pixel corner's margin, (H + 1) x (W + 1): ``thin`` where one of its four pixels is thin, ``base`` elsewhere.

    Two parts of the outline that pass a pixel apart elsewhere -- diagonally,
    round a pixel that is not thin -- are concave corners both, which the
    smoothing moves apart: holding them by the thin margin too changed no
    closest approach on the benchmark's pages (T7.4).
    """
    thin_pixel = np.pad(thin_pixels(mask), 1)  # corner (i, j) touches thin_pixel[i:i + 2, j:j + 2]
    near_thin = thin_pixel[:-1, :-1] | thin_pixel[:-1, 1:] | thin_pixel[1:, :-1] | thin_pixel[1:, 1:]
    return np.where(near_thin, thin, base)


def _boxes(mask, origin, corner_at, halfway_at, corners, joints, split, following, thin, base, page_edge):
    """Each point's box, ``(lo, hi)``: where it may move.

    A corner's reaches to its margin from the four pixel centers round it; a
    point halfway along a side, from the two either side of that side, by the
    larger of its two corners' margins. A visit to a diagonal joint reaches
    towards its own paper pixel as far as that pixel's margin allows, and
    the other way as far as the base margin does: the two ink centers on the
    diagonal are kept clear by a half-plane instead (see ``_joint_holds``).
    Where that paper pixel is thin too -- a diagonal stroke beside a diagonal
    gap -- the joint's half-width and the pixel's margin ask for more than the
    0.707 px between the ink's diagonal and that pixel's center, and both give
    way alike. Points on the page edge stay where they are -- the paper's
    edge is straight -- where ``page_edge`` (left, top, right, bottom) says
    the mask's side is the page's.
    """
    height, width = mask.shape
    margins = _corner_margins(mask, thin, base)
    column, row = np.rint(corners + 0.5).astype(np.int64).T
    corner_margin = margins[row, column]
    margin = np.empty(origin.shape[0])
    margin[corner_at] = corner_margin
    margin[halfway_at] = np.maximum(corner_margin[split], corner_margin[following[split]])
    half = (0.5 - margin)[:, None]
    lo, hi = origin - half, origin + half

    visits = np.flatnonzero(joints.any(axis=1))
    if visits.size:
        towards = joints[visits].astype(np.float64)
        paper_column, paper_row = np.rint(corners[visits] + 0.5 * towards).astype(np.int64).T
        paper_margin = np.where(thin_pixels(mask)[paper_row, paper_column], thin, base)
        share = np.minimum(1.0, _ROOM * np.sqrt(0.5) / (thin + np.sqrt(2.0) * paper_margin))
        near = (0.5 - paper_margin * share)[:, None]
        far = 0.5 - base
        index = corner_at[visits]
        lo[index] = origin[index] - np.where(towards < 0, near, far)
        hi[index] = origin[index] + np.where(towards > 0, near, far)

    left, top, right, bottom = page_edge
    x, y = origin.T
    edge = (left & (x <= -0.5)) | (top & (y <= -0.5)) | (right & (x >= width - 0.5)) | (bottom & (y >= height - 0.5))
    lo[edge] = hi[edge] = origin[edge]
    return lo, hi


def _joint_holds(origin, corner_at, halfway_at, corners, joints, split, following, lo, hi, thin):
    """The half-planes that keep each diagonal joint inked: ``(points, joint corner, normal, least offset)`` sets.

    A visit to a joint stays at least ``thin`` on its own paper pixel's side
    of the line through the two ink centers -- as far as its box lets it --
    and so does a point halfway along either side it arrives or leaves by,
    whose box reaches across that line. A halfway point between two joints
    gets one hold from each, in two sets, so no set moves a point twice.
    """
    joint = joints.any(axis=1)
    normal = joints.astype(np.float64) / np.sqrt(2.0)
    halfway_of = np.full(corners.shape[0], -1)
    halfway_of[split] = halfway_at  # the halfway point after each corner
    sets = []
    for index, of in (
        (np.concatenate([corner_at[joint], halfway_of[joint & split]]), np.concatenate([np.flatnonzero(joint), np.flatnonzero(joint & split)])),
        (halfway_of[split & joint[following]], following[split & joint[following]]),
    ):
        if index.size == 0:
            continue
        at = corners[of]
        room = np.einsum("ij,ij->i", np.where(normal[of] > 0, hi[index], lo[index]) - at, normal[of])
        sets.append((index, at, normal[of], np.minimum(thin, _ROOM * room)))
    return sets

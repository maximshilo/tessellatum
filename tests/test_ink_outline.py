"""The printed ink's outline for the vector PDF: traced round its pixels, smoothed without joining or breaking anything."""

import cv2
import numpy as np
import pytest

from tessellatum.core.ink_outline import (
    MAX_MARGIN_PX,
    THIN_MARGIN_MM,
    TOLERANCE_MM,
    ink_outline,
    thin_pixels,
    traced_rings,
)

from pdf_reading import inside_even_odd


def _segments(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every edge of every ring: its two ends, its ring and its place along it."""
    a = np.concatenate([ring[:-1] for ring in rings])
    b = np.concatenate([ring[1:] for ring in rings])
    ring_of = np.concatenate([np.full(len(ring) - 1, k) for k, ring in enumerate(rings)])
    place = np.concatenate([np.arange(len(ring) - 1) for ring in rings])
    return a, b, ring_of, place


def _distance_to_segments(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each point's distance to the nearest of the segments from ``a`` to ``b``."""
    v = b - a
    length = np.maximum((v**2).sum(axis=1), 1e-12)
    u = np.clip(((points[:, None, :] - a[None]) * v[None]).sum(axis=2) / length[None], 0, 1)
    nearest = a[None] + u[..., None] * v[None]
    return np.sqrt(((nearest - points[:, None, :]) ** 2).sum(axis=2)).min(axis=1)


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _crossings(rings: list[np.ndarray]) -> int:
    """How many pairs of edges cross, leaving out pairs that share an end."""
    a, b, ring_of, place = _segments(rings)
    count = len(a)
    found = 0
    for k in range(count):
        p, q = a[k], b[k]
        r, s = a[k + 1 :], b[k + 1 :]
        d1 = _cross(q - p, r - p)
        d2 = _cross(q - p, s - p)
        d3 = _cross(s - r, p - r)
        d4 = _cross(s - r, q - r)
        proper = (d1 * d2 < 0) & (d3 * d4 < 0)
        neighbors = (ring_of[k + 1 :] == ring_of[k]) & (
            (np.abs(place[k + 1 :] - place[k]) == 1) | (np.abs(place[k + 1 :] - place[k]) == np.bincount(ring_of)[ring_of[k]] - 1)
        )
        found += int((proper & ~neighbors).sum())
    return found


def _random_mask(seed: int, shape=(36, 44)) -> np.ndarray:
    """Specks, pinholes, strokes a pixel wide, pixels touching only at a corner, and solid patches, on and off the edge."""
    rng = np.random.default_rng(seed)
    kind = seed % 4
    if kind == 0:
        return rng.random(shape) < 0.5  # noise: every configuration of a pixel's neighbors
    if kind == 1:
        blobs = cv2.GaussianBlur(rng.random(shape), (0, 0), 1.5)
        return blobs > np.median(blobs)
    if kind == 2:
        mask = rng.random(shape) < 0.25
        return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~mask  # rings round holes
    mask = np.zeros(shape, dtype=bool)
    for _ in range(12):  # strokes a pixel wide at random slopes, crossing
        x0, y0, x1, y1 = rng.integers(0, max(shape), 4)
        cv2.line(mask.view(np.uint8), (int(x0), int(y0)), (int(x1), int(y1)), 1, 1, cv2.LINE_8)
    return mask


@pytest.mark.parametrize("seed", range(8))
def test_traced_rings_filled_even_odd_are_the_mask_exactly(seed):
    mask = _random_mask(seed)
    rings = traced_rings(mask)
    np.testing.assert_array_equal(inside_even_odd(rings, mask.shape), mask)
    # Each crack round the mask is on one ring, once, along the cracks between pixels.
    padded = np.pad(mask, 1)
    sides = np.count_nonzero(padded[1:, :] != padded[:-1, :]) + np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    assert sum(len(ring) - 1 for ring in rings) == sides
    for ring in rings:
        step = np.abs(np.diff(ring, axis=0))
        assert np.all(step.sum(axis=1) == 1) and np.all((ring + 0.5) % 1 == 0)


@pytest.mark.parametrize(
    "mask, rings",
    [
        (np.zeros((5, 7), dtype=bool), 0),
        (np.ones((5, 7), dtype=bool), 1),
        (np.eye(6, dtype=bool), 1),  # a diagonal of pixels touching corner to corner: one stroke
        (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool), 1),  # a checkerboard: one piece of ink
        (np.pad(np.zeros((3, 3), dtype=bool), 1, constant_values=True), 2),  # a frame round a hole
    ],
    ids=["empty", "full", "diagonal", "checkerboard", "hole"],
)
def test_ink_meeting_only_at_a_corner_is_one_piece(mask, rings):
    traced = traced_rings(mask)
    assert len(traced) == rings
    np.testing.assert_array_equal(inside_even_odd(traced, mask.shape), mask)
    smooth = ink_outline(mask, px_per_mm=4.0)
    assert len(smooth) == rings
    np.testing.assert_array_equal(inside_even_odd(smooth, mask.shape), mask)


@pytest.mark.parametrize("px_per_mm", [2.0, 4.0, 6.0, 12.0])
@pytest.mark.parametrize("seed", range(8))
def test_every_pixel_center_stays_on_its_side_and_no_two_edges_cross(seed, px_per_mm):
    # What makes D-053's failures impossible (T7.4): read at the page's own pixels the smooth ink is the page's ink, and
    # an outline that never crosses itself or another fills to the shapes it draws.
    mask = _random_mask(seed)
    rings = ink_outline(mask, px_per_mm=px_per_mm)
    np.testing.assert_array_equal(inside_even_odd(rings, mask.shape), mask)
    assert _crossings(rings) == 0
    # Every pixel center keeps its margin from the outline: a pixel-thin one the thin margin, but where a diagonal joint
    # and its paper pixel share the 0.707 px between them.
    a, b, _, _ = _segments(rings)
    rows, columns = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    centers = np.column_stack([columns.ravel(), rows.ravel()]).astype(np.float64)
    distance = _distance_to_segments(centers, a, b).reshape(mask.shape)
    thin = min(MAX_MARGIN_PX, (THIN_MARGIN_MM + TOLERANCE_MM) * px_per_mm)
    share = min(1.0, 0.95 * np.sqrt(0.5) / (thin * (1 + np.sqrt(2.0))))
    tolerance = TOLERANCE_MM * px_per_mm
    assert distance[thin_pixels(mask)].min() >= thin * share - tolerance - 1e-6
    assert distance.min() > 0


def test_a_speck_and_a_pinhole_come_out_round_with_their_area():
    mask = np.zeros((9, 15), dtype=bool)
    mask[4, 3] = True  # a speck
    mask[2:7, 8:13] = True
    mask[4, 10] = False  # a pinhole
    rings = ink_outline(mask, px_per_mm=4.0)
    assert len(rings) == 3
    by_center = {tuple(np.rint(ring[:-1].mean(axis=0)).astype(int)): ring for ring in rings if len(ring) < 30}
    for center, area in (((3, 4), 1.0), ((10, 4), -1.0)):
        ring = by_center[center]
        assert len(ring) - 1 >= 6  # more than a diamond of four corners
        radius = np.hypot(*(ring[:-1] - center).T)
        assert radius.max() - radius.min() < 0.1  # round: a regular octagon's corners are 0.08 further out than its sides
        assert _signed_area(ring) == pytest.approx(area, abs=0.02)  # the hole's area is negative: the ink runs round it


def _densified(points: np.ndarray, spacing: float, closed: bool = True) -> np.ndarray:
    """``points`` resampled every ``spacing`` along them (a closed ring keeps its repeated end)."""
    run = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    at = np.arange(0, run[-1], spacing) if closed else np.linspace(0, run[-1], max(2, int(run[-1] / spacing)))
    return np.column_stack([np.interp(at, run, points[:, 0]), np.interp(at, run, points[:, 1])])


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:-1, 0], ring[:-1, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2


@pytest.mark.parametrize("seed", range(4))
def test_the_outline_keeps_the_ink_s_area(seed):
    mask = _random_mask(seed, shape=(60, 80))
    rings = ink_outline(mask, px_per_mm=4.0)
    assert sum(_signed_area(ring) for ring in rings) == pytest.approx(mask.sum(), rel=0.01)


def test_a_staircase_comes_out_straight():
    # A half-plane whose edge climbs a pixel every three: traced, its outline steps a whole pixel; smoothed, it runs
    # straight down the steps' middles, held off them only by the margin.
    height, width = 40, 90
    rows, columns = np.mgrid[0:height, 0:width]
    mask = rows > 10 + columns / 3.0
    px_per_mm = 4.0
    ring = _densified(max(ink_outline(mask, px_per_mm=px_per_mm), key=len), 0.1)  # it is simplified: a straight run is two points
    edge = ring[(ring[:, 0] > 6) & (ring[:, 0] < width - 7) & (ring[:, 1] > 0) & (ring[:, 1] < height - 1)]
    exact = max(traced_rings(mask), key=len)
    exact_edge = exact[(exact[:, 0] > 6) & (exact[:, 0] < width - 7) & (exact[:, 1] > 0) & (exact[:, 1] < height - 1)]
    slope, offset = np.polyfit(exact_edge[:, 0], exact_edge[:, 1], 1)  # the line the steps climb along
    assert slope == pytest.approx(1 / 3, abs=0.01)
    off_line = lambda points: np.abs(points[:, 1] - (slope * points[:, 0] + offset)) * np.cos(np.arctan(slope))  # noqa: E731
    assert off_line(exact_edge).max() > 0.45  # each step's corners, half a pixel off
    assert off_line(edge).max() < 0.2
    # No sharp turn left along it: the staircase turns 90 degrees at every step.
    direction = np.diff(_densified(edge, 0.5, closed=False), axis=0)
    turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(direction[:, 1], direction[:, 0])))))
    assert turn.max() < 30


@pytest.mark.parametrize("px_per_mm", [4.0, 6.0, 12.0])
def test_hatching_a_pixel_apart_stays_apart_and_unbroken(px_per_mm):
    # Strokes a pixel wide a pixel apart, across and diagonal: as many pieces as strokes, and the gaps kept open by at
    # least twice the thin margin (D-053's smoothed outlines ran them together into blobs). Diagonal strokes two pixels
    # apart would touch at their corners, so these are three apart: a gap a pixel wide along the diagonal.
    mask = np.zeros((40, 90), dtype=bool)
    mask[4:36:2, 3:25] = True
    for k in range(0, 24, 3):
        mask[np.arange(4, 36), np.arange(4, 36) + 26 + k] = True
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    rings = ink_outline(mask, px_per_mm=px_per_mm)
    assert len(rings) == labels.max()
    np.testing.assert_array_equal(inside_even_odd(rings, mask.shape), mask)
    thin = min(MAX_MARGIN_PX, (THIN_MARGIN_MM + TOLERANCE_MM) * px_per_mm)
    share = min(1.0, 0.95 * np.sqrt(0.5) / (thin * (1 + np.sqrt(2.0))))
    gap = 2 * (thin * share - TOLERANCE_MM * px_per_mm)
    for k, ring in enumerate(rings):
        others = [other for j, other in enumerate(rings) if j != k]
        a, b, _, _ = _segments(others)
        assert _distance_to_segments(ring[:-1], a, b).min() >= gap - 1e-6


def test_the_page_edge_stays_put():
    mask = np.zeros((20, 30), dtype=bool)
    mask[:, :6] = True  # ink running off the page's left edge, top and bottom
    mask[5:9, 6:20] = True
    ring = ink_outline(mask, px_per_mm=4.0)[0]
    on_edge = (ring[:, 0] == -0.5) | (ring[:, 1] == -0.5) | (ring[:, 1] == 19.5)
    assert {(-0.5, -0.5), (-0.5, 19.5), (5.5, -0.5), (5.5, 19.5)} <= {tuple(p) for p in ring[on_edge]}


def test_the_same_ink_gives_the_same_outline():
    mask = _random_mask(5)
    first, second = ink_outline(mask, px_per_mm=4.0), ink_outline(mask.copy(), px_per_mm=4.0)
    assert len(first) == len(second) and all(np.array_equal(a, b) for a, b in zip(first, second))


def _least_joint_margin(px_per_mm: float) -> float:
    """The least a diagonal joint keeps on each side of the line through its two ink centers, its paper pixel thin."""
    thin = min(MAX_MARGIN_PX, (THIN_MARGIN_MM + TOLERANCE_MM) * px_per_mm)
    share = min(1.0, 0.95 * np.sqrt(0.5) / (thin * (1 + np.sqrt(2.0))))
    room = np.sqrt(2.0) * (0.5 - thin * share)
    return min(thin, 0.95 * room) - TOLERANCE_MM * px_per_mm


def _inside_points(rings: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    """Whether each point is inside ``rings`` by the even-odd rule: a ray from it to the right crosses an odd number of edges."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    for ring in rings:
        (xa, ya), (xb, yb) = ring[:-1].T, ring[1:].T
        spans = (ya <= y) != (yb <= y)
        at = xa + (y - ya) * (xb - xa) / np.where(yb == ya, 1, yb - ya)
        inside ^= (spans & (at > x)).sum(axis=1) % 2 == 1
    return inside


@pytest.mark.parametrize("px_per_mm", [4.0, 12.0])
@pytest.mark.parametrize("seed", [0, 4])
def test_a_diagonal_joint_stays_inked_round_its_corner(seed, px_per_mm):
    # Where two ink pixels meet only at a corner, the corner is the middle of the line between their centers: the ink
    # holds it, with the joint's margin round it, so the joint prints as a waist rather than a pinch.
    mask = _random_mask(seed)
    rings = ink_outline(mask, px_per_mm=px_per_mm)
    padded = np.pad(mask, 1)
    nw, ne, sw, se = padded[:-1, :-1], padded[:-1, 1:], padded[1:, :-1], padded[1:, 1:]
    rows, columns = np.nonzero((nw == se) & (ne == sw) & (nw != ne))
    joints = np.column_stack([columns - 0.5, rows - 0.5])
    assert len(joints) > 20
    assert _inside_points(rings, joints).all()
    a, b, _, _ = _segments(rings)
    assert _distance_to_segments(joints, a, b).min() >= _least_joint_margin(px_per_mm) - 1e-6


@pytest.mark.parametrize("px_per_mm", [4.0, 12.0])
@pytest.mark.parametrize("seed", [0, 1, 4])
def test_separate_pieces_stay_twice_the_thin_margin_apart(seed, px_per_mm):
    # Two outlines that pass a pixel apart -- across a gap or a stroke a pixel wide, or round the pixel between two
    # corners -- keep the thin margin each, whether or not a pixel there is thin.
    mask = _random_mask(seed)
    rings = ink_outline(mask, px_per_mm=px_per_mm)
    a, b, ring_of, _ = _segments(rings)
    nearest = np.inf
    for k, ring in enumerate(rings):
        others = ring_of != k
        nearest = min(nearest, _distance_to_segments(_densified(ring, 0.05), a[others], b[others]).min())
    assert nearest >= 2 * _least_joint_margin(px_per_mm) - 1e-6


def test_a_diagonal_stroke_keeps_its_width_on_a_fine_page():
    # A 45 degree stroke a pixel wide is 0.71 px wide on the page. At 300 dpi, where a margin is 0.4 px, each visit to a
    # joint still reaches out to the stroke's edge, its box widened towards its own paper pixel: the waist does not
    # narrow to what the margin round the stroke's thin pixels would leave it.
    mask = np.zeros((60, 60), dtype=bool)
    mask[np.arange(5, 55), np.arange(5, 55)] = True
    ring = _densified(ink_outline(mask, px_per_mm=12.0)[0], 0.05)
    middle = ring[(ring[:, 0] > 15) & (ring[:, 0] < 45)]
    off_line = np.abs(middle[:, 0] - middle[:, 1]) / np.sqrt(2.0)
    assert off_line.min() > 0.3 and off_line.max() < 0.6

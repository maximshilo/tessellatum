"""One line per boundary: what the crack graph is traced into, and what gets drawn."""

import cv2
import numpy as np
import pytest

from tessellatum.core.boundaries import crack_edges, trace_boundaries
from tessellatum.core.regions import build_regions, extract_regions
from tessellatum.core.render import OUTLINE_WIDTH, render_page


def _line_layer(region_id_map: np.ndarray) -> np.ndarray:
    """The page's lines alone, as a boolean array: True where there is ink."""
    height, width = region_id_map.shape
    rendered = render_page((width, height), [], region_id_map)
    return np.asarray(rendered.outlines) == 0


def _white_pieces(region_id_map: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """The connected white areas of the page, as the region ids each one covers."""
    white = (~_line_layer(region_id_map)).astype(np.uint8)
    count, pieces = cv2.connectedComponents(white, connectivity=4)
    return count - 1, [np.unique(region_id_map[pieces == piece]) for piece in range(1, count)]


def _walked_cracks(lines: list[np.ndarray], width: int) -> list[tuple[int, int]]:
    """Every crack edge the lines run along, as a pair of corner numbers, once per pass.

    The lines must be unsimplified, so that each of their segments is a run of
    whole crack edges along one row or column of the corner grid.
    """
    walked = []
    for line in lines:
        for (x0, y0), (x1, y1) in zip(line, line[1:]):
            assert x0 == x1 or y0 == y1, "a crack runs along a row or a column"
            steps = int(max(abs(x1 - x0), abs(y1 - y0)))
            for step in range(steps):
                t0, t1 = step / steps, (step + 1) / steps
                corners = tuple(
                    sorted(
                        int(round(y0 + t * (y1 - y0) + 0.5)) * (width + 1) + int(round(x0 + t * (x1 - x0) + 0.5))
                        for t in (t0, t1)
                    )
                )
                walked.append(corners)
    return walked


def _blobby_labels(seed: int, shape: tuple[int, int], num_colors: int, blur_sigma: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = rng.random(shape).astype(np.float32)
    if blur_sigma > 0:
        field = cv2.GaussianBlur(field, (0, 0), blur_sigma)
    edges = np.quantile(field, np.linspace(0, 1, num_colors + 1)[1:-1])
    return np.digitize(field, edges).astype(np.int32)


def test_a_boundary_between_two_regions_becomes_one_line_the_two_share():
    ids = np.zeros((6, 10), dtype=np.int32)
    ids[:, 5:] = 1

    lines = trace_boundaries(ids)

    # The shared boundary, and the page edge round each region: three lines, not
    # two closed outlines that run down the middle twice.
    shared = [line for line in lines if line.min(axis=0)[0] == line.max(axis=0)[0] == 4.5]
    assert len(lines) == 3
    np.testing.assert_array_equal(shared, [[[4.5, -0.5], [4.5, 5.5]]])


def test_lines_end_where_three_regions_meet():
    ids = np.zeros((6, 10), dtype=np.int32)
    ids[:, 5:] = 1
    ids[3:, :] = 2  # a T: the three regions meet at (4.5, 2.5)

    lines = trace_boundaries(ids)
    ends = {tuple(line[0]) for line in lines} | {tuple(line[-1]) for line in lines}

    # Lines start and end only where three regions meet, or two meet the page
    # edge -- the T itself and the three places its arms reach the edge. The
    # page corners are not junctions: the frame runs straight through them.
    assert ends == {(4.5, 2.5), (4.5, -0.5), (-0.5, 2.5), (9.5, 2.5)}
    assert len(lines) == 6


def test_a_region_lying_inside_another_gets_one_closed_line():
    ids = np.zeros((9, 9), dtype=np.int32)
    ids[3:6, 3:6] = 1

    lines = trace_boundaries(ids)

    assert len(lines) == 2  # the page edge and the island, each a loop
    for line in lines:
        np.testing.assert_array_equal(line[0], line[-1])
    inner = min(lines, key=lambda line: line.max())
    assert sorted(map(tuple, inner[:-1])) == [(2.5, 2.5), (2.5, 5.5), (5.5, 2.5), (5.5, 5.5)]


def test_the_page_edge_is_a_boundary_so_a_page_of_one_region_still_has_a_frame():
    ids = np.zeros((5, 7), dtype=np.int32)

    (frame,) = trace_boundaries(ids)

    np.testing.assert_array_equal(frame[0], frame[-1])
    assert sorted(map(tuple, frame[:-1])) == [(-0.5, -0.5), (-0.5, 4.5), (6.5, -0.5), (6.5, 4.5)]
    # It is drawn on the page, not half a pixel off it.
    assert _line_layer(ids)[0].all() and _line_layer(ids)[:, 0].all()


def test_two_regions_touching_corner_to_corner_make_a_crossing_of_four_lines():
    ids = np.zeros((4, 4), dtype=np.int32)  # region 1 in two lobes that meet at (1.5, 1.5)
    ids[:2, 2:] = 1
    ids[2:, :2] = 1

    lines = trace_boundaries(ids)

    ending_there = [line for line in lines if (1.5, 1.5) in (tuple(line[0]), tuple(line[-1]))]
    assert len(ending_there) == 4  # all four cracks around the crossing, each its own line


@pytest.mark.parametrize(
    "seed, shape, num_colors, blur_sigma", [(0, (30, 40), 5, 2.0), (1, (25, 25), 3, 0.0), (2, (40, 31), 8, 1.0)]
)
def test_every_crack_is_walked_exactly_once(seed, shape, num_colors, blur_sigma):
    region_id_map, _colors = build_regions(_blobby_labels(seed, shape, num_colors, blur_sigma), num_colors, 0)
    right, down, _degree, num_edges = crack_edges(region_id_map)

    walked = _walked_cracks(trace_boundaries(region_id_map, simplify_px=0), shape[1])

    stride = shape[1] + 1
    expected = {(c, c + 1) for c in np.flatnonzero(right)} | {(c, c + stride) for c in np.flatnonzero(down)}
    assert len(expected) == num_edges > 0
    assert sorted(walked) == sorted(expected)  # each crack once: no doubling, nothing left out


@pytest.mark.parametrize("seed, shape, num_colors, blur_sigma", [(3, (60, 80), 6, 3.0), (4, (50, 50), 4, 1.5)])
def test_no_white_area_of_the_page_belongs_to_two_regions(seed, shape, num_colors, blur_sigma):
    # Junctions are shared points, so the lines that meet there leave no gap for
    # paint to run through from one region into the next.
    region_id_map, _colors = build_regions(_blobby_labels(seed, shape, num_colors, blur_sigma), num_colors, 20, 5.0)

    pieces, ids_per_piece = _white_pieces(region_id_map)

    assert pieces >= len(np.unique(region_id_map))
    assert [len(ids) for ids in ids_per_piece] == [1] * pieces


def test_a_line_covers_the_pixels_on_both_sides_of_its_crack():
    ids = np.zeros((9, 9), dtype=np.int32)
    ids[:, 4:] = 1

    inked = np.flatnonzero(_line_layer(ids)[4])

    # The crack at x = 3.5 inks the pixels either side of it; the page edge has
    # only one side on the page, so it inks one column.
    assert OUTLINE_WIDTH == 2
    assert list(inked) == [0, 3, 4, 8]


def test_pixels_in_no_region_are_fenced_off_from_the_regions():
    labels = np.zeros((21, 21), dtype=np.int32)
    labels[6:15, 6:15] = -1  # outside the palette: these belong to no region
    region_id_map, _colors = build_regions(labels, 1, 0)
    assert (region_id_map[6:15, 6:15] == -1).all()

    pieces, ids_per_piece = _white_pieces(region_id_map)

    assert pieces == 2 and sorted(ids.tolist() for ids in ids_per_piece) == [[-1], [0]]


def test_the_numbers_do_not_depend_on_the_lines():
    # Acceptance for one line per boundary: the change is to the lines alone.
    labels = _blobby_labels(5, (120, 120), 4, 8.0)
    region_id_map, region_color = build_regions(labels, 4, 400, 9.0)
    regions = extract_regions(region_id_map, region_color)
    plain = np.zeros_like(region_id_map)

    labels_drawn = render_page((120, 120), regions, region_id_map).labels
    with_other_lines = render_page((120, 120), regions, plain).labels

    assert labels_drawn and labels_drawn == with_other_lines


def _distance_to_line(point: np.ndarray, line: np.ndarray) -> float:
    """How far ``point`` lies from the polyline ``line``."""
    starts, ends = line[:-1], line[1:]
    along = ends - starts
    length2 = np.maximum((along**2).sum(axis=1), 1e-12)
    t = np.clip(((point - starts) * along).sum(axis=1) / length2, 0.0, 1.0)
    return float(np.hypot(*(point - (starts + t[:, None] * along)).T).min())


def test_simplifying_keeps_the_junctions_and_never_moves_a_line_further_than_its_epsilon():
    region_id_map, _colors = build_regions(_blobby_labels(6, (40, 50), 5, 1.0), 5, 10)
    exact = trace_boundaries(region_id_map, simplify_px=0)

    simplified = trace_boundaries(region_id_map, simplify_px=1.2)

    assert len(simplified) == len(exact) > 0
    assert sum(len(line) for line in simplified) < sum(len(line) for line in exact)
    for rough, smooth in zip(exact, simplified):
        np.testing.assert_array_equal(smooth[[0, -1]], rough[[0, -1]])  # junctions stay put
        assert np.array_equal(smooth[0], smooth[-1]) == np.array_equal(rough[0], rough[-1])  # closed stays closed
        assert max(_distance_to_line(point, smooth) for point in rough) <= 1.2 + 1e-9

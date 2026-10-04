"""One line per boundary: what the crack graph is traced into, how it is smoothed, and what gets drawn."""

import cv2
import numpy as np
import pytest

from tessellatum.core.boundaries import (
    MAX_SHIFT_PX,
    SMOOTHING_MIN_PX,
    SMOOTHING_MM,
    crack_edges,
    smooth_boundaries,
    smoothing_length_px,
    trace_boundaries,
)
from tessellatum.core.print_size import print_scale
from tessellatum.core.regions import build_regions, extract_regions
from tessellatum.core.render import PAPER, PageStyle, render_page


def _line_layer(region_id_map: np.ndarray) -> np.ndarray:
    """The page's lines alone, as a boolean array: True where any ink falls, however little."""
    height, width = region_id_map.shape
    rendered = render_page((width, height), [], region_id_map)
    return np.asarray(rendered.outlines) != PAPER


def _white_pieces(region_id_map: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """The connected white areas of the page, as the region ids each one covers."""
    white = (~_line_layer(region_id_map)).astype(np.uint8)
    count, pieces = cv2.connectedComponents(white, connectivity=4)
    return count - 1, [np.unique(region_id_map[pieces == piece]) for piece in range(1, count)]


def _walked_cracks(lines: list[np.ndarray], width: int) -> list[tuple[int, int]]:
    """Every crack edge the lines run along, as a pair of corner numbers, once per pass.

    The lines must be unsmoothed, so that each of their segments is a run of
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


def _loop_area(loop: np.ndarray) -> float:
    """The area a closed line encloses."""
    x, y = loop[:-1, 0], loop[:-1, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2


def _length(line: np.ndarray) -> float:
    return float(np.hypot(*np.diff(line, axis=0).T).sum())


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
    # two closed outlines that run down the middle twice. The shared one runs
    # straight down the crack between the two, from page edge to page edge, and
    # smoothing leaves a straight line alone.
    shared = [line for line in lines if line.min(axis=0)[0] == line.max(axis=0)[0] == 4.5]
    assert len(lines) == 3 and len(shared) == 1
    np.testing.assert_array_equal(shared[0][[0, -1]], [[4.5, -0.5], [4.5, 5.5]])
    np.testing.assert_array_equal(np.unique(shared[0][:, 1]), np.arange(-0.5, 6.0))


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
    unsmoothed = trace_boundaries(ids, smoothing_px=0)

    assert len(lines) == 2  # the page edge and the island, each a loop
    for line in lines:
        np.testing.assert_array_equal(line[0], line[-1])
    inner = min(unsmoothed, key=lambda line: line.max())
    assert inner.min() == 2.5 and inner.max() == 5.5 and _loop_area(inner) == 9
    # Smoothing rounds the island's corners off but leaves it a shape to paint,
    # centered where it was, and no corner moves further than the corridor.
    island = min(lines, key=lambda line: line.max())
    assert len(island) == len(inner)
    assert np.abs(island[:-1].mean(axis=0) - 4.0).max() < 1e-9  # still centered on the same pixel
    assert np.hypot(*(island - inner).T).max() <= MAX_SHIFT_PX + 1e-9
    assert _loop_area(island) >= 1.0  # a shape to paint, not a stroke


@pytest.mark.parametrize("shape, label, area", [((7, 7), (3, 3), 1.0), ((9, 9), (4, slice(3, 5)), 2.0)])
def test_a_region_the_blur_would_pull_shut_keeps_the_line_it_was_traced_with(shape, label, area):
    ids = np.zeros(shape, dtype=np.int32)
    ids[label] = 1  # a pixel or two across: its outline is smaller than the corridor

    island = _island(trace_boundaries(ids), shape[1], shape[0])

    # Blurring it would leave nothing inside to paint, so it is kept as traced:
    # a region has to stay a shape, not become a stroke.
    np.testing.assert_array_equal(island, _island(trace_boundaries(ids, smoothing_px=0), shape[1], shape[0]))
    assert _loop_area(island) == area


def test_the_page_edge_is_a_boundary_so_a_page_of_one_region_still_has_a_frame():
    ids = np.zeros((5, 7), dtype=np.int32)

    (frame,) = trace_boundaries(ids)

    np.testing.assert_array_equal(frame[0], frame[-1])
    # Smoothing leaves the paper's own edge alone: the frame is the rectangle
    # round the page, corners and all, not a rounded-off version of it.
    assert set(map(tuple, frame)) == {
        (x, y) for x in np.arange(-0.5, 7.0) for y in (-0.5, 4.5)
    } | {(x, y) for x in (-0.5, 6.5) for y in np.arange(-0.5, 5.0)}
    assert _loop_area(frame) == 7 * 5
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

    walked = _walked_cracks(trace_boundaries(region_id_map, smoothing_px=0), shape[1])

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


def _thin_page(name: str) -> np.ndarray:
    """A region map full of parts a pixel or two across, which smoothing must not cross."""
    ids = np.zeros((20, 30), dtype=np.int32)
    if name == "spur":
        ids[:, 15:] = 1
        ids[5:9, 14] = 1  # 1 px wide and 4 px deep: the shape that leaked at eps 1.2
    elif name == "bar":
        ids[:, 15:] = 1
        ids[9:11, 10:20] = 2  # 2 px tall, lying across the boundary
    elif name == "comb":
        ids[:, 15:] = 1
        ids[:10, ::2] = 1  # teeth 1 px wide
    elif name == "one pixel wide":
        ids[:, 15] = 1  # a region one pixel wide, from edge to edge
    elif name == "diagonal":
        rows, columns = np.indices((20, 30))
        ids = (columns > 1.5 * rows).astype(np.int32)
    else:
        # Nothing merged and no brush pass: hundreds of regions, most of them thin.
        seed, colors, blur = {"blobs": (5, 5, 2.0), "noise": (6, 3, 0.0), "specks": (7, 8, 1.0)}[name]
        ids, _colors = build_regions(_blobby_labels(seed, (20, 30), colors, blur), colors, 0, 0.0)
    return ids.astype(np.int32)


@pytest.mark.parametrize(
    "name", ["spur", "bar", "comb", "one pixel wide", "diagonal", "blobs", "noise", "specks"]
)
def test_smoothing_leaves_no_gap_even_where_a_region_is_a_pixel_across(name):
    # A smoothed line stays inside the two pixels whose boundary it draws, so
    # however thin the regions are, no two of them share a white area to paint.
    ids = _thin_page(name)

    pieces, ids_per_piece = _white_pieces(ids)

    assert pieces > 0
    assert [len(region_ids) for region_ids in ids_per_piece] == [1] * pieces


def test_a_line_covers_the_pixels_on_both_sides_of_its_crack():
    ids = np.zeros((9, 9), dtype=np.int32)
    ids[:, 4:] = 1

    inked = np.flatnonzero(_line_layer(ids)[4])

    # The crack at x = 3.5 inks the pixels either side of it; the page edge has
    # only one side on the page, so it inks one column.
    assert list(inked) == [0, 3, 4, 8]

    # A page this small prints at a few pixels per millimetre, so the line is
    # at its floor of one pixel, laid half on each side of the crack.
    style = PageStyle()
    assert style.line_width_px((9, 9)) == style.min_line_width_px == 1.0
    ink = PAPER - np.asarray(render_page((9, 9), [], ids).outlines)[4]
    assert list(ink[[3, 4]]) == [PAPER // 2 + 1, PAPER // 2 + 1]
    assert ink[0] == PAPER // 2 + 1  # the half of the frame line that falls on the page


def test_pixels_in_no_region_are_fenced_off_from_the_regions():
    labels = np.zeros((21, 21), dtype=np.int32)
    labels[6:15, 6:15] = -1  # outside the palette: these belong to no region
    region_id_map, _colors = build_regions(labels, 1, 0)
    assert (region_id_map[6:15, 6:15] == -1).all()

    pieces, ids_per_piece = _white_pieces(region_id_map)

    assert pieces == 2 and sorted(ids.tolist() for ids in ids_per_piece) == [[-1], [0]]


def test_the_lines_do_not_depend_on_the_numbers():
    # The lines come from the region map alone. The numbers are placed around
    # them since T2.6, so it is only this way round that nothing depends.
    labels = _blobby_labels(5, (120, 120), 4, 8.0)
    region_id_map, region_color = build_regions(labels, 4, 400, 9.0)
    regions = extract_regions(region_id_map, region_color)

    numbered = render_page((120, 120), regions, region_id_map)
    bare = render_page((120, 120), [], region_id_map)

    assert numbered.labels and not bare.labels
    np.testing.assert_array_equal(np.asarray(numbered.outlines), np.asarray(bare.outlines))
    assert len(numbered.strokes) == len(bare.strokes)
    for one, other in zip(numbered.strokes, bare.strokes):
        np.testing.assert_array_equal(one, other)


def _distance_to_line(point: np.ndarray, line: np.ndarray) -> float:
    """How far ``point`` lies from the polyline ``line``."""
    starts, ends = line[:-1], line[1:]
    along = ends - starts
    length2 = np.maximum((along**2).sum(axis=1), 1e-12)
    t = np.clip(((point - starts) * along).sum(axis=1) / length2, 0.0, 1.0)
    return float(np.hypot(*(point - (starts + t[:, None] * along)).T).min())


def test_smoothing_keeps_the_junctions_and_never_moves_a_point_off_its_crack_by_more_than_the_corridor():
    region_id_map, _colors = build_regions(_blobby_labels(6, (40, 50), 5, 1.0), 5, 10)
    cracks = trace_boundaries(region_id_map, smoothing_px=0)

    smoothed = trace_boundaries(region_id_map)

    assert len(smoothed) == len(cracks) > 0
    moved = 0
    for crack, smooth in zip(cracks, smoothed):
        closed = np.array_equal(crack[0], crack[-1])
        assert len(smooth) == len(crack)  # no point is ever dropped
        assert np.array_equal(smooth[0], smooth[-1]) == closed  # closed stays closed
        if not closed:
            np.testing.assert_array_equal(smooth[[0, -1]], crack[[0, -1]])  # junctions stay put
        assert np.hypot(*(smooth - crack).T).max() <= MAX_SHIFT_PX + 1e-9
        assert _length(smooth) <= _length(crack) + 1e-9  # a staircase is the long way round
        moved += int((smooth != crack).any())
    assert moved > len(cracks) // 2  # the smoothing really does something to most lines


def test_the_smoothing_length_is_half_a_millimeter_of_paper_or_a_pixel_step_whichever_is_longer():
    preview, export = (1100, 730), (2400, 1593)  # one page, previewed and exported

    lengths = [smoothing_length_px(size) for size in (preview, export)]

    # At preview size half a millimeter is 1.99 px, so the pixel step is the longer
    # of the two; the export's grid is finer, and there the paper sets the length.
    assert lengths[0] == SMOOTHING_MIN_PX
    assert print_scale(preview).px_to_mm(lengths[0]) > SMOOTHING_MM
    assert print_scale(export).px_to_mm(lengths[1]) == pytest.approx(SMOOTHING_MM)
    # scene.png prints at 60 dpi, where a pixel is nearly half a millimeter itself.
    assert smoothing_length_px((600, 450)) == SMOOTHING_MIN_PX


def test_smoothing_blurs_a_line_along_its_length_until_the_corridor_stops_it():
    path = np.array([[float(x), 0.0] for x in range(31)])
    path[5, 1] = 1.0  # one pixel out of line: a step the grid could have made
    path[20, 1] = 5.0  # five pixels out: a shape the region map really has

    (smoothed,) = smooth_boundaries([path], SMOOTHING_MIN_PX)

    np.testing.assert_array_equal(smoothed[[0, -1]], path[[0, -1]])  # the junctions
    assert smoothed[5, 1] < 0.2  # the step is blurred away into its neighbors
    assert smoothed[20, 1] == pytest.approx(5.0 - MAX_SHIFT_PX)  # the shape stays, the corridor's worth shorter
    assert np.abs(smoothed - path).max() <= MAX_SHIFT_PX + 1e-9


def test_smoothing_takes_the_staircase_off_a_diagonal_boundary():
    # A diamond: every one of its four sides is a 45 degree staircase, which is
    # sqrt(2) times longer than the line it is drawing.
    rows, columns = np.indices((31, 31))
    ids = (np.abs(rows - 15) + np.abs(columns - 15) <= 10).astype(np.int32)

    staircase = _island(trace_boundaries(ids, smoothing_px=0), 31, 31)
    smoothed = _island(trace_boundaries(ids), 31, 31)

    # The staircase is 84 px long: 4 sides of 21 steps, each a pixel across and a
    # pixel down. Smoothed it is the diamond itself, 59.4 px round, less the four
    # corners the corridor lets it round off -- and it holds the region's area.
    assert _length(staircase) == pytest.approx(84.0)
    assert 0.9 * 4 * np.hypot(10.5, 10.5) < _length(smoothed) < 4 * np.hypot(10.5, 10.5)
    assert _loop_area(staircase) == 221 == int((ids == 1).sum())
    assert _loop_area(smoothed) == pytest.approx(221, rel=0.02)


def test_a_notch_deeper_than_the_corridor_is_not_smoothed_away():
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[:, 10:] = 1
    ids[8:12, 6:10] = 1  # a notch 4 px deep cut into region 0, twice the corridor

    # The boundary between the two regions, rather than either region's frame:
    # its ends are on the page edge and everything between them is inside.
    (into_the_notch,) = [
        line for line in trace_boundaries(ids) if line[1:-1].min() > -0.5 and line[1:-1].max() < 19.5
    ]

    # The line still turns into the notch, to within a fraction of a pixel of
    # its far side: 4 px of the region map is the region map's, not the grid's.
    assert 5.5 <= into_the_notch[:, 0].min() < 6.0


def _island(lines: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """The one closed line that doesn't run along the page edge."""
    (island,) = [
        line
        for line in lines
        if np.array_equal(line[0], line[-1])
        and line.min() > -0.5
        and line[:, 0].max() < width - 0.5
        and line[:, 1].max() < height - 0.5
    ]
    return island


def test_no_line_is_drawn_along_the_ink_and_a_boundary_ends_where_it_meets_it():
    # Two fills side by side, crossed by a band of ink two rows high: four regions, and the ink.
    ids = np.zeros((20, 30), dtype=np.int32)
    ids[:, 15:] = 1
    ids[10:, :15] = 2
    ids[10:, 15:] = 3
    ink = np.zeros(ids.shape, dtype=bool)
    ink[9:11, :] = True
    ids[ink] = -1

    lines = trace_boundaries(ids, smoothing_px=0, ink=ink)

    for line in lines:
        along = (line[:-1, 1] == line[1:, 1]) & np.isin(line[:-1, 1], (8.5, 10.5))
        assert not along.any()  # no step of a line runs along the band's edges
    # The boundary between the two fills above the band runs from the page's top edge to the band, and no further.
    assert any(np.array_equal(line, [[14.5, y - 0.5] for y in range(10)]) for line in lines)
    assert any(np.array_equal(line, [[14.5, y - 0.5] for y in range(11, 21)]) for line in lines)
    # Without the ink given, the band is a region like any other, outlined along both edges.
    assert any(((line[:-1, 1] == line[1:, 1]) & (line[:-1, 1] == 8.5)).any() for line in trace_boundaries(ids, smoothing_px=0))

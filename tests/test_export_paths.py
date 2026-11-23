"""The PDF's paths stay within what PDFium rasterizes whole (``export.PATH_CELL_BUDGET``): a path past about 4.2
million coverage cells is cut short, and an A2 PNG at 724 dpi lost every line below the top 42% (1.0.13)."""

import re

import numpy as np
import pytest

from tessellatum.core import export
from tessellatum.core.painting import Painting, Version
from tessellatum.core.render import PageDrawing, PageStyle

import pdf_reading

# A page of 1500 x 1050 px on A4, 5.4 px a millimeter.
SIZE = (1500, 1050)


def _zigzags(spacing: float = 2.0) -> list[np.ndarray]:
    """Lines across the page every ``spacing`` px, each a zig-zag of 1.5 px steps: drawn as one path at 275 dpi,
    about 6.3 million cells, half again past the limit."""
    width, height = SIZE
    xs = np.arange(1.0, width - 1.0, 1.5)
    wiggle = 0.75 * (np.arange(xs.size) % 2)
    return [np.stack([xs, y + wiggle], axis=1) for y in np.arange(2.0, height - 2.0, spacing)]


def _dense_page() -> PageDrawing:
    return PageDrawing(SIZE, PageStyle(line_width_mm=0.1), _zigzags(), [], None, 30, None)


def test_a_dense_page_prints_its_lowest_lines():
    image = np.asarray(export.rasterize(_dense_page(), 275).convert("L"), dtype=np.float64)
    assert image.shape[1] > 2500
    ink = (255 - image).mean(axis=1)
    bands = [band.mean() for band in np.array_split(ink[20:-20], 20)]  # off the page's edges, half a line wide
    assert min(bands) > 50
    assert max(bands) < 1.05 * min(bands)  # none missing its lines, nor filled across


def _paths(content: str):
    """Each path painted in a content stream: (operator, its subpaths, the line width then)."""
    width, numbers, subpaths = 1.0, [], []
    for token in content.split():
        try:
            numbers.append(float(token))
            continue
        except ValueError:
            pass
        if token == "m":
            subpaths.append([numbers[-2:]])
        elif token == "l":
            subpaths[-1].append(numbers[-2:])
        elif token == "w":
            width = numbers[-1]
        elif token in ("S", "f*", "B*", "f", "n"):
            if token in ("S", "f*", "B*") and subpaths:
                yield token, [np.array(p) for p in subpaths], width
            subpaths = []
        numbers = []


def _cells(operator, subpaths, width, scale):
    filled = export._outline_cells(subpaths, scale).sum()
    stroked = export._outline_cells(subpaths, scale, width).sum()
    return {"S": stroked, "f*": filled, "B*": max(filled, stroked)}[operator]


def _web() -> np.ndarray:
    """Line art's ink as one web: a grid of lines 2 px wide every 16 px, one ring with thousands of holes."""
    width, height = SIZE
    ink = np.zeros((height, width), dtype=bool)
    ink[np.arange(height) % 16 < 2] = True
    ink[:, np.arange(width) % 16 < 2] = True
    return ink


@pytest.mark.parametrize("version", list(Version))
def test_no_path_on_the_sheet_is_past_the_budget(tmp_path, version):
    width, height = SIZE
    ids = (np.arange(height)[:, None] // 12 * 1000 + np.arange(width)[None, :] // 12).astype(np.int32)  # 12 px squares
    colors = ((np.arange(ids.max() + 1) + np.arange(ids.max() + 1) // 1000) % 2).astype(np.int64)  # a checkerboard
    painting = Painting(ids, colors, [(200, 40, 40), (40, 160, 60)], np.zeros((height, width), np.uint8), 30)
    drawing = PageDrawing(SIZE, PageStyle(line_width_mm=0.1), _zigzags(), [], _web(), 30, None)
    path = tmp_path / "page.pdf"
    export.save_pdf(drawing, painting.palette_rgb, path, version, painting)
    scale = export._cell_scale(SIZE)
    painted = {"S": 0, "f*": 0, "B*": 0}
    for _number, (_dictionary, stream) in pdf_reading.objects(path.read_bytes()).items():
        if stream is None or b" re W n" not in stream and "/Subtype /Form" not in _dictionary:
            continue
        for operator, subpaths, line_width in _paths(stream.decode("latin-1")):
            # Read back to the hundredth of a pixel the file writes, a path's outline can come out a little longer.
            assert _cells(operator, subpaths, line_width, scale) <= 1.01 * export.PATH_CELL_BUDGET
            painted[operator] += 1
    if version is not Version.COMPLETED:
        assert painted["S"] > 10  # the lines, in groups
    assert painted["f*"] > 3  # the ink, or the paint, in bands


def _even_odd(rings: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    """Which of ``points`` the rings fill by the even-odd rule."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    for ring in rings:
        a, b = ring, np.roll(ring, -1, axis=0)
        spans = (a[:, 1] > y) != (b[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        inside ^= (spans & (x < at)).sum(axis=1) % 2 == 1
    return inside


def _square(x0, y0, x1, y1, steps=40):
    t = np.linspace(0, 1, steps, endpoint=False)
    sides = [
        np.stack([x0 + (x1 - x0) * t, np.full_like(t, y0)], 1),
        np.stack([np.full_like(t, x1), y0 + (y1 - y0) * t], 1),
        np.stack([x1 - (x1 - x0) * t, np.full_like(t, y1)], 1),
        np.stack([np.full_like(t, x0), y1 - (y1 - y0) * t], 1),
    ]
    return np.concatenate(sides)


def _circle(cx, cy, r, steps=200):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def test_a_fill_cut_into_bands_fills_what_it_did(monkeypatch):
    # A sheet with holes, islands in the holes, and holes lying just above or below wherever a cut falls.
    rings = [_square(0, 0, 400, 400)]
    rings += [_circle(60 + 70 * i, 60 + 70 * j, 25) for i in range(5) for j in range(5)]
    rings += [_circle(60 + 70 * i, 60 + 70 * j, 10) for i in range(5) for j in range(5)]
    rings += [_square(10, 5 + 13 * k, 395, 9 + 13 * k, steps=200) for k in range(30)]  # thin bars across
    monkeypatch.setattr(export, "PATH_CELL_BUDGET", 3000)
    overlap = 1.5
    bands = export._fill_bands(rings, 1.0, overlap)
    assert len(bands) > 8
    for band in bands:
        assert export._outline_cells(band, 1.0).sum() <= 3000 or np.ptp(np.concatenate(band)[:, 1]) <= 4 * overlap
    points = np.random.default_rng(0).uniform(-5, 405, size=(4000, 2))
    filled = _even_odd(rings, points)
    by_bands = np.zeros_like(filled)
    for band in bands:
        by_bands |= _even_odd(band, points)
    assert (by_bands == filled).all()


def test_a_ring_clipped_to_a_side_of_a_line_runs_along_it():
    ring = _circle(0, 0, 10)
    above = export._clip_ring(ring, 3.0, below=False)
    below = export._clip_ring(ring, 3.0, below=True)
    assert above[:, 1].max() == pytest.approx(3.0)
    assert below[:, 1].min() == pytest.approx(3.0)
    points = np.random.default_rng(1).uniform(-12, 12, size=(5000, 2))
    inside = _even_odd([ring], points)
    assert (_even_odd([above], points) == (inside & (points[:, 1] < 3))).all()
    assert (_even_odd([below], points) == (inside & (points[:, 1] > 3))).all()


def test_a_page_within_the_budget_is_written_as_before():
    ink = np.zeros_like(_web())
    ink[:200, :200] = _web()[:200, :200]
    drawing = PageDrawing(SIZE, PageStyle(line_width_mm=0.1), _zigzags(spacing=100), [], ink, 30, None)
    content = export._page_ops(drawing, 5.4)
    assert content.count("S") == 1
    assert content.count("f*") == 1


def test_a_stroke_past_the_budget_on_its_own_is_cut_into_pieces_sharing_their_ends(monkeypatch):
    path = _zigzags()[0]
    monkeypatch.setattr(export, "PATH_CELL_BUDGET", 2000)
    groups = export._stroke_groups([path], 1.0, 1.0)
    pieces = sorted((piece for group in groups for piece in group), key=lambda piece: piece[0, 0])
    assert len(pieces) > 2
    assert all(export._outline_cells(group, 1.0, 1.0).sum() <= 2000 for group in groups)
    joined = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    assert (joined == path).all()
    for a, b in zip(pieces[:-1], pieces[1:]):
        assert (a[-1] == b[0]).all()

"""Render the final coloring page: lines + numbers on a white canvas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tessellatum.core.boundaries import trace_boundaries
from tessellatum.core.labels import LEADER_REACH_MM, Label, LabelSpacing, font, min_font_size, place_labels, text_bbox
from tessellatum.core.print_size import OUTLINE_WIDTH_MM, print_scale
from tessellatum.core.regions import Region

# The page's ink, as a tone on white paper: 0 is black, 255 is invisible.
# Gray rather than black, so that a line disappears under the paint that is
# meant to cover it, and so that a number reads as part of the page's
# apparatus rather than as writing in the picture (QUALITY_BENCHMARKS.md).
LINE_GRAY = 0x59
LABEL_GRAY = 0x8C

# A line is never drawn thinner than this, whatever the paper asks for. The
# pipeline never upscales, so a small source prints at a low resolution -- 60
# dpi for a 600 px image -- where 0.3 mm is less than a pixel and the line
# would come out too faint to follow.
MIN_LINE_WIDTH_PX = 1.0

# The line layer is drawn on a grid this many times finer than the page and
# averaged back down. That is what anti-aliases it, and what lets a line be a
# fraction of a pixel wide; 4 is as fine as it can be without the page of an
# export costing hundreds of megabytes to draw.
_SUPERSAMPLE = 4
# Lines are smoothed off the pixel cracks they are traced from (see
# ``boundaries.smooth_boundaries``), by as little as a fraction of a pixel, so they
# are rasterized at 1/16 of a grid pixel: rounding them onto a coarser grid would
# put the staircase back. It is the precision the benchmark harness rasterizes with too.
_SUBPIXEL_BITS = 4

PAPER = 255  # the line layer where no ink falls at all

# A number written outside its region points into it with a leader: a line as
# wide as the page's own, in the numbers' gray, ending in a dot this many line
# widths across (see ``labels``).
LEADER_DOT_RATIO = 3.0


@dataclass(frozen=True)
class PageStyle:
    """The page's ink: how wide its lines print, and how dark they and the numbers are.

    ``line_width_mm`` is a physical width, converted for each page by the
    print model, so a preview and an export of one image print the same line.
    The grays are tones on white paper, 0 black and 255 invisible.
    """

    line_width_mm: float = OUTLINE_WIDTH_MM
    min_line_width_px: float = MIN_LINE_WIDTH_PX
    line_gray: int = LINE_GRAY
    label_gray: int = LABEL_GRAY
    leader_dot_ratio: float = LEADER_DOT_RATIO

    def line_width_px(self, size: tuple[int, int]) -> float:
        """How wide a line is on a page of ``size`` (width, height) pixels."""
        return max(self.min_line_width_px, print_scale(size).mm_to_px(self.line_width_mm))


@dataclass
class RenderedPage:
    image: Image.Image  # RGB: outlines + numbers
    outlines: Image.Image  # "L": the ink the lines alone put on the page, 0 = solid ink, 255 = bare paper
    labels: list[Label]  # every number on the page, in drawing order
    strokes: list[np.ndarray]  # every line drawn, in drawing order: Nx2 float64 (x, y); a closed one returns to its first point
    leaders: Image.Image  # "L": the ink the numbers' leader lines put on the page, as in ``outlines``


def render_page(
    size: tuple[int, int],
    regions: list[Region],
    region_id_map: np.ndarray,
    style: PageStyle = PageStyle(),
) -> RenderedPage:
    """Draw the boundaries of ``region_id_map`` + numbers for ``regions`` onto a white ``size`` canvas.

    Every boundary between two regions is drawn once, as the line its two
    regions share (see ``boundaries.trace_boundaries``), rather than as part
    of an outline around each of them. Every region in ``regions`` gets its
    number, printed at least as large as the paper needs and where no line
    runs through it (see ``labels.place_labels``). ``style`` says how wide the
    lines print and how dark they and the numbers are.

    Returns the page, plus what it was built from: the ink the lines put on it,
    the geometry each was drawn from, where each number went, and the ink of
    the leader lines that point a number written outside its region into it.
    """
    strokes = trace_boundaries(region_id_map)
    line_width = style.line_width_px(size)
    coverage = ink_coverage(size, strokes, line_width)
    outlines = Image.fromarray(PAPER - coverage, "L")

    spacing = LabelSpacing(
        min_font_size=min_font_size(size),
        label_gap_px=line_width,
        leader_width_px=line_width,
        leader_reach_px=print_scale(size).mm_to_px(LEADER_REACH_MM),
    )
    labels = place_labels(regions, region_id_map, coverage == 0, spacing)
    leader_coverage = _leader_coverage(size, labels, line_width, line_width * style.leader_dot_ratio)

    paper = _paper_under(coverage, style.line_gray)
    if any(label.leader is not None for label in labels):  # most pages have none, and white paper changes nothing
        np.minimum(paper, _paper_under(leader_coverage, style.label_gray), out=paper)
    page = Image.fromarray(paper, "L").convert("RGB")
    draw = ImageDraw.Draw(page)
    label_fill = (style.label_gray,) * 3
    for label in labels:
        bbox = text_bbox(label.text, label.font_size)
        left, top = label.box[0], label.box[1]
        draw.text((left - bbox[0], top - bbox[1]), label.text, fill=label_fill, font=font(label.font_size))

    return RenderedPage(
        image=page,
        outlines=outlines,
        labels=labels,
        strokes=strokes,
        leaders=Image.fromarray(PAPER - leader_coverage, "L"),
    )


def _leader_coverage(size: tuple[int, int], labels: list[Label], width_px: float, dot_px: float) -> np.ndarray:
    """The ink the leader lines put on a ``size`` page: 0 = bare paper, 255 = solid.

    Each leader is a line from its number to the point in the region it
    numbers, drawn with the same round pen as the page's lines, with a dot
    ``dot_px`` across at that point. Each is drawn in a window around it, since
    a page has few of them if any.
    """
    width, height = size
    coverage = np.zeros((height, width), dtype=np.uint8)
    reach = int(math.ceil(max(width_px, dot_px) / 2)) + 2
    for label in labels:
        if label.leader is None:
            continue
        points = np.array(label.leader, dtype=np.float64)
        x0 = max(0, int(math.floor(points[:, 0].min())) - reach)
        y0 = max(0, int(math.floor(points[:, 1].min())) - reach)
        x1 = min(width, int(math.ceil(points[:, 0].max())) + reach + 1)
        y1 = min(height, int(math.ceil(points[:, 1].max())) + reach + 1)
        local = points - (x0, y0)
        window = (x1 - x0, y1 - y0)
        line = ink_coverage(window, [local], width_px)
        dot = ink_coverage(window, [local[1:].repeat(2, axis=0)], dot_px)
        np.maximum(coverage[y0:y1, x0:x1], np.maximum(line, dot), out=coverage[y0:y1, x0:x1])
    return coverage


def ink_coverage(size: tuple[int, int], strokes: list[np.ndarray], width_px: float) -> np.ndarray:
    """How much ink each pixel of a ``size`` page gets from ``strokes``: 0 = bare paper, 255 = solid.

    A line is what a round pen ``width_px`` across leaves behind as it is
    dragged along its path: every point within half that width of the path.
    The pen is rasterized on a grid ``_SUPERSAMPLE`` times finer than the page
    and averaged back down, so a line that covers part of a pixel inks part of
    it -- which is what keeps a smooth line looking smooth, and what lets a
    line be thinner than a pixel.

    The pen's diameter on that grid is a whole number of grid pixels, so a
    width in between is drawn as a blend of the two diameters around it, rather
    than every line being rounded to the grid's own steps. Down a crack that
    lays down the asked-for width to a fraction of a percent. A line that
    follows neither a row nor a column is drawn along a staircase of grid
    pixels, which is not quite the line it stands for, so its band comes out
    within about a tenth of the width asked for -- a difference that shrinks
    with the grid, and that no page is drawn at a size where it can be seen.

    A line runs between pixels, not down the middle of them, so the pen is
    centered on the crack: an even diameter, reaching the same distance either
    way. The canvas has a margin, so that a line on the page edge -- whose
    other half falls off the paper -- is drawn rather than clipped away.
    """
    width, height = size
    grid = _SUPERSAMPLE
    margin = int(math.ceil(width_px / 2)) + 1
    canvas = np.zeros(((height + 2 * margin) * grid, (width + 2 * margin) * grid), np.uint8)
    if strokes:
        paths = [
            np.rint(((stroke + (margin + 0.5)) * grid - 0.5) * (1 << _SUBPIXEL_BITS)).astype(np.int32)
            for stroke in strokes
        ]
        cv2.polylines(canvas, paths, False, PAPER, thickness=1, lineType=cv2.LINE_8, shift=_SUBPIXEL_BITS)

    # Two grid pixels is the finest pen there is, so a style asking for a line
    # thinner than that gets the thinnest one the grid can draw.
    diameter = max(2.0, width_px * grid)
    thinner = 2 * int(diameter // 2)  # the widest even diameter the grid holds that is not too wide
    wider_share = (diameter - thinner) / 2
    band = np.empty_like(canvas)
    inked = _pen_coverage(canvas, band, thinner, size, margin)
    if wider_share > 0:
        wider = _pen_coverage(canvas, band, thinner + 2, size, margin)
        inked = (1 - wider_share) * inked + wider_share * wider
    return np.rint(inked).astype(np.uint8)


def _pen_coverage(
    canvas: np.ndarray, band: np.ndarray, diameter: int, size: tuple[int, int], margin: int
) -> np.ndarray:
    """``canvas``, a line one grid pixel wide, widened to ``diameter`` and averaged down to ``size``."""
    kernel, anchor = _pen(diameter)
    cv2.dilate(canvas, kernel, dst=band, anchor=anchor)
    width, height = size
    shrunk = cv2.resize(band, (width + 2 * margin, height + 2 * margin), interpolation=cv2.INTER_AREA)
    return shrunk[margin : margin + height, margin : margin + width].astype(np.float64)


@lru_cache(maxsize=None)
def _pen(diameter: int) -> tuple[np.ndarray, tuple[int, int]]:
    """A round pen ``diameter`` grid pixels across, and the anchor that centers it on a crack.

    The line it widens is one grid pixel wide and lies on the pixel right of or
    below its crack, so the pen has to reach one further up and left than down
    and right. That is what the anchor does; an even diameter is what lets the
    two halves be equal.
    """
    if diameter < 2 or diameter % 2:
        raise ValueError(f"the pen's diameter must be a positive even number of grid pixels, got {diameter}")
    half = diameter // 2
    reach = np.arange(-half, half) + 0.5  # how far each of the pen's pixels is from the crack
    dy, dx = np.meshgrid(reach, reach, indexing="ij")
    return (dx * dx + dy * dy <= half * half).astype(np.uint8), (half - 1, half - 1)


def _paper_under(coverage: np.ndarray, gray: int) -> np.ndarray:
    """White paper with ``gray`` ink laid on it as thickly as ``coverage`` says."""
    return np.rint(PAPER - coverage * ((PAPER - gray) / PAPER)).astype(np.uint8)

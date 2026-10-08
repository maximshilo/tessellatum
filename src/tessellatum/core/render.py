"""Render the final coloring page: lines + numbers on a white canvas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tessellatum.core.boundaries import SMOOTHING_MM, smoothing_length_px, trace_boundaries
from tessellatum.core.labels import (
    LEADER_REACH_MM,
    TEXT_GAP_MM,
    Label,
    LabelSpacing,
    baseline_bbox,
    cleared,
    font,
    min_font_size,
    place_labels,
    text_bbox,
)
from tessellatum.core.print_size import MIN_LABEL_SIZE_PT, OUTLINE_WIDTH_MM, print_scale
from tessellatum.core.regions import Region
from tessellatum.core.text import Lettering

# The page's ink, as a tone on white paper: 0 is black, 255 is invisible.
# Gray rather than black, so that a line disappears under the paint that is
# meant to cover it, and so that a number reads as part of the page's
# apparatus rather than as writing in the picture (QUALITY_BENCHMARKS.md).
LINE_GRAY = 0x59
LABEL_GRAY = 0x8C

# The tones the app offers for the lines and the numbers, (line gray, number gray), lightest first. Medium is the pair
# above; Light and Dark are 15 L* lighter and darker, both grays alike, so the numbers stay as much lighter than the
# lines (20 L*) in every tone.
TONES = {
    "Light": (0x7E, 0xB4),
    "Medium": (LINE_GRAY, LABEL_GRAY),
    "Dark": (0x36, 0x66),
}
DEFAULT_TONE = "Medium"
# The line widths the app offers, in mm on paper, and its step. 0.2 to 0.5 mm read well on the benchmark's pages; the
# app reaches past both, for a hairline or a bold page.
LINE_WIDTH_MM_RANGE = (0.1, 1.0)
LINE_WIDTH_MM_STEP = 0.05

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
    """The page's ink: how its lines and numbers print.

    ``line_width_mm`` is a physical width, converted for each page by the
    print model, so a preview and an export of one image print the same line.
    The grays are tones on white paper, 0 black and 255 invisible.
    ``line_smoothing_mm`` is how far along a line its pixel staircase is
    smoothed (see ``boundaries.smoothing_length_px``). ``min_label_pt`` is
    the smallest a number prints, ``leader_reach_mm`` how far from its region
    a number with a leader may go, and ``text_gap_mm`` how far a number keeps
    from a line of text found (see ``labels``). All are on paper.

    The numbers' placement decides which of line art's regions have room for
    one, and a region with none joins the area beside it (see
    ``pipeline.generate``), so on line art the number settings can change the
    regions too.
    """

    line_width_mm: float = OUTLINE_WIDTH_MM
    min_line_width_px: float = MIN_LINE_WIDTH_PX
    line_gray: int = LINE_GRAY
    label_gray: int = LABEL_GRAY
    leader_dot_ratio: float = LEADER_DOT_RATIO
    line_smoothing_mm: float = SMOOTHING_MM
    min_label_pt: float = MIN_LABEL_SIZE_PT
    leader_reach_mm: float = LEADER_REACH_MM
    text_gap_mm: float = TEXT_GAP_MM

    def line_width_px(self, size: tuple[int, int]) -> float:
        """How wide a line is on a page of ``size`` (width, height) pixels."""
        return max(self.min_line_width_px, print_scale(size).mm_to_px(self.line_width_mm))

    def smoothing_px(self, size: tuple[int, int]) -> float:
        """How far a line is smoothed along its length on a page of ``size`` (``boundaries.smoothing_length_px``)."""
        return smoothing_length_px(size, self.line_smoothing_mm)

    @classmethod
    def from_settings(
        cls, line_width_mm: float = OUTLINE_WIDTH_MM, tone: str = DEFAULT_TONE, **more: float
    ) -> PageStyle:
        """The style for a line width in mm and one of ``TONES``, as the app offers them, and other fields by name."""
        line_gray, label_gray = TONES[tone]
        return cls(line_width_mm=line_width_mm, line_gray=line_gray, label_gray=label_gray, **more)


@dataclass
class PageDrawing:
    """What a page is drawn from, for drawing it again off the pixel grid (see ``export.save_pdf``).

    In the page's pixels, ``size`` (width, height) of them, with pixel centers
    at integer coordinates, as ``render_page`` draws it, in this order: the
    lines, in ``style.line_gray``; the letters of the lines of text, the rings
    of ``lettering`` filled by the even-odd rule, solid in ``ink_gray``,
    darkening what is under them; the printed ink, solid in ``ink_gray`` over
    what is under it; the leaders, in ``style.label_gray``, darkening what is
    under them; and the numbers, in ``style.label_gray``, at the size and the
    place ``number_origin`` gives. ``ink`` and ``lettering`` are None where the
    page has none.
    """

    size: tuple[int, int]
    style: PageStyle
    strokes: list[np.ndarray]
    labels: list[Label]
    ink: np.ndarray | None
    ink_gray: int
    lettering: list[np.ndarray] | None


@dataclass
class RenderedPage:
    image: Image.Image  # RGB: outlines + numbers
    outlines: Image.Image  # "L": the ink the lines and the printed ink put on the page, 0 = solid, 255 = bare paper
    labels: list[Label]  # every number on the page, in drawing order
    strokes: list[np.ndarray]  # every line drawn, in drawing order: Nx2 float64 (x, y); a closed one returns to its first point
    leaders: Image.Image  # "L": the ink the numbers' leader lines put on the page, as in ``outlines``
    # HxW bool: the ink as printed, less the detail ink cleared behind numbers written on hatching; None without.
    printed_ink: np.ndarray | None = None
    drawing: PageDrawing | None = None  # what the page was drawn from
    # HxW uint8: the ink the page prints of the picture itself, 0 = bare paper, 255 = solid, in the drawing's
    # ``ink_gray``: the printed ink solid, and the letters in the lines of text in their own tones; none of the lines,
    # the numbers or their leaders: what still shows on the page painted in (see ``painting``).
    picture_ink: np.ndarray | None = None


def render_page(
    size: tuple[int, int],
    regions: list[Region],
    region_id_map: np.ndarray,
    style: PageStyle = PageStyle(),
    ink: np.ndarray | None = None,
    ink_gray: int = 0,
    clearable: np.ndarray | None = None,
    lettering: Lettering | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> RenderedPage:
    """Draw the boundaries of ``region_id_map`` + numbers for ``regions`` onto a white ``size`` canvas.

    Every boundary between two regions is drawn once, as the line its two
    regions share (see ``boundaries.trace_boundaries``), rather than as part
    of an outline around each of them. Every region in ``regions`` gets its
    number, printed at least as large as the paper needs and where no line
    runs through it (see ``labels.place_labels``). ``style`` says how wide the
    lines print and how dark they and the numbers are.

    ``ink`` (HxW bool) is line art's own ink, or the detail marks in a
    photograph's or a painting's faces (see ``marks``), printed solid in
    ``ink_gray``, their own tone (see ``ink.ink_gray``): part of the drawing,
    not something to paint. It is the line wherever it runs, so no line is drawn
    along it, and no number goes on it. ``clearable`` is the part of it a
    region's paint goes over whole, hatching (see ``regions.detail_ink``): a
    number with no room near its region but on that ink is written in the
    region with the ink under it, and a line's width round it, left unprinted
    (see ``labels.place_labels``).

    ``lettering`` is the letters in the lines of text found (see
    ``text.lettering``): inside the lines' boxes, ``lettering.area``, the page
    prints them in ``ink_gray``, as much as ``lettering.ink`` says -- the
    letters' own tones round their outline -- and the lines running through,
    in place of the printed ink lying in a region -- a scan's own ink would
    print its letters twice. The letters' ground is bare paper. Printed ink in no region, line art's bold ink and the
    seam down a line two regions share, keeps them apart, and still prints
    solid. No number goes in the lines' boxes, nor within
    ``style.text_gap_mm`` of them (see ``labels.place_labels``). The pixels
    the letters cover at least half of are in ``ink`` as well, which ends the
    lines crossing them and keeps label points off them. The page's drawing
    has the letters' outline, ``lettering.outline``.

    ``check_cancelled``, if given, is called as the numbers are placed, and
    stops the page by raising (see ``labels.place_labels``).

    Returns the page, plus what it was built from: the ink the lines and the
    printed ink put on it, the geometry each line was drawn from, where each
    number went, the ink of the leader lines that point a number written
    outside its region into it, the ink printed, and the ink the page prints of
    the picture itself, without the lines and numbers.
    """
    inked = ink is not None and bool(ink.any())
    strokes = trace_boundaries(region_id_map, smoothing_px=style.smoothing_px(size), ink=ink if inked else None)
    line_width = style.line_width_px(size)
    lines_only = ink_coverage(size, strokes, line_width)
    lettered = lettering is not None and bool(lettering.area.any())
    lettering_area = lettering.area if lettered else None

    def solid(printed: np.ndarray | None) -> np.ndarray | None:
        """The printed ink that prints solid: all of it but, inside the lines of text, what lies in a region."""
        if not inked or not lettered:
            return printed if inked else None
        return printed & ((region_id_map < 0) | ~lettering_area)

    def all_ink(printed: np.ndarray | None) -> np.ndarray:
        """The ink the page puts down: the lines, the printed ink solid, and inside the lines of text their letters."""
        coverage = lines_only
        if lettered:
            coverage = np.where(lettering_area, np.maximum(lines_only, lettering.ink), coverage)
        if inked:
            coverage = np.where(solid(printed), np.uint8(PAPER), coverage)
        return coverage

    coverage = all_ink(ink)

    spacing = LabelSpacing(
        min_font_size=min_font_size(size, style.min_label_pt),
        label_gap_px=line_width,
        leader_width_px=line_width,
        leader_dot_px=line_width * style.leader_dot_ratio,
        leader_reach_px=print_scale(size).mm_to_px(style.leader_reach_mm),
        text_gap_px=print_scale(size).mm_to_px(style.text_gap_mm),
    )
    # A number goes on no printed ink, and a leader runs through none, whichever region's paint goes over it.
    seen = np.where(ink, -1, region_id_map) if inked else region_id_map
    detail = np.where(clearable, region_id_map, -1).astype(np.int32) if inked and clearable is not None else None
    labels = place_labels(
        regions, seen, coverage == 0, spacing, detail, text=lettering_area if lettered else None,
        check_cancelled=check_cancelled,
    )
    if detail is not None and any(label.clears for label in labels):
        ink = ink & ~cleared(labels, detail, spacing.label_gap_px)
        coverage = all_ink(ink)
    outlines = Image.fromarray(PAPER - coverage, "L")
    leader_coverage = _leader_coverage(size, labels, line_width, spacing.leader_dot_px)

    paper = paper_under(lines_only, style.line_gray)
    if lettered:
        letters = np.minimum(paper, paper_under(lettering.ink, ink_gray))
        paper[lettering_area] = letters[lettering_area]
    if inked:
        paper[solid(ink)] = min(int(ink_gray), PAPER)
    if any(label.leader is not None for label in labels):  # most pages have none, and white paper changes nothing
        np.minimum(paper, paper_under(leader_coverage, style.label_gray), out=paper)
    page = Image.fromarray(paper, "L").convert("RGB")
    draw = ImageDraw.Draw(page)
    label_fill = (style.label_gray,) * 3
    for label in labels:
        bbox = text_bbox(label.text, label.font_size)
        left, top = label.box[0], label.box[1]
        draw.text((left - bbox[0], top - bbox[1]), label.text, fill=label_fill, font=font(label.font_size))

    picture_ink = np.where(lettering_area, lettering.ink, 0).astype(np.uint8) if lettered else np.zeros_like(lines_only)
    if inked:
        picture_ink[solid(ink)] = PAPER

    drawing = PageDrawing(
        size=size,
        style=style,
        strokes=strokes,
        labels=labels,
        ink=solid(ink) if inked else None,
        ink_gray=min(int(ink_gray), PAPER),
        lettering=list(lettering.outline) if lettered else None,
    )
    return RenderedPage(
        image=page,
        outlines=outlines,
        labels=labels,
        strokes=strokes,
        leaders=Image.fromarray(PAPER - leader_coverage, "L"),
        printed_ink=ink if inked else None,
        drawing=drawing,
        picture_ink=picture_ink,
    )


def number_origin(label: Label) -> tuple[float, float]:
    """Where ``render_page`` puts the start of a number's baseline, in the page's pixels.

    It writes the number so that its text box (``labels.text_bbox``) lands on
    ``label.box``, in ``labels.font`` at ``label.font_size``, an em of that
    many pixels.
    """
    bbox = baseline_bbox(label.text, label.font_size)
    return label.box[0] - bbox[0], label.box[1] - bbox[1]


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


def paper_under(coverage: np.ndarray, gray: int) -> np.ndarray:
    """White paper with ``gray`` ink laid on it as thickly as ``coverage`` says."""
    return np.rint(PAPER - coverage * ((PAPER - gray) / PAPER)).astype(np.uint8)

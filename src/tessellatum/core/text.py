"""Text: where a picture has lettering, for the stages that keep it readable.

A sign, a title or a caption should still read on the page, and no number
should be written on it. This module finds the lines of text, and says how
the page prints the lettering in them.

They are found by **PP-OCRv6-small**, the text detection network of
PaddleOCR (9.9 MB), run by ONNX Runtime, shipped with the app and run offline
(``resources/MODELS.md`` has its source and license). The network says how
likely each pixel is to lie in the core of a line of text -- a word, or a run
of words -- and each patch of likely pixels, grown back out to the line's
edge, is a line found (Differentiable Binarization, as PaddleOCR reads it).

It looks twice. First at the picture at half its preview size, which finds
lettering whose lines print at least about 4 mm tall. Where that finds text,
it looks again at the picture twice its preview size, taken from the source's
own pixels where it has them, which finds the smaller print beside it:
captions, small signs, down to lines about 1.7 mm tall. A picture with no
lettering big enough to see in the first look is taken to have none, which
keeps the finer look from taking window rows, shutters or fur for text, and
costs a picture without text only the small first look.

The letters in a line found are printed, and nothing else in its box (see
``lettering``): solid, in the ink's tone, as smooth outlines that a PDF draws
as vector art. Signs are lit as often as they are painted, so the letters may
be darker than their ground or lighter; either way they print as ink, and the
ground stays bare paper. Which side of a box is its letters is told by three
things about it, two of which must agree.

A line found must look like one: at most ``MAX_BOX_HEIGHT_MM`` tall on the
printed page -- lettering taller than that is big enough to paint as shapes of
its own -- and at least ``MIN_ELONGATION`` times as long as it is tall, as a
run of two letters or more is. The things the network takes for text that
aren't -- an eye, a disc, a window -- are mostly as tall as they are wide.

Text is found once per picture, so a preview and an export always agree.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import cv2
import numpy as np

from tessellatum.core import parallel
from tessellatum.core.faces import RESOURCES_DIR
from tessellatum.core.ink_outline import TOLERANCE_MM, traced_rings
from tessellatum.core.print_size import print_scale

MODEL = "PP-OCRv6_small_det.onnx"

# The first look sees the picture at GATE_SCALE times its preview size, the second at FINE_SCALE times, each side rounded
# to a multiple of SIDE_MULTIPLE pixels, as the network needs.
GATE_SCALE = 0.5
FINE_SCALE = 2.0
SIDE_MULTIPLE = 32
# The picture is fed as PaddleOCR feeds it: its BGR values scaled to [0, 1] and standardized channel by channel by these
# (ImageNet's RGB statistics, applied to BGR as PaddleOCR applies them).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# A line's core is where the network's probability is above THRESHOLD; the line is kept if its mean probability there is
# at least MIN_SCORE, and grown out on every side by its area times UNCLIP over its perimeter, back to the line's edge.
# PaddleOCR's own setting for this network grows a line by 1.4, which at twice the preview's size leaves the words of a
# small caption apart.
THRESHOLD = 0.2
MIN_SCORE = 0.4
UNCLIP = 2.0
# A line's box, grown out to the line's edge, is at most this tall on the printed page, and at least this many times as
# long as it is tall.
MAX_BOX_HEIGHT_MM = 15.0
MIN_ELONGATION = 1.5
# A line's two tones are its box's INK_PERCENTILE and PAPER_PERCENTILE of CIE L*; which is its letters' is told by
# the sides of the box halfway between them. A pixel's ink is how far it lies from its ground towards its letters' tone,
# and the letters' outline runs LETTER_CUT of the way from their tone to their ground's: a little past halfway, so that
# letters whose tone falls short of the box's -- red lettering on a sign beside white -- print solid, not hollow. A line
# prints only if the two tones Otsu's method splits its box into stand MIN_CONTRAST L* apart; a fainter one would print
# its ground's grain as boldly as letters.
PAPER_PERCENTILE = 98.0
INK_PERCENTILE = 2.0
LETTER_CUT = 0.6
MIN_CONTRAST = 20.0
# A line's ground is its lightness with the letters taken out by a disk GROUND_DISK of its box's height across, and by
# a square as wide turned to the line: wider than a stroke of the boldest letters the box holds, narrower than the line.
GROUND_DISK = 0.6
# A piece of a box's letters that runs along more than this share of the box's edge is no letter: it is ground caught
# by the box -- a sign's edge, a frame -- and is not printed.
MAX_EDGE_SHARE = 0.1
# The letters' outline is traced on a grid LETTERING_GRID times finer than the page, through the box's lightness
# interpolated bicubically between the page's pixels, and smoothed along by _OUTLINE_WEIGHTS -- a binomial filter, as
# wide as a Gaussian of one of that grid's pixels -- which takes out the grid's staircase; then simplified within
# ink_outline.TOLERANCE_MM on paper.
LETTERING_GRID = 4
_OUTLINE_WEIGHTS = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)
# Bicubic interpolation reads two pixels either side, so a box's lightness is read this many pixels beyond it.
_CONTEXT = 2
# Groups of lines whose window holds at least this many pixels are read on the shared pool.
_POOLED_PX = 4000
# Patches of the map no wider than this many of its pixels, before and after growing, are noise; and only this many
# patches are read.
_MIN_CORE_PX = 3
_MIN_BOX_PX = 5
_MAX_CANDIDATES = 1000

_lock = threading.Lock()
_session = None


@dataclass(frozen=True)
class TextLine:
    """A line of text found: its box, and how sure the network is."""

    # The box's four corners in order round it, (x, y) in pixels of the picture or page it is given for, a pixel's corner
    # at whole numbers. A rectangle turned to the line (all but: the network sees the picture with its sides rounded to
    # SIDE_MULTIPLE, a hair off its shape); it may reach past the picture's edge.
    quad: tuple[tuple[float, float], ...]
    score: float  # the network's mean probability of text over the line's core, 0-1

    @property
    def sides(self) -> tuple[float, float]:
        """The box's height and length, in pixels: its shorter and its longer side."""
        p = np.asarray(self.quad, dtype=np.float64)
        a, b = float(np.hypot(*(p[1] - p[0]))), float(np.hypot(*(p[2] - p[1])))
        return min(a, b), max(a, b)


def find_text(picture_bgr: np.ndarray, source_bgr: np.ndarray | None = None) -> list[TextLine]:
    """The lines of text in ``picture_bgr``, in its pixels, top to bottom.

    ``picture_bgr`` is an HxWx3 uint8 picture at preview size, and
    ``source_bgr`` the same picture at its own size, from which the second,
    finer look takes its pixels (``picture_bgr`` itself if None).
    """
    picture = _checked(picture_bgr)
    source = picture if source_bgr is None else _checked(source_bgr)
    height, width = picture.shape[:2]
    size = (width, height)
    if not _lines(_view(picture, picture, GATE_SCALE), size):
        return []
    return sorted(_lines(_view(picture, source, FINE_SCALE), size), key=_reading_order)


def scaled(lines: list[TextLine], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[TextLine]:
    """``lines`` found on a picture of ``from_size`` (width, height), given for the same picture at ``to_size``."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return [replace(line, quad=tuple((x * sx, y * sy) for x, y in line.quad)) for line in lines]


def mask(lines: list[TextLine], size: tuple[int, int]) -> np.ndarray:
    """HxW bool for a picture of ``size`` (width, height): the pixels whose middle lies in the box of a line.

    A middle on a box's edge is in it; a box of no area holds none.
    """
    width, height = size
    covered = np.zeros((height, width), dtype=bool)
    for line in lines:
        found = _box_pixels(np.asarray(line.quad, dtype=np.float64), size)
        if found is not None:
            (y0, y1, x0, x1), inside = found
            covered[y0:y1, x0:x1] |= inside
    return covered


def _box_pixels(quad: np.ndarray, size: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The pixels of a picture of ``size`` whose middle lies in ``quad``: ((y0, y1, x0, x1), inside), the rows and
    columns they span and which of those pixels they are; None if there are none."""
    width, height = size
    x0, x1 = max(int(np.floor(quad[:, 0].min() - 0.5)), 0), min(int(np.ceil(quad[:, 0].max() - 0.5)) + 1, width)
    y0, y1 = max(int(np.floor(quad[:, 1].min() - 0.5)), 0), min(int(np.ceil(quad[:, 1].max() - 0.5)) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    inside = _in_quad(quad, xs, ys)
    rows, columns = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    if rows.size == 0:
        return None
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1
    return (y0 + r0, y0 + r1, x0 + c0, x0 + c1), np.ascontiguousarray(inside[r0:r1, c0:c1])


def _in_quad(quad: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Whether each point (``xs``, ``ys``, broadcast together) lies in the convex ``quad`` or on its edge.

    A quad of no area holds none.
    """
    rolled = np.roll(quad, -1, axis=0)
    shape = np.broadcast_shapes(np.shape(xs), np.shape(ys))
    if abs(float(np.sum(quad[:, 0] * rolled[:, 1] - rolled[:, 0] * quad[:, 1]))) <= 1e-9:
        return np.zeros(shape, dtype=bool)  # every point would lie on the same side of all its edges, or on them
    # Inside a convex polygon: on the same side of every edge (or on it).
    sides = [(b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) for a, b in zip(quad, rolled)]
    inside = np.logical_and.reduce([s >= 0 for s in sides]) | np.logical_and.reduce([s <= 0 for s in sides])
    return np.broadcast_to(inside, shape)


@dataclass
class Lettering:
    """How a page prints the letters in the lines of text found (see ``lettering``), in its pixels."""

    # HxW bool: the pixels whose middle lies in a line's box (see ``mask``).
    area: np.ndarray
    # HxW uint8: the ink the page prints, 0 bare paper to 255 solid: the letters' own tones, 0 at their ground's, 128
    # where their outline runs and 255 at the letters' tone, on the pixels the outline reaches and those beside them; 0
    # elsewhere. 128 and up is where a pixel's middle lies inside the outline, or all but.
    ink: np.ndarray
    # The letters' outline: closed rings of (x, y) points, pixel centers at whole numbers, each repeating its first
    # point at the end. Filled by the even-odd rule, they are the letters, solid.
    outline: list[np.ndarray]
    # HxW bool: in the boxes that print, their side darker than halfway between their two tones -- the letters, or
    # their ground where the letters are the lighter side. The tone the lettering prints in is taken from it.
    dark: np.ndarray


def lettering(page_bgr: np.ndarray, lines: list[TextLine]) -> Lettering:
    """How a page of ``page_bgr`` prints the letters in ``lines``, given in its pixels.

    A line's letters are the side of its box's lightness (CIE L*) away from
    their ground: darker or lighter, as two of three things about the box say
    (see ``_light_letters``), or, where boxes overlap -- one sign or caption
    read twice -- as the boxes holding most of their pixels say. The letters
    alone print, dark letters and light letters alike as ink on paper; their
    ground is bare paper.

    The ground is found round every pixel, not once for the box: it is the
    lightness with the letters taken out (see ``_read``), so a ground shading
    from light to dark, or a band, a panel or a glow wider than a stroke, is
    ground and prints nothing. A pixel's ink is how far it lies from its
    ground towards the letters' tone, the box's ``INK_PERCENTILE`` or its
    ``PAPER_PERCENTILE``; the letters' outline runs ``LETTER_CUT`` of the way
    from that tone to the ground's. A piece of the letters that runs along
    more than ``MAX_EDGE_SHARE`` of the box's edge is ground caught by the
    box, a sign's edge or a frame, and does not print. A line whose two tones
    stand less than ``MIN_CONTRAST`` apart prints nothing, nor one all one
    tone but specks. Where two boxes overlap, a point is a letter if either
    box says so.

    The outline is traced on a grid ``LETTERING_GRID`` times finer than the
    page, through the lightness interpolated between the page's pixels, and
    smoothed off that grid, so it runs along the letters' edges rather than
    round the page's pixels. ``Lettering.ink`` is the letters' own tones, at
    and round the outline: from bare paper at the ground's tone to solid at
    the letters', half inked at the outline.
    """
    picture = _checked(page_bgr)
    height, width = picture.shape[:2]
    area = np.zeros((height, width), dtype=bool)
    dark = np.zeros((height, width), dtype=bool)
    printed = []
    found = parallel.map_balanced(
        lambda line: _line_box(picture, line),
        lines,
        [abs(_area(line.quad)) for line in lines],
        min_pooled_cost=_POOLED_PX,
    )
    for (y0, y1, x0, x1), inside, box, darker in filter(None, found):
        area[y0:y1, x0:x1] |= inside
        if box is not None:
            dark[y0:y1, x0:x1] |= darker
            printed.append(box)

    ink = np.zeros((height, width), dtype=np.uint8)
    outline: list[np.ndarray] = []
    tolerance = TOLERANCE_MM * print_scale((width, height)).px_per_mm
    # Boxes that overlap are one sign or caption read twice, so their letters lie on one side: the side the boxes
    # holding most of their pixels say. A shaded ground can fool one box, seldom all of them.
    groups = [(window, _one_side(members)) for window, members in _groups(printed, (width, height))]
    # Each box is read, and each group's letters traced, the large ones on the shared pool (OpenCV and the tracer let
    # go of the GIL); what comes back is in the boxes' and the groups' order, whichever finishes first.
    boxes = [box for _, members in groups for box in members]
    reads = iter(
        parallel.map_balanced(
            lambda box: _read(picture, box), boxes, [box.inside.size for box in boxes], min_pooled_cost=_POOLED_PX
        )
    )
    groups = [(window, [(box, next(reads)) for box in members]) for window, members in groups]
    traced = parallel.map_balanced(
        lambda group: _group_letters(*group, tolerance),
        groups,
        [(y1 - y0) * (x1 - x0) for (y0, y1, x0, x1), _ in groups],
        min_pooled_cost=_POOLED_PX,
    )
    for ((y0, y1, x0, x1), _), (rings, local) in zip(groups, traced):
        np.maximum(ink[y0:y1, x0:x1], local, out=ink[y0:y1, x0:x1])
        outline.extend(rings)
    return Lettering(area=area, ink=ink, outline=outline, dark=dark)


def _one_side(members: list[_Box]) -> list[_Box]:
    """``members``, a group's boxes, with their letters on the side the boxes holding most of their pixels say.

    Half and half, each box keeps its own.
    """
    weights = [int(box.inside.sum()) for box in members]
    light_share = sum(w for w, box in zip(weights, members) if box.light) / sum(weights)
    if light_share == 0.5:
        return members
    return [replace(box, light=light_share > 0.5) for box in members]


def _group_letters(
    window: tuple[int, int, int, int], reads: list, tolerance: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """A group's letters, from each of its boxes as ``_read`` reads it: their outline, in the page's pixels, and the
    ink they put on the group's window (uint8)."""
    y0, y1, x0, x1 = window
    grid = LETTERING_GRID
    letters = np.zeros(((y1 - y0) * grid, (x1 - x0) * grid), dtype=bool)
    tones = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for box, (fine, (row, column), ramp) in reads:
        row, column = row - y0 * grid, column - x0 * grid
        letters[row : row + fine.shape[0], column : column + fine.shape[1]] |= fine
        by0, by1, bx0, bx1 = box.window
        local = tones[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0]
        np.maximum(local, np.where(box.inside, ramp, np.float32(0)), out=local)
    rows, columns = np.flatnonzero(letters.any(axis=1)), np.flatnonzero(letters.any(axis=0))
    rings = []
    if rows.size:
        r0, c0 = max(0, int(rows[0]) - 1), max(0, int(columns[0]) - 1)
        part = letters[r0 : int(rows[-1]) + 2, c0 : int(columns[-1]) + 2]
        rings = [ring + (c0, r0) for ring in _smoothed(traced_rings(part), tolerance)]
    # The page prints the letters' tones at and round the letters traced, and nothing elsewhere in their boxes.
    near = cv2.dilate((_filled(rings, (x1 - x0, y1 - y0)) > 0).view(np.uint8), np.ones((3, 3), np.uint8))
    # From the grid's pixels to the page's: grid pixel u's middle is (u + 0.5) / LETTERING_GRID of a page pixel in.
    outline = [(ring + 0.5) / grid - 0.5 + (x0, y0) for ring in rings]
    return outline, np.rint(tones * 255).astype(np.uint8) * near


def _line_box(picture: np.ndarray, line: TextLine):
    """A line's box as ``lettering`` reads it: (window, inside, box, darker side).

    None if no pixel's middle lies in the box. ``window`` is the rows and
    columns its pixels lie in, (y0, y1, x0, x1), and ``inside`` which of the
    window's pixels they are. ``box`` is the line as it prints, and ``darker
    side`` its pixels darker than halfway between its two tones; both None if
    it prints nothing.
    """
    quad = np.asarray(line.quad, dtype=np.float64)
    found = _box_pixels(quad, picture.shape[1::-1])
    if found is None:
        return None
    window, inside = found
    y0, y1, x0, x1 = window
    lightness = _lightness(picture[y0:y1, x0:x1])
    tones = _tones(lightness[inside])
    if tones is None:
        return window, inside, None, None
    middle = sum(tones) / 2
    box = _Box(quad, window, inside, tones, _light_letters(lightness, inside, middle))
    return window, inside, box, inside & (lightness < middle)


def _area(quad) -> float:
    """A quad's signed area, by the shoelace formula."""
    p = np.asarray(quad, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2


@dataclass(frozen=True)
class _Box:
    """A line that prints, as ``lettering`` reads it: its box, its pixels, its two tones and its letters' side."""

    quad: np.ndarray
    window: tuple[int, int, int, int]  # (y0, y1, x0, x1): the rows and columns of the page its pixels lie in
    inside: np.ndarray  # the window's pixels whose middle lies in the box
    tones: tuple[float, float]  # (dark, light): see ``_tones``
    light: bool  # whether the letters are the light side

    @property
    def ground_reach(self) -> int:
        """Half the width, in pixels, its ground is found with: ``GROUND_DISK`` of the box's height across."""
        p = self.quad
        height = min(float(np.hypot(*(p[1] - p[0]))), float(np.hypot(*(p[2] - p[1]))))
        return max(1, int(round(GROUND_DISK * height / 2)))

    @property
    def angle(self) -> float:
        """The line's slope, in radians: of its box's longer side."""
        p = self.quad
        along = max((p[1] - p[0], p[2] - p[1]), key=lambda side: float(np.hypot(*side)))
        return float(np.arctan2(along[1], along[0]))


def _disk(reach: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))


def _square(reach: int, angle: float) -> np.ndarray:
    """A square ``2 * reach + 1`` pixels across, turned by ``angle``: the pixels whose middle lies in it."""
    extent = int(np.ceil((reach + 0.5) * np.sqrt(2)))
    dy, dx = np.mgrid[-extent : extent + 1, -extent : extent + 1].astype(np.float64)
    cos, sin = np.cos(angle), np.sin(angle)
    inside = (np.abs(dx * cos + dy * sin) <= reach + 0.5) & (np.abs(-dx * sin + dy * cos) <= reach + 0.5)
    return inside.astype(np.uint8)


def _read(picture: np.ndarray, box: _Box) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """A box's letters: (letters, (row, column), ramp).

    Its ground is the picture's lightness round it with the letters taken
    out: closed (dark letters) or opened (light letters) by a disk wider than
    their strokes, which fills each stroke in with the ground either side of
    it and leaves anything wider as it is -- and by a square as wide, turned
    to the line, which keeps the corners of a band or a panel that the disk
    would round off; whichever keeps more of the picture as it is. A pixel's
    ink is how far it lies from its ground towards the letters' tone, in the
    contrast between the two -- never taken as less than ``MIN_CONTRAST``, so
    a ground as dark as the letters, or as light, prints nothing of its grain.

    ``letters`` is on the grid ``LETTERING_GRID`` times finer than the page,
    over the box's window and a page pixel round it (as far as the page goes),
    its top-left grid pixel at ``(row, column)`` of the page's grid: every
    grid pixel the box can hold lies in it. It holds the grid pixels whose
    middle lies in the box and whose ink, from the lightness and its ground
    interpolated bicubically there, is more than ``1 - LETTER_CUT``, less the
    pieces running along the box's edge. ``ramp`` is the ink of the box's
    window's pixels, 0-1: none at the ground's tone, half at the cut, solid at
    the letters' tone, and in proportion between.
    """
    height, width = picture.shape[:2]
    grid = LETTERING_GRID
    y0, y1, x0, x1 = box.window
    reach = box.ground_reach
    margin = 2 * (_square(reach, box.angle).shape[0] // 2) + _CONTEXT + 1  # what closing or opening reads round the box
    oy0, oy1, ox0, ox1 = max(0, y0 - margin), min(height, y1 + margin), max(0, x0 - margin), min(width, x1 + margin)
    lightness = _lightness(picture[oy0:oy1, ox0:ox1])
    # A disk keeps a round patch of ground, but not a square one's corners, which a square turned to the line keeps:
    # the ground is the more of the two that is ground -- the one nearer the lightness.
    operation, nearer = (cv2.MORPH_OPEN, np.maximum) if box.light else (cv2.MORPH_CLOSE, np.minimum)
    ground = nearer(
        *(
            cv2.morphologyEx(lightness, operation, element, borderType=cv2.BORDER_REPLICATE)
            for element in (_disk(reach), _square(reach, box.angle))
        )
    )
    sign = np.float32(1 if box.light else -1)  # the letters' side of their ground
    letters_tone = box.tones[1] if box.light else box.tones[0]
    cut = 1 - LETTER_CUT

    def ink(lightness: np.ndarray, ground: np.ndarray) -> np.ndarray:
        return sign * (lightness - ground) / np.maximum(sign * (letters_tone - ground), MIN_CONTRAST)

    towards = np.clip(ink(lightness, ground)[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0], 0, 1)
    ramp = np.where(towards < cut, 0.5 * towards / cut, 0.5 + 0.5 * (towards - cut) / (1 - cut))

    gy0, gy1, gx0, gx1 = max(y0 - 1, oy0), min(y1 + 1, oy1), max(x0 - 1, ox0), min(x1 + 1, ox1)
    block = (gy0 - oy0, gy1 - oy0, gx0 - ox0, gx1 - ox0)
    fine_lightness, fine_ground = _finer(lightness, *block), _finer(ground, *block)
    # The grid pixels whose middle lies in the box: grid pixel u's middle is gx0 + (u + 0.5) / grid on the page.
    inside = np.zeros(fine_lightness.shape, dtype=np.uint8)
    corners = (box.quad - (gx0, gy0)) * grid - 0.5
    cv2.fillPoly(inside, [np.rint(corners * 256).astype(np.int32)], 1, cv2.LINE_8, shift=8)
    inside = inside.view(bool)
    fine = inside & (ink(fine_lightness, fine_ground) > cut)
    count, pieces = cv2.connectedComponents(fine.view(np.uint8), connectivity=8)
    edge = inside & ~_eroded(inside)
    along = np.bincount(pieces[edge], minlength=count)
    along[0] = 0  # what is no letter
    fine &= ~(along > MAX_EDGE_SHARE * edge.sum())[pieces]
    return fine, (gy0 * grid, gx0 * grid), ramp.astype(np.float32)


def _tones(values: np.ndarray) -> tuple[float, float] | None:
    """A box's two tones, (dark, light), from its pixels' lightness ``values``; None if it has no lettering to speak of.

    Its ``INK_PERCENTILE`` and ``PAPER_PERCENTILE``. None where the two tones
    Otsu's method splits the box into stand less than ``MIN_CONTRAST`` apart,
    or where the box is all one tone but specks.
    """
    if _otsu_contrast(values) < MIN_CONTRAST:
        return None
    ink_level, paper_level = np.percentile(values, [INK_PERCENTILE, PAPER_PERCENTILE])
    if paper_level <= ink_level:
        return None
    return float(ink_level), float(paper_level)


def _lightness(window_bgr: np.ndarray) -> np.ndarray:
    """CIE L* of a BGR window, 0-100, float32."""
    window = np.ascontiguousarray(window_bgr, dtype=np.float32) / np.float32(255)
    return cv2.cvtColor(window, cv2.COLOR_BGR2Lab)[:, :, 0]


def _light_letters(lightness: np.ndarray, inside: np.ndarray, level: float) -> bool:
    """Whether a box's letters are its side lighter than ``level``, as two of three things about it say.

    Letters take up less of their box than their ground does; the ground runs
    along the box's edge, which the line found was grown out to; and the ground
    is one piece round the letters, which are many, so the largest piece of the
    ground's side holds more of it. Each is wrong on some signs -- bold lit
    letters fill most of their box, a box catches the next line or a frame
    along its edge -- but seldom two at once.
    """
    light = inside & (lightness > level)
    darker = inside & ~light
    edge = inside & ~_eroded(inside)
    votes = (
        int(light.sum() < darker.sum())
        + int(light[edge].mean() < 0.5)
        + int(_largest_share(light) < _largest_share(darker))
    )
    return votes >= 2


def _largest_share(side: np.ndarray) -> float:
    """The share of ``side``'s pixels its largest piece (8-connected) holds; 0 if it has none."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(side.view(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max() / areas.sum()) if count > 1 else 0.0


def _eroded(inside: np.ndarray) -> np.ndarray:
    """``inside`` less its edge: the pixels whose eight neighbors are all in it, off the array counting as out."""
    padded = np.pad(inside, 1).view(np.uint8)
    return cv2.erode(padded, np.ones((3, 3), np.uint8))[1:-1, 1:-1].view(bool)


def _groups(printed: list[_Box], size: tuple[int, int]):
    """The lines that print, in groups whose windows overlap: ((y0, y1, x0, x1), members) for each.

    A line's window is its box's pixels and ``_CONTEXT`` more round them, cut
    to the page; a group's is the smallest holding its lines'. A group's
    letters are traced together, so letters two boxes share are traced once.
    """
    width, height = size
    windows = np.zeros((height, width), dtype=np.uint8)
    for box in printed:
        y0, y1, x0, x1 = box.window
        windows[max(0, y0 - _CONTEXT) : y1 + _CONTEXT, max(0, x0 - _CONTEXT) : x1 + _CONTEXT] = 1
    _, labels, stats, _ = cv2.connectedComponentsWithStats(windows, connectivity=4)
    groups: dict[int, list] = {}
    for member in printed:
        y0, _, x0, _ = member.window
        groups.setdefault(int(labels[y0, x0]), []).append(member)
    for label, members in sorted(groups.items()):
        x, y, w, h = (int(v) for v in stats[label, :4])
        yield (y, y + h, x, x + w), members


def _smoothed(rings: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    """Rings traced round grid pixels, smoothed off their staircase and simplified within ``tolerance`` page pixels.

    In the grid's pixels, as ``ink_outline.traced_rings`` gives them, a point
    at every corner, each repeating its first point at the end. A ring that
    simplifies to fewer than three points is left out.
    """
    if not rings:
        return []
    counts = np.array([len(ring) - 1 for ring in rings])
    points = np.concatenate([ring[:-1] for ring in rings])
    first = np.repeat(np.cumsum(counts) - counts, counts)
    length = np.repeat(counts, counts)
    at = np.arange(len(points)) - first  # each point's place along its own ring
    smooth = np.zeros_like(points)
    for offset, weight in zip(range(-2, 3), _OUTLINE_WEIGHTS):
        # Each point's neighbor ``offset`` along its own ring, round its end and back to its start.
        smooth += weight * points[first + (at + offset) % length]
    smooth = smooth.astype(np.float32)
    kept_rings = []
    for begin, count in zip(np.cumsum(counts) - counts, counts):
        kept = cv2.approxPolyDP(smooth[begin : begin + count].reshape(-1, 1, 2), tolerance * LETTERING_GRID, True)
        if len(kept) >= 3:
            kept = kept.reshape(-1, 2).astype(np.float64)
            kept_rings.append(np.vstack([kept, kept[:1]]))
    return kept_rings


def _finer(values: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """``values[r0:r1, c0:c1]`` interpolated bicubically on the grid ``LETTERING_GRID`` times finer.

    As interpolating all of ``values`` and cutting the block out would give
    it: the block is read with the ``_CONTEXT`` pixels round it that the
    interpolation reads.
    """
    grid = LETTERING_GRID
    a0, a1 = max(0, r0 - _CONTEXT), min(values.shape[0], r1 + _CONTEXT)
    b0, b1 = max(0, c0 - _CONTEXT), min(values.shape[1], c1 + _CONTEXT)
    fine = cv2.resize(values[a0:a1, b0:b1], None, fx=grid, fy=grid, interpolation=cv2.INTER_CUBIC)
    return fine[(r0 - a0) * grid : (r1 - a0) * grid, (c0 - b0) * grid : (c1 - b0) * grid]


def _filled(rings: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """``rings``, in the grid's pixels, filled by the even-odd rule on the grid, averaged down to a ``size`` window."""
    width, height = size
    grid = LETTERING_GRID
    canvas = np.zeros((height * grid, width * grid), dtype=np.uint8)
    if rings:
        cv2.fillPoly(canvas, [np.rint(ring[:-1] * 16).astype(np.int32) for ring in rings], 255, cv2.LINE_8, shift=4)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def _otsu_contrast(values: np.ndarray) -> float:
    """How far apart the means of the two classes Otsu's method splits ``values`` into are; 0 for fewer than two."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = ordered.size
    if count < 2:
        return 0.0
    sums = np.cumsum(ordered)
    below = np.arange(1, count)  # how many values fall in the lower class, for each place the split can go
    lower = sums[:-1] / below
    upper = (sums[-1] - sums[:-1]) / (count - below)
    share = below / count
    split = int(np.argmax(share * (1 - share) * (lower - upper) ** 2))
    return float(upper[split] - lower[split])


def _checked(picture_bgr: np.ndarray) -> np.ndarray:
    picture = np.asarray(picture_bgr)
    if picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3 or 0 in picture.shape[:2]:
        raise ValueError(f"expected an HxWx3 uint8 picture, got {picture.dtype} {picture.shape}")
    return picture


def _input_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    """(width, height) the network sees a picture of ``size`` at ``scale``: each side rounded to ``SIDE_MULTIPLE``."""
    return tuple(max(SIDE_MULTIPLE, int(round(side * scale / SIDE_MULTIPLE)) * SIDE_MULTIPLE) for side in size)


def _view(picture: np.ndarray, source: np.ndarray, scale: float) -> np.ndarray:
    """The picture as the network sees it at ``scale`` times ``picture``'s size: from ``source``, shrunk or stretched."""
    width, height = _input_size(picture.shape[1::-1], scale)
    if source is picture:
        return cv2.resize(picture, (width, height))
    shrink = source.shape[1] >= width
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def _probability(view: np.ndarray) -> np.ndarray:
    """The network's map for ``view``: how likely each of its pixels is to lie in the core of a line of text."""
    global _session
    blob = ((view.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()
    with _lock:
        if _session is None:
            import onnxruntime  # here, so that importing the pipeline doesn't pay for it

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = parallel.worker_count()
            # Memory goes back after each look rather than staying in ONNX Runtime's arena, which grows to ~1 GB over
            # pictures of a few sizes and keeps it; it costs ~10% of the time.
            options.enable_cpu_mem_arena = False
            options.log_severity_level = 3  # errors only
            _session = onnxruntime.InferenceSession(
                (RESOURCES_DIR / MODEL).read_bytes(), options, providers=["CPUExecutionProvider"]
            )
        session = _session
    # One session runs from several threads at once; its result doesn't depend on how many threads it uses.
    probability = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return probability.reshape(probability.shape[-2:])


def _lines(view: np.ndarray, size: tuple[int, int]) -> list[TextLine]:
    """The lines the network finds in ``view``, given in the pixels of the picture of ``size`` (width, height) it shows.

    Only lines that look like one are kept (see ``MAX_BOX_HEIGHT_MM``).
    """
    probability = _probability(view)
    map_height, map_width = probability.shape
    sx, sy = size[0] / map_width, size[1] / map_height
    max_height_px = print_scale(size).mm_to_px(MAX_BOX_HEIGHT_MM)
    # Grown by a pixel right and down, so that a core's outline, through its pixels' middles, spans its pixels' corners.
    cores = cv2.dilate((probability > THRESHOLD).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(cores, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours[:_MAX_CANDIDATES]:
        center, (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) < _MIN_CORE_PX:
            continue
        score = _mean_inside(probability, cv2.boxPoints((center, (w, h), angle)))
        if score < MIN_SCORE:
            continue
        grow = w * h * UNCLIP / (2 * (w + h))
        w, h = w + 2 * grow, h + 2 * grow
        if min(w, h) < _MIN_BOX_PX:
            continue
        corners = cv2.boxPoints((center, (w, h), angle)).astype(np.float64) * (sx, sy)
        line = TextLine(quad=tuple((float(x), float(y)) for x, y in corners), score=score)
        height, length = line.sides
        if height <= max_height_px and length >= MIN_ELONGATION * height:
            lines.append(line)
    return lines


def _mean_inside(probability: np.ndarray, corners: np.ndarray) -> float:
    """The mean of ``probability`` over the box with these corners, filled as PaddleOCR fills it (corners truncated)."""
    height, width = probability.shape
    x0 = int(np.clip(np.floor(corners[:, 0].min()), 0, width - 1))
    x1 = int(np.clip(np.ceil(corners[:, 0].max()), 0, width - 1))
    y0 = int(np.clip(np.floor(corners[:, 1].min()), 0, height - 1))
    y1 = int(np.clip(np.ceil(corners[:, 1].max()), 0, height - 1))
    inside = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillPoly(inside, [(corners - (x0, y0)).astype(np.int32)], 1)
    return float(cv2.mean(probability[y0 : y1 + 1, x0 : x1 + 1], inside)[0])


def _reading_order(line: TextLine) -> tuple[float, float]:
    quad = np.asarray(line.quad)
    return float(quad[:, 1].mean()), float(quad[:, 0].mean())

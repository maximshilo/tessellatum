"""Where each region's number goes on the page, and how large it prints.

A number is legible only if it is large enough on paper and nothing else is
drawn through it. So every number is at least ``print_size.MIN_LABEL_SIZE_PT``,
and it goes where its text box holds no line ink at all -- not even the pale,
anti-aliased edge of a line -- nor any other number.

Most numbers go where they always did: centered on the point of the region
farthest from its edge, at a size that grows with that distance. A number
that does not fit there is moved to wherever in its region it keeps farthest
from the lines, and made smaller, down to the smallest legible size, if it
still does not fit. A region too small to hold even that has its number
written just outside it, in a neighbor, with a leader line pointing in. On a
page laid out for vectors, a number with no room for that either goes smaller
inside its region, down to half the smallest size (``CROWDED_SIZE_RATIO``),
before it is written where lines run through it.

No number goes on a line of text either, nor right beside one, where it
would read as part of a sign or a caption: every number keeps about
``TEXT_GAP_MM`` from the lines of text found. A region lying in one has its
number written outside it, with a leader that crosses the lettering to reach
it.

A region pinched so thin somewhere that it reads on paper as two regions
gets its number on each side (see ``find_sides``): its own number where it
always went, then, once every region has that, one more on each other side,
in the room left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core import kernels
from tessellatum.core.print_size import A4, MIN_LABEL_SIZE_PT, PaperFormat, print_scale

MAX_FONT_SIZE = 40  # px
FONT_SIZE_RADIUS_RATIO = 0.85
# A number drawn in pixels is never smaller than this, whatever the paper asks
# for. A preview is made at as few as 1100 px, about 100 dpi on A4 and 50 on
# A2, where the smallest legible size on paper is too few pixels to draw a
# digit with: at 10 px a digit is 8 px tall. A page laid out to print, to be
# drawn as vectors (every export, see ``min_font_size``), has no such floor.
MIN_FONT_SIZE = 10  # px

# How finely a number laid out for vectors shrinks to fit, in pixels an em: its
# sizes are fractions of a pixel, and of those only the largest with each text
# box is tried (see ``_sizes``).
EXACT_SIZE_STEP = 0.125  # px
# A number laid out for vectors that finds no room anywhere at the page's smallest size may go this much smaller inside
# its own region, never under the smallest size a number can be asked for (the setting's 1 pt), rather than be written
# where lines run through it: a smaller number reads, one crossed by lines doesn't (T2c, D-014).
CROWDED_SIZE_RATIO = 0.5
CROWDED_MIN_LABEL_PT = 1.0
# The em Pillow measures a number's outline at, for a layout in exact sizes:
# large enough that rounding its measurements to whole pixels is a thousandth
# of the em.
MEASURING_EM_PX = 1000

# How far from the region a number with a leader may be written, on paper. Any
# farther and the line pointing back stops reading as belonging to the region.
LEADER_REACH_MM = 8.0

# How far a number keeps from a line of text found, on paper: far enough not
# to read as one more word of it. At twice that, a number on the benchmark's
# Times Square found no room left and landed on a line. It is measured as the
# page's pixels see it, between the middles of the number's pixels and of the
# pixels whose middle lies in the line's box, so the bare paper between the
# number and the box itself can be up to about a pixel and a half less (0.14 mm
# at the least on the benchmark's previews, where a pixel is 0.25 mm).
TEXT_GAP_MM = 0.5

# A region whose paper narrows to less than this between its lines, on paper, reads as two regions there, and each side
# gets its number (see ``find_sides``). Three quarters of the narrowest neck the user judged to read as one region, on
# the lion at 2047 px: its paper was 3 px across there, 0.41-0.54 mm.
SPLIT_NECK_MM = 0.35


@dataclass
class Label:
    """A region's number as drawn on the page."""

    region_id: int
    text: str
    font_size: float  # px: the font's em size; whole pixels unless ``exact``
    box: tuple[float, float, float, float]  # (x0, y0, x1, y1): the text's bounding box on the page
    # A number written outside its region: the leader line from the number to the point in the region it
    # points at, as ((x, y), (x, y)) with pixel centers at integer coordinates, like the page's lines.
    leader: tuple[tuple[float, float], tuple[float, float]] | None = None
    # A number written on hatching: the region's own detail ink under its box, and a gap round it, is not printed
    # (see ``cleared``).
    clears: bool = False
    # A number that found no room anywhere, written where lines may run through it (the last resort of ``place_labels``).
    cramped: bool = False
    # 0 for the region's own number; k for the number on the k-th of its other sides, where it is pinched into more
    # than one (see ``find_sides``). Only the region's own number decides anything about its region.
    side: int = 0
    # Laid out to be drawn as vectors (the PDF): ``box`` is the number's outline at ``font_size``, any fraction of a
    # pixel, measured exactly and rounded out to whole pixels, the number in its middle (see ``text_box_size``).
    # Otherwise it is the box Pillow draws the number in at a whole-pixel size.
    exact: bool = False


@dataclass(frozen=True)
class LabelSpacing:
    """How much room the placement leaves, in pixels of the page it is placing numbers on."""

    min_font_size: float  # the smallest em size a number is drawn at (see ``min_font_size``)
    label_gap_px: float  # the least space between a number written outside its region and any other number
    leader_width_px: float  # how wide a leader line is inked
    leader_dot_px: float  # how wide the dot a leader ends in, at the point of its region it points at, is inked
    leader_reach_px: float  # how far from its region a number with a leader may go
    text_gap_px: float = 0.0  # how far a number keeps from a line of text
    # A neck of a region's paper narrower than this splits it into sides, each numbered (see ``find_sides``); None
    # numbers every region once.
    split_neck_px: float | None = None
    min_side_px: float = 0.0  # the smallest area, in pixels, a side has to have to count as one
    # Numbers laid out to be drawn as vectors, in sizes of any fraction of a pixel (see ``Label.exact``); otherwise
    # in whole pixels.
    exact: bool = False
    # The smallest em size a number with no room anywhere at ``min_font_size`` shrinks to inside its region before it
    # is written across lines (see ``CROWDED_SIZE_RATIO``); None, never smaller than ``min_font_size``.
    crowded_font_size: float | None = None


def min_font_size(
    size: tuple[int, int], min_pt: float = MIN_LABEL_SIZE_PT, exact: bool = False, paper: PaperFormat = A4
) -> float:
    """The smallest em size, in pixels, a number on a page of ``size`` made for ``paper`` is drawn at.

    That is the smallest that prints at ``min_pt`` -- by default the smallest
    that prints legibly (``MIN_LABEL_SIZE_PT``). On a page drawn in pixels it
    is a whole number of them, and never less than ``MIN_FONT_SIZE``; on one
    laid out to be drawn as vectors (``exact``, see ``Label.exact``) it is any
    fraction of a pixel, as small as ``min_pt`` asks. The size in points is
    checked with the same arithmetic the benchmark measures it with, so a
    number of this size is never scored a hair under it.
    """
    scale = print_scale(size, paper)
    if exact:
        exact_size = scale.pt_to_px(min_pt)
        while scale.px_to_pt(exact_size) < min_pt:
            exact_size = math.nextafter(exact_size, math.inf)
        return exact_size
    font_size = max(MIN_FONT_SIZE, math.floor(scale.pt_to_px(min_pt)))
    while scale.px_to_pt(font_size) < min_pt:
        font_size += 1
    return font_size


def place_labels(
    regions,
    region_id_map: np.ndarray,
    free: np.ndarray,
    spacing: LabelSpacing,
    clearable: np.ndarray | None = None,
    text: np.ndarray | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> list[Label]:
    """A number for every region in ``regions``, placed where no line runs through it.

    ``free`` is True where the page's lines put no ink at all. A number's text
    box is kept inside one region's free pixels, so two numbers written inside
    their own regions can never touch: a line lies between them. The numbers
    that need a leader are placed after all the others, in whatever room those
    leave (see ``_LeaderRoom``).

    ``clearable`` (HxW int32) is line art's detail ink, which a region's paint
    goes over whole (see ``regions.detail_ink``): the id of the region it is
    in, -1 elsewhere. ``region_id_map`` has all the ink the page prints in no
    region, detail ink too, since no number or leader may go on it. Where a
    number has no room anywhere near without landing on ink -- in hatching --
    it goes in its own region with the detail ink under it cleared (see
    ``cleared``), rather than on the ink.

    ``text`` (HxW bool) is the lines of text on the page, their boxes: no
    number goes on them, nor within ``spacing.text_gap_px`` of them (from the
    middle of any of its pixels to the middle of any of theirs), whether in
    its region, beside it or on hatching. A leader may cross them, to reach
    a region lying in one. Only a number with no room anywhere, which lands
    where lines may run through it, may land on text too.

    ``check_cancelled``, if given, is called before each number is tried in
    its region, and again before each with no room there is written beside
    it, and stops the placement by raising: a page of thousands of regions
    takes seconds to number.

    With ``spacing.split_neck_px``, a region whose paper is pinched into more
    than one side (see ``find_sides``) gets a number on each side but the one
    its own number went on, ``Label.side`` 1, 2, ...: placed after every
    region's own number, which is therefore placed exactly as without them,
    in its side as a region's would be -- at its middle, or where it keeps
    farthest from the lines, else outside with a leader -- in the room the
    numbers before it left.

    Returns the labels in drawing order: the numbers inside their regions, in
    the order of ``regions``, then the others, then the numbers on other sides.
    """
    # Where a region's sides are found: the page's paper, before the gap round the text is taken off it, and the detail
    # ink a region's paint goes over, which is part of the region and parts nothing (hatching is not a boundary).
    side_ids, paper = ids_for_sides(region_map=region_id_map, clearable=clearable), free
    if clearable is not None:
        paper = free | (np.asarray(clearable) >= 0)
    if text is not None and text.any():
        near_text = _grown(np.asarray(text, dtype=bool), spacing.text_gap_px)
        free = free & ~near_text
        if clearable is not None:
            clearable = np.where(near_text, np.int32(-1), clearable)
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    height, width = ids.shape
    labels: list[Label] = []
    homeless = []
    bounds = None  # every region's bounding box, found the first time a number does not fit at its region's middle
    for region in regions:
        if check_cancelled is not None:
            check_cancelled()
        label = _label_at_middle(region, ids, free, spacing.min_font_size, spacing.exact)
        if label is None:
            if bounds is None:
                # Sized for every region asked about, in case one is not on the map: it then has no room of its own.
                num_ids = max(int(ids.max()), max(r.region_id for r in regions)) + 1
                bounds = kernels.region_bounds(ids.reshape(-1), height, width, num_ids)[0]
            label = _label_inside(region, ids, free, bounds, spacing.min_font_size, spacing.exact)
        if label is None:
            homeless.append(region)
        else:
            labels.append(label)

    room = None
    if homeless:
        room = _LeaderRoom(ids, free, labels, [region.interior_point for region in homeless], spacing, clearable)
        for region in homeless:
            if check_cancelled is not None:
                check_cancelled()
            labels.append(room.place(region))

    if spacing.split_neck_px is not None:
        sides = find_sides(side_ids, paper, spacing.split_neck_px, spacing.min_side_px)
        if sides:
            if room is None:
                room = _LeaderRoom(ids, free, labels, [], spacing, clearable)
            labels += _side_labels(regions, sides, labels, room, check_cancelled)
    return labels


def ids_for_sides(region_map: np.ndarray, clearable: np.ndarray | None) -> np.ndarray:
    """``region_map`` as ``place_labels`` takes it, all printed ink -1, with the detail ink back in its region."""
    ids = np.ascontiguousarray(region_map, dtype=np.int32)
    if clearable is None:
        return ids
    clearable = np.asarray(clearable)
    return np.where(clearable >= 0, clearable, ids).astype(np.int32)


@dataclass(frozen=True)
class Sides:
    """The sides a region is pinched into (see ``find_sides``), in a window of the page from (``x0``, ``y0``)."""

    x0: int
    y0: int
    map: np.ndarray  # int32, the window's size: k + 1 on the paper of the region's k-th side, 0 elsewhere
    points: list[tuple[int, int]]  # each side's point farthest from its lines, (x, y) on the page
    radii: list[float]  # how far that point is from them, in pixels
    boxes: list[tuple[int, int, int, int]]  # each side's inclusive (x0, y0, x1, y1) on the page

    def side(self, k: int) -> tuple[np.ndarray, int, int]:
        """Side ``k``'s pixels (bool) in its box, and the box's (x0, y0) on the page."""
        x0, y0, x1, y1 = self.boxes[k]
        window = (slice(y0 - self.y0, y1 + 1 - self.y0), slice(x0 - self.x0, x1 + 1 - self.x0))
        return self.map[window] == k + 1, x0, y0

    def side_at(self, point: tuple[float, float]) -> int:
        """The side ``point``, (x, y) on the page, is on, or the nearest to it."""
        x, y = int(round(point[0])) - self.x0, int(round(point[1])) - self.y0
        rows, columns = self.map.shape
        if 0 <= y < rows and 0 <= x < columns and self.map[y, x] > 0:
            return int(self.map[y, x]) - 1
        ys, xs = np.nonzero(self.map)
        nearest = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return int(self.map[ys[nearest], xs[nearest]]) - 1


def find_sides(
    ids: np.ndarray, paper: np.ndarray, neck_px: float, min_area_px: float
) -> dict[int, Sides]:
    """The regions of ``ids`` that read as more than one region on paper, by region id, and their sides.

    A region's paper is its pixels the lines leave bare (``paper``). Where the
    lines close it off, or narrow it to less than ``neck_px`` across, it reads
    as two regions. So its sides are the pieces of its paper the lines part,
    each split again at every neck narrower than ``neck_px``: the parts of it
    a disk that wide fits in (an opening's core, the paper eroded by the disk),
    grown back over the piece, each pixel to the nearest. A piece with no such
    part is a side whole. A side counts only if its paper is at least
    ``min_area_px``, as large as a region has to be. Regions with fewer than
    two sides are left out.

    A neck is measured in whole pixels of paper across: one at least
    ``neck_px`` across, rounded up, holds the region together. A region's
    side map covers its sides' boxes: its paper there can run past the region
    map's box, as ``ids`` gives detail ink back to its region.
    """
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    height, width = ids.shape
    # The regions' paper, cut where two regions meet should no line run between them: of every two pixels side by side
    # in different regions, the one to the right or below is left out, so that no piece of paper runs into two regions.
    bare = paper & (ids >= 0)
    bare[:, 1:] &= (ids[:, 1:] == ids[:, :-1]) | (ids[:, :-1] < 0)
    bare[1:, :] &= (ids[1:, :] == ids[:-1, :]) | (ids[:-1, :] < 0)
    bare8 = bare.astype(np.uint8)
    if not bare8.any():
        return {}
    clearance = _clearance(bare)  # for each side's middle
    # Eroded by a disk as many pixels across as the setting, rounded up: a neck of fewer pixels of paper is cut. (A
    # distance from the paper's edge would tell a neck of 2n - 1 pixels from one of 2n no better than a pixel.)
    across = int(math.ceil(neck_px - 1e-9))
    core = bare8
    if across > 1:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (across, across))
        core = cv2.erode(bare8, disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    num_pieces, pieces = cv2.connectedComponents(bare8, connectivity=4)
    num_cores, cores = cv2.connectedComponents(core, connectivity=4)

    # Every pixel of paper's side, numbered over the page: a piece with fewer than two cores is a side, numbered as
    # the piece; one with more is parted between its cores, each pixel to the nearest along a path through the piece.
    flat_ids = ids.ravel()
    flat_side, side_area, side_region, side_box = kernels.paper_sides(
        pieces.ravel(), cores.ravel(), flat_ids, height, width, num_pieces, num_cores
    )
    side = flat_side.reshape(height, width)
    # The sides that count, by region: those of the regions with two or more.
    counted = np.nonzero((side_area >= min_area_px) & (side_region >= 0))[0]
    counted = counted[np.argsort(side_region[counted], kind="stable")]
    starts = np.flatnonzero(np.r_[True, np.diff(side_region[counted]) != 0, True])
    groups = [counted[a:b] for a, b in zip(starts[:-1], starts[1:]) if b - a >= 2]
    if not groups:
        return {}

    found = {}
    for its_sides in groups:
        region_id = int(side_region[its_sides[0]])
        x0, y0 = (int(v) for v in side_box[its_sides, :2].min(axis=0))
        x1, y1 = (int(v) for v in side_box[its_sides, 2:].max(axis=0))
        side_map = np.zeros((y1 + 1 - y0, x1 + 1 - x0), dtype=np.int32)
        points, radii, boxes = [], [], []
        for index, k in enumerate(its_sides, start=1):
            sx0, sy0, sx1, sy1 = (int(v) for v in side_box[k])
            box = (slice(sy0, sy1 + 1), slice(sx0, sx1 + 1))
            on_side = side[box] == k
            side_map[sy0 - y0 : sy1 + 1 - y0, sx0 - x0 : sx1 + 1 - x0][on_side] = index
            y, x = _farthest(clearance[box], on_side)
            points.append((x + sx0, y + sy0))
            radii.append(float(clearance[box][y, x]))
            boxes.append((sx0, sy0, sx1, sy1))
        found[region_id] = Sides(x0, y0, side_map, points, radii, boxes)
    return found


def _side_labels(
    regions, sides: dict[int, Sides], labels: list[Label], room: _LeaderRoom, check_cancelled: Callable[[], None] | None
) -> list[Label]:
    """A number on each side of a pinched region but the side its own number is on (see ``place_labels``)."""
    at = {}  # where each region's own number is: the point its leader points at, or its box's middle
    for label in labels:
        if label.side == 0:
            box = label.box
            at[label.region_id] = label.leader[1] if label.leader is not None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    placed = []
    for region in regions:
        found = sides.get(region.region_id)
        if found is None or region.region_id not in at:
            continue
        own = found.side_at(at[region.region_id])
        number = 0
        for index, (point, radius) in enumerate(zip(found.points, found.radii)):
            if index == own:
                continue
            if check_cancelled is not None:
                check_cancelled()
            number += 1  # counted whether or not it finds room, so a number names its side
            side = replace(region, interior_point=point, interior_radius=radius)
            label = room.place_on_side(side, *found.side(index))
            if label is not None:
                placed.append(replace(label, side=number))
    return placed


def cleared(labels: list[Label], clearable: np.ndarray, gap_px: float) -> np.ndarray:
    """HxW bool: the detail ink the numbers written on hatching clear, their box and ``gap_px`` round it.

    ``clearable`` is as ``place_labels`` took it, and ``gap_px`` the
    ``LabelSpacing.label_gap_px`` the numbers were placed with. A number
    clears only its own region's detail ink.
    """
    clear = np.zeros(clearable.shape, dtype=bool)
    for label in labels:
        if label.clears:
            spaced = _spaced_slices(label.box, gap_px)
            clear[spaced] |= clearable[spaced] == label.region_id
    return clear


def _spaced_slices(box, gap_px: float) -> tuple[slice, slice]:
    """The rows and columns of a number's ``box`` and the gap of ``gap_px`` kept round it, cut to the page's top left."""
    x0, y0, x1, y1 = (int(math.floor(box[0])), int(math.floor(box[1])), int(math.ceil(box[2])), int(math.ceil(box[3])))
    gap = int(math.ceil(gap_px))
    return slice(max(0, y0 - gap), y1 + gap), slice(max(0, x0 - gap), x1 + gap)


def text_box_size(text: str, font_size: float, exact: bool = False) -> tuple[int, int]:
    """(width, height) of ``text``'s bounding box at ``font_size``, in whole pixels.

    That is the box Pillow draws it in at that size or, with ``exact`` (see
    ``Label.exact``), its outline at that size, rounded out.
    """
    if exact:
        left, top, right, bottom = outline_bbox(text, font_size)
        return max(1, math.ceil(right - left - 1e-9)), max(1, math.ceil(bottom - top - 1e-9))
    left, top, right, bottom = text_bbox(text, font_size)
    return right - left, bottom - top


def outline_bbox(text: str, font_size: float) -> tuple[float, float, float, float]:
    """The bounding box of ``text``'s outline at ``font_size``, any fraction of a pixel, from the start of its
    baseline, y down."""
    scale = font_size / MEASURING_EM_PX
    left, top, right, bottom = baseline_bbox(text, MEASURING_EM_PX)
    return left * scale, top * scale, right * scale, bottom * scale


def _preferred_size(region, min_size: float) -> float:
    """The size a region's number is drawn at if it fits: larger the farther its middle is from its edge, in whole
    pixels above ``min_size``."""
    return max(min_size, int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))


def _sizes(text: str, largest: float, smallest: float, exact: bool):
    """(font size, box width, box height) of each size a number shrinks through to fit, ``largest`` to ``smallest``.

    In whole pixels; or with ``exact``, every ``EXACT_SIZE_STEP`` and
    ``smallest`` last, each the largest size of its box -- a smaller size
    with the same box fits nowhere the larger one doesn't.
    """
    if not exact:
        for font_size in range(int(largest), int(smallest) - 1, -1):
            yield (font_size, *text_box_size(text, font_size))
        return
    steps = max(0, math.ceil((largest - smallest) / EXACT_SIZE_STEP - 1e-9))
    last = None
    for k in range(steps + 1):
        font_size = largest - k * EXACT_SIZE_STEP if k < steps else smallest
        box = text_box_size(text, font_size, exact=True)
        if box != last:
            last = box
            yield (font_size, *box)


def _centered_box(point, box_width: int, box_height: int, page_size: tuple[int, int]) -> tuple[int, int]:
    """(left, top) of a ``box_width`` x ``box_height`` box centered on ``point``, kept on a ``page_size`` page if it fits."""
    x, y = point
    width, height = page_size
    return max(0, min(x - box_width // 2, width - box_width)), max(0, min(y - box_height // 2, height - box_height))


def _label_at_middle(
    region, ids: np.ndarray, free: np.ndarray, min_size: float, exact: bool = False
) -> Label | None:
    """``region``'s number where it always went, if it fits there: at its preferred size, on the region's middle."""
    text = str(region.color_index + 1)
    font_size = _preferred_size(region, min_size)
    box_width, box_height = text_box_size(text, font_size, exact)
    height, width = ids.shape
    if box_width > width or box_height > height:
        return None
    left, top = _centered_box(region.interior_point, box_width, box_height, (width, height))
    window = (slice(top, top + box_height), slice(left, left + box_width))
    if free[window].all() and (ids[window] == region.region_id).all():
        return Label(region.region_id, text, font_size, (left, top, left + box_width, top + box_height), exact=exact)
    return None


def _label_inside(
    region, ids: np.ndarray, free: np.ndarray, bounds, min_size: float, exact: bool = False
) -> Label | None:
    """``region``'s number wherever in the region it keeps farthest from the lines, shrinking it if it has to.

    None if it does not fit anywhere in the region even at ``min_size``.
    """
    text = str(region.color_index + 1)
    x0, y0, x1, y1 = (int(v) for v in bounds[region.region_id])
    room = free[y0 : y1 + 1, x0 : x1 + 1] & (ids[y0 : y1 + 1, x0 : x1 + 1] == region.region_id)
    clearance = _clearance(room)
    for font_size, box_width, box_height in _sizes(text, _preferred_size(region, min_size), min_size, exact):
        spot = _roomiest_box(clearance, box_width, box_height)
        if spot is not None:
            left, top = x0 + spot[0], y0 + spot[1]
            return Label(region.region_id, text, font_size, (left, top, left + box_width, top + box_height), exact=exact)
    return None


class _LeaderRoom:
    """The page's room for numbers written outside their regions, and what each one placed takes of it.

    Every leader ends in a dot at its region's most interior point, so all the
    dots are known before any of these numbers is placed. A number goes:

    * in the nearest room outside its region, within the leader's reach, out
      of reach of every dot -- its own too -- and clear of every other number
      and leader, whose leader runs from the region into the number's region
      and nowhere else, keeping clear of every other number, leader and dot;
    * failing that, the same with a leader that may cross other regions;
    * failing that, inside its own region with the detail ink under it
      cleared, if it has room there once that ink is gone: its box and the gap
      kept round it lie on the region's free pixels and its detail ink
      (``clearable``) alone, clear of every other number, leader and dot, as
      near the region's middle as it can be;
    * failing that -- there is no room at all -- as near the region's middle as
      it can be without landing on another number or leader, lines or not,
      which the benchmark reports.
    """

    def __init__(
        self,
        ids: np.ndarray,
        free: np.ndarray,
        labels: list[Label],
        anchors,
        spacing: LabelSpacing,
        clearable: np.ndarray | None = None,
    ) -> None:
        self.ids, self.free, self.spacing, self.clearable = ids, free, spacing, clearable
        self.anchors = list(anchors)
        # A line's ink reaches half its width and a pixel of anti-aliasing from its middle, and a dot's likewise.
        self.line_reach = spacing.leader_width_px / 2 + 1
        self.dot_reach = spacing.leader_dot_px / 2 + 1
        # A leader is checked at the pixels its middle passes through, which can lie most of a pixel off it, so what
        # it must keep clear of is kept a pixel farther away than its ink reaches.
        self.line_clear = self.line_reach + 1
        shape = ids.shape
        self.boxes = np.zeros(shape, dtype=bool)  # every number's box
        self.spaced = np.zeros(shape, dtype=bool)  # every number's box and the gap kept around it
        self.leaders = np.zeros(shape, dtype=bool)  # where the leaders placed so far can put ink, dots aside
        for label in labels:
            self._take_box(label.box)
        # How many dots put ink on each pixel, and how many a leader through it would come too near: counted once, so
        # that a number's own dot can be told from the others' by taking it away.
        self.dot_count = _disk_count(shape, self.anchors, self.dot_reach)
        self.near_dot_count = _disk_count(shape, self.anchors, self.dot_reach + self.line_clear)
        self.dots = self.dot_count > 0  # where every dot puts ink

    def place(self, region, last_resort: bool = True) -> Label | None:
        """``region``'s number, its leader pointing at its ``interior_point``, which is one of the room's anchors.

        Without ``last_resort``, None where the number has no room at all, rather than written where lines run through
        it.
        """
        ids, spacing = self.ids, self.spacing
        text = str(region.color_index + 1)
        font_size, exact = spacing.min_font_size, spacing.exact
        box_width, box_height = text_box_size(text, font_size, exact)
        height, width = ids.shape
        x, y = region.interior_point
        anchor = (float(x), float(y))

        # Every box and every leader that could be chosen lies in this window around the point the leader points at.
        reach = int(math.ceil(spacing.leader_reach_px))
        wx0, wy0 = max(0, x - reach - box_width), max(0, y - reach - box_height)
        wx1, wy1 = min(width, x + reach + box_width + 1), min(height, y + reach + box_height + 1)
        window = (slice(wy0, wy1), slice(wx0, wx1))
        local_ids = ids[window]
        busy = self.spaced[window] | self.leaders[window]

        room = self.free[window] & ~busy & ~self.dots[window] & (local_ids != region.region_id)
        tops, lefts, distance = _nearest_boxes(room, box_width, box_height, (x - wx0, y - wy0))
        within = distance <= spacing.leader_reach_px
        tops, lefts, distance = tops[within], lefts[within], distance[within]
        local_anchor = [(x - wx0, y - wy0)]
        own_dot = _disks(busy.shape, local_anchor, self.dot_reach)

        if len(tops):  # on a page crowded with numbers, many have no room in reach at all
            # A leader keeps its ink off every number, every leader placed so far and every other dot -- those just
            # outside the window too, so they are grown from a margin around it.
            margin = int(math.ceil(self.line_clear))
            mx0, my0 = max(0, wx0 - margin), max(0, wy0 - margin)
            around = (slice(my0, wy1 + margin), slice(mx0, wx1 + margin))
            grown = _grown(self.boxes[around] | self.leaders[around], self.line_clear)
            blocked = grown[wy0 - my0 : wy1 - my0, wx0 - mx0 : wx1 - mx0]
            # Beyond its own dot, that is: two points close enough have dots that touch whatever their leaders do.
            near_own = _disks(blocked.shape, local_anchor, self.dot_reach + self.line_clear)
            blocked |= (self.near_dot_count[window] > near_own) & ~own_dot
            # A leader that runs from the region straight into the number's region, and nowhere else, can only end in a
            # region touching this one: it steps from pixel to pixel, diagonals included.
            own = (local_ids == region.region_id).astype(np.uint8)
            neighbors = np.unique(local_ids[cv2.dilate(own, np.ones((3, 3), np.uint8)).astype(bool) & ~own.astype(bool)])
            region_of = local_ids[tops, lefts]
            next_door = np.isin(region_of, neighbors)

            # The boxes in order, first those next door whose leader crosses only the two regions, then any whose leader
            # is clear; a box too near to leave room for a leader is passed over (only a dot smaller than its line
            # allows it). Tried in compiled code: a page crowded with numbers tries hundreds of thousands of leaders.
            k = kernels.first_leader(
                float(x), float(y), lefts.astype(np.int64), tops.astype(np.int64), box_width, box_height,
                float(self.line_reach), np.ascontiguousarray(blocked), np.ascontiguousarray(local_ids, dtype=np.int32),
                int(region.region_id), region_of.astype(np.int32), next_door, wx0, wy0,
            )
            if k >= 0:
                left, top = int(lefts[k]) + wx0, int(tops[k]) + wy0
                box = (left, top, left + box_width, top + box_height)
                end = _leader_end(anchor, box, self.line_reach)
                self._take_box(box)
                self._take_leader(_pixels_along(anchor, end))
                return Label(region.region_id, text, font_size, box, leader=(end, anchor), exact=exact)

        others_dots = self.dot_count[window] > own_dot
        if self.clearable is not None:
            # In hatching: in the region itself, where clearing its detail ink leaves room.
            gap = int(math.ceil(spacing.label_gap_px))
            own = ((local_ids == region.region_id) & self.free[window]) | (self.clearable[window] == region.region_id)
            room = own & ~busy & ~others_dots
            nearest = _nearest_box(room, box_width + 2 * gap, box_height + 2 * gap, (x - wx0, y - wy0))
            if nearest is not None:
                left, top = nearest[1] + gap + wx0, nearest[0] + gap + wy0
                box = (left, top, left + box_width, top + box_height)
                self._take_box(box)
                return Label(region.region_id, text, font_size, box, clears=True, exact=exact)

        # No room anywhere near: at the region's middle, or as near it as the number can be without landing on
        # another number, a leader or a dot.
        if not last_resort:
            return None
        clear = ~busy & ~others_dots
        nearest = _nearest_box(clear, box_width, box_height, (x - wx0, y - wy0))
        if nearest is not None and nearest[2] <= spacing.leader_reach_px:
            left, top = nearest[1] + wx0, nearest[0] + wy0
        else:
            left, top = _centered_box((x, y), box_width, box_height, (width, height))
        box = (left, top, left + box_width, top + box_height)
        if spacing.crowded_font_size is not None and not self.free[top : top + box_height, left : left + box_width].all():
            # Lines would run through it: smaller, in the region's own room, if it fits there.
            own = (local_ids == region.region_id) & self.free[window] & ~busy & ~others_dots
            clearance = _clearance(own)
            largest = font_size - EXACT_SIZE_STEP
            for smaller, small_width, small_height in _sizes(text, largest, spacing.crowded_font_size, exact):
                spot = _roomiest_box(clearance, small_width, small_height)
                if spot is not None:
                    left, top = spot[0] + wx0, spot[1] + wy0
                    box = (left, top, left + small_width, top + small_height)
                    self._take_box(box)
                    return Label(region.region_id, text, smaller, box, exact=exact)
        self._take_box(box)
        return Label(region.region_id, text, font_size, box, cramped=True, exact=exact)

    def place_on_side(self, side, room: np.ndarray, x0: int, y0: int) -> Label | None:
        """The number of a region's other ``side``, placed after every region's own (see ``place_labels``).

        ``side`` is the region with its ``interior_point`` and ``interior_radius`` those of the side, and ``room``
        (bool) the side's pixels, in a window of the page from (``x0``, ``y0``). As a region's own number goes: at the
        side's middle at its preferred size, else where in the side it keeps farthest from the lines, shrinking to
        the smallest size, else outside with a leader (see ``place``) -- clear of every number, leader and dot placed so far.
        None if it has no room in any of those: a number written over the lines would make the page harder to read,
        not easier. A leader points at the side's middle, or if its dot would touch a number, leader or dot placed
        before, at the side's pixel farthest from its lines that keeps the dot clear of them.
        """
        on_side = room
        rows, columns = room.shape
        window = (slice(y0, y0 + rows), slice(x0, x0 + columns))
        room = room & self.free[window] & ~self.spaced[window] & ~self.leaders[window] & ~self.dots[window]
        text = str(side.color_index + 1)
        min_size, exact = self.spacing.min_font_size, self.spacing.exact
        font_size = _preferred_size(side, min_size)
        box_width, box_height = text_box_size(text, font_size, exact)
        x, y = side.interior_point
        left, top = x - x0 - box_width // 2, y - y0 - box_height // 2
        spot = None
        if 0 <= left and left + box_width <= columns and 0 <= top and top + box_height <= rows:
            if room[top : top + box_height, left : left + box_width].all():
                spot = (left, top)
        if spot is None:
            clearance = _clearance(room)
            for font_size, box_width, box_height in _sizes(text, font_size, min_size, exact):
                spot = _roomiest_box(clearance, box_width, box_height)
                if spot is not None:
                    break
        if spot is not None:
            box = (spot[0] + x0, spot[1] + y0, spot[0] + x0 + box_width, spot[1] + y0 + box_height)
            self._take_box(box)
            return Label(side.region_id, text, font_size, box, exact=exact)
        point = self._side_anchor(on_side, x0, y0)
        if point is None:
            return None
        side = replace(side, interior_point=point)
        self._anchor(point, 1)
        label = self.place(side, last_resort=False)
        if label is None:
            self._anchor(point, -1)
        return label

    def _side_anchor(self, on_side: np.ndarray, x0: int, y0: int) -> tuple[int, int] | None:
        """Where a leader to a side (``on_side``, bool, in a window of the page from (``x0``, ``y0``)) points: the
        side's pixel farthest from its lines whose dot keeps clear of every number, leader and dot placed so far.
        None if there is none."""
        rows, columns = on_side.shape
        height, width = self.ids.shape
        reach = int(math.ceil(self.dot_reach)) + 1  # what lies just outside the window can be in a dot's way too
        top, left = max(0, y0 - reach), max(0, x0 - reach)
        around = (slice(top, min(height, y0 + rows + reach)), slice(left, min(width, x0 + columns + reach)))
        taken = self.boxes[around] | self.leaders[around] | self.dots[around]
        away = cv2.distanceTransform((~taken).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        away = away[y0 - top : y0 - top + rows, x0 - left : x0 - left + columns]
        paper = on_side & self.free[y0 : y0 + rows, x0 : x0 + columns]
        fits = paper & (away > self.dot_reach)
        if not fits.any():
            return None
        y, x = _farthest(_clearance(paper), fits)
        return x + x0, y + y0

    def _anchor(self, point: tuple[int, int], change: int) -> None:
        """Make ``point`` one of the room's anchors (``change`` 1), a point a leader will point at, after the room was
        made, or no longer one (-1)."""
        if change > 0:
            self.anchors.append(point)
        else:
            self.anchors.remove(point)
        _add_disks(self.dot_count, [point], self.dot_reach, change)
        _add_disks(self.near_dot_count, [point], self.dot_reach + self.line_clear, change)
        reach = int(math.ceil(self.dot_reach))
        x, y = point
        around = (slice(max(0, y - reach), y + reach + 1), slice(max(0, x - reach), x + reach + 1))
        self.dots[around] = self.dot_count[around] > 0

    def _take_box(self, box) -> None:
        self.boxes[_spaced_slices(box, 0)] = True
        self.spaced[_spaced_slices(box, self.spacing.label_gap_px)] = True

    def _take_leader(self, path: tuple[np.ndarray, np.ndarray]) -> None:
        """Mark where a leader along ``path``, (rows, columns) of its middle, puts ink."""
        rows, columns = path
        reach = int(math.ceil(self.line_clear))
        height, width = self.ids.shape
        y0, x0 = max(0, int(rows.min()) - reach), max(0, int(columns.min()) - reach)
        y1, x1 = min(height, int(rows.max()) + reach + 1), min(width, int(columns.max()) + reach + 1)
        along = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        along[rows - y0, columns - x0] = True
        self.leaders[y0:y1, x0:x1] |= _grown(along, self.line_clear)


def _nearest_boxes(room: np.ndarray, box_width: int, box_height: int, point) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tops, lefts, distances) of every box that fits inside ``room``, nearest ``point`` first; ties in reading order.

    A box's distance is from ``point``, (x, y), to the nearest of its pixels, in whole pixel steps.
    """
    tops, lefts = np.nonzero(_box_fits(room, box_width, box_height))
    x, y = point
    dx = np.maximum(np.maximum(lefts - x, x - (lefts + box_width - 1)), 0)
    dy = np.maximum(np.maximum(tops - y, y - (tops + box_height - 1)), 0)
    distance = np.hypot(dx, dy)
    order = np.lexsort((lefts, tops, distance))
    return tops[order], lefts[order], distance[order]


def _nearest_box(room: np.ndarray, box_width: int, box_height: int, point) -> tuple[int, int, float] | None:
    """(top, left, distance) of the box ``_nearest_boxes`` gives first, None if none fits, without sorting the others."""
    fits = _box_fits(room, box_width, box_height)
    if not fits.any():
        return None
    rows, columns = room.shape
    x, y = point
    lefts, tops = np.arange(columns), np.arange(rows)
    dx = np.maximum(np.maximum(lefts - x, x - (lefts + box_width - 1)), 0)
    dy = np.maximum(np.maximum(tops - y, y - (tops + box_height - 1)), 0)
    distance = np.where(fits, np.hypot(dx[None, :], dy[:, None]), np.inf)
    # The nearest, and of those the first in reading order, which is the first in the array's own order.
    top, left = divmod(int(np.argmin(distance)), columns)
    return top, left, float(distance[top, left])


def _clearance(room: np.ndarray) -> np.ndarray:
    """How far each pixel of ``room`` is from the nearest pixel outside it; 0 outside it."""
    padded = np.zeros((room.shape[0] + 2, room.shape[1] + 2), dtype=np.uint8)  # the window's edge is outside the room
    padded[1:-1, 1:-1] = room
    return cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def _farthest(clearance: np.ndarray, where: np.ndarray) -> tuple[int, int]:
    """The (row, column) of the pixel of ``where`` (bool, not all False) with the most ``clearance``, the first in
    reading order of those with as much.

    A precise distance transform's distances are square roots of whole numbers, but worked out to within a float's
    last bit, which can differ from one call to the next on the very same input: squared and rounded, equal distances
    tie, so the same page always gets the same point.
    """
    squared = np.where(where, np.rint(np.square(clearance, dtype=np.float64)), -1.0)
    y, x = np.unravel_index(int(np.argmax(squared)), squared.shape)
    return int(y), int(x)


def _roomiest_box(clearance: np.ndarray, box_width: int, box_height: int) -> tuple[int, int] | None:
    """The (left, top) of the ``box_width`` x ``box_height`` box inside the room that keeps farthest from its edge.

    A box's clearance is the least clearance of its pixels, which is how near
    it comes to anything outside the room. Ties go to the first box in reading
    order. None if no box fits.
    """
    rows, columns = clearance.shape
    if box_width > columns or box_height > rows:
        return None
    least = cv2.erode(
        clearance, np.ones((box_height, box_width), np.uint8), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    best = int(np.argmax(least))
    if least.flat[best] <= 0:
        return None
    top, left = divmod(best, columns)
    return left, top


def _box_fits(room: np.ndarray, box_width: int, box_height: int) -> np.ndarray:
    """For every pixel, whether a ``box_width`` x ``box_height`` box with its top left corner there lies inside ``room``."""
    rows, columns = room.shape
    if box_width > columns or box_height > rows:
        return np.zeros(room.shape, dtype=bool)
    inside = cv2.erode(
        room.astype(np.uint8),
        np.ones((box_height, box_width), np.uint8),
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return inside.astype(bool)


def _leader_end(
    anchor: tuple[float, float], box: tuple[int, int, int, int], clear_px: float
) -> tuple[float, float] | None:
    """Where a leader from ``anchor`` to the number in ``box`` stops: ``clear_px`` short of the nearest point of the box.

    ``anchor`` has pixel centers at integer coordinates; ``box`` is (x0, y0,
    x1, y1) in the page's own coordinates, where a pixel spans a unit square
    from its corner, so the box's pixels run from x0 - 0.5 to x1 - 0.5. None
    if the box is no farther than ``clear_px`` from the anchor, which leaves no
    leader to draw.
    """
    x, y = anchor
    left, top, right, bottom = box[0] - 0.5, box[1] - 0.5, box[2] - 0.5, box[3] - 0.5
    near = (min(max(x, left), right), min(max(y, top), bottom))
    dx, dy = x - near[0], y - near[1]
    length = math.hypot(dx, dy)
    if length <= clear_px:
        return None
    return (near[0] + dx * clear_px / length, near[1] + dy * clear_px / length)


def _pixels_along(start: tuple[float, float], end: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The (rows, columns) of the pixels a straight line from ``start`` to ``end`` passes through."""
    steps = int(math.ceil(2 * max(abs(end[0] - start[0]), abs(end[1] - start[1])))) + 1
    # np.linspace(0.0, 1.0, steps), worked out as it works it out, without its overhead: a page crowded with numbers
    # tries tens of thousands of leaders.
    t = np.arange(steps, dtype=np.float64)
    if steps > 1:
        t *= 1.0 / (steps - 1)
        t[-1] = 1.0
    columns = np.rint(start[0] + t * (end[0] - start[0])).astype(np.intp)
    rows = np.rint(start[1] + t * (end[1] - start[1])).astype(np.intp)
    return rows, columns


def _disks(shape: tuple[int, int], centers, radius_px: float) -> np.ndarray:
    """The pixels of a ``shape`` page within ``radius_px`` of any of ``centers``, (x, y) pixels, which may lie off it."""
    return _disk_count(shape, centers, radius_px) > 0


def _disk_count(shape: tuple[int, int], centers, radius_px: float) -> np.ndarray:
    """How many of ``centers``, (x, y) pixels that may lie off a ``shape`` page, each of its pixels is within ``radius_px``
    of, as uint16: a pixel is within reach of at most as many as fit in a disk that wide, far fewer than its limit."""
    counts = np.zeros(shape, dtype=np.uint16)
    _add_disks(counts, centers, radius_px)
    return counts


def _add_disks(counts: np.ndarray, centers, radius_px: float, change: int = 1) -> None:
    """Add ``change`` (1 or -1, taking back a 1 added before) to ``counts`` (uint16) on every pixel within ``radius_px``
    of each of ``centers`` (see ``_disk_count``)."""
    shape = counts.shape
    reach = int(math.ceil(radius_px))
    offsets = np.arange(-reach, reach + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius_px * radius_px
    height, width = shape
    for x, y in centers:
        y0, x0 = y - reach, x - reach
        rows = slice(max(0, y0), max(0, min(height, y + reach + 1)))
        columns = slice(max(0, x0), max(0, min(width, x + reach + 1)))
        if rows.stop > rows.start and columns.stop > columns.start:
            stamp = disk[rows.start - y0 : rows.stop - y0, columns.start - x0 : columns.stop - x0]
            if change > 0:
                counts[rows, columns] += stamp
            else:
                counts[rows, columns] -= stamp


def _grown(mask: np.ndarray, reach_px: float) -> np.ndarray:
    """``mask`` grown by every pixel within ``reach_px`` of it."""
    reach = int(math.ceil(reach_px))
    if reach <= 0 or not mask.any():
        return mask.copy()
    offsets = np.arange(-reach, reach + 1)
    disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2 <= reach_px * reach_px).astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), disk).astype(bool)


@lru_cache(maxsize=4096)
def font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


_MEASURING_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@lru_cache(maxsize=4096)
def text_bbox(text: str, font_size: float) -> tuple[int, int, int, int]:
    """``textbbox`` of ``text`` at the origin, as measured on an RGB page."""
    return _MEASURING_DRAW.textbbox((0, 0), text, font=font(font_size))


@lru_cache(maxsize=4096)
def baseline_bbox(text: str, font_size: float) -> tuple[int, int, int, int]:
    """``text_bbox`` measured from the start of the text's baseline instead of from the top of its line."""
    return _MEASURING_DRAW.textbbox((0, 0), text, font=font(font_size), anchor="ls")

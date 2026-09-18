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
written just outside it, in a neighbor, with a leader line pointing in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core import kernels
from tessellatum.core.print_size import MIN_LABEL_SIZE_PT, print_scale

MAX_FONT_SIZE = 40  # px
FONT_SIZE_RADIUS_RATIO = 0.85
# A number is never drawn smaller than this, whatever the paper asks for. The
# pipeline never upscales, so a small source prints at a low resolution, where
# the smallest legible size on paper is too few pixels to draw a digit with:
# at 10 px a digit is 8 px tall.
MIN_FONT_SIZE = 10  # px

# How far from the region a number with a leader may be written, on paper. Any
# farther and the line pointing back stops reading as belonging to the region.
LEADER_REACH_MM = 8.0


@dataclass
class Label:
    """A region's number as drawn on the page."""

    region_id: int
    text: str
    font_size: int  # px: the font's em size
    box: tuple[float, float, float, float]  # (x0, y0, x1, y1): the text's bounding box on the page
    # A number written outside its region: the leader line from the number to the point in the region it
    # points at, as ((x, y), (x, y)) with pixel centers at integer coordinates, like the page's lines.
    leader: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class LabelSpacing:
    """How much room the placement leaves, in pixels of the page it is placing numbers on."""

    min_font_size: int  # the smallest em size a number is drawn at (see ``min_font_size``)
    label_gap_px: float  # the least space between a number written outside its region and any other number
    leader_width_px: float  # how wide a leader line is inked
    leader_reach_px: float  # how far from its region a number with a leader may go


def min_font_size(size: tuple[int, int]) -> int:
    """The smallest em size, in whole pixels, a number on a page of ``size`` is drawn at.

    That is the smallest that prints legibly (``MIN_LABEL_SIZE_PT``), and
    never less than ``MIN_FONT_SIZE``. The size in points is checked with the
    same arithmetic the benchmark measures it with, so a number of this size is
    never scored a hair under it.
    """
    scale = print_scale(size)
    font_size = max(MIN_FONT_SIZE, math.floor(scale.pt_to_px(MIN_LABEL_SIZE_PT)))
    while scale.px_to_pt(font_size) < MIN_LABEL_SIZE_PT:
        font_size += 1
    return font_size


def place_labels(regions, region_id_map: np.ndarray, free: np.ndarray, spacing: LabelSpacing) -> list[Label]:
    """A number for every region in ``regions``, placed where no line runs through it.

    ``free`` is True where the page's lines put no ink at all. A number's text
    box is kept inside one region's free pixels, so two numbers written inside
    their own regions can never touch: a line lies between them. The numbers
    that need a leader are placed after all the others, in whatever room those
    leave.

    Returns the labels in drawing order: the numbers inside their regions, in
    the order of ``regions``, then those with a leader.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    height, width = ids.shape
    if not regions:
        return []
    # Sized for every region asked about, in case one is not on the map: it then has no room of its own.
    num_ids = max(int(ids.max()), max(region.region_id for region in regions)) + 1
    bounds = kernels.region_bounds(ids.reshape(-1), height, width, num_ids)[0]

    labels: list[Label] = []
    homeless = []
    for region in regions:
        label = _label_inside(region, ids, free, bounds, spacing.min_font_size)
        if label is None:
            homeless.append(region)
        else:
            labels.append(label)

    if homeless:
        taken = np.zeros((height, width), dtype=bool)
        for label in labels:
            _mark_box(taken, label.box, spacing.label_gap_px)
        for region in homeless:
            labels.append(_label_with_leader(region, ids, free, taken, spacing))
    return labels


def text_box_size(text: str, font_size: int) -> tuple[int, int]:
    """(width, height) of ``text``'s bounding box at ``font_size``, in whole pixels."""
    left, top, right, bottom = text_bbox(text, font_size)
    return right - left, bottom - top


def _label_inside(region, ids: np.ndarray, free: np.ndarray, bounds, min_size: int) -> Label | None:
    """``region``'s number inside the region, as large as it can be up to its preferred size; None if none fits."""
    text = str(region.color_index + 1)
    preferred = max(min_size, int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))
    height, width = ids.shape

    # Where it always went, if it fits there: centered on the point farthest from the region's edge.
    box_width, box_height = text_box_size(text, preferred)
    x, y = region.interior_point
    left = min(max(x - box_width // 2, 0), width - box_width)
    top = min(max(y - box_height // 2, 0), height - box_height)
    if left >= 0 and top >= 0:
        window = (slice(top, top + box_height), slice(left, left + box_width))
        if free[window].all() and (ids[window] == region.region_id).all():
            return Label(region.region_id, text, preferred, (left, top, left + box_width, top + box_height))

    # Otherwise wherever in the region it keeps farthest from the lines, shrinking it if it has to.
    x0, y0, x1, y1 = (int(v) for v in bounds[region.region_id])
    room = free[y0 : y1 + 1, x0 : x1 + 1] & (ids[y0 : y1 + 1, x0 : x1 + 1] == region.region_id)
    clearance = _clearance(room)
    for font_size in range(preferred, min_size - 1, -1):
        box_width, box_height = text_box_size(text, font_size)
        spot = _roomiest_box(clearance, box_width, box_height)
        if spot is not None:
            left, top = x0 + spot[0], y0 + spot[1]
            return Label(region.region_id, text, font_size, (left, top, left + box_width, top + box_height))
    return None


def _label_with_leader(region, ids: np.ndarray, free: np.ndarray, taken: np.ndarray, spacing: LabelSpacing) -> Label:
    """``region``'s number written just outside it, with a leader line pointing in.

    The number goes in the nearest room outside the region, within
    ``spacing.leader_reach_px``, whose leader runs from the region into the
    number's own region and nowhere else, clear of every other number. If no
    such room exists, the leader may cross other regions too; if there is no
    room at all, the number goes inside the region regardless of the lines,
    which the benchmark reports.

    ``taken`` marks the room other numbers and leaders hold; it is updated
    with this one.
    """
    text = str(region.color_index + 1)
    font_size = spacing.min_font_size
    box_width, box_height = text_box_size(text, font_size)
    height, width = ids.shape
    x, y = region.interior_point
    anchor = (float(x), float(y))

    reach = int(math.ceil(spacing.leader_reach_px))
    wx0, wy0 = max(0, x - reach - box_width), max(0, y - reach - box_height)
    wx1, wy1 = min(width, x + reach + box_width + 1), min(height, y + reach + box_height + 1)
    room = free[wy0:wy1, wx0:wx1] & ~taken[wy0:wy1, wx0:wx1] & (ids[wy0:wy1, wx0:wx1] != region.region_id)
    fits = _box_fits(room, box_width, box_height)
    tops, lefts = np.nonzero(fits)
    lefts, tops = lefts + wx0, tops + wy0
    # How far each box is from the point the leader points at, nearest first; ties in reading order.
    dx = np.maximum(np.maximum(lefts - x, x - (lefts + box_width - 1)), 0)
    dy = np.maximum(np.maximum(tops - y, y - (tops + box_height - 1)), 0)
    distance = np.hypot(dx, dy)
    order = np.lexsort((lefts, tops, distance))
    order = order[distance[order] <= spacing.leader_reach_px]

    # A leader keeps a pen's half width and a pixel of anti-aliasing clear of every number.
    clear_px = spacing.leader_width_px / 2 + 1
    blocked = _grown(taken, clear_px)
    for crossing_others in (False, True):
        for index in order:
            left, top = int(lefts[index]), int(tops[index])
            box = (left, top, left + box_width, top + box_height)
            end = _leader_end(anchor, box, clear_px)
            path = _pixels_along(anchor, end)
            if blocked[path].any():
                continue
            if not crossing_others:
                along = ids[path]
                if not np.isin(along, (region.region_id, ids[top, left])).all():
                    continue
            _mark_box(taken, box, spacing.label_gap_px)
            _mark_path(taken, path, clear_px)
            return Label(region.region_id, text, font_size, box, leader=(end, anchor))

    left = min(max(x - box_width // 2, 0), width - box_width)
    top = min(max(y - box_height // 2, 0), height - box_height)
    box = (left, top, left + box_width, top + box_height)
    _mark_box(taken, box, spacing.label_gap_px)
    return Label(region.region_id, text, font_size, box)


def _clearance(room: np.ndarray) -> np.ndarray:
    """How far each pixel of ``room`` is from the nearest pixel outside it; 0 outside it."""
    padded = np.pad(room, 1).astype(np.uint8)  # the edge of the window is outside the room
    return cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


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


def _leader_end(anchor: tuple[float, float], box: tuple[int, int, int, int], clear_px: float) -> tuple[float, float]:
    """Where a leader from ``anchor`` to the number in ``box`` stops: ``clear_px`` short of the nearest point of the box.

    ``anchor`` has pixel centers at integer coordinates; ``box`` is (x0, y0,
    x1, y1) in the page's own coordinates, where a pixel spans a unit square
    from its corner, so the box's pixels run from x0 - 0.5 to x1 - 0.5.
    """
    x, y = anchor
    left, top, right, bottom = box[0] - 0.5, box[1] - 0.5, box[2] - 0.5, box[3] - 0.5
    near = (min(max(x, left), right), min(max(y, top), bottom))
    dx, dy = x - near[0], y - near[1]
    length = math.hypot(dx, dy)
    if length <= clear_px:
        return near
    return (near[0] + dx * clear_px / length, near[1] + dy * clear_px / length)


def _pixels_along(start: tuple[float, float], end: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The (rows, columns) of the pixels a straight line from ``start`` to ``end`` passes through."""
    steps = int(math.ceil(2 * max(abs(end[0] - start[0]), abs(end[1] - start[1])))) + 1
    t = np.linspace(0.0, 1.0, steps)
    columns = np.rint(start[0] + t * (end[0] - start[0])).astype(np.intp)
    rows = np.rint(start[1] + t * (end[1] - start[1])).astype(np.intp)
    return rows, columns


def _mark_box(taken: np.ndarray, box, gap_px: float) -> None:
    """Mark ``box`` in ``taken``, grown by ``gap_px`` on every side."""
    gap = int(math.ceil(gap_px))
    x0, y0, x1, y1 = (int(math.floor(box[0])), int(math.floor(box[1])), int(math.ceil(box[2])), int(math.ceil(box[3])))
    taken[max(0, y0 - gap) : y1 + gap, max(0, x0 - gap) : x1 + gap] = True


def _mark_path(taken: np.ndarray, path: tuple[np.ndarray, np.ndarray], reach_px: float) -> None:
    """Mark the pixels within ``reach_px`` of ``path`` in ``taken``."""
    along = np.zeros(taken.shape, dtype=bool)
    along[path] = True
    taken |= _grown(along, reach_px)


def _grown(mask: np.ndarray, reach_px: float) -> np.ndarray:
    """``mask`` grown by every pixel within ``reach_px`` of it."""
    reach = int(math.ceil(reach_px))
    if reach <= 0 or not mask.any():
        return mask.copy()
    offsets = np.arange(-reach, reach + 1)
    disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2 <= reach_px * reach_px).astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), disk).astype(bool)


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


_MEASURING_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@lru_cache(maxsize=4096)
def text_bbox(text: str, font_size: int) -> tuple[int, int, int, int]:
    """``textbbox`` of ``text`` at the origin, as measured on an RGB page."""
    return _MEASURING_DRAW.textbbox((0, 0), text, font=font(font_size))

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
    leader_dot_px: float  # how wide the dot a leader ends in, at the point of its region it points at, is inked
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
    leave (see ``_LeaderRoom``).

    Returns the labels in drawing order: the numbers inside their regions, in
    the order of ``regions``, then the others.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    height, width = ids.shape
    labels: list[Label] = []
    homeless = []
    bounds = None  # every region's bounding box, found the first time a number does not fit at its region's middle
    for region in regions:
        label = _label_at_middle(region, ids, free, spacing.min_font_size)
        if label is None:
            if bounds is None:
                # Sized for every region asked about, in case one is not on the map: it then has no room of its own.
                num_ids = max(int(ids.max()), max(r.region_id for r in regions)) + 1
                bounds = kernels.region_bounds(ids.reshape(-1), height, width, num_ids)[0]
            label = _label_inside(region, ids, free, bounds, spacing.min_font_size)
        if label is None:
            homeless.append(region)
        else:
            labels.append(label)

    if homeless:
        room = _LeaderRoom(ids, free, labels, [region.interior_point for region in homeless], spacing)
        labels.extend(room.place(region) for region in homeless)
    return labels


def text_box_size(text: str, font_size: int) -> tuple[int, int]:
    """(width, height) of ``text``'s bounding box at ``font_size``, in whole pixels."""
    left, top, right, bottom = text_bbox(text, font_size)
    return right - left, bottom - top


def _preferred_size(region, min_size: int) -> int:
    """The size a region's number is drawn at if it fits: larger the farther its middle is from its edge."""
    return max(min_size, int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))


def _centered_box(point, box_width: int, box_height: int, page_size: tuple[int, int]) -> tuple[int, int]:
    """(left, top) of a ``box_width`` x ``box_height`` box centered on ``point``, kept on a ``page_size`` page if it fits."""
    x, y = point
    width, height = page_size
    return max(0, min(x - box_width // 2, width - box_width)), max(0, min(y - box_height // 2, height - box_height))


def _label_at_middle(region, ids: np.ndarray, free: np.ndarray, min_size: int) -> Label | None:
    """``region``'s number where it always went, if it fits there: at its preferred size, on the region's middle."""
    text = str(region.color_index + 1)
    font_size = _preferred_size(region, min_size)
    box_width, box_height = text_box_size(text, font_size)
    height, width = ids.shape
    if box_width > width or box_height > height:
        return None
    left, top = _centered_box(region.interior_point, box_width, box_height, (width, height))
    window = (slice(top, top + box_height), slice(left, left + box_width))
    if free[window].all() and (ids[window] == region.region_id).all():
        return Label(region.region_id, text, font_size, (left, top, left + box_width, top + box_height))
    return None


def _label_inside(region, ids: np.ndarray, free: np.ndarray, bounds, min_size: int) -> Label | None:
    """``region``'s number wherever in the region it keeps farthest from the lines, shrinking it if it has to.

    None if it does not fit anywhere in the region even at ``min_size``.
    """
    text = str(region.color_index + 1)
    x0, y0, x1, y1 = (int(v) for v in bounds[region.region_id])
    room = free[y0 : y1 + 1, x0 : x1 + 1] & (ids[y0 : y1 + 1, x0 : x1 + 1] == region.region_id)
    clearance = _clearance(room)
    for font_size in range(_preferred_size(region, min_size), min_size - 1, -1):
        box_width, box_height = text_box_size(text, font_size)
        spot = _roomiest_box(clearance, box_width, box_height)
        if spot is not None:
            left, top = x0 + spot[0], y0 + spot[1]
            return Label(region.region_id, text, font_size, (left, top, left + box_width, top + box_height))
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
    * failing that -- there is no room at all -- as near the region's middle as
      it can be without landing on another number or leader, lines or not,
      which the benchmark reports.
    """

    def __init__(self, ids: np.ndarray, free: np.ndarray, labels: list[Label], anchors, spacing: LabelSpacing) -> None:
        self.ids, self.free, self.spacing = ids, free, spacing
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
        self.dots = _disks(shape, self.anchors, self.dot_reach)  # where every dot puts ink
        self.placed = 0

    def place(self, region) -> Label:
        index, self.placed = self.placed, self.placed + 1
        others = self.anchors[:index] + self.anchors[index + 1 :]
        ids, spacing = self.ids, self.spacing
        text = str(region.color_index + 1)
        font_size = spacing.min_font_size
        box_width, box_height = text_box_size(text, font_size)
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

        # A leader keeps its ink off every number, every leader placed so far and every other dot -- those just
        # outside the window too, so they are grown from a margin around it.
        margin = int(math.ceil(self.line_clear))
        mx0, my0 = max(0, wx0 - margin), max(0, wy0 - margin)
        around = (slice(my0, wy1 + margin), slice(mx0, wx1 + margin))
        grown = _grown(self.boxes[around] | self.leaders[around], self.line_clear)
        blocked = grown[wy0 - my0 : wy1 - my0, wx0 - mx0 : wx1 - mx0]
        # Beyond its own dot, that is: two points close enough have dots that touch whatever their leaders do.
        other_dots = _disks(blocked.shape, [(ox - wx0, oy - wy0) for ox, oy in others], self.dot_reach + self.line_clear)
        blocked |= other_dots & ~_disks(blocked.shape, [(x - wx0, y - wy0)], self.dot_reach)
        # A leader that runs from the region straight into the number's region, and nowhere else, can only end in a
        # region touching this one: it steps from pixel to pixel, diagonals included.
        own = (local_ids == region.region_id).astype(np.uint8)
        neighbors = np.unique(local_ids[cv2.dilate(own, np.ones((3, 3), np.uint8)).astype(bool) & ~own.astype(bool)])
        region_of = local_ids[tops, lefts]
        next_door = np.isin(region_of, neighbors)

        known_blocked = np.zeros(len(tops), dtype=bool)
        for crossing_others in (False, True):
            for k in range(len(tops)):
                if known_blocked[k] or not (crossing_others or next_door[k]):
                    continue
                left, top = int(lefts[k]) + wx0, int(tops[k]) + wy0
                box = (left, top, left + box_width, top + box_height)
                end = _leader_end(anchor, box, self.line_reach)
                if end is None:  # too near to leave room for a leader: only a dot smaller than its line allows it
                    continue
                rows, columns = _pixels_along(anchor, end)
                rows, columns = rows - wy0, columns - wx0
                if blocked[rows, columns].any():
                    known_blocked[k] = True
                    continue
                if not crossing_others:
                    along = local_ids[rows, columns]
                    if not ((along == region.region_id) | (along == region_of[k])).all():
                        continue
                self._take_box(box)
                self._take_leader((rows + wy0, columns + wx0))
                return Label(region.region_id, text, font_size, box, leader=(end, anchor))

        # No room anywhere near: at the region's middle, or as near it as the number can be without landing on
        # another number, a leader or a dot.
        clear = ~busy & ~_disks(busy.shape, [(ox - wx0, oy - wy0) for ox, oy in others], self.dot_reach)
        tops, lefts, distance = _nearest_boxes(clear, box_width, box_height, (x - wx0, y - wy0))
        if len(tops) and distance[0] <= spacing.leader_reach_px:
            left, top = int(lefts[0]) + wx0, int(tops[0]) + wy0
        else:
            left, top = _centered_box((x, y), box_width, box_height, (width, height))
        box = (left, top, left + box_width, top + box_height)
        self._take_box(box)
        return Label(region.region_id, text, font_size, box)

    def _take_box(self, box) -> None:
        x0, y0, x1, y1 = (int(math.floor(box[0])), int(math.floor(box[1])), int(math.ceil(box[2])), int(math.ceil(box[3])))
        self.boxes[max(0, y0) : y1, max(0, x0) : x1] = True
        gap = int(math.ceil(self.spacing.label_gap_px))
        self.spaced[max(0, y0 - gap) : y1 + gap, max(0, x0 - gap) : x1 + gap] = True

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
    t = np.linspace(0.0, 1.0, steps)
    columns = np.rint(start[0] + t * (end[0] - start[0])).astype(np.intp)
    rows = np.rint(start[1] + t * (end[1] - start[1])).astype(np.intp)
    return rows, columns


def _disks(shape: tuple[int, int], centers, radius_px: float) -> np.ndarray:
    """The pixels of a ``shape`` page within ``radius_px`` of any of ``centers``, (x, y) pixels, which may lie off it."""
    disks = np.zeros(shape, dtype=bool)
    reach = int(math.ceil(radius_px))
    offsets = np.arange(-reach, reach + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius_px * radius_px
    height, width = shape
    for x, y in centers:
        y0, x0 = y - reach, x - reach
        rows = slice(max(0, y0), max(0, min(height, y + reach + 1)))
        columns = slice(max(0, x0), max(0, min(width, x + reach + 1)))
        if rows.stop > rows.start and columns.stop > columns.start:
            disks[rows, columns] |= disk[rows.start - y0 : rows.stop - y0, columns.start - x0 : columns.stop - x0]
    return disks


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

"""The original (slow) region + render implementations, kept as a test oracle.

Straightforward whole-image versions of ``build_regions``,
``extract_regions``, ``trace_boundaries`` and ``render_page``, written from
the implementations before the performance rewrite. The optimized versions in
``tessellatum.core`` must produce exactly the same output;
``test_regions_equivalence.py`` checks that.

A change meant to alter that output updates this file deliberately, so it
keeps saying what the stages should do rather than what they used to. Since
the rewrite: ``_merge_same_color_neighbors``, then ``_absorb_thin_parts``
with the rebuild that follows it, then ``trace_boundaries`` and the
``render_page`` that draws its lines instead of outlining every region, and
now ``_smooth`` in place of the Douglas-Peucker pass those lines used to get,
``line_layer``, which lays those lines down with a round pen of the width the
paper asks for instead of a two-pixel band, and ``_place_labels``, which gives
every region a number no line runs through, at least as large as the paper
needs, instead of numbering only the regions roomy enough at their middle.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.boundaries import MAX_SHIFT_PX, SMOOTHING_MIN_PX, SMOOTHING_MM, _MIN_LOOP_AREA_PX, _SMOOTHING_STEP
from tessellatum.core.labels import FONT_SIZE_RADIUS_RATIO, LEADER_REACH_MM, MAX_FONT_SIZE, MIN_FONT_SIZE
from tessellatum.core.print_size import MIN_LABEL_SIZE_PT, print_scale
from tessellatum.core.regions import Region
from tessellatum.core.render import PAPER, PageStyle, _SUPERSAMPLE

_OUTSIDE = -2  # the label of everything off the page: its edge is a boundary like any other

_NEIGHBOR_KERNEL = np.ones((3, 3), np.uint8)


def build_regions(
    labels: np.ndarray, num_colors: int, min_area_px: int, min_width_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    region_id_map, region_color = _regions_from_labels(labels, num_colors, min_area_px)
    if min_width_px > 0 and region_color.size:
        widened = _absorb_thin_parts(region_id_map, region_color, labels, min_width_px)
        if widened is not None:
            region_id_map, region_color = _regions_from_labels(widened, num_colors, min_area_px)
    return region_id_map, region_color


def _regions_from_labels(labels: np.ndarray, num_colors: int, min_area_px: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = labels.shape
    region_id_map = np.full((h, w), -1, dtype=np.int32)
    region_color: list[int] = []
    next_id = 0

    for c in range(num_colors):
        mask = (labels == c).astype(np.uint8)
        if not mask.any():
            continue
        num_components, components = cv2.connectedComponents(mask, connectivity=8)
        for comp_id in range(1, num_components):
            comp_mask = components == comp_id
            region_id_map[comp_mask] = next_id
            region_color.append(c)
            next_id += 1

    if next_id == 0:
        return region_id_map, np.array([], dtype=np.int32)

    region_color_arr = np.array(region_color, dtype=np.int32)
    areas = np.bincount(region_id_map.ravel()[region_id_map.ravel() >= 0], minlength=next_id).astype(np.int64)
    active = np.ones(next_id, dtype=bool)

    _merge_small_regions(region_id_map, areas, active, min_area_px)
    _merge_same_color_neighbors(region_id_map, region_color_arr)

    return region_id_map, region_color_arr


def _merge_small_regions(region_id_map: np.ndarray, areas: np.ndarray, active: np.ndarray, min_area_px: int) -> None:
    if min_area_px <= 0 or active.sum() <= 1:
        return

    while True:
        candidates = np.where(active & (areas < min_area_px))[0]
        if candidates.size == 0:
            return
        rid = int(candidates[np.argmin(areas[candidates])])

        mask = (region_id_map == rid).astype(np.uint8)
        dilated = cv2.dilate(mask, _NEIGHBOR_KERNEL, iterations=1)
        ring = (dilated.astype(bool)) & (~mask.astype(bool))
        neighbor_ids = region_id_map[ring]
        neighbor_ids = neighbor_ids[neighbor_ids >= 0]
        neighbor_ids = neighbor_ids[neighbor_ids != rid]

        if neighbor_ids.size == 0:
            active[rid] = False
            continue

        counts = np.bincount(neighbor_ids)
        target = int(np.argmax(counts))

        region_id_map[mask.astype(bool)] = target
        areas[target] += areas[rid]
        areas[rid] = 0
        active[rid] = False


def _merge_same_color_neighbors(region_id_map: np.ndarray, region_color: np.ndarray) -> None:
    """Union 8-adjacent regions of one color, keeping the lowest id (see ``kernels``)."""
    inside = region_id_map >= 0
    pixel_color = np.where(inside, region_color[np.where(inside, region_id_map, 0)], -1)

    for c in np.unique(pixel_color[inside]):
        mask = (pixel_color == c).astype(np.uint8)
        num_components, components = cv2.connectedComponents(mask, connectivity=8)
        for comp_id in range(1, num_components):
            comp_mask = components == comp_id
            region_id_map[comp_mask] = region_id_map[comp_mask].min()


def _absorb_thin_parts(
    region_id_map: np.ndarray, region_color: np.ndarray, labels: np.ndarray, min_width_px: float
) -> np.ndarray | None:
    """Every pixel takes the color of the nearest region a brush ``min_width_px`` wide fits in (see ``regions``)."""
    radius = min_width_px / 2
    fits = np.zeros(region_id_map.shape, dtype=bool)
    for rid in np.unique(region_id_map):
        if rid < 0:
            continue
        mask = (region_id_map == rid).astype(np.uint8)
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # How far each of the region's pixels is from the nearest pixel of
        # another region, or from off the page.
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        fits |= mask.astype(bool) & (distance > radius)
    if not fits.any():
        return None

    _distance, nearest = cv2.distanceTransformWithLabels(
        (~fits).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    widened = np.array(labels, dtype=np.int32, copy=True)
    color_of_label = {int(nearest[y, x]): int(region_color[region_id_map[y, x]]) for y, x in zip(*np.nonzero(fits))}
    for y, x in zip(*np.nonzero(region_id_map >= 0)):
        widened[y, x] = color_of_label[int(nearest[y, x])]
    return widened


def extract_regions(region_id_map: np.ndarray, region_color: np.ndarray, min_contour_area: float = 1.0) -> list[Region]:
    regions: list[Region] = []
    unique_ids = np.unique(region_id_map)

    for rid in unique_ids:
        if rid < 0:
            continue
        mask = (region_id_map == rid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_contour_area:
            continue
        contour = cv2.approxPolyDP(contour, epsilon=1.2, closed=True)

        pad = 1
        padded_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)

        regions.append(
            Region(
                region_id=int(rid),
                color_index=int(region_color[rid]),
                contour=contour,
                interior_point=(int(max_loc[0]) - pad, int(max_loc[1]) - pad),
                interior_radius=float(max_val),
                area=int((mask > 0).sum()),
            )
        )

    return regions


def trace_boundaries(
    region_id_map: np.ndarray, smoothing_px: float | None = None, max_shift_px: float = MAX_SHIFT_PX
) -> list[np.ndarray]:
    """One line per boundary between two regions, as ``boundaries.py`` describes it.

    The crack graph is built as a dict of pixel corners, each holding the
    corners it shares a crack with. Corners with three or more of them are
    junctions; the rest have two, and following those from junction to
    junction gives one boundary at a time. Corners are visited in raster
    order and their edges in the order right, down, left, up, which is the
    order the kernel walks them in too, so the lines come out the same way
    round.
    """
    h, w = region_id_map.shape
    if smoothing_px is None:
        # The longer of a pixel step and what the printed page can show.
        smoothing_px = max(SMOOTHING_MIN_PX, print_scale((w, h)).mm_to_px(SMOOTHING_MM))

    def pixel(y: int, x: int) -> int:
        return int(region_id_map[y, x]) if 0 <= y < h and 0 <= x < w else _OUTSIDE

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def join(a: tuple[int, int], b: tuple[int, int]) -> None:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    for i in range(h + 1):
        for j in range(w + 1):
            if j < w and pixel(i - 1, j) != pixel(i, j):  # the crack right of this corner
                join((i, j), (i, j + 1))
            if i < h and pixel(i, j - 1) != pixel(i, j):  # the crack below it
                join((i, j), (i + 1, j))

    def steps(corner: tuple[int, int]) -> list[tuple[int, int]]:
        i, j = corner
        order = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]  # right, down, left, up
        return [c for c in order if c in neighbors.get(corner, ())]

    walked: set[frozenset] = set()

    def walk(corner: tuple[int, int], to: tuple[int, int]) -> list[tuple[int, int]]:
        path = [corner]
        while True:
            walked.add(frozenset((corner, to)))
            path.append(to)
            corner = to
            if len(neighbors[corner]) != 2:
                break  # a junction: the next boundary is another line
            onwards = [c for c in steps(corner) if c != path[-2]]
            if not onwards or frozenset((corner, onwards[0])) in walked:
                break  # back where this line started: a closed boundary
            to = onwards[0]
        return path

    corners = sorted(neighbors)
    paths: list[list[tuple[int, int]]] = []
    for junctions_first in (True, False):
        for corner in corners:
            if (len(neighbors[corner]) >= 3) != junctions_first:
                continue
            for to in steps(corner):
                if frozenset((corner, to)) not in walked:
                    paths.append(walk(corner, to))

    return [
        np.array(_smooth([(j - 0.5, i - 0.5) for i, j in path], smoothing_px, max_shift_px, w, h), dtype=np.float64)
        for path in paths
    ]


def _smooth(
    path: list[tuple[float, float]], smoothing_px: float, max_shift_px: float, width: int, height: int
) -> list[tuple[float, float]]:
    """``path`` blurred along its length, no point further than ``max_shift_px`` from where it started.

    Every point moves a fixed step of the way towards the middle of its two
    neighbors, all of them at once, over and over. A closed path -- one whose
    first and last point are the same -- is smoothed round its own ring; an
    open path's ends are junctions and stay where they are, as does any point
    on the page edge. A closed path that smoothing would pull shut, leaving
    nothing inside to paint, is kept as it was.
    """
    passes = int(round(smoothing_px**2 / _SMOOTHING_STEP))
    if passes < 1:
        return path
    closed = len(path) > 2 and path[0] == path[-1]
    crack = path[:-1] if closed else path

    def on_the_page_edge(point: tuple[float, float]) -> bool:
        return point[0] in (-0.5, width - 0.5) or point[1] in (-0.5, height - 0.5)

    def neighbors(at: int) -> tuple[int, int]:
        if closed:
            return (at - 1) % len(crack), (at + 1) % len(crack)
        return max(at - 1, 0), min(at + 1, len(crack) - 1)

    movable = [
        (closed or 0 < at < len(crack) - 1) and not on_the_page_edge(point) for at, point in enumerate(crack)
    ]
    smoothed = list(crack)
    for _ in range(passes):
        moved = []
        for at, (x, y) in enumerate(smoothed):
            if movable[at]:
                before, after = neighbors(at)
                middle_x = 0.5 * (smoothed[before][0] + smoothed[after][0])
                middle_y = 0.5 * (smoothed[before][1] + smoothed[after][1])
                x, y = x + _SMOOTHING_STEP * (middle_x - x), y + _SMOOTHING_STEP * (middle_y - y)
                shift_x, shift_y = x - crack[at][0], y - crack[at][1]
                distance = math.sqrt(shift_x * shift_x + shift_y * shift_y)
                if distance > max_shift_px:
                    pull = max_shift_px / distance
                    x, y = crack[at][0] + shift_x * pull, crack[at][1] + shift_y * pull
            moved.append((x, y))
        smoothed = moved
    if not closed:
        return smoothed
    loop = smoothed + smoothed[:1]
    return loop if _enclosed_area(smoothed) >= _MIN_LOOP_AREA_PX else path


def _enclosed_area(loop: list[tuple[float, float]]) -> float:
    """How much a closed line encloses, by the shoelace formula."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(loop, loop[1:] + loop[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def line_layer(size: tuple[int, int], region_id_map: np.ndarray, style: PageStyle) -> np.ndarray:
    """The ink the page's lines put on it, 0 solid and 255 bare paper: ``RenderedPage.outlines``."""
    return PAPER - _ink(size, trace_boundaries(region_id_map), style.line_width_px(size))


def _ink(size: tuple[int, int], strokes: list[np.ndarray], width_px: float) -> np.ndarray:
    """The ink ``strokes`` put on a ``size`` page, 0 bare paper and 255 solid.

    A round pen ``width_px`` wide, dragged along every stroke on a grid
    ``_SUPERSAMPLE`` times finer than the page, and averaged back down. A
    width between two whole grid pixels is a blend of the two pens around it.
    """
    width, height = size
    margin = int(math.ceil(width_px / 2)) + 1
    grid = _SUPERSAMPLE
    canvas = np.zeros(((height + 2 * margin) * grid, (width + 2 * margin) * grid), dtype=np.uint8)
    for stroke in strokes:
        # 1/16 of a grid pixel, as the renderer draws; the half pixel is the
        # step from a page pixel's middle to the grid pixel that starts it.
        points = np.rint(((stroke + (margin + 0.5)) * grid - 0.5) * 16).astype(np.int32)
        cv2.polylines(canvas, [points], False, 1, thickness=1, lineType=cv2.LINE_8, shift=4)

    diameter = max(2.0, width_px * grid)  # two grid pixels is the finest pen there is
    thinner = 2 * int(diameter // 2)
    wider_share = (diameter - thinner) / 2
    covered = _pen_coverage(canvas, thinner, size, margin).astype(np.float64)
    if wider_share > 0:
        wider = _pen_coverage(canvas, thinner + 2, size, margin).astype(np.float64)
        covered = (1 - wider_share) * covered + wider_share * wider
    return np.rint(covered).astype(np.uint8)


def _pen_coverage(canvas: np.ndarray, diameter: int, size: tuple[int, int], margin: int) -> np.ndarray:
    """``canvas``, a line one grid pixel wide, widened by a pen ``diameter`` across and averaged down."""
    half = diameter // 2
    # The line lies on the grid pixel right of or below its crack, so the pen
    # reaches one further up and left than down and right, and lands centred.
    reach = range(-half, half)
    offsets = [(dy, dx) for dy in reach for dx in reach if (dx + 0.5) ** 2 + (dy + 0.5) ** 2 <= half * half]
    padded = np.pad(canvas, half)
    band = np.zeros_like(canvas)
    for dy, dx in offsets:
        y0, x0 = half - dy, half - dx
        band |= padded[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]]

    grid = _SUPERSAMPLE
    rows, columns = band.shape[0] // grid, band.shape[1] // grid
    inked = band.reshape(rows, grid, columns, grid).sum(axis=(1, 3), dtype=np.uint16)
    covered = np.rint(inked.astype(np.float64) * PAPER / grid**2).astype(np.uint8)
    width, height = size
    return covered[margin : margin + height, margin : margin + width]


def render_page(
    size: tuple[int, int], regions: list[Region], region_id_map: np.ndarray, style: PageStyle = PageStyle()
) -> Image.Image:
    width, height = size
    lines = PAPER - line_layer(size, region_id_map, style)
    labels = _place_labels(size, regions, region_id_map, lines == 0, style)

    # Each leader and its dot, drawn over the whole page; where two overlap, the darker ink.
    line_width = style.line_width_px(size)
    leaders = np.zeros((height, width), dtype=np.uint8)
    for _region_id, _text, _font_size, _box, leader in labels:
        if leader is not None:
            end, anchor = (np.array(point, dtype=np.float64) for point in leader)
            stroke = np.array([end, anchor])
            leaders = np.maximum(leaders, _ink(size, [stroke], line_width))
            leaders = np.maximum(leaders, _ink(size, [np.array([anchor, anchor])], line_width * style.leader_dot_ratio))

    def paper(ink: np.ndarray, gray: int) -> np.ndarray:
        return np.rint(PAPER - ink.astype(np.float64) * ((PAPER - gray) / PAPER)).astype(np.uint8)

    page = Image.fromarray(np.minimum(paper(lines, style.line_gray), paper(leaders, style.label_gray)), "L").convert("RGB")
    draw = ImageDraw.Draw(page)
    for _region_id, text, font_size, (left, top, _right, _bottom), _leader in labels:
        font = ImageFont.load_default(size=font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((left - bbox[0], top - bbox[1]), text, fill=(style.label_gray,) * 3, font=font)
    return page


def _place_labels(size, regions, region_id_map, free, style):
    """(region id, text, font size, box, leader) for every region, in drawing order, as ``labels.place_labels``."""
    width, height = size
    scale = print_scale(size)
    smallest = MIN_FONT_SIZE
    while scale.px_to_pt(smallest) < MIN_LABEL_SIZE_PT:
        smallest += 1

    placed, without_room = [], []
    for region in regions:
        text = str(region.color_index + 1)
        room = free & (region_id_map == region.region_id)
        preferred = max(smallest, int(min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))
        box_width, box_height = _box_size(text, preferred)
        x, y = region.interior_point
        left = min(max(x - box_width // 2, 0), width - box_width)
        top = min(max(y - box_height // 2, 0), height - box_height)
        if left >= 0 and top >= 0 and room[top : top + box_height, left : left + box_width].all():
            placed.append((region.region_id, text, preferred, (left, top, left + box_width, top + box_height), None))
            continue
        # Every box's clearance: the least, over its pixels, of their distance to the nearest pixel outside the room.
        clearance = cv2.distanceTransform(np.pad(room, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        for font_size in range(preferred, smallest - 1, -1):
            box_width, box_height = _box_size(text, font_size)
            if box_width > width or box_height > height:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(clearance, (box_height, box_width)).min(axis=(2, 3))
            top, left = np.unravel_index(np.argmax(windows), windows.shape)  # the first of the best in reading order
            if windows[top, left] > 0:
                top, left = int(top), int(left)
                placed.append((region.region_id, text, font_size, (left, top, left + box_width, top + box_height), None))
                break
        else:
            without_room.append(region)

    line_width = style.line_width_px(size)
    gap = int(math.ceil(line_width))
    taken = np.zeros((height, width), dtype=bool)
    for *_rest, box, _leader in placed:
        _take(taken, box, gap)
    clear_px = line_width / 2 + 1
    reach_px = scale.mm_to_px(LEADER_REACH_MM)
    for region in without_room:
        placed.append(_with_leader(region, region_id_map, free, taken, smallest, clear_px, reach_px, gap))
    return placed


def _with_leader(region, region_id_map, free, taken, font_size, clear_px, reach_px, gap):
    height, width = region_id_map.shape
    text = str(region.color_index + 1)
    box_width, box_height = _box_size(text, font_size)
    x, y = region.interior_point
    room = free & ~taken & (region_id_map != region.region_id)

    # How far every box on the page would be from the point, then the ones near enough that fit.
    tops, lefts = np.mgrid[: height - box_height + 1, : width - box_width + 1]
    dx = np.maximum(np.maximum(lefts - x, x - (lefts + box_width - 1)), 0)
    dy = np.maximum(np.maximum(tops - y, y - (tops + box_height - 1)), 0)
    distances = np.hypot(dx, dy)
    candidates = []
    for top, left in zip(*np.nonzero(distances <= reach_px)):
        if room[top : top + box_height, left : left + box_width].all():
            candidates.append((float(distances[top, left]), int(top), int(left)))
    candidates.sort()

    blocked = _grow(taken, clear_px)
    for crossing_others in (False, True):
        for _distance, top, left in candidates:
            box = (left, top, left + box_width, top + box_height)
            end = _stop_short((float(x), float(y)), box, clear_px)
            path = _straight_path((float(x), float(y)), end)
            if any(blocked[row, column] for row, column in path):
                continue
            allowed = {region.region_id, int(region_id_map[top, left])}
            if not crossing_others and any(int(region_id_map[row, column]) not in allowed for row, column in path):
                continue
            _take(taken, box, gap)
            along = np.zeros_like(taken)
            for row, column in path:
                along[row, column] = True
            taken |= _grow(along, clear_px)
            return (region.region_id, text, font_size, box, (end, (float(x), float(y))))

    left = min(max(x - box_width // 2, 0), width - box_width)
    top = min(max(y - box_height // 2, 0), height - box_height)
    box = (left, top, left + box_width, top + box_height)
    _take(taken, box, gap)
    return (region.region_id, text, font_size, box, None)


def _box_size(text: str, font_size: int) -> tuple[int, int]:
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=ImageFont.load_default(size=font_size)
    )
    return right - left, bottom - top


def _take(taken: np.ndarray, box, gap: int) -> None:
    left, top, right, bottom = box
    taken[max(0, top - gap) : bottom + gap, max(0, left - gap) : right + gap] = True


def _grow(mask: np.ndarray, reach_px: float) -> np.ndarray:
    """Every pixel within ``reach_px`` of ``mask``, counting whole pixel steps."""
    reach = int(math.ceil(reach_px))
    grown = mask.copy()
    height, width = mask.shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > reach_px * reach_px:
                continue
            shifted = np.zeros_like(mask)
            shifted[max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)] = mask[
                max(0, -dy) : height + min(0, -dy), max(0, -dx) : width + min(0, -dx)
            ]
            grown |= shifted
    return grown


def _stop_short(anchor, box, clear_px: float) -> tuple[float, float]:
    """The point ``clear_px`` short of ``box``'s nearest point, on the way from ``anchor``; pixel centers at integers."""
    x, y = anchor
    near_x = min(max(x, box[0] - 0.5), box[2] - 0.5)
    near_y = min(max(y, box[1] - 0.5), box[3] - 0.5)
    length = math.hypot(x - near_x, y - near_y)
    if length <= clear_px:
        return (near_x, near_y)
    return (near_x + (x - near_x) * clear_px / length, near_y + (y - near_y) * clear_px / length)


def _straight_path(start, end) -> list[tuple[int, int]]:
    """(row, column) of each pixel nearest a point on the line, sampled at most half a pixel apart."""
    steps = int(math.ceil(2 * max(abs(end[0] - start[0]), abs(end[1] - start[1])))) + 1
    path = []
    for t in np.linspace(0.0, 1.0, steps):
        path.append((int(np.rint(start[1] + t * (end[1] - start[1]))), int(np.rint(start[0] + t * (end[0] - start[0])))))
    return path

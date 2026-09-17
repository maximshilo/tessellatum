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
now ``_smooth`` in place of the Douglas-Peucker pass those lines used to get.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.boundaries import MAX_SHIFT_PX, SMOOTHING_PX, _MIN_LOOP_AREA_PX, _SMOOTHING_STEP
from tessellatum.core.regions import Region
from tessellatum.core.render import FONT_SIZE_RADIUS_RATIO, MAX_FONT_SIZE, MIN_FONT_SIZE, MIN_LABEL_RADIUS_PX, OUTLINE_WIDTH

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
    region_id_map: np.ndarray, smoothing_px: float = SMOOTHING_PX, max_shift_px: float = MAX_SHIFT_PX
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


def render_page(size: tuple[int, int], regions: list[Region], region_id_map: np.ndarray) -> Image.Image:
    width, height = size
    margin = OUTLINE_WIDTH
    canvas = np.zeros((height + 2 * margin, width + 2 * margin), dtype=np.uint8)
    for stroke in trace_boundaries(region_id_map):
        points = np.rint((stroke + margin) * 16).astype(np.int32)  # 1/16 px, as the renderer draws
        cv2.polylines(canvas, [points], False, 1, thickness=1, lineType=cv2.LINE_8, shift=4)

    # The line is one pixel wide, on the pixel right of or below its crack.
    # Widening it up and left puts it on both sides of the crack instead.
    spread = (OUTLINE_WIDTH - 1) // 2
    padded = np.pad(canvas, OUTLINE_WIDTH)
    band = np.zeros_like(canvas)
    for dy in range(-spread, OUTLINE_WIDTH - spread):
        for dx in range(-spread, OUTLINE_WIDTH - spread):
            y0, x0 = OUTLINE_WIDTH + dy, OUTLINE_WIDTH + dx
            band |= padded[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]]
    inked = band[margin : margin + height, margin : margin + width] > 0

    page = Image.new("RGB", size, "white")
    page.paste(Image.new("RGB", size, "black"), (0, 0), Image.fromarray((inked * 255).astype(np.uint8), "L"))
    draw = ImageDraw.Draw(page)

    for region in regions:
        if region.interior_radius < MIN_LABEL_RADIUS_PX:
            continue
        font_size = int(max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO)))
        font = ImageFont.load_default(size=font_size)
        text = str(region.color_index + 1)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = region.interior_point
        draw_x = min(max(x - tw / 2, 0), size[0] - tw) - bbox[0]
        draw_y = min(max(y - th / 2, 0), size[1] - th) - bbox[1]
        draw.text((draw_x, draw_y), text, fill="black", font=font)

    return page

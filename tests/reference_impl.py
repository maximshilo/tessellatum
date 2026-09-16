"""The original (slow) region + render implementations, kept as a test oracle.

Straightforward whole-image versions of ``build_regions``,
``extract_regions`` and ``render_page``, written from the implementations
before the performance rewrite. The optimized versions in
``tessellatum.core`` must produce exactly the same output;
``test_regions_equivalence.py`` checks that.

A change meant to alter that output updates this file deliberately, so it
keeps saying what the stages should do rather than what they used to. Since
the rewrite: ``_merge_same_color_neighbors`` (T2.1).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.regions import Region
from tessellatum.core.render import FONT_SIZE_RADIUS_RATIO, MAX_FONT_SIZE, MIN_FONT_SIZE, MIN_LABEL_RADIUS_PX, OUTLINE_WIDTH

_NEIGHBOR_KERNEL = np.ones((3, 3), np.uint8)


def build_regions(labels: np.ndarray, num_colors: int, min_area_px: int) -> tuple[np.ndarray, np.ndarray]:
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


def render_page(size: tuple[int, int], regions: list[Region]) -> Image.Image:
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)

    for region in regions:
        points = [(int(p[0][0]), int(p[0][1])) for p in region.contour]
        if len(points) >= 2:
            draw.polygon(points, outline="black", width=OUTLINE_WIDTH)
        elif len(points) == 1:
            draw.point(points[0], fill="black")

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

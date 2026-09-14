"""Connected-component region extraction, small-region merging, and contour extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_NEIGHBOR_KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Region:
    """One contiguous, single-colored region of the final coloring page."""

    region_id: int
    color_index: int
    contour: np.ndarray  # Nx1x2 int32 points (OpenCV contour format), outer boundary
    interior_point: tuple[int, int]  # (x, y) safe point for a number label
    interior_radius: float  # px clearance at interior_point, used for font sizing
    area: int


def build_regions(labels: np.ndarray, num_colors: int, min_area_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``labels`` into connected regions and merge tiny ones into neighbors.

    Returns:
        (region_id_map, region_color): region_id_map is HxW int32 (each pixel's
        region id), region_color is region_count-length int32 array mapping a
        region id to its color index in the palette.
    """
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
            # No neighbor found (region touches only image border / is the only
            # region left) -- nothing sensible to merge into, so keep it as-is.
            active[rid] = False
            continue

        counts = np.bincount(neighbor_ids)
        target = int(np.argmax(counts))

        region_id_map[mask.astype(bool)] = target
        areas[target] += areas[rid]
        areas[rid] = 0
        active[rid] = False


def extract_regions(region_id_map: np.ndarray, region_color: np.ndarray, min_contour_area: float = 1.0) -> list[Region]:
    """Extract one outer contour + label point per surviving region id."""
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

        # Pad with a zero border first: cv2.distanceTransform does not treat
        # the image edge itself as a boundary, so a region touching the edge
        # could otherwise report a "safe" interior point right at the
        # physical corner of the canvas, clipping the drawn number.
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

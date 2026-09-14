"""Connected-component region extraction, small-region merging, and contour extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tessellatum.core import kernels, parallel

# Regions with a smaller bounding box than this are extracted on the calling
# thread: for them, handing off to a worker (and contending for the GIL) costs
# more than the OpenCV work itself.
_MIN_POOLED_BOX_PX = 32_000


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

    Regions are 8-connected runs of one color, numbered by color and then by
    raster position. Regions below ``min_area_px`` are merged, smallest first,
    into the neighbor sharing the most boundary (see
    ``kernels.merge_small_regions``).

    Returns:
        (region_id_map, region_color): region_id_map is HxW int32 (each pixel's
        region id), region_color is region_count-length int32 array mapping a
        region id to its color index in the palette.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    h, w = labels.shape
    region_id_map = np.empty((h, w), dtype=np.int32)
    flat_ids = region_id_map.reshape(-1)

    region_color, areas = kernels.label_components(labels.reshape(-1), h, w, int(num_colors), flat_ids)
    if region_color.size:
        kernels.merge_small_regions(flat_ids, h, w, areas, int(min_area_px))
    return region_id_map, region_color


def extract_regions(region_id_map: np.ndarray, region_color: np.ndarray, min_contour_area: float = 1.0) -> list[Region]:
    """Extract one outer contour + label point per surviving region id.

    Each region is processed inside its own one-pixel-padded bounding box
    rather than across the whole image (same result: nothing outside the box
    can affect its contour or distance transform), spread over worker threads.
    Regions come back in region-id order.
    """
    ids = np.ascontiguousarray(region_id_map, dtype=np.int32)
    h, w = ids.shape
    num_ids = int(ids.max()) + 1 if ids.size else 0
    if num_ids <= 0:
        return []
    bounds_array, areas_array = kernels.region_bounds(ids.reshape(-1), h, w, num_ids)
    # Plain lists: much cheaper than NumPy scalar indexing in the per-region loop.
    bounds = bounds_array.tolist()
    areas = areas_array.tolist()
    colors = np.asarray(region_color).tolist()
    present = np.flatnonzero(areas_array > 0).tolist()
    box_px = [(bounds[r][2] - bounds[r][0] + 1) * (bounds[r][3] - bounds[r][1] + 1) for r in present]

    def extract(rid: int) -> Region | None:
        x0, y0, x1, y1 = bounds[rid]
        mask = np.zeros((y1 - y0 + 3, x1 - x0 + 3), dtype=np.uint8)
        np.equal(ids[y0 : y1 + 1, x0 : x1 + 1], rid, out=mask[1:-1, 1:-1], casting="unsafe")
        origin = (x0 - 1, y0 - 1)  # image coordinates of mask[0, 0]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=origin)
        if not contours:
            return None
        contour = contours[0] if len(contours) == 1 else max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_contour_area:
            return None
        contour = cv2.approxPolyDP(contour, epsilon=1.2, closed=True)

        # The zero padding also makes the distance transform treat the image
        # edge as a boundary, so a region touching the edge can't report a
        # "safe" label point right at the canvas corner.
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)

        return Region(
            region_id=rid,
            color_index=colors[rid],
            contour=contour,
            interior_point=(origin[0] + max_loc[0], origin[1] + max_loc[1]),
            interior_radius=float(max_val),
            area=areas[rid],
        )

    extracted = parallel.map_balanced(extract, present, box_px, min_pooled_cost=_MIN_POOLED_BOX_PX)
    return [region for region in extracted if region is not None]

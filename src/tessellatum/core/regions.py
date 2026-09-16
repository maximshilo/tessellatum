"""Connected-component region extraction, region merging, and contour extraction."""

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


def build_regions(
    labels: np.ndarray, num_colors: int, min_area_px: int, min_width_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``labels`` into connected regions and merge away what can't be painted.

    Regions are 8-connected runs of one color, numbered by color and then by
    raster position. Regions below ``min_area_px`` are merged, smallest first,
    into the neighbor sharing the most boundary (see
    ``kernels.merge_small_regions``). That merge ignores color, so it can
    leave two neighbors sharing one: those are then unioned into a single
    region (``kernels.merge_same_color_neighbors``), so no boundary on the
    page separates two areas the painter fills with the same color.

    With ``min_width_px`` set, every part of a region narrower than that --
    which a brush that wide cannot paint without crossing a line -- is then
    given away to the region whose paint reaches it first, and the regions
    are rebuilt from the result (see ``absorb_thin_parts``). A region thinner
    than the brush everywhere disappears into its neighbors.

    Returns:
        (region_id_map, region_color): region_id_map is HxW int32 (each pixel's
        region id), region_color is region_count-length int32 array mapping a
        region id to its color index in the palette.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    region_id_map, region_color = _regions_from_labels(labels, num_colors, min_area_px)
    if min_width_px > 0 and region_color.size:
        widened = absorb_thin_parts(region_id_map, region_color, labels, min_width_px)
        if widened is not None:
            region_id_map, region_color = _regions_from_labels(widened, num_colors, min_area_px)
    return region_id_map, region_color


def _regions_from_labels(labels: np.ndarray, num_colors: int, min_area_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``labels``, with the two merges that always apply."""
    h, w = labels.shape
    region_id_map = np.empty((h, w), dtype=np.int32)
    flat_ids = region_id_map.reshape(-1)

    region_color, areas = kernels.label_components(labels.reshape(-1), h, w, int(num_colors), flat_ids)
    if region_color.size:
        kernels.merge_small_regions(flat_ids, h, w, areas, int(min_area_px))
        kernels.merge_same_color_neighbors(flat_ids, h, w, region_color, areas)
    return region_id_map, region_color


def absorb_thin_parts(
    region_id_map: np.ndarray, region_color: np.ndarray, labels: np.ndarray, min_width_px: float
) -> np.ndarray | None:
    """Give every pixel the color of the region whose core lies nearest, or None if there is no core.

    A region's *core* is the pixels where a round brush ``min_width_px``
    across fits inside the region: the places a painter can put the brush
    without crossing into a neighbor. Every pixel then takes the color of the
    nearest core pixel, which leaves each region its core plus everything the
    brush sweeps around it, and hands the parts too thin to paint -- and
    regions with no core at all -- to whichever neighbor reaches them first.

    A pixel its own region's brush can reach keeps its color: no other
    region's core can be nearer than its own. So only thin parts move, and a
    thin part between two regions is split down its middle rather than given
    to one side.

    Pixels in no region (``labels`` outside the palette) keep their label and
    take no part.

    Returns the new HxW int32 label map, or None when the brush fits nowhere
    on the page, leaving nothing to grow from.
    """
    fits = _brush_fits(region_id_map, int(region_color.size), min_width_px / 2)
    if not fits.any():
        return None

    # For every pixel, the label of the nearest zero pixel: of the nearest core pixel.
    _distance, nearest = cv2.distanceTransformWithLabels(
        (~fits).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    color_of_label = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    color_of_label[nearest[fits]] = region_color[region_id_map[fits]]
    widened = color_of_label[nearest]

    outside = region_id_map < 0
    if outside.any():
        widened[outside] = labels[outside]
    return widened


def _brush_fits(region_id_map: np.ndarray, num_regions: int, radius: float) -> np.ndarray:
    """Pixels where a round brush of ``radius`` sits inside one region.

    A brush fits where the nearest pixel of another region is farther away
    than its radius, so one distance transform per class of
    ``kernels.edge_adjacency_classes`` measures that for every region in the
    class at once. Pixels off the page count as another region: the page edge
    is a line like any other.
    """
    h, w = region_id_map.shape
    class_map = np.empty((h, w), dtype=np.int8)
    num_classes = kernels.edge_adjacency_classes(
        region_id_map.reshape(-1), h, w, num_regions, class_map.reshape(-1)
    )

    in_class = np.zeros((h + 2, w + 2), dtype=np.uint8)  # the border stays 0: off the page is another region
    interior = in_class[1:-1, 1:-1]
    distance = np.empty((h + 2, w + 2), dtype=np.float32)
    far_enough = np.empty((h, w), dtype=bool)
    fits = np.zeros((h, w), dtype=bool)
    for region_class in range(num_classes):
        np.equal(class_map, region_class, out=interior, casting="unsafe")
        cv2.distanceTransform(in_class, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=distance)
        np.greater(distance[1:-1, 1:-1], radius, out=far_enough)
        far_enough &= interior.view(bool)  # only pixels of this class: the rest measure another distance
        fits |= far_enough
    return fits


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

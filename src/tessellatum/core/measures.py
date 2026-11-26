"""Measurements of a generated page, from what it is made of: the regions, their colors and the numbers.

What the app shows of a page (the palette inspector's coverage, the region inspector's sizes) is measured here, on the
region map at the page's own size, and turned into millimeters on paper by the page's scale. The module depends on
NumPy and OpenCV only, never on the rest of the package, so that the benchmark can load this file by its path and judge
every version of the pipeline with the same code (see ``benchmarks/bench_metrics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorCoverage:
    """How much of a page each palette color paints: ``pixels[i]`` of the page's pixels and ``regions[i]`` regions in
    color ``i``, for every color (the legend's first). ``page_pixels`` is the page's size in pixels; ``painted`` the
    pixels in any region, which ``pixels`` sums to."""

    pixels: np.ndarray  # int64, one per color
    regions: np.ndarray  # int64, one per color
    page_pixels: int
    painted: int

    def share(self, color: int) -> float:
        """Color ``color``'s share of the page, 0 to 1."""
        return float(self.pixels[color]) / self.page_pixels if self.page_pixels else 0.0

    def area_mm2(self, color: int, px_per_mm: float) -> float:
        """Color ``color``'s area on paper, in square millimeters, on a page of ``px_per_mm``."""
        return float(self.pixels[color]) / (px_per_mm * px_per_mm)


def region_areas(region_id_map: np.ndarray) -> np.ndarray:
    """Each region id's area in pixels (0 for an id merged away); the pixels in no region (-1) are left out."""
    ids = np.asarray(region_id_map)
    inside = ids[ids >= 0]
    return np.bincount(inside.ravel(), minlength=0).astype(np.int64)


def color_coverage(region_id_map: np.ndarray, region_color: np.ndarray, colors: int) -> ColorCoverage:
    """The pixels and the regions of each of ``colors`` palette colors on a page: a region id's color is
    ``region_color[id]``. A color past ``colors`` (a region keeping a color of its own) widens the counts to it, so
    they always sum to the painted pixels and the regions there are."""
    ids = np.asarray(region_id_map)
    areas = region_areas(ids)
    present = np.flatnonzero(areas)
    color_of = np.asarray(region_color, dtype=np.int64)[present]
    count = max(int(colors), int(color_of.max()) + 1 if color_of.size else 0)
    pixels = np.bincount(color_of, weights=areas[present], minlength=count).astype(np.int64)
    regions = np.bincount(color_of, minlength=count).astype(np.int64)
    return ColorCoverage(pixels=pixels, regions=regions, page_pixels=int(ids.size), painted=int(areas.sum()))


def regions_of_color(region_id_map: np.ndarray, region_color: np.ndarray, color: int) -> np.ndarray:
    """The ids of the regions on the page in ``color``, largest first (ties by id)."""
    areas = region_areas(region_id_map)
    present = np.flatnonzero(areas)
    ids = present[np.asarray(region_color, dtype=np.int64)[present] == color]
    order = np.lexsort((ids, -areas[ids]))
    return ids[order]


def color_mask(region_id_map: np.ndarray, region_color: np.ndarray, color: int) -> np.ndarray:
    """HxW bool: the pixels painted in ``color``."""
    ids = np.asarray(region_id_map)
    lookup = np.asarray(region_color, dtype=np.int64) == color
    mask = np.zeros(ids.shape, dtype=bool)
    inside = ids >= 0
    mask[inside] = lookup[ids[inside]]
    return mask


@dataclass(frozen=True)
class RegionStats:
    """Every region's area and bounding box on a page, by region id: ``areas[id]`` pixels within the box
    ``x0[id] <= x < x1[id]``, ``y0[id] <= y < y1[id]``. An id merged away has area 0 and an empty box."""

    areas: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray

    def box(self, region_id: int) -> tuple[int, int, int, int]:
        return int(self.x0[region_id]), int(self.y0[region_id]), int(self.x1[region_id]), int(self.y1[region_id])

    def __contains__(self, region_id: int) -> bool:
        return 0 <= region_id < len(self.areas) and self.areas[region_id] > 0


def region_stats(region_id_map: np.ndarray) -> RegionStats:
    """Each region's area and bounding box (see ``RegionStats``), in one pass over the page."""
    ids = np.asarray(region_id_map)
    height, width = ids.shape
    flat = ids.ravel()
    where = np.flatnonzero(flat >= 0)
    count = int(flat[where].max()) + 1 if where.size else 0
    areas = np.bincount(flat[where], minlength=count).astype(np.int64)
    x0 = np.zeros(count, dtype=np.int64)
    y0 = np.zeros(count, dtype=np.int64)
    x1 = np.zeros(count, dtype=np.int64)
    y1 = np.zeros(count, dtype=np.int64)
    if where.size:
        order = where[np.argsort(flat[where], kind="stable")]
        present = np.flatnonzero(areas)
        starts = np.concatenate(([0], np.cumsum(areas[present])[:-1]))
        ys, xs = np.divmod(order, width)
        y0[present] = ys[starts]  # the pixels are in reading order within a region: its first is in its top row
        y1[present] = np.maximum.reduceat(ys, starts) + 1
        x0[present] = np.minimum.reduceat(xs, starts)
        x1[present] = np.maximum.reduceat(xs, starts) + 1
    return RegionStats(areas=areas, x0=x0, y0=y0, x1=x1, y1=y1)


def widest_brush_px(region_id_map: np.ndarray, region_id: int, box: tuple[int, int, int, int]) -> float:
    """The width, in pixels, of the widest round brush that fits somewhere in region ``region_id`` (whose bounding
    box is ``box``): twice the farthest any pixel of it is from outside it, less the pixel the brush is centered on."""
    import cv2

    x0, y0, x1, y1 = box
    crop = np.asarray(region_id_map)[y0:y1, x0:x1] == region_id
    if not crop.any():
        return 0.0
    padded = np.pad(crop, 1).astype(np.uint8)  # the page's edge bounds a region as its neighbors do
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return max(2.0 * float(distance.max()) - 1.0, 1.0)


def region_mask(region_id_map: np.ndarray, region_id: int) -> np.ndarray:
    """HxW bool: the pixels of region ``region_id``."""
    return np.asarray(region_id_map) == region_id

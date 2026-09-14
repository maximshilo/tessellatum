"""Top-level orchestration: image -> quantize -> regions -> render -> legend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.legend import render_legend
from tessellatum.core.quantize import quantize
from tessellatum.core.regions import build_regions, extract_regions
from tessellatum.core.render import render_page

PREVIEW_LONG_EDGE = 1100
EXPORT_LONG_EDGE = 2400

# Rough share of total generation time each stage takes, used to report real
# (if coarse-grained) percentage progress rather than a fake animation.
_STAGE_PROGRESS = {
    "resize": 5,
    "quantize": 50,
    "regions": 75,
    "contours": 90,
    "render": 100,
}


class PipelineCancelled(Exception):
    """Raised internally to unwind ``generate`` when the caller cancels it."""


@dataclass
class GeneratedPage:
    page: Image.Image
    legend: Image.Image
    palette_rgb: list[tuple[int, int, int]]
    num_colors_used: int
    num_regions: int


def load_image_bgr(path: Path) -> np.ndarray:
    """Load any image format via Pillow, return an OpenCV-style BGR array."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def resize_to_long_edge(image_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    current_long_edge = max(h, w)
    if current_long_edge <= long_edge:
        return image_bgr
    scale = long_edge / current_long_edge
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def generate(
    image_bgr: np.ndarray,
    params: DifficultyParams,
    long_edge: int,
    progress_callback: Callable[[int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> GeneratedPage:
    """Run the full pipeline on ``image_bgr`` and produce a coloring page + legend.

    ``progress_callback``, if given, is called after each pipeline stage with
    a 0-100 percentage reflecting real work completed (not a fake animation).
    ``should_cancel``, if given, is polled between stages; when it returns
    True, ``PipelineCancelled`` is raised and no more work is done.
    """

    def report(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(_STAGE_PROGRESS[stage])

    def check_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise PipelineCancelled()

    check_cancelled()
    resized = resize_to_long_edge(image_bgr, long_edge)
    h, w = resized.shape[:2]
    report("resize")

    check_cancelled()
    labels, palette_bgr = quantize(resized, params.num_colors, params.blur_sigma)
    report("quantize")

    check_cancelled()
    min_area_px = max(4, int(round(params.min_region_fraction * h * w)))
    region_id_map, region_color = build_regions(labels, params.num_colors, min_area_px)
    report("regions")

    check_cancelled()
    regions = extract_regions(region_id_map, region_color)
    report("contours")

    check_cancelled()
    # Quantizing to more colors than the image actually has can leave some
    # k-means clusters with no (or a merged-away) region. Drop those from the
    # legend and renumber the rest contiguously so "1..N" always matches what
    # is actually drawn on the page.
    used_color_indices = sorted({r.color_index for r in regions})
    remap = {old: new for new, old in enumerate(used_color_indices)}
    for region in regions:
        region.color_index = remap[region.color_index]
    used_palette_bgr = palette_bgr[used_color_indices]

    page = render_page((w, h), regions)
    legend = render_legend(used_palette_bgr, width=w)
    report("render")

    palette_rgb = [(int(b[2]), int(b[1]), int(b[0])) for b in used_palette_bgr]

    return GeneratedPage(
        page=page,
        legend=legend,
        palette_rgb=palette_rgb,
        num_colors_used=len(used_color_indices),
        num_regions=len(regions),
    )

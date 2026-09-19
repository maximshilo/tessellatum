"""Top-level orchestration: image -> quantize -> regions -> render -> legend."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, TypeVar

import cv2
import numpy as np
from PIL import Image

from tessellatum.core import ink, kernels
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.legend import render_legend
from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM, MIN_REGION_AREA_MM2, print_scale
from tessellatum.core.quantize import quantize
from tessellatum.core.regions import Region, build_regions, extract_regions
from tessellatum.core.render import Label, PageStyle, render_page

PREVIEW_LONG_EDGE = 1100
EXPORT_LONG_EDGE = 2400

# Rough share of total generation time each stage takes, used to report real
# (if coarse-grained) percentage progress rather than a fake animation.
_STAGE_PROGRESS = {
    "resize": 2,
    "quantize": 60,
    "regions": 80,
    "contours": 90,
    "render": 100,
}

T = TypeVar("T")


class PipelineCancelled(Exception):
    """Raised internally to unwind ``generate`` when the caller cancels it."""


@dataclass
class PageAnalysis:
    """What a generated page is made of, for measuring it (see ``generate``).

    Colors are indices into ``palette_bgr``. Its first ``legend_size`` colors
    are the legend's, in legend order, so color ``i`` is numbered ``i + 1`` on
    the page. The rest are quantized colors no drawn region has.

    A region in ``region_id_map`` with no entry in ``regions`` has an outline
    that encloses no area (e.g. it is one pixel wide), so it gets no number.
    Its boundaries are still drawn: the lines come from the region map, not
    from the regions.
    """

    region_id_map: np.ndarray  # HxW int32: each pixel's region id, -1 for none
    region_color: np.ndarray  # color of each region id (merged-away ids keep an entry)
    palette_bgr: np.ndarray  # Kx3 uint8, legend colors first
    legend_size: int
    min_region_area_px: int  # merge threshold: smaller regions merge into a neighbor, if they have one
    min_paintable_width_px: float  # brush width: narrower parts of a region are given to a neighbor
    regions: list[Region]  # regions drawn on the page, in region-id order
    labels: list[Label]  # numbers drawn on the page
    outlines: np.ndarray  # HxW uint8: the ink the lines alone put on the page, 0 = solid ink, 255 = bare paper
    # Every line drawn, in drawing order: Nx2 float64 (x, y) points with pixel centers at integer coordinates.
    # One line per boundary between two regions, traced along the pixel cracks and smoothed off them
    # by at most boundaries.MAX_SHIFT_PX.
    # A closed line repeats its first point at the end.
    strokes: list[np.ndarray]
    # HxW uint8: the ink the leader lines of numbers written outside their regions put on the page, as in outlines.
    leaders: np.ndarray
    # Whether the picture is line art, decided on it at preview size (see ``ink``), and HxW bool: the pixels on its
    # ink lines, all False unless it is. Found for measuring only: nothing on the page uses them yet.
    line_art: ink.LineArt
    ink_lines: np.ndarray


@dataclass
class GeneratedPage:
    page: Image.Image
    legend: Image.Image
    palette_rgb: list[tuple[int, int, int]]
    num_colors_used: int
    num_regions: int
    analysis: PageAnalysis | None = None  # only with generate(..., collect_analysis=True)


class _StageCache:
    """Recent stage results for the most recently used source image.

    Tuning difficulty re-runs the pipeline on one image over and over, and
    the expensive early stages depend on only some parameters (smoothing and
    k-means don't care about the minimum region size, resizing only about the
    output size). Entries are tied to the image *object* through a weak
    reference, so a different or garbage-collected image never gets stale
    results. Image arrays must not be modified in place after being passed in.
    """

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._image_ref: weakref.ref | None = None
        self._entries: OrderedDict[Hashable, object] = OrderedDict()
        self._max_entries = max_entries

    def get_or_compute(self, image: np.ndarray, key: Hashable, compute: Callable[[], T]) -> T:
        with self._lock:
            if self._image_ref is None or self._image_ref() is not image:
                self._image_ref = weakref.ref(image)
                self._entries.clear()
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]  # type: ignore[return-value]

        value = compute()  # outside the lock: this is the slow part

        with self._lock:
            if self._image_ref() is image:
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._image_ref = None
            self._entries.clear()


_cache = _StageCache(max_entries=6)


def clear_cache() -> None:
    """Forget cached intermediate results (see ``_StageCache``)."""
    _cache.clear()


def _paintable_limits(params: DifficultyParams, size: tuple[int, int]) -> tuple[int, float]:
    """The region stage's limits for a page of ``size`` (width, height) in pixels.

    The difficulty sets the smallest region and the narrowest part of one on
    the printed page, and the printed page sets how small either may get at
    any difficulty: the brush, and its footprint (see ``print_size``). The
    printed page's size follows the image's shape rather than its pixel
    count, so a preview and an export of one image are held to the same
    physical sizes.
    """
    scale = print_scale(size)
    min_area_px = max(4, int(round(scale.mm2_to_px(max(params.min_region_area_mm2, MIN_REGION_AREA_MM2)))))
    return min_area_px, scale.mm_to_px(max(params.min_width_mm, MIN_PAINTABLE_WIDTH_MM))


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


def warm_up() -> None:
    """Pay one-time start-up costs before the first real generation.

    Loads the compiled region kernels (compiling them if this is the first run
    since install, which takes a few seconds) and runs the whole pipeline once
    on a tiny synthetic image. Meant for a background thread at app start.
    """
    kernels.warm_up()
    tiny = np.zeros((48, 64, 3), dtype=np.uint8)
    tiny[:, 32:] = (40, 160, 220)
    tiny[12:36, 8:24] = (200, 60, 60)
    generate(tiny, DifficultyParams(num_colors=4, min_region_area_mm2=500.0, blur_sigma=1.0), long_edge=64)


def generate(
    image_bgr: np.ndarray,
    params: DifficultyParams,
    long_edge: int,
    progress_callback: Callable[[int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    collect_analysis: bool = False,
    style: PageStyle = PageStyle(),
) -> GeneratedPage:
    """Run the full pipeline on ``image_bgr`` and produce a coloring page + legend.

    ``progress_callback``, if given, is called after each pipeline stage with
    a 0-100 percentage reflecting real work completed (not a fake animation).
    ``should_cancel``, if given, is polled between stages; when it returns
    True, ``PipelineCancelled`` is raised and no more work is done.
    ``collect_analysis`` also returns what the page is made of in
    ``GeneratedPage.analysis`` (see ``PageAnalysis``), for benchmarks and
    tests. The page itself is the same either way. ``style`` says how the page
    is drawn -- line width and the tone of the ink (see ``PageStyle``); it
    changes nothing about which regions the page has.

    Resizing and quantization results are cached per image object, so
    regenerating the same image with a different minimum region size, or
    going back to earlier settings, skips straight to the region stages.
    """

    def report(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(_STAGE_PROGRESS[stage])

    def check_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise PipelineCancelled()

    check_cancelled()
    resized = _cache.get_or_compute(
        image_bgr, ("resize", long_edge), lambda: resize_to_long_edge(image_bgr, long_edge)
    )
    h, w = resized.shape[:2]
    report("resize")

    check_cancelled()
    labels, palette_bgr = _cache.get_or_compute(
        image_bgr,
        ("quantize", long_edge, params.num_colors, params.blur_sigma),
        lambda: quantize(resized, params.num_colors, params.blur_sigma),
    )
    report("quantize")

    check_cancelled()
    min_area_px, min_width_px = _paintable_limits(params, (w, h))
    # The palette can be shorter than the difficulty asked for: colors too
    # close to tell apart are merged (see ``quantize``). Passing the count the
    # difficulty asked for gives the same regions -- the labeling only needs an
    # upper bound -- but not the same meaning, and it sizes its arrays for
    # colors that do not exist.
    region_id_map, region_color = build_regions(labels, len(palette_bgr), min_area_px, min_width_px)
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

    rendered = render_page((w, h), regions, region_id_map, style)
    legend = render_legend(used_palette_bgr, width=w)
    report("render")

    palette_rgb = [(int(b[2]), int(b[1]), int(b[0])) for b in used_palette_bgr]

    analysis = None
    if collect_analysis:
        # Legend colors first, so a color index means the same color in
        # region_color as in the renumbered regions.
        order = used_color_indices + [i for i in range(len(palette_bgr)) if i not in remap]
        new_index = np.empty(len(order), dtype=np.int32)
        new_index[order] = np.arange(len(order), dtype=np.int32)
        # Line art is decided on the picture at preview size, so every size of it gets the same answer.
        picture = resized if long_edge == PREVIEW_LONG_EDGE else resize_to_long_edge(image_bgr, PREVIEW_LONG_EDGE)
        line_art, ink_lines = ink.find_ink(resized, picture)
        analysis = PageAnalysis(
            region_id_map=region_id_map,
            region_color=new_index[region_color],
            palette_bgr=palette_bgr[order],  # a copy: palette_bgr belongs to the stage cache
            legend_size=len(used_color_indices),
            min_region_area_px=min_area_px,
            min_paintable_width_px=min_width_px,
            regions=regions,
            labels=rendered.labels,
            outlines=np.asarray(rendered.outlines),
            strokes=rendered.strokes,
            leaders=np.asarray(rendered.leaders),
            line_art=line_art,
            ink_lines=ink_lines,
        )

    return GeneratedPage(
        page=rendered.image,
        legend=legend,
        palette_rgb=palette_rgb,
        num_colors_used=len(used_color_indices),
        num_regions=len(regions),
        analysis=analysis,
    )

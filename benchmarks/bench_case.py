"""Run one benchmark case in a fresh interpreter.

Spawned by ``bench.py run`` -- not meant to be invoked by hand. Imports
tessellatum from ``--src`` (so any version can be measured), times the
pipeline, scores its output with ``bench_metrics`` and writes ``case.json``
plus the rendered outputs into ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Stage functions the probe times, if the pipeline module exposes them
# (``generate`` looks them up as module globals, so wrapping works). For
# versions whose ``generate`` can't collect analysis, the probe also captures
# what scoring needs. Keep these names when restructuring the pipeline, or
# extend this list.
PROBED_STAGES = (
    "resize_to_long_edge",
    "quantize",
    "build_regions",
    "extract_regions",
    "render_page",
    "render_legend",
)

# Benchmark-only presets on top of the app's own. "Max" is the most granular
# setting the Custom sliders allow -- the worst case for region handling.
EXTRA_PRESETS = {"Max": dict(num_colors=40, min_region_fraction=0.0002, blur_sigma=0.0)}

# How render_page numbered regions in the versions before the analysis payload,
# for when their render module doesn't say (see ``page_data_from_probe``).
RENDER_DEFAULTS = {"MIN_LABEL_RADIUS_PX": 9.0, "MIN_FONT_SIZE": 10, "MAX_FONT_SIZE": 40, "FONT_SIZE_RADIUS_RATIO": 0.85}


class Probe:
    """Wraps the pipeline's stage functions to time them and keep their inputs/outputs."""

    def __init__(self, module) -> None:
        self.timings: dict[str, float] = {}
        self.captured: dict[str, tuple] = {}
        for name in PROBED_STAGES:
            fn = getattr(module, name, None)
            if callable(fn):
                setattr(module, name, self._wrap(name, fn))

    def reset(self) -> None:
        self.timings = {}
        self.captured = {}

    def _wrap(self, name, fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - start
            self.captured[name] = (args, kwargs, result)
            return result

        return wrapper


@dataclass
class PageData:
    """What scoring reads from a generated page, whichever way its pipeline version provides it."""

    source: str  # "analysis": generate's analysis payload; "probe": captured stage calls
    region_id_map: np.ndarray  # HxW region id per pixel
    region_color: np.ndarray  # color of each region id, as an index into palette_bgr
    palette_bgr: np.ndarray
    min_region_area_px: int
    regions: list  # regions drawn on the page
    labeled_region_ids: set[int]  # regions that carry a number
    label_font_sizes_px: list[int]  # em size of every number on the page
    strokes: list[np.ndarray]  # every line drawn: (x, y) polylines, pixel centers at integers; a closed one returns to its start


def page_data_from_analysis(analysis) -> PageData:
    """Read the payload of ``generate(..., collect_analysis=True)``.

    Versions before 0.1.12 don't list the lines they draw; like the probe, their
    lines are rebuilt as the outlines of the drawn regions.
    """
    return PageData(
        source="analysis",
        region_id_map=analysis.region_id_map,
        region_color=analysis.region_color,
        palette_bgr=analysis.palette_bgr,
        min_region_area_px=analysis.min_region_area_px,
        regions=analysis.regions,
        labeled_region_ids={label.region_id for label in analysis.labels},
        label_font_sizes_px=[label.font_size for label in analysis.labels],
        strokes=(
            analysis.strokes
            if hasattr(analysis, "strokes")
            else [outline_polyline(r.contour) for r in analysis.regions if len(r.contour)]
        ),
    )


def page_data_from_probe(captured: dict, params, size: tuple[int, int], render_module) -> PageData | None:
    """Rebuild page data from the stage calls of a version without the analysis payload.

    Relies on how those versions worked: ``generate`` derived the merge
    threshold from the difficulty as below, and ``render_page`` outlined every
    region it was given by drawing its contour as a polygon, and numbered
    exactly the regions with at least ``MIN_LABEL_RADIUS_PX`` of clearance, at
    a font size of that clearance times ``FONT_SIZE_RADIUS_RATIO``, clamped to
    ``MIN_FONT_SIZE``..``MAX_FONT_SIZE``.
    """
    if not all(stage in captured for stage in ("quantize", "build_regions", "render_page")):
        return None
    w, h = size
    region_id_map, region_color = captured["build_regions"][2]
    regions = captured["render_page"][0][1]

    def constant(name: str):
        return getattr(render_module, name, RENDER_DEFAULTS[name])

    labeled = [r for r in regions if r.interior_radius >= constant("MIN_LABEL_RADIUS_PX")]
    smallest, largest, ratio = constant("MIN_FONT_SIZE"), constant("MAX_FONT_SIZE"), constant("FONT_SIZE_RADIUS_RATIO")
    return PageData(
        source="probe",
        region_id_map=region_id_map,
        region_color=region_color,
        palette_bgr=captured["quantize"][2][1],
        min_region_area_px=max(4, int(round(params.min_region_fraction * h * w))),
        regions=regions,
        labeled_region_ids={r.region_id for r in labeled},
        label_font_sizes_px=[int(max(smallest, min(largest, r.interior_radius * ratio))) for r in labeled],
        strokes=[outline_polyline(r.contour) for r in regions if len(r.contour)],
    )


def outline_polyline(contour) -> np.ndarray:
    """A region contour drawn as a polygon outline, as (x, y) points that return to the first one."""
    import numpy as np  # imported late in this module, after the measured version's package

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    return np.vstack([points, points[:1]]) if len(points) >= 2 else points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--long-edge", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src_dir = args.src.resolve()
    sys.path.insert(0, str(src_dir))

    start = time.perf_counter()
    import tessellatum
    from tessellatum.core import difficulty, pipeline

    import_s = time.perf_counter() - start

    if src_dir not in Path(tessellatum.__file__).resolve().parents:
        print("tessellatum was not imported from --src (is something shadowing it?)", file=sys.stderr)
        return 2

    import cv2
    import numpy as np
    from PIL import Image

    import bench_metrics as bm

    if args.threads:
        cv2.setNumThreads(args.threads)

    if args.preset in EXTRA_PRESETS:
        params = difficulty.DifficultyParams(**EXTRA_PRESETS[args.preset])
    else:
        params = difficulty.params_for_preset(args.preset)

    image_bgr = pipeline.load_image_bgr(args.image)
    probe = Probe(pipeline)
    has_analysis = "collect_analysis" in inspect.signature(pipeline.generate).parameters

    def run_once(collect_analysis: bool = False):
        probe.reset()
        # A fresh copy per run, so a pipeline that caches work per image object
        # can't turn repeats into cache hits.
        fresh = image_bgr.copy()
        options = {"collect_analysis": True} if collect_analysis else {}
        started = time.perf_counter()
        result = pipeline.generate(
            fresh, params, args.long_edge, progress_callback=lambda _pct: None, should_cancel=lambda: False, **options
        )
        total = time.perf_counter() - started
        return result, {"total_s": total, "stages_s": dict(probe.timings)}

    warmup_runs = [run_once()[1] for _ in range(args.warmup)]
    runs = []
    page_hashes = set()
    for _ in range(max(1, args.repeats)):
        result, timing = run_once()
        runs.append(timing)
        page_hashes.add(hashlib.sha1(result.page.tobytes()).hexdigest())
    peak_rss_mb = _peak_rss_mb()  # before scoring, which allocates plenty of its own

    totals = [r["total_s"] for r in runs]
    stage_names = [name for name in PROBED_STAGES if all(name in r["stages_s"] for r in runs)]
    median_stages = {name: statistics.median(r["stages_s"][name] for r in runs) for name in stage_names}
    median_stages["other"] = statistics.median(
        r["total_s"] - sum(r["stages_s"].get(name, 0.0) for name in stage_names) for r in runs
    )

    reference = bm.reference_resize(image_bgr, args.long_edge)
    h, w = reference.shape[:2]
    if has_analysis:
        # The timed runs leave analysis off, as the app does; one more run collects it.
        result = run_once(collect_analysis=True)[0]
        page_hashes.add(hashlib.sha1(result.page.tobytes()).hexdigest())
        page_data = page_data_from_analysis(result.analysis)
    else:  # from the stage calls of the last measured run
        render_module = sys.modules.get("tessellatum.core.render")
        page_data = page_data_from_probe(probe.captured, params, (w, h), render_module)

    print_scale = bm.print_size.print_scale(result.page.size)
    page_rgb = np.asarray(result.page.convert("RGB"))
    quality: dict[str, float | int | None] = {
        "colors_used": int(result.num_colors_used),
        "regions": int(result.num_regions),
        "ink_fraction": bm.ink_fraction(page_rgb),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    result.page.save(args.out / "page.png")

    if page_data is not None:
        painted = bm.paint(page_data.region_id_map, page_data.region_color, page_data.palette_bgr)
        quality.update(bm.fidelity(reference, bm.fit_to(painted, (w, h))))
        quality["undersized_regions"] = bm.count_undersized(page_data.region_id_map, page_data.min_region_area_px)
        quality.update(
            bm.label_coverage(page_data.regions, page_data.labeled_region_ids, page_rgb.shape[0] * page_rgb.shape[1])
        )
        quality.update(bm.unlabeled_regions(page_data.region_id_map, page_data.labeled_region_ids))
        brush_px = print_scale.mm_to_px(bm.print_size.MIN_PAINTABLE_WIDTH_MM)
        quality["sliver_area_fraction"] = bm.sliver_share(page_data.region_id_map, brush_px)
        quality.update(bm.label_sizes(page_data.label_font_sizes_px, print_scale))
        quality.update(bm.compactness_stats(page_data.region_id_map))
        quality.update(bm.boundary_lines(page_data.region_id_map, page_data.strokes))
        quality["same_color_boundary_fraction"] = bm.same_color_boundary_share(
            page_data.region_id_map, page_data.region_color
        )
        quality["jaggedness"] = bm.jaggedness(
            page_data.strokes, page_data.region_id_map, print_scale.mm_to_px(bm.JAGGEDNESS_SMOOTHING_MM)
        )
        edges = bm.source_edges(
            bm.fit_to(reference, page_data.region_id_map.shape[::-1]), print_scale.mm_to_px(bm.EDGE_SMOOTHING_MM)
        )
        quality.update(bm.edge_alignment(page_data.region_id_map, edges, print_scale.mm_to_px(bm.EDGE_TOLERANCE_MM)))
        Image.fromarray(np.ascontiguousarray(painted[:, :, ::-1])).save(args.out / "painted.png")
        np.savez_compressed(args.out / "regions.npz", region_id_map=page_data.region_id_map)

    case = {
        "case": args.out.name,
        "image": args.image.name,
        "preset": args.preset,
        "params": {
            "num_colors": params.num_colors,
            "min_region_fraction": params.min_region_fraction,
            "blur_sigma": params.blur_sigma,
        },
        "long_edge": args.long_edge,
        "status": "ok",
        "version": getattr(tessellatum, "__version__", "unknown"),
        "output_size": [result.page.width, result.page.height],
        "print": {
            "landscape": print_scale.landscape,
            "printed_size_mm": list(print_scale.printed_size_mm),
            "px_per_mm": print_scale.px_per_mm,
            "dpi": print_scale.dpi,
        },
        "import_s": import_s,
        "first_run_s": (warmup_runs or runs)[0]["total_s"],
        "median_total_s": statistics.median(totals),
        "min_total_s": min(totals),
        "median_stages_s": median_stages,
        "warmup_runs": warmup_runs,
        "runs": runs,
        "deterministic": len(page_hashes) == 1,
        "page_sha1": sorted(page_hashes)[0],
        "peak_rss_mb": peak_rss_mb,
        "scored_from": page_data.source if page_data else None,
        "quality": quality,
    }
    (args.out / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return 0


def _peak_rss_mb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PeakWorkingSetSize / 2**20

        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except Exception:  # noqa: BLE001 - memory stats are nice-to-have
        return None


if __name__ == "__main__":
    sys.exit(main())

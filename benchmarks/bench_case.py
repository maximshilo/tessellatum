"""Run one benchmark case in a fresh interpreter.

Spawned by ``bench.py run`` -- not meant to be invoked by hand. Imports
tessellatum from ``--src`` (so any version can be measured), times the
pipeline, scores its output with ``bench_metrics`` and writes ``case.json``
plus the rendered outputs into ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

# Stage functions the probe times and captures, if the pipeline module exposes
# them (``generate`` looks them up as module globals, so wrapping works).
# Keep these names when restructuring the pipeline, or extend this list.
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

DEFAULT_LABEL_RADIUS_PX = 9.0


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

    def run_once():
        probe.reset()
        # A fresh copy per run, so a pipeline that caches work per image object
        # can't turn repeats into cache hits.
        fresh = image_bgr.copy()
        started = time.perf_counter()
        result = pipeline.generate(
            fresh, params, args.long_edge, progress_callback=lambda _pct: None, should_cancel=lambda: False
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

    captured = probe.captured  # from the last measured run
    reference = bm.reference_resize(image_bgr, args.long_edge)
    h, w = reference.shape[:2]
    print_scale = bm.print_size.print_scale(result.page.size)
    page_rgb = np.asarray(result.page.convert("RGB"))
    quality: dict[str, float | int] = {
        "colors_used": int(result.num_colors_used),
        "regions": int(result.num_regions),
        "ink_fraction": bm.ink_fraction(page_rgb),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    result.page.save(args.out / "page.png")

    if "build_regions" in captured and "quantize" in captured:
        region_id_map, region_color = captured["build_regions"][2]
        palette_bgr = captured["quantize"][2][1]
        painted = bm.paint(region_id_map, region_color, palette_bgr)
        quality.update(bm.fidelity(reference, bm.fit_to(painted, (w, h))))
        min_area_px = max(4, int(round(params.min_region_fraction * h * w)))
        quality["undersized_regions"] = bm.count_undersized(region_id_map, min_area_px)
        Image.fromarray(np.ascontiguousarray(painted[:, :, ::-1])).save(args.out / "painted.png")
        np.savez_compressed(args.out / "regions.npz", region_id_map=region_id_map)

    if "render_page" in captured:
        regions = captured["render_page"][0][1]
        render_module = sys.modules.get("tessellatum.core.render")
        label_radius = getattr(render_module, "MIN_LABEL_RADIUS_PX", DEFAULT_LABEL_RADIUS_PX)
        quality.update(bm.label_coverage(regions, label_radius, page_rgb.shape[0] * page_rgb.shape[1]))

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

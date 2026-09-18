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

# Line-art fields, None unless the image's manifest entry has both flat and ink colors.
LINE_ART_KEYS = ("ink_line_precision", "ink_line_recall", "ink_line_f1", "tube_regions", "tube_ink_fraction")
# Face fields, None unless the image's manifest entry has faces.
FACE_KEYS = ("face_de00_mean", "face_ssim", "features_lost", "feature_edge_recall", "labels_on_features")
# Text fields, None unless the image's manifest entry has text; the character error rates also without the OCR engine.
TEXT_KEYS = ("text_cer_source", "text_cer_page", "text_cer_painting", "labels_on_text")
# Enclosure fields, None for versions that report no line layer (before 0.1.10).
ENCLOSURE_KEYS = ("unenclosed_area_fraction", "unenclosed_areas", "split_regions")

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
    legend_bgr: np.ndarray  # the legend's colors, in legend order: the colors of the drawn regions
    min_region_area_px: int
    regions: list  # regions drawn on the page
    labeled_region_ids: set[int]  # regions that carry a number
    label_font_sizes_px: list[int]  # em size of every number on the page
    label_boxes: list[tuple[float, float, float, float]]  # every number's (x0, y0, x1, y1) text box on the page
    strokes: list[np.ndarray]  # every line drawn: (x, y) polylines, pixel centers at integers; a closed one returns to its start
    outlines: np.ndarray | None  # HxW: the line layer alone, 0 where there is a line; None before 0.1.10
    leaders: np.ndarray | None  # HxW: the ink of the numbers' leader lines, as outlines; None for versions that draw none
    leader_labels: int  # numbers written outside their region, with a leader pointing in


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
        legend_bgr=analysis.palette_bgr[: analysis.legend_size],
        min_region_area_px=analysis.min_region_area_px,
        regions=analysis.regions,
        labeled_region_ids={label.region_id for label in analysis.labels},
        label_font_sizes_px=[label.font_size for label in analysis.labels],
        label_boxes=[tuple(label.box) for label in analysis.labels],
        strokes=(
            analysis.strokes
            if hasattr(analysis, "strokes")
            else [outline_polyline(r.contour) for r in analysis.regions if len(r.contour)]
        ),
        outlines=analysis.outlines,
        leaders=getattr(analysis, "leaders", None),  # before 0.1.25 no number had a leader
        leader_labels=sum(getattr(label, "leader", None) is not None for label in analysis.labels),
    )


def page_data_from_probe(captured: dict, params, size: tuple[int, int], render_module) -> PageData | None:
    """Rebuild page data from the stage calls of a version without the analysis payload.

    Relies on how those versions worked: ``generate`` derived the merge
    threshold from the difficulty as below, and ``render_page`` outlined every
    region it was given by drawing its contour as a polygon, and numbered
    exactly the regions with at least ``MIN_LABEL_RADIUS_PX`` of clearance, at
    a font size of that clearance times ``FONT_SIZE_RADIUS_RATIO``, clamped to
    ``MIN_FONT_SIZE``..``MAX_FONT_SIZE``, centered on the region's label point
    and kept on the page. The legend listed the drawn regions' colors, in
    quantizer order.
    """
    if not all(stage in captured for stage in ("quantize", "build_regions", "render_page")):
        return None
    w, h = size
    region_id_map, region_color = captured["build_regions"][2]
    regions = captured["render_page"][0][1]
    palette_bgr = captured["quantize"][2][1]

    def constant(name: str):
        return getattr(render_module, name, RENDER_DEFAULTS[name])

    labeled = [r for r in regions if r.interior_radius >= constant("MIN_LABEL_RADIUS_PX")]
    smallest, largest, ratio = constant("MIN_FONT_SIZE"), constant("MAX_FONT_SIZE"), constant("FONT_SIZE_RADIUS_RATIO")
    font_sizes = [int(max(smallest, min(largest, r.interior_radius * ratio))) for r in labeled]
    return PageData(
        source="probe",
        region_id_map=region_id_map,
        region_color=region_color,
        palette_bgr=palette_bgr,
        # Regions' own color_index is renumbered to legend order before rendering, so read their colors from the map's.
        legend_bgr=palette_bgr[sorted({int(region_color[r.region_id]) for r in regions})],
        min_region_area_px=max(4, int(round(params.min_region_fraction * h * w))),
        regions=regions,
        labeled_region_ids={r.region_id for r in labeled},
        label_font_sizes_px=font_sizes,
        # Regions' color_index is in legend order by the time render_page runs, so it gives each number's text.
        label_boxes=[
            label_box(str(r.color_index + 1), font_size, r.interior_point, (w, h)) for r, font_size in zip(labeled, font_sizes)
        ],
        strokes=[outline_polyline(r.contour) for r in regions if len(r.contour)],
        outlines=None,  # those versions report no layers; enclosure goes unscored for them
        leaders=None,
        leader_labels=0,  # those versions wrote every number inside its region
    )


def outline_polyline(contour) -> np.ndarray:
    """A region contour drawn as a polygon outline, as (x, y) points that return to the first one."""
    import numpy as np  # imported late in this module, after the measured version's package

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    return np.vstack([points, points[:1]]) if len(points) >= 2 else points


def label_box(text: str, font_size: int, point, page_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """The (x0, y0, x1, y1) box where ``render_page`` put a number.

    That is the number's text box in the default font, centered on ``point``
    and kept on the page.
    """
    from PIL import Image, ImageDraw, ImageFont  # imported late in this module, after the measured version's package

    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=ImageFont.load_default(size=font_size)
    )
    tw, th = right - left, bottom - top
    x, y = point
    x0 = min(max(x - tw / 2, 0), page_size[0] - tw)
    y0 = min(max(y - th / 2, 0), page_size[1] - th)
    return (x0, y0, x0 + tw, y0 + th)


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

    import bench_manifest
    import bench_metrics as bm

    if args.threads:
        cv2.setNumThreads(args.threads)

    if args.preset in EXTRA_PRESETS:
        params = difficulty.DifficultyParams(**EXTRA_PRESETS[args.preset])
    else:
        params = difficulty.params_for_preset(args.preset)

    image_bgr = pipeline.load_image_bgr(args.image)
    image_info = bench_manifest.find_image(args.image)
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
    face_features = None
    ocr = text_blocks = None

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
        quality.update(bm.label_clearance(page_data.label_boxes, page_data.outlines, page_data.leaders))
        quality["leader_labels"] = page_data.leader_labels
        quality.update(bm.compactness_stats(page_data.region_id_map))
        quality.update(bm.boundary_lines(page_data.region_id_map, page_data.strokes))
        quality.update(
            bm.enclosure(page_data.region_id_map, page_data.outlines)
            if page_data.outlines is not None
            else dict.fromkeys(ENCLOSURE_KEYS)
        )
        quality["same_color_boundary_fraction"] = bm.same_color_boundary_share(
            page_data.region_id_map, page_data.region_color
        )
        quality["jaggedness"] = bm.jaggedness(
            page_data.strokes, page_data.region_id_map, print_scale.mm_to_px(bm.JAGGEDNESS_SMOOTHING_MM)
        )
        source = bm.fit_to(reference, page_data.region_id_map.shape[::-1])
        edges = bm.source_edges(source, print_scale.mm_to_px(bm.EDGE_SMOOTHING_MM))
        quality.update(bm.edge_alignment(page_data.region_id_map, edges, print_scale.mm_to_px(bm.EDGE_TOLERANCE_MM)))
        quality.update(bm.palette_separation(page_data.legend_bgr))
        # Line art is scored against the image's manifest entry: its flat colors, and its ink lines if it has ink colors.
        flat_colors, ink_colors = (
            np.array(getattr(image_info, name, ()), dtype=np.uint8).reshape(-1, 3)[:, ::-1]  # the manifest's are RGB
            for name in ("flat_colors", "ink_colors")
        )
        quality.update(bm.flat_color_match(flat_colors, page_data.legend_bgr))
        quality.update(dict.fromkeys(LINE_ART_KEYS))
        if len(flat_colors) and len(ink_colors):
            ink_width_px = print_scale.mm_to_px(bm.INK_MAX_WIDTH_MM)
            ink = bm.source_ink(source, flat_colors, ink_colors, ink_width_px)
            quality.update(bm.ink_line_match(page_data.strokes, ink, print_scale.mm_to_px(bm.INK_LINE_TOLERANCE_MM)))
            quality.update(bm.tube_regions(page_data.region_id_map, ink, ink_width_px))
        # Faces are scored inside the manifest's face boxes, and on whether their features survive on the page.
        quality.update(dict.fromkeys(FACE_KEYS))
        annotations = image_info.scaled_to(source.shape[1::-1]) if image_info else None
        faces = annotations.faces if annotations else ()
        if faces:
            quality.update(bm.face_fidelity(source, painted, [_xywh(face.box) for face in faces]))
            features = [feature for face in faces for feature in face.features]
            feature_boxes = [_xywh(feature.box) for feature in features]
            scores = bm.feature_survival(
                feature_boxes,
                page_data.region_id_map,
                {region.region_id for region in page_data.regions},
                page_data.strokes,
                edges,
                print_scale.mm_to_px(bm.EDGE_TOLERANCE_MM),
            )
            quality.update(bm.lost_features(scores))
            quality["labels_on_features"] = bm.labels_on_boxes(page_data.label_boxes, feature_boxes) if features else None
            face_features = [{"part": feature.part, **score} for feature, score in zip(features, scores)]
        # Text is read by OCR inside the manifest's text boxes, on the source, the page and the painting.
        blocks = annotations.text if annotations else ()
        reader = bm.text_reader() if blocks else None
        ocr = reader.name if reader else None
        layers = {"source": source, "page": np.ascontiguousarray(page_rgb[:, :, ::-1]), "painting": painted}
        text_quality, text_blocks = text_scores(blocks, layers, page_data.label_boxes, reader)
        quality.update(text_quality)
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
        "source_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
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
        "face_features": face_features,  # each annotated feature's part and feature_survival score
        "ocr": ocr,  # the OCR engine and version that read the text, None without text or without the engine
        "text_blocks": text_blocks,  # each annotated text block's string, and what OCR read on each layer
    }
    (args.out / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return 0


def text_scores(blocks, layers: dict, label_boxes, reader) -> tuple[dict, list | None]:
    """The text fields of a case's quality, and what OCR read in each block.

    ``blocks`` are the manifest's text blocks at output size, ``layers`` the
    images to read them on (named as the ``text_cer_`` fields), and ``reader``
    the OCR engine (``bench_metrics.text_reader``). Without blocks every field is
    None; without a reader only ``labels_on_text`` is scored, and no block's
    reading is returned.
    """
    import bench_metrics as bm  # imported late in this module, after the measured version's package

    quality = dict.fromkeys(TEXT_KEYS)
    if not blocks:
        return quality, None
    boxes = [_xywh(block.box) for block in blocks]
    quality["labels_on_text"] = bm.labels_on_boxes(label_boxes, boxes)
    if reader is None:
        return quality, None
    scores = bm.text_legibility(layers, [(box, block.string, block.rotation) for box, block in zip(boxes, blocks)], reader)
    quality.update({f"text_cer_{name}": cer for name, cer in scores["cer"].items()})
    return quality, scores["blocks"]


def _xywh(box) -> tuple[int, int, int, int]:
    """A manifest box as (x, y, width, height)."""
    return (box.x, box.y, box.w, box.h)


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

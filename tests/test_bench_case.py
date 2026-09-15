"""The benchmark case runner scores the pipeline's analysis payload, or what its stage probe captures for older versions."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import bench_case  # noqa: E402
import bench_metrics as bm  # noqa: E402
from tessellatum.core import difficulty, pipeline, render  # noqa: E402


def test_probe_fallback_reads_the_same_page_data_as_the_analysis(speckled_image_bgr, monkeypatch):
    for name in bench_case.PROBED_STAGES:
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name))  # undoes the probe's wrapping afterwards
    probe = bench_case.Probe(pipeline)
    params = difficulty.DifficultyParams(num_colors=3, min_region_fraction=0.01, blur_sigma=0.0)

    result = pipeline.generate(speckled_image_bgr, params, long_edge=80, collect_analysis=True)
    from_analysis = bench_case.page_data_from_analysis(result.analysis)
    from_probe = bench_case.page_data_from_probe(probe.captured, params, result.page.size, render)

    assert (from_analysis.source, from_probe.source) == ("analysis", "probe")
    # The analysis numbers colors legend first, the probe sees quantizer order; the painting is the same.
    assert not np.array_equal(from_analysis.region_color, from_probe.region_color)
    np.testing.assert_array_equal(from_analysis.region_id_map, from_probe.region_id_map)
    np.testing.assert_array_equal(
        bm.paint(from_analysis.region_id_map, from_analysis.region_color, from_analysis.palette_bgr),
        bm.paint(from_probe.region_id_map, from_probe.region_color, from_probe.palette_bgr),
    )
    assert from_analysis.min_region_area_px == from_probe.min_region_area_px
    assert from_analysis.labeled_region_ids == from_probe.labeled_region_ids != set()
    assert from_analysis.label_font_sizes_px == from_probe.label_font_sizes_px
    assert [r.region_id for r in from_analysis.regions] == [r.region_id for r in from_probe.regions]
    assert len(from_analysis.strokes) == len(from_probe.strokes) == len(from_analysis.regions)
    for from_payload, rebuilt in zip(from_analysis.strokes, from_probe.strokes):
        np.testing.assert_array_equal(from_payload, rebuilt)


def test_probe_fallback_rebuilds_font_sizes_without_the_render_module():
    dot = np.array([[[2, 3]]], dtype=np.int32)
    regions = [
        SimpleNamespace(region_id=i, interior_radius=radius, contour=dot) for i, radius in enumerate([5.0, 9.0, 30.0, 60.0])
    ]
    captured = {
        "quantize": ((), {}, (None, np.zeros((2, 3), dtype=np.uint8))),
        "build_regions": ((), {}, (np.zeros((4, 4), dtype=np.int32), np.zeros(4, dtype=np.int32))),
        "render_page": (((4, 4), regions), {}, None),
    }
    params = difficulty.DifficultyParams(num_colors=2, min_region_fraction=0.5, blur_sigma=0.0)

    page_data = bench_case.page_data_from_probe(captured, params, (4, 4), render_module=None)

    # Clearance of at least 9 px gets a number, at 0.85 x the clearance, between 10 and 40 px.
    assert page_data.labeled_region_ids == {1, 2, 3}
    assert page_data.label_font_sizes_px == [10, 25, 40]


def test_case_runner_scores_the_current_pipeline_from_its_analysis(tmp_path, sample_image_bgr):
    image = tmp_path / "blocks.png"
    Image.fromarray(np.ascontiguousarray(sample_image_bgr[:, :, ::-1])).save(image)
    out = tmp_path / "case"

    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "benchmarks" / "bench_case.py"),
            "--src", str(REPO_ROOT / "src"),
            "--image", str(image),
            "--preset", "Hard",
            "--long-edge", "200",
            "--repeats", "2",
            "--warmup", "0",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )  # fmt: skip

    assert proc.returncode == 0, proc.stderr
    case = json.loads((out / "case.json").read_text(encoding="utf-8"))
    assert case["scored_from"] == "analysis"
    assert len(case["runs"]) == 2  # the analysis run isn't timed...
    assert case["deterministic"]  # ...but its page must match the timed runs'
    assert {
        "de00_mean",
        "ssim",
        "undersized_regions",
        "labeled_area_fraction",
        "unlabeled_regions",
        "unlabeled_area_fraction",
        "sliver_area_fraction",
        "small_label_fraction",
        "min_label_pt",
        "compactness_median",
        "compactness_p10",
        "lines_per_boundary",
        "doubled_boundary_fraction",
        "undrawn_boundary_fraction",
        "same_color_boundary_fraction",
        "jaggedness",
        "edge_precision",
        "edge_recall",
        "edge_f1",
    } <= case["quality"].keys()
    assert (out / "painted.png").is_file() and (out / "regions.npz").is_file()

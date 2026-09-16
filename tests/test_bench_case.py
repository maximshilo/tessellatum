"""The benchmark case runner scores the pipeline's analysis payload, or what its stage probe captures for older versions."""

import importlib.util
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
    assert from_analysis.label_boxes == from_probe.label_boxes
    assert [r.region_id for r in from_analysis.regions] == [r.region_id for r in from_probe.regions]
    assert len(from_analysis.strokes) == len(from_probe.strokes) == len(from_analysis.regions)
    for from_payload, rebuilt in zip(from_analysis.strokes, from_probe.strokes):
        np.testing.assert_array_equal(from_payload, rebuilt)
    # Both read the legend's colors in legend order, without the specks' color, which no drawn region has.
    np.testing.assert_array_equal(from_analysis.legend_bgr, from_probe.legend_bgr)
    assert len(from_probe.legend_bgr) == result.num_colors_used < len(from_probe.palette_bgr)
    assert [tuple(color) for color in from_probe.legend_bgr[:, ::-1].tolist()] == result.palette_rgb


def test_probe_fallback_rebuilds_font_sizes_without_the_render_module():
    dot = np.array([[[2, 3]]], dtype=np.int32)
    regions = [
        SimpleNamespace(region_id=i, interior_radius=radius, contour=dot, color_index=0, interior_point=point)
        for i, (radius, point) in enumerate([(5.0, (100, 100)), (9.0, (100, 100)), (30.0, (100, 100)), (60.0, (199, 199))])
    ]
    captured = {
        "quantize": ((), {}, (None, np.zeros((2, 3), dtype=np.uint8))),
        "build_regions": ((), {}, (np.zeros((4, 4), dtype=np.int32), np.zeros(4, dtype=np.int32))),
        "render_page": (((4, 4), regions), {}, None),
    }
    params = difficulty.DifficultyParams(num_colors=2, min_region_fraction=0.5, blur_sigma=0.0)

    page_data = bench_case.page_data_from_probe(captured, params, (200, 200), render_module=None)

    # Clearance of at least 9 px gets a number, at 0.85 x the clearance, between 10 and 40 px.
    assert page_data.labeled_region_ids == {1, 2, 3}
    assert page_data.label_font_sizes_px == [10, 25, 40]
    # Each number, "1", is centered on its label point, but kept on the page.
    small, medium, large = page_data.label_boxes
    assert [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in (small, medium)] == [(100, 100), (100, 100)]
    assert small[3] - small[1] < medium[3] - medium[1]
    assert large[2:] == (200, 200)


def _drawing(tmp_path: Path) -> Path:
    """Line art with a face and a text block in its manifest: a fill inside a black outline 2 px wide.

    The outline is narrower than the widest ink line at this size (5.3 px),
    and also narrower than the brush the region stage paints with (3.2 px),
    so the page prints no shape of its own for it. The eye is a black square
    wide enough to paint, which stays a region.
    """
    drawing = np.full((200, 200, 3), 255, dtype=np.uint8)
    drawing[40:160, 40:160] = 0
    drawing[42:158, 42:158] = (230, 150, 90)
    drawing[70:84, 70:84] = 0
    image = tmp_path / "drawing.png"
    Image.fromarray(drawing).save(image)
    manifest = {
        "size": [200, 200],
        "categories": ["cartoon", "face", "text"],
        "faces": [{"kind": "cartoon", "box": [30, 30, 140, 140], "features": [{"part": "eye", "box": [66, 66, 22, 22]}]}],
        "text": [{"box": [60, 90, 80, 20], "string": "INK"}],
        "flat_colors": ["#ffffff", "#e6965a"],
        "ink_colors": ["#000000"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": 1, "images": {"drawing.png": manifest}}), encoding="utf-8")
    return image


def _case_arguments(image: Path, out: Path, repeats: int) -> list[str]:
    return [
        "--src", str(REPO_ROOT / "src"),
        "--image", str(image),
        "--preset", "Hard",
        "--long-edge", "200",
        "--repeats", str(repeats),
        "--warmup", "0",
        "--out", str(out),
    ]  # fmt: skip


def test_case_runner_scores_the_current_pipeline_from_its_analysis(tmp_path):
    image = _drawing(tmp_path)
    out = tmp_path / "case"

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "bench_case.py"), *_case_arguments(image, out, repeats=2)],
        capture_output=True,
        text=True,
        timeout=300,
    )

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
        "palette_min_de00",
        "palette_close_pairs",
        "ink_line_precision",
        "ink_line_recall",
        "ink_line_f1",
        "tube_regions",
        "tube_ink_fraction",
        "flat_color_de00_mean",
        "flat_color_de00_max",
        "face_de00_mean",
        "face_ssim",
        "features_lost",
        "feature_edge_recall",
        "labels_on_features",
        "text_cer_source",
        "text_cer_page",
        "text_cer_painting",
        "labels_on_text",
    } <= case["quality"].keys()
    # Scored against the drawing's manifest entry: the outline is ink, and the legend has the fill's color.
    assert case["quality"]["ink_line_f1"] is not None and case["quality"]["tube_ink_fraction"] is not None
    assert case["quality"]["flat_color_de00_mean"] < 1.0
    # The "eye" is the black square, which fills enough of its box to count as still on the page.
    assert case["quality"]["face_de00_mean"] is not None and case["quality"]["labels_on_features"] is not None
    assert case["quality"]["features_lost"] == 0
    assert [feature["part"] for feature in case["face_features"]] == ["eye"]
    # The text box holds no text; with the OCR package installed, it is read on the source, the page and the painting.
    assert isinstance(case["quality"]["labels_on_text"], int)
    if importlib.util.find_spec("rapidocr"):
        assert case["ocr"].startswith("rapidocr ")
        assert [(block["string"], sorted(block["read"])) for block in case["text_blocks"]] == [("INK", ["page", "painting", "source"])]
        assert case["quality"]["text_cer_source"] is not None
    assert (out / "painted.png").is_file() and (out / "regions.npz").is_file()


# Runs bench_case.py as its own script would, with the OCR package made impossible to import.
WITHOUT_OCR = (
    "import runpy, sys; from pathlib import Path; script = sys.argv[1]; sys.argv = sys.argv[1:]; "
    "sys.path.insert(0, str(Path(script).parent)); sys.modules['rapidocr'] = None; runpy.run_path(script, run_name='__main__')"
)


def test_case_runner_scores_only_labels_on_text_without_the_ocr_engine(tmp_path):
    image = _drawing(tmp_path)
    out = tmp_path / "case"

    proc = subprocess.run(
        [sys.executable, "-c", WITHOUT_OCR, str(REPO_ROOT / "benchmarks" / "bench_case.py"), *_case_arguments(image, out, repeats=1)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, proc.stderr
    case = json.loads((out / "case.json").read_text(encoding="utf-8"))
    assert case["ocr"] is None and case["text_blocks"] is None
    assert [case["quality"][f"text_cer_{layer}"] for layer in ("source", "page", "painting")] == [None, None, None]
    assert isinstance(case["quality"]["labels_on_text"], int)
    assert case["quality"]["face_de00_mean"] is not None  # everything else is still scored


def test_text_scores_without_text_blocks_are_blank():
    quality, blocks = bench_case.text_scores((), {}, [(0, 0, 10, 10)], reader=None)

    assert quality == dict.fromkeys(bench_case.TEXT_KEYS) and blocks is None

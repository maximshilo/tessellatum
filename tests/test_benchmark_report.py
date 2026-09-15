"""The benchmark comparison report, rendered from small synthetic result sets."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_report  # noqa: E402

PAINTABILITY_KEYS = ("unlabeled_regions", "sliver_area_fraction", "small_label_fraction", "compactness_p10", "compactness_median")
LINE_KEYS = ("lines_per_boundary", "same_color_boundary_fraction", "jaggedness", "edge_f1")
PALETTE_KEYS = ("palette_min_de00", "palette_close_pairs")
LINE_ART_KEYS = ("ink_line_f1", "tube_regions", "tube_ink_fraction", "flat_color_de00_mean")
FACE_KEYS = ("face_de00_mean", "face_ssim", "features_lost", "labels_on_features")


def _case(image: str, categories: list[str], de00: float) -> dict:
    return {
        "case": f"{Path(image).stem}__Easy__1100",
        "image": image,
        "categories": categories,
        "preset": "Easy",
        "long_edge": 1100,
        "status": "ok",
        "settings": {"repeats": 1, "warmup": 0, "timeout_s": 60.0},
        "import_s": 0.1,
        "first_run_s": 0.3,
        "median_total_s": 0.2,
        "median_stages_s": {"quantize": 0.1, "other": 0.1},
        "deterministic": True,
        "quality": {
            "de00_mean": de00,
            "de00_p95": 10.0,
            "ssim": 0.8,
            "regions": 100,
            "labeled_area_fraction": 0.5,
            "unlabeled_regions": 3,
            "sliver_area_fraction": 0.1,
            "small_label_fraction": 0.0,
            "compactness_p10": 0.1,
            "compactness_median": 0.4,
            "lines_per_boundary": 2.0,
            "same_color_boundary_fraction": 0.01,
            "jaggedness": 1.1,
            "edge_f1": 0.5,
            "palette_min_de00": 4.5,
            "palette_close_pairs": 2,
            "ink_line_f1": 0.25,
            "tube_regions": 2,
            "tube_ink_fraction": 0.6,
            "flat_color_de00_mean": 3.5,
            "face_de00_mean": 7.5,
            "face_ssim": 0.6,
            "features_lost": 1,
            "labels_on_features": 2,
            "undersized_regions": 0,
            "ink_fraction": 0.1,
        },
    }


def _write_set(path: Path, cases: list[dict]) -> Path:
    path.mkdir(parents=True)
    data = {
        "schema": 1,
        "label": path.name,
        "target": {"ref": "WORKTREE", "commit": "abc1234", "dirty": False},
        "version": "0.0.0",
        "machine": {"system": "Test", "release": "1", "machine": "x86", "cpu_count": 1, "python": "3"},
        "settings": {"threads": None},
        "cases": cases,
    }
    (path / "results.json").write_text(json.dumps(data), encoding="utf-8")
    return path


def _sets(tmp_path: Path) -> list[Path]:
    # Run order puts the cartoon first; the report should still list photos first.
    ref = _write_set(
        tmp_path / "ref",
        [_case("cartoon.png", ["cartoon", "text"], 4.0), _case("lion.jpg", ["photo", "face"], 5.0), _case("old.png", [], 3.0)],
    )
    cand = _write_set(
        tmp_path / "cand",
        [_case("cartoon.png", ["cartoon", "text"], 4.0), _case("lion.jpg", ["photo", "face"], 6.0), _case("old.png", [], 3.0)],
    )
    return [ref, cand]


def test_quality_is_grouped_by_primary_category(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())
    quality = report[report.index("## Quality") : report.index("## Agreement")]

    headings = [line for line in quality.splitlines() if line.startswith("#### ")]
    assert headings == ["#### photo", "#### cartoon", "#### uncategorized"]
    assert "| lion / Easy / 1100 | 5.00 → 6.00 |" in quality.split("#### photo")[1].split("#### cartoon")[0]


def test_category_summary_counts_images_in_every_category(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())

    assert "| category | images | cases | ΔE00 mean ↓ | SSIM ↑ | labeled area ↑ | slivers ↓ | flagged cases |" in report
    assert "| face | 1 | 1 | 5.00 → 6.00 | 0.800 | 50.0% | 10.0% | 1 |" in report
    assert "| text | 1 | 1 | 4.00 | 0.800 | 50.0% | 10.0% | 0 |" in report
    assert "| uncategorized | 1 | 1 | 3.00 | 0.800 | 50.0% | 10.0% | 0 |" in report


def test_every_case_table_uses_the_category_order(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())
    speed = report[report.index("## Speed") : report.index("## Where the time goes")]

    assert speed.index("lion / Easy") < speed.index("cartoon / Easy") < speed.index("old / Easy")
    assert "quality regressions in 1 case(s)**: lion / Easy / 1100 (ΔE00 worse)" in report


def test_single_result_set_report_has_summary_without_flags(tmp_path):
    report = bench_report.build_report(_sets(tmp_path)[:1], bench_report.Tolerances())

    assert "| category | images | cases | ΔE00 mean ↓ | SSIM ↑ | labeled area ↑ | slivers ↓ |\n" in report
    assert "| photo | 1 | 1 | 5.00 | 0.800 | 50.0% | 10.0% |" in report
    assert "#### cartoon" in report
    assert "## Verdict" not in report


def test_columns_added_since_a_result_set_was_recorded_are_blank_for_it(tmp_path):
    before = _case("lion.jpg", ["photo"], 5.0)
    for key in PAINTABILITY_KEYS + LINE_KEYS + PALETTE_KEYS + LINE_ART_KEYS + FACE_KEYS:
        del before["quality"][key]
    old = _write_set(tmp_path / "old", [before])
    new = _write_set(tmp_path / "new", [_case("lion.jpg", ["photo"], 5.0)])

    old_alone = bench_report.build_report([old], bench_report.Tolerances())
    old_vs_new = bench_report.build_report([old, new], bench_report.Tolerances())

    header = (
        "| case | ΔE00 mean ↓ | ΔE00 p95 ↓ | SSIM ↑ | regions | labeled area ↑ | unlabeled ↓ | slivers ↓ | labels < 6 pt ↓ "
        "| compactness p10 ↑ | compactness median ↑ | lines per boundary | same-color boundary ↓ | jaggedness ↓ "
        "| edge F1 ↑ | palette min ΔE00 ↑ | color pairs < 10 ΔE00 ↓ | ink line F1 ↑ | tubes ↓ | ink in shapes < 5 mm ↓ "
        "| flat colors ΔE00 ↓ | face ΔE00 ↓ | face SSIM ↑ | features lost ↓ | labels on features ↓ | undersized ↓ | ink |"
    )
    assert header in old_alone
    assert (
        "| lion / Easy / 1100 | 5.00 | 10.0 | 0.800 | 100 | 50.0% | – | – | – | – | – | – | – | – | – | – | – | – | – | – "
        "| – | – | – | – | – | 0 | 10.0% |"
    ) in old_alone
    assert (
        "| lion / Easy / 1100 | 5.00 | 10.0 | 0.800 | 100 | 50.0% | 3 | 10.0% | 0.0% | 0.10 | 0.40 | 2.00 | 1.0% | 1.100 "
        "| 0.50 | 4.5 | 2 | 0.25 | 2 | 60.0% | 3.50 | 7.50 | 0.600 | 1 | 2 | 0 | 10.0% | ok |"
    ) in old_vs_new

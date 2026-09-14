"""The benchmark comparison report, rendered from small synthetic result sets."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_report  # noqa: E402


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

    assert "| category | images | cases | ΔE00 mean ↓ | SSIM ↑ | labeled area ↑ | flagged cases |" in report
    assert "| face | 1 | 1 | 5.00 → 6.00 | 0.800 | 50.0% | 1 |" in report
    assert "| text | 1 | 1 | 4.00 | 0.800 | 50.0% | 0 |" in report
    assert "| uncategorized | 1 | 1 | 3.00 | 0.800 | 50.0% | 0 |" in report


def test_every_case_table_uses_the_category_order(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())
    speed = report[report.index("## Speed") : report.index("## Where the time goes")]

    assert speed.index("lion / Easy") < speed.index("cartoon / Easy") < speed.index("old / Easy")
    assert "quality regressions in 1 case(s)**: lion / Easy / 1100 (ΔE00 worse)" in report


def test_single_result_set_report_has_summary_without_flags(tmp_path):
    report = bench_report.build_report(_sets(tmp_path)[:1], bench_report.Tolerances())

    assert "| category | images | cases | ΔE00 mean ↓ | SSIM ↑ | labeled area ↑ |\n" in report
    assert "| photo | 1 | 1 | 5.00 | 0.800 | 50.0% |" in report
    assert "#### cartoon" in report
    assert "## Verdict" not in report

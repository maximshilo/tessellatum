"""The benchmark comparison report, rendered from small synthetic result sets."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_report  # noqa: E402

PAINTABILITY_KEYS = ("unlabeled_regions", "sliver_area_fraction", "small_label_fraction", "compactness_p10", "compactness_median")
LINE_KEYS = ("lines_per_boundary", "same_color_boundary_fraction", "jaggedness", "edge_f1")
PALETTE_KEYS = ("palette_min_de00", "palette_close_pairs")
LINE_ART_KEYS = ("ink_line_f1", "tube_regions", "tube_ink_fraction", "flat_color_de00_mean")
FACE_KEYS = ("face_de00_mean", "face_ssim", "features_lost", "labels_on_features")
TEXT_KEYS = ("text_cer_source", "text_cer_page", "text_cer_painting", "labels_on_text")
METRICS = bench_report.METRICS_BY_KEY


def _case(image: str, categories: list[str], de00: float, preset: str = "Easy", long_edge: int = 1100, **quality) -> dict:
    case = {
        "case": f"{Path(image).stem}__{preset}__{long_edge}",
        "image": image,
        "categories": categories,
        "preset": preset,
        "long_edge": long_edge,
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
            "text_cer_source": 0.1,
            "text_cer_page": 0.95,
            "text_cer_painting": 0.98,
            "labels_on_text": 1,
            "undersized_regions": 0,
            "ink_fraction": 0.1,
        },
    }
    case["quality"].update(quality)
    return case


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
        [
            _case("cartoon.png", ["cartoon", "text"], 4.0),
            _case("lion.jpg", ["photo", "face"], 5.0, features_lost=0),
            _case("old.png", [], 3.0),
        ],
    )
    cand = _write_set(
        tmp_path / "cand",
        [
            _case("cartoon.png", ["cartoon", "text"], 4.0, sliver_area_fraction=0.005),
            _case("lion.jpg", ["photo", "face"], 8.0),
            _case("old.png", [], 3.0),
        ],
    )
    return [ref, cand]


def _section(report: str, start: str, end: str | None = None) -> str:
    return report[report.index(start) : report.index(end) if end else len(report)]


def test_report_renders_every_section(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())

    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## Speed",
        "## Where the time goes",
        "## Scorecard",
        "## Quality",
        "## Agreement with the reference",
        "## Verdict",
    ]
    scorecard = _section(report, "## Scorecard", "## Quality")
    assert "### cand vs ref" in scorecard
    assert [line for line in scorecard.splitlines() if line.startswith("#### ")] == [
        "#### resembles",
        "#### paintable",
        "#### clean drawing",
        "#### palette",
    ]


def test_quality_is_grouped_by_primary_category(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())
    quality = _section(report, "## Quality", "## Agreement")

    headings = [line for line in quality.splitlines() if line.startswith("#### ")]
    assert headings == ["#### photo", "#### cartoon", "#### uncategorized"]
    assert "| lion / Easy / 1100 | 5.00 → 8.00 |" in quality.split("#### photo")[1].split("#### cartoon")[0]


def test_scorecard_scores_each_job_per_category_and_over_all_cases(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances(sigma={"de00_mean": 0.05}))
    scorecard = _section(report, "## Scorecard", "## Quality")

    # Categories in manifest order, an image in every category it has, then all cases.
    assert "| category | images | cases | resembles | paintable | clean drawing | palette |" in scorecard
    rows = [line.split(" |")[0] for line in scorecard.split("#### resembles")[0].splitlines() if line.startswith("| ")]
    assert rows == ["| category", "| photo", "| cartoon", "| face", "| text", "| uncategorized", "| all"]
    assert "| photo | 1 | 1 | 1/1 → 0/1 met; **ΔE00 mean worse** | 0/1 met | 0/1 met | 0/1 met |" in scorecard
    assert "| cartoon | 1 | 1 | 0/1 met | 0/1 met | 0/1 met | 0/1 met |" in scorecard
    assert "| all | 3 | 3 | 1/3 → 0/3 met; **ΔE00 mean worse** | 0/3 met | 0/3 met | 0/3 met |" in scorecard

    # Each job's metrics, with their targets; a regressed mean in bold.
    resembles = _section(scorecard, "#### resembles", "#### paintable")
    assert (
        "| category | cases | ΔE00 mean ↓ | ΔE00 p95 ↓ | SSIM ↑ | face ΔE00 ↓ | face SSIM ↑ | features lost ↓ (0) "
        "| text CER painting ↓ | targets met |"
    ) in resembles
    assert "| photo | 1 | **5.00 → 8.00** | 10.0 | 0.800 | 7.50 | 0.600 | 0.0 → 1.0 | 0.98 | 1/1 → 0/1 |" in resembles
    assert "| all | 3 | **4.00 → 5.00** | 10.0 | 0.800 | 7.50 | 0.600 | 0.7 → 1.0 | 0.98 | 1/3 → 0/3 |" in resembles
    paintable = _section(scorecard, "#### paintable", "#### clean drawing")
    assert (
        "| category | cases | labeled area ↑ | unlabeled ↓ (0) | slivers ↓ (≤ 1%) | labels < 6 pt ↓ (0) | compactness p10 ↑ "
        "| compactness median ↑ | undersized ↓ | targets met |"
    ) in paintable
    assert "| cartoon | 1 | 50.0% | 3.0 | 10.0% → 0.5% | 0.0% | 0.10 | 0.40 | 0.0 | 0/1 |" in paintable
    drawing = _section(scorecard, "#### clean drawing", "#### palette")
    assert (
        "| category | cases | lines per boundary (1 ± 0.05) | same-color boundary ↓ (0) | jaggedness ↓ (≤ 1.02) | edge F1 ↑ "
        "| ink line F1 ↑ (≥ 0.9) | tubes ↓ (0) | ink in shapes < 5 mm ↓ | labels on features ↓ "
        "| text CER page ↓ (≤ text CER source + 0.1) | labels on text ↓ (0) | targets met |"
    ) in drawing
    palette = _section(scorecard, "#### palette")
    assert "| category | cases | palette min ΔE00 ↑ (≥ 10) | color pairs < 10 ΔE00 ↓ | flat colors ΔE00 ↓ | targets met |" in palette


def test_verdict_lists_regressions_target_misses_and_cases_to_look_at_separately(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances(sigma={"de00_mean": 0.05}))
    verdict = _section(report, "## Verdict")

    # +60% on one photo of three: over n cases the mean may rise 3 × 5% / √n, i.e. 15% alone and 8.7% over all three.
    assert "- **cand**: 1.00× geometric-mean speedup; **quality regressions on 1 metric(s)**." in verdict
    assert (
        "  - **Regressions:** ΔE00 mean worse in photo (by 60.0%, tolerance 15.0%, n = 1), "
        "face (by 60.0%, tolerance 15.0%, n = 1), all (by 20.0%, tolerance 8.7%, n = 3)."
    ) in verdict
    # Losing the lion's feature is a new target miss, but not a regression: 1 feature is within 3 × 0.35.
    assert "slivers (≤ 1%) 3/3 → 2/3" in verdict
    assert "features lost (0) 2/3 → 3/3" in verdict
    assert "labels < 6 pt (0) 0/3;" in verdict
    assert "text CER page (≤ text CER source + 0.1) 3/3" in verdict
    assert "  - **New target misses in 1 case(s):** lion / Easy / 1100 (features lost)." in verdict
    assert "  - Targets newly met: 1 (case, target) pair(s)." in verdict
    assert "  - Cases to look at, worse alone by more than 3 tolerances: lion / Easy / 1100 (ΔE00 mean)." in verdict
    assert "Tolerances overridden: ΔE00 mean σ = 0.05." in verdict

    quality = _section(report, "## Quality", "## Agreement")
    assert quality.count("| ΔE00 mean worse |") == 1


def test_every_case_table_uses_the_category_order(tmp_path):
    report = bench_report.build_report(_sets(tmp_path), bench_report.Tolerances())
    speed = _section(report, "## Speed", "## Where the time goes")

    assert speed.index("lion / Easy") < speed.index("cartoon / Easy") < speed.index("old / Easy")


def test_single_result_set_report_has_scorecard_without_verdict(tmp_path):
    lion = _case("lion.jpg", ["photo"], 5.0, unlabeled_regions=0, sliver_area_fraction=0.005)
    cartoon = _case("cartoon.png", ["cartoon"], 4.0, features_lost=None)
    report = bench_report.build_report([_write_set(tmp_path / "one", [lion, cartoon])], bench_report.Tolerances())

    assert "### one\n" in report
    # A job's targets are met when all of them that have a value are; the cartoon has no face to lose features on.
    assert "| photo | 1 | 1 | 0/1 met | 1/1 met | 0/1 met | 0/1 met |" in report
    assert "| cartoon | 1 | 1 | no targets | 0/1 met | 0/1 met | 0/1 met |" in report
    assert "| all | 2 | 2 | 0/1 met | 1/2 met | 0/2 met | 0/2 met |" in report
    assert "| lion / Easy / 1100 | 5.00 |" in report
    assert "worse**" not in report
    assert "flags" not in _section(report, "## Quality").splitlines()[-1]
    assert "## Verdict" not in report


def test_a_group_regresses_when_its_mean_change_is_worse_than_three_standard_errors():
    tol = bench_report.Tolerances(sigma={"jaggedness": 0.1, "de00_mean": 0.05, "ssim": 0.1, "lines_per_boundary": 0.1})

    def worse(key, pairs, tolerances=tol):
        found = bench_report.regressions([METRICS[key]], pairs, "group", tolerances)
        return [(r.cases, round(r.change, 9), round(r.allowed, 9)) for r in found]

    def pair(key, before, after):
        return ({key: before}, {key: after})

    # Over 4 cases the mean may get worse by 3 × 0.1 / √4 = 0.15; one case alone by 0.3.
    assert worse("jaggedness", [pair("jaggedness", 1.0, 1.16)] * 4) == [(4, 0.16, 0.15)]
    assert worse("jaggedness", [pair("jaggedness", 1.0, 1.14)] * 4) == []
    assert worse("jaggedness", [pair("jaggedness", 1.0, 1.25)]) == []
    assert worse("jaggedness", [pair("jaggedness", 1.0, 1.35)]) == [(1, 0.35, 0.3)]
    # Better cases offset worse ones, and only cases with a value in both count: mean 0.1 over n = 2, 0.212 allowed.
    mixed = [pair("jaggedness", 1.0, 1.4), pair("jaggedness", 1.2, 1.0), pair("jaggedness", None, 2.0), ({}, {"jaggedness": 2.0})]
    assert worse("jaggedness", mixed) == []
    assert worse("jaggedness", mixed[:1] * 2 + mixed[2:]) == [(2, 0.4, 0.212132034)]
    # Higher is better: a drop is worse.
    assert worse("ssim", [pair("ssim", 0.9, 0.5)]) == [(1, 0.4, 0.3)]
    assert worse("ssim", [pair("ssim", 0.5, 0.9)]) == []
    # A relative metric changes by a share of the reference's value.
    assert worse("de00_mean", [pair("de00_mean", 10.0, 11.0)]) == []
    assert worse("de00_mean", [pair("de00_mean", 10.0, 12.0)]) == [(1, 0.2, 0.15)]
    # Lines per boundary are judged by their distance from 1.
    assert worse("lines_per_boundary", [pair("lines_per_boundary", 1.0, 0.6)]) == [(1, 0.4, 0.3)]
    assert worse("lines_per_boundary", [pair("lines_per_boundary", 2.0, 1.2)]) == []
    # Undersized regions may not rise at all; fields that only inform are never judged.
    default = bench_report.Tolerances()
    assert worse("undersized_regions", [pair("undersized_regions", 0, 1)], default) == [(1, 1, 0.0)]
    assert worse("undersized_regions", [pair("undersized_regions", 2, 2)], default) == []
    assert worse("regions", [pair("regions", 10, 1000)], default) == []


def test_a_case_misses_a_target_on_its_worse_side_where_it_has_a_value():
    def miss(key, **quality):
        return bench_report.misses_target(METRICS[key], quality)

    assert miss("sliver_area_fraction", sliver_area_fraction=0.01) is False
    assert miss("sliver_area_fraction", sliver_area_fraction=0.0101) is True
    assert miss("palette_min_de00", palette_min_de00=10.0) is False
    assert miss("palette_min_de00", palette_min_de00=9.99) is True
    assert miss("jaggedness", jaggedness=1.02) is False
    assert miss("jaggedness", jaggedness=1.021) is True
    assert miss("unlabeled_regions", unlabeled_regions=0) is False
    assert miss("unlabeled_regions", unlabeled_regions=1) is True
    # One line per boundary: too few lines miss as much as too many.
    assert [miss("lines_per_boundary", lines_per_boundary=v) for v in (0.94, 0.96, 1.04, 1.06)] == [True, False, False, True]
    # The page's text counts from what OCR reads on the source.
    assert miss("text_cer_page", text_cer_page=0.3, text_cer_source=0.25) is False
    assert miss("text_cer_page", text_cer_page=0.36, text_cer_source=0.25) is True
    assert miss("text_cer_page", text_cer_page=0.3, text_cer_source=None) is None
    # No value, or no target.
    assert miss("tube_regions", tube_regions=None) is None
    assert miss("tube_regions") is None
    assert miss("edge_f1", edge_f1=0.1) is None


def test_tolerance_overrides_take_judged_metrics_by_key():
    assert bench_report.parse_sigma_overrides(["jaggedness=0.02", " de00_mean=0.1"]) == {"jaggedness": 0.02, "de00_mean": 0.1}
    for bad in ("jaggedness", "regions=3", "nonsense=1", "jaggedness=abc", "jaggedness=-1", "jaggedness=nan"):
        with pytest.raises(ValueError):
            bench_report.parse_sigma_overrides([bad])


def test_noise_sigma_is_the_root_mean_square_change_between_sizes(tmp_path):
    base = _write_set(tmp_path / "base", [_case("lion.jpg", ["photo"], 10.0, jaggedness=1.10), _case("cat.jpg", ["photo"], 5.0)])
    sizes = _write_set(
        tmp_path / "sizes",
        [
            _case("lion.jpg", ["photo"], 11.0, long_edge=1099, jaggedness=1.13),
            _case("lion.jpg", ["photo"], 9.0, long_edge=1101, jaggedness=1.06),
            _case("lion.jpg", ["photo"], 20.0, preset="Hard", long_edge=1099, jaggedness=2.0),  # no other size at Hard
        ],
    )

    sigmas = bench_report.noise_sigmas([bench_report.ResultSet(base), bench_report.ResultSet(sizes)])

    # Pairs, smaller size first: 1099 → 1100, 1099 → 1101, 1100 → 1101.
    assert sigmas["jaggedness"] == (3, pytest.approx(math.sqrt((0.03**2 + 0.07**2 + 0.04**2) / 3)))
    assert sigmas["de00_mean"] == (3, pytest.approx(math.sqrt(((1 / 11) ** 2 + (2 / 11) ** 2 + 0.1**2) / 3)))
    assert sigmas["ssim"] == (3, 0.0)
    assert "regions" not in sigmas
    assert "| jaggedness | jaggedness | 3 | 0.05 | 0.0081 | no |" in bench_report.noise_report([base, sizes])


def test_columns_added_since_a_result_set_was_recorded_are_blank_for_it(tmp_path):
    before = _case("lion.jpg", ["photo"], 5.0)
    for key in PAINTABILITY_KEYS + LINE_KEYS + PALETTE_KEYS + LINE_ART_KEYS + FACE_KEYS + TEXT_KEYS:
        del before["quality"][key]
    old = _write_set(tmp_path / "old", [before])
    new = _write_set(tmp_path / "new", [_case("lion.jpg", ["photo"], 5.0)])

    old_alone = bench_report.build_report([old], bench_report.Tolerances())
    old_vs_new = bench_report.build_report([old, new], bench_report.Tolerances())

    header = (
        "| case | ΔE00 mean ↓ | ΔE00 p95 ↓ | SSIM ↑ | regions | labeled area ↑ | unlabeled ↓ | slivers ↓ | labels < 6 pt ↓ "
        "| compactness p10 ↑ | compactness median ↑ | lines per boundary | same-color boundary ↓ | jaggedness ↓ "
        "| edge F1 ↑ | palette min ΔE00 ↑ | color pairs < 10 ΔE00 ↓ | ink line F1 ↑ | tubes ↓ | ink in shapes < 5 mm ↓ "
        "| flat colors ΔE00 ↓ | face ΔE00 ↓ | face SSIM ↑ | features lost ↓ | labels on features ↓ | text CER source "
        "| text CER page ↓ | text CER painting ↓ | labels on text ↓ | undersized ↓ | ink | targets missed |"
    )
    assert header in old_alone
    assert (
        "| lion / Easy / 1100 | 5.00 | 10.0 | 0.800 | 100 | 50.0% | – | – | – | – | – | – | – | – | – | – | – | – | – | – "
        "| – | – | – | – | – | – | – | – | – | 0 | 10.0% | – |"
    ) in old_alone
    assert "| all | 1 | 1 | no targets | no targets | no targets | no targets |" in old_alone
    assert (
        "| lion / Easy / 1100 | 5.00 | 10.0 | 0.800 | 100 | 50.0% | 3 | 10.0% | 0.0% | 0.10 | 0.40 | 2.00 | 1.0% | 1.100 "
        "| 0.50 | 4.5 | 2 | 0.25 | 2 | 60.0% | 3.50 | 7.50 | 0.600 | 1 | 2 | 0.10 | 0.95 | 0.98 | 1 | 0 | 10.0% | – → 11/12 | ok |"
    ) in old_vs_new
    assert "| all | 1 | 1 | 0/1 met | 0/1 met | 0/1 met | 0/1 met |" in old_vs_new

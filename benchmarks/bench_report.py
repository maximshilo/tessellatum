"""Markdown comparison report for benchmark result sets (``bench.py compare``)."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import bench_manifest
import bench_metrics as bm

STAGE_ORDER = (
    "resize_to_long_edge",
    "quantize",
    "build_regions",
    "extract_regions",
    "render_page",
    "render_legend",
    "other",
)

# (quality key, column title, format)
QUALITY_COLUMNS = (
    ("de00_mean", "ΔE00 mean ↓", "{:.2f}"),
    ("de00_p95", "ΔE00 p95 ↓", "{:.1f}"),
    ("ssim", "SSIM ↑", "{:.3f}"),
    ("regions", "regions", "{:d}"),
    ("labeled_area_fraction", "labeled area ↑", "{:.1%}"),
    ("unlabeled_regions", "unlabeled ↓", "{:d}"),
    ("sliver_area_fraction", "slivers ↓", "{:.1%}"),
    ("small_label_fraction", f"labels < {bm.print_size.MIN_LABEL_SIZE_PT:g} pt ↓", "{:.1%}"),
    ("compactness_p10", "compactness p10 ↑", "{:.2f}"),
    ("compactness_median", "compactness median ↑", "{:.2f}"),
    ("undersized_regions", "undersized ↓", "{:d}"),
    ("ink_fraction", "ink", "{:.1%}"),
)

# Metrics averaged per image category.
SUMMARY_COLUMNS = (
    ("de00_mean", "ΔE00 mean ↓", "{:.2f}"),
    ("ssim", "SSIM ↑", "{:.3f}"),
    ("labeled_area_fraction", "labeled area ↑", "{:.1%}"),
    ("sliver_area_fraction", "slivers ↓", "{:.1%}"),
)


@dataclass(frozen=True)
class Tolerances:
    """How much worse than the reference a candidate may score before it's flagged."""

    de00_rel: float = 0.03  # mean CIEDE2000 (source vs painting) may rise by 3%
    ssim_abs: float = 0.01  # SSIM (source vs painting) may drop by 0.01
    labeled_area_abs: float = 0.02  # share of page area carrying a number may drop 2 points
    region_drift_rel: float = 0.15  # region-count change worth a note (not a failure)


class ResultSet:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = json.loads((path / "results.json").read_text(encoding="utf-8"))
        self.label: str = self.data["label"]
        self.cases: dict[str, dict] = {c["case"]: c for c in self.data["cases"]}

    def ok_case(self, case_id: str) -> dict | None:
        case = self.cases.get(case_id)
        return case if case is not None and case["status"] == "ok" else None

    def case_dir(self, case_id: str) -> Path:
        return self.path / "cases" / case_id


def build_report(paths: list[Path], tol: Tolerances, detail: bool = False) -> str:
    sets = [ResultSet(p) for p in paths]
    case_ids = list(dict.fromkeys(cid for s in sets for cid in s.cases))
    categories = _case_categories(sets, case_ids)
    # Every table lists cases grouped by their image's primary category (stable sort keeps the run order within one).
    case_ids.sort(key=lambda cid: _category_key(categories[cid][0]))
    lines = ["# Tessellatum benchmark comparison", ""]
    lines += _header(sets)
    lines += _speed_section(sets, case_ids)
    lines += _stage_section(sets, case_ids, detail)
    lines += _quality_section(sets, case_ids, categories, tol)
    if len(sets) > 1:
        lines += _agreement_section(sets, case_ids)
        lines += _verdict_section(sets, case_ids, tol)
    return "\n".join(lines).rstrip() + "\n"


def _header(sets: list[ResultSet]) -> list[str]:
    lines = []
    for i, s in enumerate(sets):
        target, settings = s.data["target"], s.data["settings"]
        role = "reference" if i == 0 else "candidate"
        dirty = " + uncommitted changes" if target.get("dirty") else ""
        threads = f", threads capped at {settings['threads']}" if settings.get("threads") else ""
        case_settings = [c["settings"] for c in s.cases.values() if "settings" in c]
        repeats = _range_text([cs["repeats"] for cs in case_settings])
        warmups = _range_text([cs["warmup"] for cs in case_settings])
        lines.append(
            f"- **{s.label}** ({role}): tessellatum {s.data['version']}, `{target['ref']}` @ {target['commit']}{dirty}; "
            f"median of {repeats} run(s) after {warmups} warm-up run(s){threads}"
        )
    machine = sets[0].data["machine"]
    lines.append(
        f"- Machine: {machine['system']} {machine['release']} {machine['machine']}, "
        f"{machine['cpu_count']} logical CPUs, Python {machine['python']}"
    )
    if any(s.data["machine"] != machine for s in sets[1:]):
        lines.append("- **Warning:** result sets come from different machines or library versions; timings may not be comparable.")
    return lines + [""]


def _speed_section(sets: list[ResultSet], case_ids: list[str]) -> list[str]:
    ref = sets[0]
    rows = []
    ratios: dict[str, list[float]] = {s.label: [] for s in sets[1:]}
    for cid in case_ids:
        row = [_case_title(cid)]
        for i, s in enumerate(sets):
            case = s.cases.get(cid)
            cell = _time_cell(case)
            if i > 0 and case is not None and case["status"] == "ok":
                ratio, lower_bound = _speedup(ref.cases.get(cid), case)
                if ratio is not None:
                    cell += f" ({'≥ ' if lower_bound else ''}{_fmt_ratio(ratio)})"
                    if not lower_bound:
                        ratios[s.label].append(ratio)
            row.append(cell)
        rows.append(row)

    lines = ["## Speed", "", "Median wall time of one `generate()` call (lower is better); speedup vs. the reference.", ""]
    lines += _table(["case", *(s.label for s in sets)], rows)
    lines.append("")
    for label, values in ratios.items():
        if values:
            lines.append(
                f"- **{label}**: geometric-mean speedup **{_fmt_ratio(_geomean(values))}** over {len(values)} "
                f"comparable case(s) (range {_fmt_ratio(min(values))} to {_fmt_ratio(max(values))})"
            )
    for s in sets:
        ok = [c for c in s.cases.values() if c["status"] == "ok"]
        if not ok:
            continue
        import_s = statistics.median(c["import_s"] for c in ok)
        overhead = statistics.median(c["first_run_s"] - c["median_total_s"] for c in ok)
        rss = [c["peak_rss_mb"] for c in ok if c.get("peak_rss_mb")]
        memory = f", peak RSS up to {max(rss):,.0f} MB" if rss else ""
        lines.append(f"- {s.label}: import {import_s:.2f} s, first-run overhead {overhead:+.2f} s (medians){memory}")
    return lines + [""]


def _stage_section(sets: list[ResultSet], case_ids: list[str], detail: bool) -> list[str]:
    common = [cid for cid in case_ids if all(s.ok_case(cid) for s in sets)]
    if not common:
        return []

    def stage_s(s: ResultSet, cid: str, stage: str) -> float:
        return s.ok_case(cid)["median_stages_s"].get(stage, 0.0)

    stages = [st for st in STAGE_ORDER if any(stage_s(s, cid, st) for s in sets for cid in common)]
    totals = {s.label: sum(stage_s(s, cid, st) for cid in common for st in stages) for s in sets}
    rows = []
    for st in stages:
        row = [st]
        for s in sets:
            value = sum(stage_s(s, cid, st) for cid in common)
            share = value / totals[s.label] if totals[s.label] else 0.0
            row.append(f"{_fmt_ms(value)} ({share:.0%})")
        rows.append(row)
    rows.append(["**total**", *(f"**{_fmt_ms(totals[s.label])}**" for s in sets)])

    lines = ["## Where the time goes", "", f"Sum of per-stage medians over the {len(common)} case(s) every set completed.", ""]
    lines += _table(["stage", *(s.label for s in sets)], rows) + [""]
    if detail:
        for cid in common:
            rows = [[st, *(_fmt_ms(stage_s(s, cid, st)) for s in sets)] for st in stages]
            lines += [f"### {_case_title(cid)}", ""] + _table(["stage", *(s.label for s in sets)], rows) + [""]
    return lines


def _quality_section(
    sets: list[ResultSet], case_ids: list[str], categories: dict[str, tuple[str, ...]], tol: Tolerances
) -> list[str]:
    lines = [
        "## Quality",
        "",
        "Scored on the *finished painting* (every region filled with its legend color) against the source image. "
        "**ΔE00**: CIEDE2000 color error, mean and 95th percentile. **SSIM**: structural similarity of luma. "
        "**labeled area**: share of the page inside regions that carry a number. **unlabeled**: regions without a "
        f"number. **slivers**: share of the page a round brush {bm.print_size.MIN_PAINTABLE_WIDTH_MM:g} mm wide can't "
        "paint without crossing into another region. "
        f"**labels < {bm.print_size.MIN_LABEL_SIZE_PT:g} pt**: share of numbers printing smaller than that. "
        "**compactness**: 4πA/P² of the regions (1 = disk), 10th percentile and median. "
        "**undersized**: regions left below the merge threshold. **ink**: share of dark outline/number pixels.",
        "",
        "Millimeters and points are at print size on A4 (see `benchmarks/README.md`). The paintability metrics "
        "(unlabeled, slivers, label size, compactness) have no tolerances yet and don't affect the verdict.",
        "",
        "Categories come from the image manifest. The first table averages each category (an image counts in every "
        "category it has); the per-case tables list each image under its primary category.",
        "",
    ]
    ref = sets[0]
    pairs = [(s, ref) for s in sets[1:]] or [(ref, None)]
    for cand, base in pairs:
        lines += [f"### {cand.label}" + (f" vs {base.label}" if base else ""), ""]
        lines += _category_summary(cand, base, case_ids, categories, tol)
        header = ["case", *(title for _key, title, _fmt in QUALITY_COLUMNS)] + (["flags"] if base else [])
        for category in _category_order(categories.values(), primary_only=True):
            rows = []
            for cid in case_ids:
                c = cand.ok_case(cid)
                if c is None or categories[cid][0] != category:
                    continue
                b = base.ok_case(cid) if base else None
                row = [_case_title(cid)]
                for key, _title, fmt in QUALITY_COLUMNS:
                    row.append(_pair_cell(b["quality"].get(key) if b else None, c["quality"].get(key), fmt))
                if base:
                    notes = _quality_flags(b["quality"], c["quality"], tol) + _drift_notes(b["quality"], c["quality"], tol) if b else []
                    row.append(", ".join(notes) or ("ok" if b else "no reference"))
                rows.append(row)
            if rows:
                lines += [f"#### {category}", ""] + _table(header, rows) + [""]
    return lines


def _category_summary(
    cand: ResultSet, base: ResultSet | None, case_ids: list[str], categories: dict[str, tuple[str, ...]], tol: Tolerances
) -> list[str]:
    rows = []
    for category in _category_order(categories.values()):
        cids = [cid for cid in case_ids if category in categories[cid] and cand.ok_case(cid)]
        if not cids:
            continue
        row = [category, str(len({cand.ok_case(cid)["image"] for cid in cids})), str(len(cids))]
        # With a reference, average only the cases both sets completed, so before and after cover the same cases.
        paired = [cid for cid in cids if base.ok_case(cid)] if base else cids
        for key, _title, fmt in SUMMARY_COLUMNS:
            after = _mean([cand.ok_case(cid)["quality"].get(key) for cid in paired])
            before = _mean([base.ok_case(cid)["quality"].get(key) for cid in paired]) if base else None
            row.append(_pair_cell(before, after, fmt))
        if base:
            flagged = [cid for cid in paired if _quality_flags(base.ok_case(cid)["quality"], cand.ok_case(cid)["quality"], tol)]
            row.append(str(len(flagged)))
        rows.append(row)
    header = ["category", "images", "cases", *(title for _key, title, _fmt in SUMMARY_COLUMNS)]
    header += ["flagged cases"] if base else []
    return _table(header, rows) + [""]


def _case_categories(sets: list[ResultSet], case_ids: list[str]) -> dict[str, tuple[str, ...]]:
    """Each case's image categories as ``bench.py run`` recorded them; result sets from before the manifest have none."""
    categories = {}
    for cid in case_ids:
        recorded = next((s.cases[cid]["categories"] for s in sets if s.cases.get(cid, {}).get("categories")), None)
        categories[cid] = tuple(recorded) if recorded else (bench_manifest.UNCATEGORIZED,)
    return categories


def _category_order(category_lists, primary_only: bool = False) -> list[str]:
    present = {cats[0] for cats in category_lists} if primary_only else {c for cats in category_lists for c in cats}
    return sorted(present, key=_category_key)


def _category_key(category: str) -> tuple[int, str]:
    """Manifest order, then uncategorized, then anything unknown alphabetically."""
    order = (*bench_manifest.CATEGORIES, bench_manifest.UNCATEGORIZED)
    return (order.index(category) if category in order else len(order), category)


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _agreement_section(sets: list[ResultSet], case_ids: list[str]) -> list[str]:
    ref = sets[0]
    lines = [
        "## Agreement with the reference",
        "",
        "**page px differing**: share of page pixels that differ (0 = identical page). **same partition**: exactly "
        "the same regions. **boundary F1**: how well region outlines line up (±2 px; 1 = same). "
        "**painting ΔE00**: color difference between the two finished paintings.",
        "",
    ]
    for s in sets[1:]:
        rows = []
        for cid in case_ids:
            agreement = _agreement(ref, s, cid)
            if agreement is None:
                continue
            rows.append(
                [
                    _case_title(cid),
                    _fmt_opt(agreement.get("page_diff"), "{:.2%}"),
                    {True: "yes", False: "no", None: "–"}[agreement.get("partition_identical")],
                    _fmt_opt(agreement.get("boundary_f1"), "{:.3f}"),
                    _fmt_opt(agreement.get("painted_de00"), "{:.2f}"),
                ]
            )
        header = ["case", "page px differing", "same partition", "boundary F1", "painting ΔE00"]
        lines += [f"### {s.label} vs {ref.label}", ""] + _table(header, rows) + [""]
    return lines


def _verdict_section(sets: list[ResultSet], case_ids: list[str], tol: Tolerances) -> list[str]:
    ref = sets[0]
    lines = ["## Verdict", ""]
    for s in sets[1:]:
        ratios, regressions, problems = [], [], []
        for cid in case_ids:
            case = s.cases.get(cid)
            if case is None:
                continue
            if case["status"] != "ok":
                problems.append(f"{_case_title(cid)}: {case['status']}")
                continue
            if not case.get("deterministic", True):
                problems.append(f"{_case_title(cid)}: non-deterministic output")
            ratio, lower_bound = _speedup(ref.cases.get(cid), case)
            if ratio is not None and not lower_bound:
                ratios.append(ratio)
            base = ref.ok_case(cid)
            if base:
                flags = _quality_flags(base["quality"], case["quality"], tol)
                if flags:
                    regressions.append(f"{_case_title(cid)} ({', '.join(flags)})")
        speed = f"{_fmt_ratio(_geomean(ratios))} geometric-mean speedup" if ratios else "no comparable timings"
        if regressions:
            quality = f"**quality regressions in {len(regressions)} case(s)**: " + "; ".join(regressions)
        else:
            quality = "no quality regressions beyond tolerance"
        line = f"- **{s.label}**: {speed}; {quality}."
        if problems:
            line += " Problems: " + "; ".join(problems) + "."
        lines.append(line)
    lines += [
        "",
        f"Tolerances: mean ΔE00 may rise {tol.de00_rel:.0%}, SSIM may drop {tol.ssim_abs}, labeled area may drop "
        f"{tol.labeled_area_abs:.0%} points, undersized regions may not increase.",
    ]
    return lines + [""]


def _agreement(ref: ResultSet, cand: ResultSet, cid: str) -> dict | None:
    if not (ref.ok_case(cid) and cand.ok_case(cid)):
        return None
    ref_dir, cand_dir = ref.case_dir(cid), cand.case_dir(cid)
    out: dict = {}
    ref_page, cand_page = _load_rgb(ref_dir / "page.png"), _load_rgb(cand_dir / "page.png")
    if ref_page is not None and cand_page is not None:
        out["page_diff"] = bm.page_diff_fraction(ref_page, cand_page)
    ref_map, cand_map = _load_region_map(ref_dir), _load_region_map(cand_dir)
    if ref_map is not None and cand_map is not None:
        out["partition_identical"] = bm.partition_identical(ref_map, cand_map)
        out["boundary_f1"] = bm.boundary_f1(ref_map, cand_map)
    ref_paint, cand_paint = _load_rgb(ref_dir / "painted.png"), _load_rgb(cand_dir / "painted.png")
    if ref_paint is not None and cand_paint is not None and ref_paint.shape == cand_paint.shape:
        out["painted_de00"] = bm.mean_de00(ref_paint[:, :, ::-1], cand_paint[:, :, ::-1])
    return out


def _quality_flags(ref_q: dict, cand_q: dict, tol: Tolerances) -> list[str]:
    def both(key: str) -> bool:
        return ref_q.get(key) is not None and cand_q.get(key) is not None

    flags = []
    if both("de00_mean") and cand_q["de00_mean"] > ref_q["de00_mean"] * (1 + tol.de00_rel):
        flags.append("ΔE00 worse")
    if both("ssim") and cand_q["ssim"] < ref_q["ssim"] - tol.ssim_abs:
        flags.append("SSIM worse")
    if both("labeled_area_fraction") and cand_q["labeled_area_fraction"] < ref_q["labeled_area_fraction"] - tol.labeled_area_abs:
        flags.append("less labeled area")
    if both("undersized_regions") and cand_q["undersized_regions"] > ref_q["undersized_regions"]:
        flags.append("more undersized regions")
    return flags


def _drift_notes(ref_q: dict, cand_q: dict, tol: Tolerances) -> list[str]:
    before, after = ref_q.get("regions"), cand_q.get("regions")
    if before and after is not None and abs(after - before) / before > tol.region_drift_rel:
        return [f"note: regions {before} → {after}"]
    return []


def _speedup(ref_case: dict | None, cand_case: dict) -> tuple[float | None, bool]:
    """(ratio, is_lower_bound) of reference time over candidate time."""
    if ref_case is None:
        return None, False
    if ref_case["status"] == "ok":
        return ref_case["median_total_s"] / cand_case["median_total_s"], False
    if ref_case["status"] == "timeout":
        return ref_case["timeout_s"] / cand_case["median_total_s"], True
    return None, False


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_region_map(case_dir: Path) -> np.ndarray | None:
    path = case_dir / "regions.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return data["region_id_map"]


def _pair_cell(before, after, fmt: str) -> str:
    if after is None:
        return "–"
    after_s = fmt.format(after)
    if before is None:
        return after_s
    before_s = fmt.format(before)
    return after_s if before_s == after_s else f"{before_s} → {after_s}"


def _time_cell(case: dict | None) -> str:
    if case is None:
        return "–"
    if case["status"] == "timeout":
        return f"timeout (> {case['timeout_s']:.0f} s)"
    if case["status"] != "ok":
        return "error"
    return _fmt_ms(case["median_total_s"])


def _fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:,.0f} ms"


def _fmt_ratio(ratio: float) -> str:
    return f"{ratio:.2f}×" if ratio < 10 else f"{ratio:,.0f}×"


def _fmt_opt(value, fmt: str) -> str:
    return "–" if value is None else fmt.format(value)


def _range_text(values: list[int]) -> str:
    if not values:
        return "?"
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}–{high}"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _case_title(case_id: str) -> str:
    return case_id.replace("__", " / ")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines

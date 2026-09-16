"""Markdown comparison report for benchmark result sets (``bench.py compare``), and metric noise (``bench.py noise``)."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass, field
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

# The four jobs a page does (benchmarks/QUALITY_BENCHMARKS.md), in the scorecard's order.
JOBS = ("resembles", "paintable", "clean drawing", "palette")
# The scorecard's group of every case, listed after the image categories.
ALL_CASES = "all"
# A group of cases regresses on a metric when its mean change is worse than this many standard errors of the mean.
STANDARD_ERRORS = 3.0
# A case whose page is at least this share of its image's own long edge is judged with `sigma_export`.
EXPORT_SIGMA_MIN_PAGE_SHARE = 0.99
# Without `source_size`, a case counts as an export when the size asked for is at least this large: a quarter above the
# 1100 px preview, so that a preview rendered a pixel or two wide of it is still judged with `sigma`.
EXPORT_SIGMA_MIN_LONG_EDGE = 1375


@dataclass(frozen=True)
class Target:
    """A value every case should reach, where the metric has a value for it.

    ``limit`` is the worst value that still meets the target: the most for a
    metric where lower is better, the least where higher is, and the farthest
    from the ideal for a metric that has one. With ``relative_to``, the limit is
    counted from that quality field of the same case.
    """

    limit: float
    relative_to: str | None = None


@dataclass(frozen=True)
class Metric:
    """A quality field of ``case.json``: its column in the report, and how the scorecard and the verdict judge it."""

    key: str
    title: str
    fmt: str
    job: str | None = None  # one of JOBS; None for a field that only informs
    better: str | None = None  # "lower" or "higher"; None for a field that only informs or has an ideal value
    ideal: float | None = None  # judged by the distance from this value
    sigma: float | None = None  # the tolerance at preview size: typical change of one case between equally good pages
    sigma_export: float | None = None  # the tolerance at an image's own size, where a page keeps the source's detail
    relative: bool = False  # sigma and changes are shares of the reference's value
    target: Target | None = None

    @property
    def name(self) -> str:
        return self.title.rstrip(" ↑↓")

    def sigma_at(self, size: str) -> float | None:
        """The tolerance for a case of this size, ``"preview"`` or ``"export"`` (see ``_sigma_size``)."""
        return self.sigma_export if size == "export" and self.sigma_export is not None else self.sigma


# Every quality column, in the report's order. A tolerance is the root mean square change of one case between two sizes
# of the same image and preset (`bench.py noise`), measured on `main` at 36d75ed (T1.8):
# - `sigma`, at 1099, 1100 and 1101 px: 132 pairs over the 12 benchmark images at Easy / Medium / Hard / Max, fewer for
#   the metrics that need annotations;
# - `sigma_export`, at 1, 2 and 3 px below each image's own long edge, where a page keeps the source's detail: 144 pairs.
#   A case rendered at an image's own size is judged with it (`_sigma_size`).
# Targets are the plan's definition of done.
METRICS = (
    Metric("de00_mean", "ΔE00 mean ↓", "{:.2f}", "resembles", "lower", sigma=0.069, sigma_export=0.073, relative=True),
    Metric("de00_p95", "ΔE00 p95 ↓", "{:.1f}", "resembles", "lower", sigma=0.073, sigma_export=0.12, relative=True),
    Metric("ssim", "SSIM ↑", "{:.3f}", "resembles", "higher", sigma=0.0066, sigma_export=0.0070),
    Metric("regions", "regions", "{:d}"),
    Metric("labeled_area_fraction", "labeled area ↑", "{:.1%}", "paintable", "higher", sigma=0.024, sigma_export=0.0064),
    Metric("unlabeled_regions", "unlabeled ↓", "{:d}", "paintable", "lower", sigma=69, sigma_export=18, target=Target(0)),
    Metric(
        "sliver_area_fraction", "slivers ↓", "{:.1%}", "paintable", "lower", sigma=0.015, sigma_export=0.014, target=Target(0.01)
    ),
    Metric(
        "small_label_fraction",
        f"labels < {bm.print_size.MIN_LABEL_SIZE_PT:g} pt ↓",
        "{:.1%}",
        "paintable",
        "lower",
        sigma=0.021,
        sigma_export=0.022,
        target=Target(0),
    ),
    Metric("compactness_p10", "compactness p10 ↑", "{:.2f}", "paintable", "higher", sigma=0.023, sigma_export=0.020),
    Metric("compactness_median", "compactness median ↑", "{:.2f}", "paintable", "higher", sigma=0.036, sigma_export=0.029),
    Metric("lines_per_boundary", "lines per boundary", "{:.2f}", "clean drawing", ideal=1.0, sigma=0.10, sigma_export=0.048),
    Metric(
        "lines_per_boundary_clear",
        "lines per boundary (clear)",
        "{:.2f}",
        "clean drawing",
        ideal=1.0,
        sigma=0.10,  # provisional: inherited from lines per boundary until `noise` measures it (D-027)
        sigma_export=0.048,
        target=Target(0.05),
    ),
    Metric(
        "unenclosed_area_fraction",
        "unenclosed ↓",
        "{:.1%}",
        "clean drawing",
        "lower",
        sigma=0,  # a page whose lines close has none of it, so any is a regression
        sigma_export=0,
        target=Target(0),
    ),
    Metric(
        "same_color_boundary_fraction",
        "same-color boundary ↓",
        "{:.1%}",
        "clean drawing",
        "lower",
        sigma=0.028,
        sigma_export=0.014,
        target=Target(0),
    ),
    Metric("jaggedness", "jaggedness ↓", "{:.3f}", "clean drawing", "lower", sigma=0.0085, sigma_export=0.011, target=Target(1.02)),
    Metric("edge_f1", "edge F1 ↑", "{:.2f}", "clean drawing", "higher", sigma=0.023, sigma_export=0.020),
    Metric(
        "palette_min_de00",
        "palette min ΔE00 ↑",
        "{:.1f}",
        "palette",
        "higher",
        sigma=1.3,
        sigma_export=0.93,
        target=Target(bm.PALETTE_MIN_DE00),
    ),
    Metric(
        "palette_close_pairs",
        f"color pairs < {bm.PALETTE_MIN_DE00:g} ΔE00 ↓",
        "{:d}",
        "palette",
        "lower",
        sigma=3.2,
        sigma_export=3.1,
    ),
    Metric("ink_line_f1", "ink line F1 ↑", "{:.2f}", "clean drawing", "higher", sigma=0.021, sigma_export=0.023, target=Target(0.9)),
    Metric("tube_regions", "tubes ↓", "{:d}", "clean drawing", "lower", sigma=2.6, sigma_export=1.9, target=Target(0)),
    Metric(
        "tube_ink_fraction",
        f"ink in shapes < {bm.INK_MAX_WIDTH_MM:g} mm ↓",
        "{:.1%}",
        "clean drawing",
        "lower",
        sigma=0.031,
        sigma_export=0.028,
    ),
    Metric("flat_color_de00_mean", "flat colors ΔE00 ↓", "{:.2f}", "palette", "lower", sigma=1.2, sigma_export=0.81),
    Metric("face_de00_mean", "face ΔE00 ↓", "{:.2f}", "resembles", "lower", sigma=0.078, sigma_export=0.082, relative=True),
    Metric("face_ssim", "face SSIM ↑", "{:.3f}", "resembles", "higher", sigma=0.016, sigma_export=0.038),
    Metric("features_lost", "features lost ↓", "{:d}", "resembles", "lower", sigma=0.35, sigma_export=0.29, target=Target(0)),
    Metric("labels_on_features", "labels on features ↓", "{:d}", "clean drawing", "lower", sigma=1.6, sigma_export=3.6),
    Metric("text_cer_source", "text CER source", "{:.2f}"),
    Metric(
        "text_cer_page",
        "text CER page ↓",
        "{:.2f}",
        "clean drawing",
        "lower",
        sigma=0.017,
        sigma_export=0.018,
        target=Target(0.1, relative_to="text_cer_source"),
    ),
    Metric("text_cer_painting", "text CER painting ↓", "{:.2f}", "resembles", "lower", sigma=0.0096, sigma_export=0.017),
    Metric("labels_on_text", "labels on text ↓", "{:d}", "clean drawing", "lower", sigma=1.3, sigma_export=2.7, target=Target(0)),
    Metric("undersized_regions", "undersized ↓", "{:d}", "paintable", "lower", sigma=0.0, sigma_export=0.0),
    Metric("ink_fraction", "ink", "{:.1%}"),
)
METRICS_BY_KEY = {metric.key: metric for metric in METRICS}
JUDGED = tuple(metric for metric in METRICS if metric.sigma is not None)
TARGETED = tuple(metric for metric in METRICS if metric.target is not None)


@dataclass(frozen=True)
class Tolerances:
    """How far a candidate may drift from the reference before the report flags it."""

    sigma: dict[str, float] = field(default_factory=dict)  # overrides of a metric's tolerances, at every size, by quality key
    region_drift_rel: float = 0.15  # region-count change worth a note (not a failure)

    def sigma_of(self, metric: Metric, size: str = "preview") -> float | None:
        return self.sigma.get(metric.key, metric.sigma_at(size))


@dataclass(frozen=True)
class Regression:
    """A metric whose mean change over a group of cases is worse than its tolerance allows."""

    metric: Metric
    group: str  # an image category, ALL_CASES, or a case
    cases: int  # cases with a value in both result sets
    change: float  # mean change, positive when worse; for a relative metric a share of the reference's value
    allowed: float


def parse_sigma_overrides(items) -> dict[str, float]:
    """``METRIC=SIGMA`` strings, as ``bench.py compare --tol`` takes them, as {quality key: sigma}."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        metric = METRICS_BY_KEY.get(key.strip())
        if not sep or metric is None or metric.sigma is None:
            raise ValueError(f"--tol takes METRIC=SIGMA, METRIC one of {', '.join(m.key for m in JUDGED)}; got {item!r}")
        try:
            sigma = float(value)
        except ValueError:
            raise ValueError(f"--tol {item!r}: {value!r} is not a number") from None
        if not sigma >= 0:
            raise ValueError(f"--tol {item!r}: sigma must be 0 or more")
        overrides[metric.key] = sigma
    return overrides


def regressions(metrics, pairs, group: str, tol: Tolerances) -> list[Regression]:
    """The metrics whose mean change over ``pairs`` of (reference quality, candidate quality, size) is worse than allowed.

    A case counts for a metric where both have a value (and, for a relative
    metric, the reference's isn't 0). Its tolerance is the metric's sigma at its
    size, ``"preview"`` or ``"export"``. Over n such cases the mean change may be
    worse by ``STANDARD_ERRORS`` standard errors of the mean, sqrt(sum of their
    sigmas squared) / n, which is sigma / sqrt(n) where they share one sigma.
    """
    found = []
    for metric in metrics:
        changes, sigmas = [], []
        for before, after, size in pairs:
            sigma = tol.sigma_of(metric, size)
            change = None if sigma is None else _change(metric, before.get(metric.key), after.get(metric.key))
            if change is not None:
                changes.append(change)
                sigmas.append(sigma)
        if not changes:
            continue
        mean = sum(changes) / len(changes)
        allowed = STANDARD_ERRORS * math.sqrt(sum(sigma * sigma for sigma in sigmas)) / len(changes)
        if mean > allowed:
            found.append(Regression(metric, group, len(changes), mean, allowed))
    return found


def misses_target(metric: Metric, quality: dict) -> bool | None:
    """Whether a case misses the metric's target; None where the metric has no target or no value."""
    target = metric.target
    value = quality.get(metric.key)
    if target is None or value is None:
        return None
    limit = target.limit
    if target.relative_to is not None:
        base = quality.get(target.relative_to)
        if base is None:
            return None
        limit += base
    if metric.ideal is not None:
        return abs(value - metric.ideal) > limit
    return value < limit if metric.better == "higher" else value > limit


# bench.py noise pairs output sizes at most this share apart (1099 and 1101 px), never a preview with an export.
NOISE_MAX_SIZE_CHANGE = 0.01


def noise_sigmas(sets: list[ResultSet]) -> dict[str, tuple[int, float]]:
    """Each judged metric's (pairs, sigma): the root mean square change between cases that differ only in output size.

    Every two completed cases of the same image and preset whose pages' long
    edges differ, by at most ``NOISE_MAX_SIZE_CHANGE`` of the smaller one, are a
    pair, within a result set and across them, the smaller size first. So previews
    and exports in the same result sets each pair only with sizes near their own.
    Sizes are the pages' own, not the ones asked for, and a size rendered in
    several result sets counts once.

    Pages at an image's own size are left out. The pipeline never upscales, so
    every size asked for at or above it renders that one page, and resizing
    changes a page by more than chance does: on the benchmark images, SSIM changes
    by 0.05 between an image's own size and 1 px less, and by 0.008 between two
    sizes below it. A relative metric's change is a share of the smaller size's
    value.
    """
    by_image: dict[tuple[str, str], dict[int, dict]] = {}
    for result_set in sets:
        for case in result_set.cases.values():
            if case["status"] == "ok" and not _unscaled(case):
                by_image.setdefault((case["image"], case["preset"]), {}).setdefault(_page_long_edge(case), case)
    changes: dict[str, list[float]] = {metric.key: [] for metric in JUDGED}
    for cases in by_image.values():
        for (smaller_edge, smaller), (larger_edge, larger) in itertools.combinations(sorted(cases.items()), 2):
            if larger_edge - smaller_edge > NOISE_MAX_SIZE_CHANGE * smaller_edge:
                continue
            for metric in JUDGED:
                change = _change(metric, smaller["quality"].get(metric.key), larger["quality"].get(metric.key))
                if change is not None:
                    changes[metric.key].append(change)
    return {key: (len(values), math.sqrt(sum(v * v for v in values) / len(values))) for key, values in changes.items() if values}


def _page_long_edge(case: dict) -> int:
    """The long edge of a case's page in px; result sets from before ``output_size`` was recorded give the size asked for."""
    return max(case["output_size"]) if case.get("output_size") else case["long_edge"]


def _unscaled(case: dict) -> bool:
    """Whether a case's page is its image at the image's own size.

    Result sets from before ``source_size`` was recorded tell it only where a
    larger size was asked for than the page has.
    """
    if case.get("source_size"):
        return _page_long_edge(case) >= max(case["source_size"])
    return case["long_edge"] > _page_long_edge(case)


def _sigma_size(case: dict) -> str:
    """Which tolerance judges a case: ``"export"`` where its page keeps the image's own size, else ``"preview"``.

    Result sets from before ``source_size`` was recorded go by the size asked
    for, so that a preview rendered a pixel wide of 1100 px still counts as one.
    """
    if case.get("source_size"):
        return "export" if _page_long_edge(case) >= EXPORT_SIGMA_MIN_PAGE_SHARE * max(case["source_size"]) else "preview"
    return "export" if case["long_edge"] >= EXPORT_SIGMA_MIN_LONG_EDGE else "preview"


def noise_report(paths: list[Path]) -> str:
    sets = [ResultSet(p) for p in paths]
    sigmas = noise_sigmas(sets)
    rows = []
    for metric in JUDGED:
        if metric.key in sigmas:
            pairs, sigma = sigmas[metric.key]
            rows.append(
                [
                    metric.key,
                    metric.name,
                    str(pairs),
                    f"{sigma:.2g}",
                    f"{metric.sigma:g}",
                    "–" if metric.sigma_export is None else f"{metric.sigma_export:g}",
                    "yes" if metric.relative else "no",
                ]
            )
    lines = [
        "# Tessellatum benchmark noise",
        "",
        f"Root mean square change of one case between the same image and preset at page sizes up to "
        f"{NOISE_MAX_SIZE_CHANGE:.0%} apart, in {', '.join(s.label for s in sets)}. Sizes are the pages' own, each counted "
        "once, and pages at an image's own size, which aren't resized, are left out. For pages that should be equally good, "
        "it is a metric's tolerance "
        "(`bench.py compare --tol METRIC=SIGMA`). A relative metric's change is a share of its value.",
        "",
    ]
    header = ["metric", "name", "pairs", "sigma", "preview now", "export now", "relative"]
    return "\n".join(lines + _table(header, rows)) + "\n"


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
    lines += _scorecard_section(sets, case_ids, categories, tol)
    lines += _quality_section(sets, case_ids, categories, tol)
    if len(sets) > 1:
        lines += _agreement_section(sets, case_ids)
        lines += _verdict_section(sets, case_ids, categories, tol)
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


def _scorecard_section(
    sets: list[ResultSet], case_ids: list[str], categories: dict[str, tuple[str, ...]], tol: Tolerances
) -> list[str]:
    lines = [
        "## Scorecard",
        "",
        "A page does four jobs (`benchmarks/QUALITY_BENCHMARKS.md`): it **resembles** the image, is **paintable**, reads "
        "as a **clean drawing**, and has a **palette** that works. Each job's metrics are averaged over each image "
        f"category (an image counts in every category it has) and over **{ALL_CASES}** cases. A metric's target, what "
        "every case should reach, is in brackets after it; **targets met** counts the cases meeting all of the job's "
        "targets that apply to them. Against a reference, both sets are averaged over the cases both completed, and a "
        "mean in bold got worse by more than its tolerance allows (see the verdict).",
        "",
    ]
    ref = sets[0]
    for cand, base in [(s, ref) for s in sets[1:]] or [(ref, None)]:
        groups = _groups(cand, base, case_ids, categories)
        if not groups:
            continue
        worse = {name: regressions(JUDGED, _quality_pairs(base, cand, cids), name, tol) if base else [] for name, cids in groups}
        lines += [f"### {cand.label}" + (f" vs {base.label}" if base else ""), ""]
        rows = []
        for name, cids in groups:
            row = [name, str(len({cand.ok_case(cid)["image"] for cid in cids})), str(len(cids))]
            for job in JOBS:
                met = _met_cell(_job_metrics(job), cand, base, cids)
                cell = "no targets" if met is None else f"{met} met"
                names = [found.metric.name for found in worse[name] if found.metric.job == job]
                row.append(cell + (f"; **{', '.join(names)} worse**" if names else ""))
            rows.append(row)
        lines += _table(["category", "images", "cases", *JOBS], rows) + [""]
        for job in JOBS:
            metrics = _job_metrics(job)
            rows = []
            for name, cids in groups:
                worse_keys = {found.metric.key for found in worse[name]}
                row = [name, str(len(cids))]
                for metric in metrics:
                    after = _mean(cand.ok_case(cid)["quality"].get(metric.key) for cid in cids)
                    before = _mean(base.ok_case(cid)["quality"].get(metric.key) for cid in cids) if base else None
                    cell = _pair_cell(before, after, _mean_fmt(metric.fmt))
                    row.append(f"**{cell}**" if metric.key in worse_keys else cell)
                row.append(_met_cell(metrics, cand, base, cids) or "–")
                rows.append(row)
            header = ["category", "cases", *(_metric_header(metric) for metric in metrics), "targets met"]
            lines += [f"#### {job}", ""] + _table(header, rows) + [""]
    return lines


def _quality_section(
    sets: list[ResultSet], case_ids: list[str], categories: dict[str, tuple[str, ...]], tol: Tolerances
) -> list[str]:
    lines = [
        "## Quality",
        "",
        "**ΔE00** and **SSIM** score the *finished painting* (every region filled with its legend color) against the "
        "source image: CIEDE2000 color error (mean and 95th percentile) and structural similarity of luma. The other "
        "metrics score the page's regions, lines, numbers and legend colors, on line art how they keep the "
        "artwork's ink lines and flat colors, on faces whether their features survive, and on text whether OCR still "
        "reads it, as the image manifest gives them. "
        "**labeled area**: share of the page inside regions that carry a number. **unlabeled**: regions without a "
        f"number. **slivers**: share of the page a round brush {bm.print_size.MIN_PAINTABLE_WIDTH_MM:g} mm wide can't "
        "paint without crossing into another region. "
        f"**labels < {bm.print_size.MIN_LABEL_SIZE_PT:g} pt**: share of numbers printing smaller than that. "
        "**compactness**: 4πA/P² of the regions (1 = disk), 10th percentile and median. "
        "**lines per boundary**: lines drawn along each boundary between regions (1 = one line per boundary), and "
        f"**(clear)** the same over the boundary more than {bm.JUNCTION_CLEARANCE_PX:g} px from a junction, where "
        "the lines of the boundaries that end there are not counted too. "
        "**unenclosed**: share of the page in a white area covering more than one region, where two regions' "
        "paint would run together. "
        "**same-color boundary**: share of boundary length between two regions of the same color. "
        f"**jaggedness**: length of the drawn lines over their length with wiggles under "
        f"{bm.JAGGEDNESS_SMOOTHING_MM:g} mm smoothed away (1 = smooth). "
        f"**edge F1**: how well region boundaries and the source's edges line up, within {bm.EDGE_TOLERANCE_MM:g} mm "
        "(1 = every boundary on an edge and every edge on a boundary). "
        "**palette min ΔE00**: smallest CIEDE2000 difference between two legend colors. "
        f"**color pairs < {bm.PALETTE_MIN_DE00:g} ΔE00**: pairs of legend colors closer than that. "
        f"**ink line F1**: how well drawn lines run down the middle of the artwork's ink lines, within "
        f"{bm.INK_LINE_TOLERANCE_MM:g} mm (lines away from the ink don't count). "
        "**tubes**: regions at least half made of ink lines. "
        f"**ink in shapes < {bm.INK_MAX_WIDTH_MM:g} mm**: share of the ink lines lying in parts of regions that narrow, "
        "to be painted instead of printed. "
        "**flat colors ΔE00**: mean CIEDE2000 from each of the artwork's flat colors to the nearest legend color. "
        "**face ΔE00** and **face SSIM**: ΔE00 mean and SSIM inside the image's face boxes. "
        f"**features lost**: annotated eyes, noses and mouths with neither drawn lines along at least "
        f"{bm.FEATURE_MIN_EDGE_RECALL:.0%} of their edges nor a region of their own covering at least "
        f"{bm.FEATURE_MIN_REGION_SHARE:.0%} of their box. "
        "**labels on features**: numbers overlapping a feature box. "
        "**text CER**: character error rate of OCR inside the image's text boxes against their annotated text, on the "
        "source (how much OCR reads there at all), the page and the painting (0 = read exactly, 1 = nothing read). "
        "**labels on text**: numbers overlapping a text box. "
        "**undersized**: regions left below the merge threshold. **ink**: share of dark outline/number pixels.",
        "",
        "Millimeters and points are at print size on A4 (see `benchmarks/README.md`). **targets missed**: the scorecard's "
        "targets a case misses, out of those that apply to it. **flags**: metrics on which the case alone got worse "
        f"than on the reference by more than {STANDARD_ERRORS:g} times their tolerance, which are cases to look at, not "
        "regressions (see the verdict); and region-count changes worth a note.",
        "",
        "Categories come from the image manifest; each table lists the images whose primary category it is.",
        "",
    ]
    ref = sets[0]
    pairs = [(s, ref) for s in sets[1:]] or [(ref, None)]
    for cand, base in pairs:
        lines += [f"### {cand.label}" + (f" vs {base.label}" if base else ""), ""]
        header = ["case", *(metric.title for metric in METRICS), "targets missed"] + (["flags"] if base else [])
        for category in _category_order(categories.values(), primary_only=True):
            rows = []
            for cid in case_ids:
                c = cand.ok_case(cid)
                if c is None or categories[cid][0] != category:
                    continue
                b = base.ok_case(cid) if base else None
                row = [_case_title(cid)]
                for metric in METRICS:
                    row.append(_pair_cell(b["quality"].get(metric.key) if b else None, c["quality"].get(metric.key), metric.fmt))
                row.append(_misses_cell(b["quality"] if b else None, c["quality"]))
                if base:
                    row.append(_flags_cell(b, c, cid, tol) if b else "no reference")
                rows.append(row)
            if rows:
                lines += [f"#### {category}", ""] + _table(header, rows) + [""]
    return lines


def _groups(
    cand: ResultSet, base: ResultSet | None, case_ids: list[str], categories: dict[str, tuple[str, ...]]
) -> list[tuple[str, list[str]]]:
    """(name, case ids) for each image category, then for all cases: the cases the candidate, and the reference if any, completed."""
    done = [cid for cid in case_ids if cand.ok_case(cid) and (base is None or base.ok_case(cid))]
    groups = [(category, [cid for cid in done if category in categories[cid]]) for category in _category_order(categories[cid] for cid in done)]
    return groups + [(ALL_CASES, done)] if done else []


def _job_metrics(job: str) -> list[Metric]:
    return [metric for metric in METRICS if metric.job == job]


def _quality_pairs(base: ResultSet, cand: ResultSet, case_ids: list[str]) -> list[tuple[dict, dict, str]]:
    return [
        (base.ok_case(cid)["quality"], cand.ok_case(cid)["quality"], _sigma_size(cand.ok_case(cid))) for cid in case_ids
    ]


def _change(metric: Metric, before, after) -> float | None:
    """How much worse ``after`` is than ``before``, negative when better; for a relative metric a share of ``before``."""
    if before is None or after is None:
        return None
    if metric.ideal is not None:
        worse = abs(after - metric.ideal) - abs(before - metric.ideal)
    else:
        worse = before - after if metric.better == "higher" else after - before
    if metric.relative:
        return worse / abs(before) if before else None
    return worse


def _meets_targets(metrics, quality: dict) -> bool | None:
    """Whether a case meets every target of ``metrics`` that applies to it; None where none does."""
    misses = [miss for miss in (misses_target(metric, quality) for metric in metrics) if miss is not None]
    return not any(misses) if misses else None


def _met_cell(metrics, cand: ResultSet, base: ResultSet | None, case_ids: list[str]) -> str | None:
    """"met/scored": cases meeting all their targets among ``metrics``, of the cases any applies to; None if none does."""

    def count(result_set: ResultSet) -> tuple[int, int]:
        verdicts = [_meets_targets(metrics, result_set.ok_case(cid)["quality"]) for cid in case_ids]
        scored = [verdict for verdict in verdicts if verdict is not None]
        return sum(scored), len(scored)

    after = count(cand)
    before = count(base) if base else (0, 0)
    if not after[1] and not before[1]:
        return None
    after_s = f"{after[0]}/{after[1]}"
    before_s = f"{before[0]}/{before[1]}"
    return after_s if not before[1] or before_s == after_s else f"{before_s} → {after_s}"


def _misses_cell(before_q: dict | None, after_q: dict) -> str:
    def text(quality: dict) -> str:
        misses = [miss for miss in (misses_target(metric, quality) for metric in TARGETED) if miss is not None]
        return f"{sum(misses)}/{len(misses)}" if misses else "–"

    after = text(after_q)
    before = text(before_q) if before_q is not None else after
    return after if before == after else f"{before} → {after}"


def _flags_cell(before: dict, after: dict, case_id: str, tol: Tolerances) -> str:
    pairs = [(before["quality"], after["quality"], _sigma_size(after))]
    notes = [f"{found.metric.name} worse" for found in regressions(JUDGED, pairs, _case_title(case_id), tol)]
    return ", ".join(notes + _drift_notes(before["quality"], after["quality"], tol)) or "ok"


def _metric_header(metric: Metric) -> str:
    return metric.title + (f" ({_target_text(metric)})" if metric.target else "")


def _target_text(metric: Metric) -> str:
    target = metric.target
    limit = f"{target.limit * 100:g}%" if "%" in metric.fmt else f"{target.limit:g}"
    if metric.ideal is not None:
        return f"{metric.ideal:g} ± {limit}"
    sign = "≥" if metric.better == "higher" else "≤"
    if target.relative_to is not None:
        return f"{sign} {METRICS_BY_KEY[target.relative_to].name} + {limit}"
    return "0" if target.limit == 0 and metric.better == "lower" else f"{sign} {limit}"


def _mean_fmt(fmt: str) -> str:
    """A mean of counts gets a decimal place."""
    return "{:.1f}" if fmt == "{:d}" else fmt


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


def _verdict_section(
    sets: list[ResultSet], case_ids: list[str], categories: dict[str, tuple[str, ...]], tol: Tolerances
) -> list[str]:
    ref = sets[0]
    lines = ["## Verdict", ""]
    for s in sets[1:]:
        ratios, problems = [], []
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
        groups = _groups(s, ref, case_ids, categories)
        found: dict[str, list[Regression]] = {}
        for name, cids in groups:
            for regression in regressions(JUDGED, _quality_pairs(ref, s, cids), name, tol):
                found.setdefault(regression.metric.key, []).append(regression)
        paired = dict(groups).get(ALL_CASES, [])

        speed = f"{_fmt_ratio(_geomean(ratios))} geometric-mean speedup" if ratios else "no comparable timings"
        quality = f"**quality regressions on {len(found)} metric(s)**" if found else "no quality regressions beyond tolerance"
        lines.append(f"- **{s.label}**: {speed}; {quality}.")
        if found:
            texts = []
            for key in (metric.key for metric in JUDGED if metric.key in found):
                metric = METRICS_BY_KEY[key]
                where = ", ".join(
                    f"{r.group} (by {_fmt_amount(metric, r.change)}, tolerance {_fmt_amount(metric, r.allowed)}, n = {r.cases})"
                    for r in found[key]
                )
                texts.append(f"{metric.name} worse in {where}")
            lines.append("  - **Regressions:** " + "; ".join(texts) + ".")
        counts, new_misses, newly_met = _target_changes(ref, s, paired)
        lines.append("  - **Target misses** (cases missing each target): " + ("; ".join(counts) or "none") + ".")
        if new_misses:
            texts = [f"{_case_title(cid)} ({', '.join(names)})" for cid, names in new_misses]
            lines.append(f"  - **New target misses in {len(new_misses)} case(s):** " + "; ".join(texts) + ".")
        if newly_met:
            lines.append(f"  - Targets newly met: {newly_met} (case, target) pair(s).")
        look_at = []
        for cid in paired:
            names = [r.metric.name for r in regressions(JUDGED, _quality_pairs(ref, s, [cid]), _case_title(cid), tol)]
            if names:
                look_at.append(f"{_case_title(cid)} ({', '.join(names)})")
        if look_at:
            lines.append(f"  - Cases to look at, worse alone by more than {STANDARD_ERRORS:g} tolerances: " + "; ".join(look_at) + ".")
        if problems:
            lines.append("  - Problems: " + "; ".join(problems) + ".")
    lines += [
        "",
        f"A metric regresses in a category, or over all cases, when its mean change against the reference is worse than "
        f"{STANDARD_ERRORS:g} standard errors of the mean, {STANDARD_ERRORS:g} √(Σσ²)/n: σ is a case's tolerance, the "
        "typical change between equally good pages at its size (preview, or export where the page is the image at its own "
        f"size), and n the number of cases with a value in both sets, so it is {STANDARD_ERRORS:g} σ/√n where they are all "
        "of one size (see `benchmarks/README.md`). Target misses are counted separately, on the reference → the candidate; "
        "a new miss is a case that met the target on the reference.",
    ]
    if tol.sigma:
        lines.append("Tolerances overridden: " + ", ".join(f"{METRICS_BY_KEY[k].name} σ = {v:g}" for k, v in tol.sigma.items()) + ".")
    return lines + [""]


def _target_changes(ref: ResultSet, cand: ResultSet, case_ids: list[str]) -> tuple[list[str], list[tuple[str, list[str]]], int]:
    """(each target's miss count, reference → candidate; the cases newly missing targets, with their names; newly met pairs)."""
    verdicts = {
        cid: [(misses_target(m, ref.ok_case(cid)["quality"]), misses_target(m, cand.ok_case(cid)["quality"])) for m in TARGETED]
        for cid in case_ids
    }
    counts = []
    for i, metric in enumerate(TARGETED):
        before = [verdicts[cid][i][0] for cid in case_ids if verdicts[cid][i][0] is not None]
        after = [verdicts[cid][i][1] for cid in case_ids if verdicts[cid][i][1] is not None]
        if not before and not after:
            continue
        before_s, after_s = f"{sum(before)}/{len(before)}", f"{sum(after)}/{len(after)}"
        change = after_s if not before or before_s == after_s else f"{before_s} → {after_s}"
        counts.append(f"{metric.name} ({_target_text(metric)}) {change}")
    new_misses = []
    for cid in case_ids:
        names = [m.name for m, (before, after) in zip(TARGETED, verdicts[cid]) if before is False and after]
        if names:
            new_misses.append((cid, names))
    newly_met = sum(before is True and after is False for cid in case_ids for before, after in verdicts[cid])
    return counts, new_misses, newly_met


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


def _fmt_amount(metric: Metric, value: float) -> str:
    """A change or tolerance in a metric's unit: a share for a relative metric, points for a share of the page."""
    if metric.relative:
        return f"{value:.1%}"
    if "%" in metric.fmt:
        return f"{value * 100:.2g} points"
    return f"{value:.2g}" if abs(value) < 100 else f"{value:.0f}"


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

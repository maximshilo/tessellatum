#!/usr/bin/env python
"""Speed + quality benchmarks for Tessellatum's page-generation pipeline.

  run      Benchmark targets -- git refs, or WORKTREE for the current checkout
           (uncommitted changes included) -- over images x presets x sizes.
  compare  Speedups, quality scorecard and verdict of result sets against the first one.
  noise    Each quality metric's typical change between cases that differ only in output size (its tolerance).

See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from importlib import metadata
from pathlib import Path

import bench_manifest
import bench_report

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DEFAULT_IMAGES_DIR = REPO_ROOT / "tests" / "sample_images"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
WORKTREE = "WORKTREE"
TRACKED_PACKAGES = ("numpy", "opencv-python", "pillow", "numba")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="benchmark one or more targets")
    run_p.add_argument(
        "targets",
        nargs="*",
        default=[WORKTREE],
        metavar="TARGET[=LABEL]",
        help=f"git ref (branch, tag, commit) or {WORKTREE} for the current checkout; "
        f"optionally name the result set with =LABEL (default: {WORKTREE})",
    )
    run_p.add_argument("--images", nargs="+", type=Path, default=[DEFAULT_IMAGES_DIR], help="image files and/or directories")
    run_p.add_argument(
        "--category",
        nargs="+",
        choices=bench_manifest.CATEGORIES,
        help="only images in any of these categories, per the manifest next to the images",
    )
    run_p.add_argument(
        "--presets", nargs="+", default=["Easy", "Medium", "Hard"], help="difficulty presets (also: Max, the most granular Custom setting)"
    )
    run_p.add_argument("--long-edge", nargs="+", type=int, default=[1100], dest="long_edges", help="output long edge(s) in px (preview is 1100, export 2400)")
    run_p.add_argument("--repeats", type=int, default=3, help="measured runs per case (the median is reported)")
    run_p.add_argument("--warmup", type=int, default=1, help="unmeasured warm-up runs per case")
    run_p.add_argument("--threads", type=int, help="cap worker threads (OpenCV, numba, tessellatum's own pools)")
    run_p.add_argument("--timeout", type=float, default=1800.0, help="seconds before a case is killed and recorded as a timeout")
    run_p.add_argument(
        "--append",
        action="store_true",
        help="add to (or re-run cases in) an existing result set instead of replacing it, "
        "e.g. to use fewer repeats or a shorter timeout for very slow cases",
    )
    run_p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    cmp_p = sub.add_parser("compare", help="compare result sets; the first is the reference")
    cmp_p.add_argument("result_sets", nargs="+", help="result set labels (under --results-dir) or directories")
    cmp_p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    cmp_p.add_argument("--output", type=Path, help="also write the Markdown report here")
    cmp_p.add_argument("--detail", action="store_true", help="per-case stage breakdown")
    cmp_p.add_argument(
        "--tol",
        action="append",
        default=[],
        metavar="METRIC=SIGMA",
        help="use SIGMA as a metric's tolerance, by its case.json key, e.g. jaggedness=0.02 (repeatable; see benchmarks/README.md)",
    )

    noise_p = sub.add_parser("noise", help="each metric's typical change between cases that differ only in output size")
    noise_p.add_argument("result_sets", nargs="+", help="result set labels (under --results-dir) or directories")
    noise_p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args(argv)
    return {"run": _cmd_run, "compare": _cmd_compare, "noise": _cmd_noise}[args.command](args)


# -- run --------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    try:
        images, manifest = select_images(_collect_images(args.images), args.category)
    except bench_manifest.ManifestError as exc:
        print(f"Invalid image manifest: {exc}", file=sys.stderr)
        return 1
    if not images:
        print(f"No images in categories {', '.join(args.category)}." if args.category else "No images found.", file=sys.stderr)
        return 1
    cases = list(itertools.product(images, args.presets, args.long_edges))

    for spec in args.targets:
        ref, _, label = spec.partition("=")
        with _materialize(ref) as (src_dir, target):
            label = label or _default_label(target)
            out_dir = args.results_dir / label
            results_by_case: dict[str, dict] = {}
            if args.append and (out_dir / "results.json").is_file():
                previous = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
                if previous["target"]["commit"] != target["commit"]:
                    print(
                        f"Cannot append to {label!r}: it was recorded at commit {previous['target']['commit']}, "
                        f"not {target['commit']}.",
                        file=sys.stderr,
                    )
                    return 1
                results_by_case = {c["case"]: c for c in previous["cases"]}
            elif out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"== {label}: {len(cases)} case(s) -> {out_dir}", flush=True)

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src_dir), env.get("PYTHONPATH", "")) if p)
            if args.threads:
                for var in ("TESSELLATUM_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
                    env[var] = str(args.threads)

            for image, preset, long_edge in cases:
                case_id = f"{image.stem}__{preset}__{long_edge}"
                print(f"   {case_id:<45} ", end="", flush=True)
                case_dir = out_dir / "cases" / case_id
                if case_dir.exists():
                    shutil.rmtree(case_dir)
                categories = list(manifest[image].categories) if image in manifest else []
                case = _run_case(args, src_dir, env, image, preset, long_edge, case_dir, categories)
                results_by_case[case_id] = case
                print(_case_summary(case), flush=True)

            results = list(results_by_case.values())
            versions = {c["version"] for c in results if c["status"] == "ok"}
            payload = {
                "schema": 1,
                "label": label,
                "target": target,
                "version": ", ".join(sorted(versions)) or "unknown",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "machine": _machine_info(),
                "settings": {"threads": args.threads},
                "cases": results,
            }
            (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def _run_case(
    args, src_dir: Path, env: dict, image: Path, preset: str, long_edge: int, case_dir: Path, categories: list[str]
) -> dict:
    settings = {"repeats": args.repeats, "warmup": args.warmup, "timeout_s": args.timeout}
    base = {
        "case": case_dir.name,
        "image": image.name,
        "categories": categories,
        "preset": preset,
        "long_edge": long_edge,
        "settings": settings,
    }
    cmd = [
        sys.executable,
        str(BENCH_DIR / "bench_case.py"),
        "--src", str(src_dir),
        "--image", str(image),
        "--preset", preset,
        "--long-edge", str(long_edge),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--out", str(case_dir),
    ]  # fmt: skip
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout", "timeout_s": args.timeout}
    if proc.returncode != 0:
        return {**base, "status": "error", "error": "\n".join(proc.stderr.strip().splitlines()[-15:])}
    return {**json.loads((case_dir / "case.json").read_text(encoding="utf-8")), "categories": categories, "settings": settings}


def _case_summary(case: dict) -> str:
    if case["status"] == "timeout":
        return f"TIMEOUT (> {case['timeout_s']:.0f} s)"
    if case["status"] != "ok":
        return "ERROR\n" + case.get("error", "")
    q = case["quality"]
    parts = [f"{case['median_total_s'] * 1000:,.0f} ms", f"{q['regions']} regions"]
    if "de00_mean" in q:
        parts.append(f"ΔE00 {q['de00_mean']:.2f}")
    return ", ".join(parts)


def _collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            images.append(path)
        else:
            print(f"warning: {path} does not exist, skipping", file=sys.stderr)
    return [p.resolve() for p in images]


def select_images(
    images: list[Path], categories: list[str] | None
) -> tuple[list[Path], dict[Path, bench_manifest.ImageInfo]]:
    """Look up each image in the manifest next to it; keep those in any of ``categories`` (all, if none given)."""
    by_dir = {directory: bench_manifest.load_directory(directory) for directory in {p.parent for p in images}}
    manifest = {p: by_dir[p.parent][p.name] for p in images if p.name in by_dir[p.parent]}
    unlisted = [p.name for p in images if p not in manifest]
    if unlisted:
        print(f"warning: no manifest entry, counted as {bench_manifest.UNCATEGORIZED}: {', '.join(unlisted)}", file=sys.stderr)
    if categories:
        wanted = set(categories)
        images = [p for p in images if p in manifest and wanted & set(manifest[p].categories)]
    return images, manifest


@contextlib.contextmanager
def _materialize(ref: str):
    """Yield ``(src_dir, target_info)`` for a git ref or the working tree."""
    if ref == WORKTREE:
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--", "src"))
        yield REPO_ROOT / "src", {"ref": WORKTREE, "commit": commit, "dirty": dirty}
        return

    commit = _git("rev-parse", "--verify", "--short", f"{ref}^{{commit}}")
    tmp = Path(tempfile.mkdtemp(prefix="tessellatum-bench-"))
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", commit, "src"], cwd=REPO_ROOT, check=True, capture_output=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:  # pragma: no cover - Python without extraction filters
                tar.extractall(tmp)
        yield tmp / "src", {"ref": ref, "commit": commit, "dirty": False}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _default_label(target: dict) -> str:
    if target["ref"] == WORKTREE:
        return f"worktree-{target['commit']}" + ("-dirty" if target["dirty"] else "")
    safe_ref = re.sub(r"[^A-Za-z0-9._-]+", "_", target["ref"])
    return target["commit"] if safe_ref.startswith(target["commit"]) else f"{safe_ref}-{target['commit']}"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _machine_info() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": packages,
    }


# -- compare, noise ---------------------------------------------------------
def _cmd_compare(args: argparse.Namespace) -> int:
    paths = _result_set_paths(args)
    if paths is None:
        return 1
    try:
        tolerances = bench_report.Tolerances(sigma=bench_report.parse_sigma_overrides(args.tol))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    report = bench_report.build_report(paths, tolerances, detail=args.detail)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    return 0


def _cmd_noise(args: argparse.Namespace) -> int:
    paths = _result_set_paths(args)
    if paths is None:
        return 1
    print(bench_report.noise_report(paths))
    return 0


def _result_set_paths(args: argparse.Namespace) -> list[Path] | None:
    paths = []
    for item in args.result_sets:
        path = Path(item)
        if not (path / "results.json").is_file():
            path = args.results_dir / item
        if not (path / "results.json").is_file():
            print(f"No results.json for {item!r} (looked in {path})", file=sys.stderr)
            return None
        paths.append(path)
    return paths


if __name__ == "__main__":
    sys.exit(main())

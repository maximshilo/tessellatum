# Benchmarks

Speed **and** quality benchmarks for the page-generation pipeline, so a change
can be judged on numbers: is it faster, and did the output get any worse?

What counts as a good page for each kind of image (photos, bold-line
cartoons, faces, text), and the metrics still needed to measure it, is
described in [`QUALITY_BENCHMARKS.md`](QUALITY_BENCHMARKS.md).

## Quick start

```
# Benchmark main and the current working tree (uncommitted changes included)
.venv\Scripts\python benchmarks\bench.py run main WORKTREE

# Compare result sets -- the first one is the reference
.venv\Scripts\python benchmarks\bench.py compare main-<commit> worktree-<commit>-dirty
```

Result sets land in `benchmarks/results/<label>/` (git-ignored). Labels default
to `<ref>-<commit>` or `worktree-<commit>[-dirty]`; name your own with
`TARGET=LABEL`, e.g. `run WORKTREE=try-smaller-kernel`.

## How it works

`run` benchmarks each **target** -- any git ref (extracted with `git archive`,
so your checkout is left alone) or `WORKTREE` -- over every combination of
images x presets x output sizes:

- Each case runs in a **fresh Python process** with that version's `src/` on
  the path, so versions never share imports, caches or JIT state. Import time
  and first-run overhead are recorded separately from steady-state timings.
- `--warmup` unmeasured runs (default 1), then `--repeats` measured runs
  (default 3); the median is reported. Every run gets a fresh copy of the
  image, so caching inside the pipeline can't turn repeats into cache hits.
- Per-stage timings come from wrapping the stage functions that
  `tessellatum.core.pipeline.generate` calls: `resize_to_long_edge`,
  `quantize`, `build_regions`, `extract_regions`, `render_page`,
  `render_legend`. The same wrappers capture the region map and palette the
  quality metrics need. Keep those names if you restructure the pipeline, or
  update `PROBED_STAGES` in `bench_case.py`.
- Cases exceeding `--timeout` (default 30 min) are killed and recorded as
  timeouts; speedups against them are reported as lower bounds.

Defaults: every image in `tests/sample_images/`, presets Easy/Medium/Hard, at
preview size (1100 px). `--presets Max` adds the most granular Custom setting
(the worst case for region handling); `--long-edge 1100 2400` adds export size;
`--threads N` caps worker threads to test scaling.

## Quality metrics

Absolute metrics, per result, scored on the *finished painting* (every region
filled with its legend color) against the source image at output size:

| metric | meaning | better |
|---|---|---|
| ΔE00 mean / p95 | CIEDE2000 color error between painting and source | lower |
| SSIM | structural similarity of luma between painting and source | higher |
| labeled area | share of the page inside regions big enough to carry a number | higher |
| undersized | regions still below the difficulty's minimum size | lower (0) |
| regions, ink | region count and share of dark outline/number pixels | informational |

Agreement metrics, candidate vs. reference (computed by `compare` from the
saved `page.png`, `painted.png` and `regions.npz` of each case):

| metric | meaning |
|---|---|
| page px differing | share of page pixels that differ; 0 means a bit-identical page |
| same partition | whether both produced exactly the same regions |
| boundary F1 | how well region outlines line up, within 2 px (1 = same) |
| painting ΔE00 | mean color difference between the two finished paintings |

## Verdict

`compare` flags a case as a **quality regression** when, versus the reference,
mean ΔE00 rises more than 3% (`--tol-de00`), SSIM drops more than 0.01
(`--tol-ssim`), labeled area drops more than 2 points (`--tol-labeled`), or
undersized regions increase. Region-count changes over 15% are noted but not
failures: region count is a difficulty trait, not a quality score.

Timings are wall-clock: close heavy apps while benchmarking, and only compare
result sets recorded on the same machine (the report warns when they're not).

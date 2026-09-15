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
  `render_legend`. Keep those names if you restructure the pipeline, or
  update `PROBED_STAGES` in `bench_case.py`.
- Quality metrics read what the page is made of from the pipeline itself.
  `generate(..., collect_analysis=True)` returns it as `GeneratedPage.analysis`
  (`PageAnalysis` in `pipeline.py`):
  - the region map, each region's color and the palette, legend colors first;
  - the drawn regions, with their outlines and label points;
  - every number, with its font size and bounding box;
  - the outline layer on its own.

  The timed runs don't collect it, as in the app. One more run after them
  does, and its page must match theirs for the case to count as
  deterministic. Versions from before 0.1.10 have no analysis; the stage
  wrappers capture the region map, palette and regions for them instead.
  `case.json` records which source was used under `scored_from` (`analysis`
  or `probe`).
- Cases exceeding `--timeout` (default 30 min) are killed and recorded as
  timeouts; speedups against them are reported as lower bounds.

Defaults: every image in `tests/sample_images/`, presets Easy/Medium/Hard, at
preview size (1100 px). `--presets Max` adds the most granular Custom setting
(the worst case for region handling); `--long-edge 1100 2400` adds export size;
`--threads N` caps worker threads to test scaling; `--category face text` runs
only the images in those categories (see the image manifest below).

## Image manifest

`tests/sample_images/manifest.json` records what each benchmark image is and
where its important parts are, so metrics that need that (faces, text, line
art) know what to score. `run` stores each image's categories with its cases,
and the report groups cases by category.

Images that exist only in your checkout get their entries in a git-ignored
`manifest.local.json` next to `manifest.json`, with the same schema. Its
entries are added to the committed ones, and replace committed entries of the
same name. `tests/test_bench_manifest.py` fails if an image in
`tests/sample_images/` has no entry, a size that doesn't match its file, or
lacks an annotation its categories need.

To check annotations by eye, draw them onto copies of the images (written to
`benchmarks/results/annotations/`):

```
.venv\Scripts\python benchmarks\draw_annotations.py
```

### Schema (version 1)

Coordinates are pixels of the source file. A box is `[x, y, width, height]`,
with `x, y` its top-left corner.

```json
{
  "schema": 1,
  "images": {
    "m-cartoon-bold-lines-girl.png": {
      "size": [997, 1600],
      "categories": ["cartoon", "face"],
      "faces": [
        {"kind": "cartoon", "box": [140, 270, 690, 560],
         "features": [{"part": "eye", "box": [160, 410, 250, 225]}]}
      ],
      "text": [{"box": [10, 10, 200, 40], "string": "FIRST LINE\nSECOND LINE", "rotation": 0}],
      "flat_colors": ["#ffffff", "#ffdfc9"],
      "ink_colors": ["#000000"],
      "areas": [{"kind": "gradient", "box": [0, 0, 997, 200]}],
      "notes": "free text"
    }
  }
}
```

Only `size` and `categories` are required. The first category is the image's
primary one: the report's per-case tables list the image under it.

| category | the image | needs |
|---|---|---|
| `photo` | is a continuous-tone photograph | – |
| `cartoon` | is line art: dark ink lines around flat fills | `flat_colors` (2 or more) and `ink_colors` |
| `flat` | is flat fills without ink lines | `flat_colors` (2 or more) |
| `face` | has faces people will look at | `faces`, each with `features` |
| `text` | has readable text | `text` |
| `gradient` | has smooth gradients, e.g. sky or calm water | `areas` of kind `gradient` |
| `texture` | has fine texture, e.g. fur, foliage or ripples | `areas` of kind `texture` |

- **`size`** must match the file, so annotations can't silently go stale when
  an image is replaced.
- **`faces`**: `kind` is `human`, `animal` or `cartoon`. Each feature's `part`
  is `eye`, `nose` or `mouth`, and its box lies inside the face box.
- **`text`**: one block per sign, title or caption, in reading order.
  - `string` is the ground truth: lines separated by `\n`, plain ASCII
    punctuation (straight quotes, `-` for dashes).
  - `rotation` is how far the text is turned counterclockwise from upright, in
    degrees: `90` reads bottom to top, `180` is upside down.
  - Only text that reads clearly at full size is annotated. Lettering that's
    cut off, hidden or strongly slanted is left out and mentioned in `notes`.
- **`flat_colors`** are the artwork's fill colors and **`ink_colors`** its line
  colors, as `#rrggbb`; a color can't be both. For scans they are cluster
  centers of the printed colors, not exact values.
- **`areas`** are boxes lying inside a gradient or a textured part of the
  image.

Code reads the manifest with `bench_manifest.load_directory(images_dir)` or
`bench_manifest.find_image(path)`; `ImageInfo.scaled_to((width, height))`
converts the annotations to an output size.

## Print scale

Paintability is physical: a brush needs a region a few millimeters wide, and a
number has to be legible on paper. So paintability thresholds are set in
millimeters and points, and converted to pixels for each output image by the
print-size model in `src/tessellatum/core/print_size.py`. The harness loads that
file from its own checkout (`bench_metrics.print_size`), not from the version
being measured, so every version is judged at the same scale, including
versions older than the model.

- **Paper:** A4, 210 × 297 mm, with 10 mm margins, leaving a 190 × 277 mm
  printable area. Print resolution 300 dpi.
- **Placement:** the page prints on a sheet of its own (the legend gets
  another), scaled to fill the printable area with its aspect ratio kept, on a
  landscape sheet if the image is wider than tall. Its physical size depends
  only on its shape, so the preview and the export of an image are judged at
  the same physical size.
- **Label size** is the font size (the em) in points. A digit is about 0.73 em
  tall in the default font.

| threshold | print size | 825 × 1100 px (3:4) | 1800 × 2400 px (3:4) | at 300 dpi |
|---|---|---|---|---|
| paintable width | ≥ 3 mm | 13 px | 28 px | 35 px |
| region number | ≥ 6 pt | 9.2 px | 20 px | 25 px |
| outline | ≈ 0.3 mm | 1.3 px | 2.8 px | 3.5 px |

`print_scale((width, height))` gives an output image's scale: `px_per_mm`,
`px_per_pt`, `dpi`, `printed_size_mm` and `landscape`, plus the conversions
`mm_to_px`, `pt_to_px`, `mm2_to_px`, `px_to_mm` and `px_to_pt`. Each case's
`case.json` records it under `print`.

The pipeline never upscales, so a small source prints at a low resolution:
`scene.png` (600 × 450 px) comes out at 60 dpi, where a 0.3 mm line is less
than a pixel wide.

## Quality metrics

Absolute metrics, per result. Fidelity is scored on the *finished painting*
(every region filled with its legend color) against the source image at output
size. Paintability is scored on the region map and the numbers, at print size
(see "Print scale" above):

| metric | meaning | better |
|---|---|---|
| ΔE00 mean / p95 | CIEDE2000 color error between painting and source | lower |
| SSIM | structural similarity of luma between painting and source | higher |
| labeled area | share of the page inside regions that carry a number | higher |
| unlabeled | regions without a number | lower (0) |
| slivers | share of the page a round brush 3 mm wide can't paint without crossing into another region | lower |
| labels < 6 pt | share of numbers printing smaller than 6 pt; `case.json` also records the smallest, as `min_label_pt` | lower (0) |
| compactness p10 / median | 4πA/P² over the regions: 1 for a disk, lower for stretched or ragged ones | higher |
| undersized | regions still below the difficulty's minimum size | lower (0) |
| regions, ink | region count and share of dark outline/number pixels | informational |

How the paintability metrics are defined:

- **Slivers.** The brush is the disk of pixels within half the paintable width
  of a pixel center. A pixel is paintable if the brush fits entirely inside the
  pixel's region somewhere that covers it: the region's morphological opening
  by that disk. The page edge counts as a boundary.
  - Thin parts of regions are slivers, and so are the corners a round brush
    can't reach: a square region loses a few pixels at each corner.
  - The brush is an odd number of pixels across (13 px for a 13.2 px width,
    15 px for 14 px), so a bar is judged to within a pixel of the width.
- **Unlabeled** regions are counted from the region map, so a region too small
  to get an outline counts too.
- **Label size** is each number's em size in points. Versions before 0.1.10
  don't report their numbers, so the harness rebuilds the sizes from the
  renderer's formula, which all of those versions share.
- **Compactness.** A is the region's pixel count. P, the boundary length
  including holes and the page edge, is estimated with the Cauchy–Crofton
  formula: from how often rows, columns and both diagonals of pixel centers
  cross the boundary.
  - Counting pixel edges instead would make diagonal boundaries √2 times too
    long.
  - A digital disk scores 0.94 at a radius of 10 px and 0.98 at 30 px. A
    square scores about 0.88 (exactly π/4 ≈ 0.79 in the continuous plane),
    whether upright or turned 45°.
  - Values are capped at 1, which the estimate exceeds for regions of a few
    pixels.

The paintability metrics have no tolerances yet, so they don't affect the
verdict.

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

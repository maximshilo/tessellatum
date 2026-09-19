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
  `detect_ink` (from 0.1.28), `quantize`, `build_regions`, `extract_regions`,
  `render_page`, `render_legend`. Keep those names if you restructure the pipeline, or
  update `PROBED_STAGES` in `bench_case.py`.
- Quality metrics read what the page is made of from the pipeline itself.
  `generate(..., collect_analysis=True)` returns it as `GeneratedPage.analysis`
  (`PageAnalysis` in `pipeline.py`):
  - the region map, each region's color and the palette, legend colors first;
  - the drawn regions, with their outlines and label points;
  - every number, with its font size and bounding box, and for a number written
    outside its region the leader line pointing into it;
  - the ink the lines (and, from 0.1.28, the printed ink) put on the page, 0
    solid and 255 bare paper, and the ink of the leader lines, the same way;
  - every line drawn, as a polyline;
  - whether the picture is line art, and the ink lines it was drawn with;
  - the ink the page prints (from 0.1.28: line art's own ink, printed solid,
    in no region) and the gray it prints in.

  The timed runs don't collect it, as in the app. One more run after them
  does, and its page must match theirs for the case to count as
  deterministic. Versions from before 0.1.10 have no analysis; the stage
  wrappers capture the region map, palette and regions for them instead.
  Versions from before 0.1.12 don't list their lines, so the harness rebuilds
  them as the outline of every drawn region, which is what those versions draw.
  `case.json` records which source was used under `scored_from` (`analysis`
  or `probe`).
- Cases exceeding `--timeout` (default 30 min) are killed and recorded as
  timeouts; speedups against them are reported as lower bounds.

Defaults: every image in `tests/sample_images/`, presets Easy/Medium/Hard, at
preview size (1100 px). `--presets Max` adds the most granular Custom setting
(the worst case for region handling); `--long-edge 1100 2400` adds export size;
`--threads N` caps worker threads to test scaling; `--category face text` runs
only the images in those categories (see the image manifest below).

`Max` is the finest setting of the version being measured, as its own
`difficulty.finest_params()` gives it: from 0.1.26, 40 colors, regions of at
least 30 mm² on paper, no smoothing. Older versions don't have that function,
and their sliders stopped at 40 colors, 0.0002 of the image and no smoothing,
which is what they get. So a comparison at `Max` compares each version's own
finest page. `case.json` records the settings a case ran with under `params`,
with the version's own field names.

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
      "exact_colors": true,
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
  - `box` fits the lines tightly: the text metrics split it into that many lines
    of equal height.
  - `rotation` is how far the text is turned counterclockwise from upright, in
    degrees: `0`, `90` (reads bottom to top), `180` (upside down) or `270`.
  - Only text that reads clearly at full size is annotated. Lettering that's
    cut off, hidden or strongly slanted is left out and mentioned in `notes`.
- **`flat_colors`** are the artwork's fill colors and **`ink_colors`** its line
  colors, as `#rrggbb`; a color can't be both. For scans they are cluster
  centers of the printed colors, not exact values.
- **`exact_colors`** is `true` where the flat and ink colors are the file's own
  pixel values, as in digital artwork (the two bold-line PNGs and `scene.png`:
  81–100% of their pixels are exactly one of them). Scans leave it out, which
  means `false`.
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
(every region filled with its legend color, and what the page prints kept as it
is: line art's printed ink in its gray, bare paper white) against the source
image at output size. Paintability is scored on the region map and the numbers, at print size
(see "Print scale" above). Line quality is scored on the lines drawn, the region
map and the source image, and the palette on the legend's colors. Line art, faces
and text are also scored against the image's manifest entry: on how the page keeps
the artwork's ink lines and flat colors, on how the painting matches inside the faces
and whether their features survive, and on whether OCR still reads the text:

| metric | meaning | better |
|---|---|---|
| ΔE00 mean / p95 | CIEDE2000 color error between painting and source | lower |
| SSIM | structural similarity of luma between painting and source | higher |
| labeled area | share of the area to paint (the page less what it prints) inside regions that carry a number | higher |
| unlabeled | regions without a number | lower (0) |
| slivers | share of the page a round brush 3 mm wide can't paint without crossing into another region | lower |
| labels < 6 pt | share of numbers printing smaller than 6 pt; `case.json` also records the smallest, as `min_label_pt` | lower (0) |
| labels on lines | numbers with any ink of a line in their box: of the page's lines, or of the leader lines that point a number into its region | lower (0) |
| overlapping labels | numbers whose box overlaps another number's | lower (0) |
| leaders | numbers written outside their region, with a leader pointing in | informational |
| compactness p10 / median | 4πA/P² over the regions: 1 for a disk, lower for stretched or ragged ones | higher |
| lines per boundary | lines drawn along each boundary between two regions; `case.json` also records the shares with two or more (`doubled_boundary_fraction`) and with none (`undrawn_boundary_fraction`) | 1 |
| lines per boundary (clear) | the same, over the boundary clear of junctions, where the count means what it says; `case.json` also records how much of the boundary that is, as `clear_boundary_fraction` | 1 |
| unenclosed | share of the page in a white area covering more than one region: where two regions' paint would run together; `case.json` also records how many such areas there are (`unenclosed_areas`), and how many regions the lines pinch into more than one white piece (`split_regions`) | lower (0) |
| same-color boundary | share of the boundary length that lies between two regions of the same color | lower (0) |
| jaggedness | length of the drawn lines over their length with wiggles under 0.5 mm smoothed away | lower (1) |
| edge F1 | how well region boundaries and the source's edges line up, within 0.5 mm; `case.json` also records `edge_precision` and `edge_recall` | higher |
| palette min ΔE00 | smallest CIEDE2000 color difference between two colors on the legend | higher |
| color pairs < 10 ΔE00 | pairs of legend colors that differ by less than 10 ΔE00 | lower (0) |
| ink line F1 | how well drawn lines, and the centerlines of the ink the page prints, run down the middle of the artwork's ink lines, within 0.5 mm; `case.json` also records `ink_line_precision` and `ink_line_recall` | higher |
| ink printed recall / precision | on line art, how much of the artwork's ink lines the page prints, and how much of what it prints is the artwork's ink, each pixel within 0.5 mm (from 0.1.28); `case.json` also records `ink_print_f1` | higher |
| tubes | regions at least half made of the artwork's ink lines | lower (0) |
| ink in shapes < 5 mm | share of the ink lines lying in parts of regions narrower than 5 mm: ink to paint instead of print | lower (0) |
| flat colors ΔE00 | mean CIEDE2000 from each of the artwork's flat colors to the nearest legend color; `case.json` also records the largest, as `flat_color_de00_max` | lower |
| ink found | share of the page the pipeline takes for the artwork's ink lines (from 0.1.27; printed from 0.1.28) | informational |
| ink found recall / precision | on line art, how much of the artwork's ink lines the pipeline found, and how much of what it found is on them, each within 0.5 mm; `case.json` also records `ink_found_f1`, and whether the manifest's colors are exact as `ink_reference_exact` | higher |
| stray ink | the share of the page found to be ink lines on a picture that isn't line art | lower (0) |
| face ΔE00, face SSIM | ΔE00 mean and SSIM inside the image's face boxes | lower, higher |
| features lost | annotated eyes, noses and mouths the page no longer shows, as lines along their edges or as a region of their own; `case.json` also records the mean share of their edges drawn, as `feature_edge_recall`, and each feature's scores under `face_features` | lower (0) |
| labels on features | numbers overlapping a feature box | lower (0) |
| text CER source / page / painting | character error rate of OCR inside the image's text boxes, against their annotated text: on the source (how much OCR reads there at all), the page and the painting; `case.json` records the OCR engine under `ocr`, and what it read in each block under `text_blocks` | lower (0) |
| labels on text | numbers overlapping a text box | lower (0) |
| undersized | regions still below the difficulty's minimum size that another region touches, so that they had a neighbor to merge into | lower (0) |
| colors, regions, ink | how many colors the legend lists, which is fewer than the difficulty asked for wherever colors had to be merged to keep the palette apart; the region count; and how much of the page the lines and numbers cover in ink (a pixel counts by how far it is from bare paper) | informational |

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
- **Labels on lines** reads the line layer (`PageAnalysis.outlines`) and, from
  0.1.25, the leaders' layer (`PageAnalysis.leaders`), as `unenclosed` reads
  them: any ink at all is a line, since the pale edge of an anti-aliased line
  is the line. A box covers every pixel it reaches into, so a number centred
  on a half pixel, as before 0.1.25, is judged by all the pixels its ink can
  touch. A number's own leader counts too: it should end short of the number.
  Versions before 0.1.10 report no line layer, and get no value.
- **Overlapping labels** compares the numbers' boxes pairwise; boxes that only
  touch don't overlap.
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

How the line metrics are defined:

- **Lines per boundary** reads the lines the renderer reports drawing
  (`PageAnalysis.strokes`), not the printed pixels. Two outlines drawn side by
  side merge into one thick band, which the pixels can't tell apart from a
  single thick line.
  - Boundaries are measured per pixel edge between two regions; the page edge
    doesn't count. A line runs along a pixel edge if its rasterized centerline
    passes through or next to (8-neighborhood) either of the edge's two pixels.
    It counts once however often it passes.
  - A renderer that outlines every region on its own scores 2 on a boundary
    between two regions. Around a region lying inside another it scores 1,
    because outlines don't trace holes.
  - Where a region is only 1–2 px wide, the lines on its two sides are within
    reach of each other, so even one line per boundary scores 2 or more there.
- **Lines per boundary (clear)** is the same count over the boundary more than
  2.5 px from a junction. Within that reach the count cannot tell one
  boundary's line from the lines of the boundaries that end at the junction, so
  it counts all of them however few a page draws.
  - The clearance is in pixels, not on paper, because it corrects for how a line
    is rounded onto the pixel grid: half a pixel from a crack to the center of
    the pixel beside it, one pixel (√2 at the corners) for the neighborhood the
    count reaches into, and half a pixel of rounding.
  - It is not a way round the doubling the metric is there to catch. Over the
    96 baseline cases, a renderer that outlines every region scores 1.183–2.000
    on it against 1.187–2.099 plain: the cut moves it by 0.029 on average and
    0.114 at worst, because doubling is everywhere on such a page and not only
    near junctions. On pages drawn one line per boundary the same cut is worth
    0.080 on average and 0.323 at worst. What it removes is the part no renderer
    can do anything about.
    (Those pages score below 2 for a separate reason the plain count shares:
    outlines don't trace holes, so a region lying inside another gets one line,
    which is why the line art at Easy scores 1.18–1.43 either way.)
  - `clear_boundary_fraction`, how much of the boundary is clear, falls as a
    page gets more regions: 95.5% on `scene.png` at Hard (7 regions), 75.1% on
    the lion at Max (778). That is why the plain count rises with region count
    however the page is drawn.
- **Unenclosed** area reads the page's line layer (`PageAnalysis.outlines`), not
  the lines' geometry: what is filled is pixels. Its white is split into
  4-connected areas, and an area covering more than one region is one the paint
  can run out of — a gap in the lines. Pixels in no region count as one more
  region, so a region is not allowed to leak into them either.
  - It is the flood-fill test in plain terms: fill the white from any point and
    you should never reach out of the region that point is in.
  - White is bare paper: any ink at all is a line, however faint. What the
    metric looks for is a gap where no line was drawn, and the pale edge of an
    anti-aliased line is the line, not a gap. On a page of solid black lines
    both readings give the same answer, so this is how every result set from
    0.1.10 on is scored.
  - `split_regions` counts the opposite fault, a region whose white the lines
    pinch into more than one piece where it is narrow. A line takes ink from
    each side of its crack, so a region much narrower than the line closes up.
    Those places are slivers already, so this only informs.
  - Versions before 0.1.10 report no line layer, and get no value.
- **Same-color boundary** estimates lengths as compactness does. Merging a small
  region into a neighbor can leave two regions of one color touching.
- **Jaggedness** smooths each drawn line along its length by a Gaussian of
  0.5 mm (2.0–2.4 px at preview size).
  - Lines are cut 0.5 mm short of junctions, where three regions meet or two
    regions meet the page edge, and where they run along the page edge. So the
    corners where lines meet don't count.
  - Each piece keeps its ends fixed. A closed line that meets no junction is
    smoothed all the way round.
  - The score is the pieces' total length over their total smoothed length, so
    longer lines weigh more.
  - At 2 px of smoothing, a straight line scores 1 and a staircase of 1 px steps
    √2 (1.414). A staircase of 20 px steps scores 1.063, a lone right angle with
    legs of about 30 and 40 px 1.018, and a circle of radius 50 px 1.0008.
- **Edge F1** compares the region boundaries, as `boundary_map` marks them, with
  the source's edges.
  - The edges come from Canny on the source at output size, in CIE Lab, after a
    Gaussian blur of 0.5 mm. Its thresholds are clean color steps of 5 and 10
    Lab units (L runs 0–100) in whichever channel changes most: a lightness step
    of 11 is an edge, and one of 9 isn't.
  - Precision is the share of boundary pixels within 0.5 mm of an edge: lines on
    real edges. Recall is the share of edge pixels within 0.5 mm of a boundary:
    real edges that got a line.
  - Softly shaded images have few edges, so the boundaries between the bands of
    a gradient count as off-edge.

How the palette metrics are defined:

- **Palette min ΔE00** and **color pairs** compare every two colors on the
  legend: the colors of the drawn regions, which the painter mixes. A color
  k-means found that no drawn region has isn't on the legend and doesn't count.
  Versions before 0.1.10 don't report their legend, so the harness rebuilds it
  from the drawn regions' colors.
- Colors are compared with CIEDE2000, after converting sRGB to CIE Lab (D65)
  exactly as the standards define it. Fidelity uses OpenCV's conversion, whose
  lookup tables put a color up to about 0.5 ΔE00 from its exact value; that
  averages out over an image, but not over a few colors near a threshold. 10 ΔE00
  is the clear margin `QUALITY_BENCHMARKS.md` asks for.
- Every pair counts: five near-identical browns make 10 close pairs.
- Since 0.1.24 the pipeline keeps that margin itself, with its own copy of this
  math (`tessellatum.core.color`): the harness has to score versions that
  predate it, so the two cannot be one file. `tests/test_color.py` pins them to
  the same values, because a page is judged on the margin measured here.
- **Colors** is the legend's length, as the pipeline reports it
  (`GeneratedPage.num_colors_used`). It can be well below the difficulty's color
  count on an image whose colors crowd together; see D-031 in the plan.
- Whether the colors can be printed isn't checked. They are 8-bit sRGB, so they
  always display; which of them paper and ink can reproduce depends on the
  printer, and checking that needs its color profile.

How the line-art metrics are defined:

- They read the image's manifest entry: the ink metrics need its `ink_colors` and
  `flat_colors` (cartoons), flat colors ΔE00 its `flat_colors` (cartoons and flat
  art). Other images get no value.
- **Ink lines.** Every pixel of the source at output size takes the nearest of
  the manifest's flat and ink colors (CIEDE2000, exact Lab).
  - Anti-aliasing between two ink colors is ink too: mixes of every two ink
    colors, in sRGB steps of 1/8, count as ink colors, even where a flat color
    lies nearer.
  - Ink lines are the ink narrower than 5 mm. Parts of it a 5 mm disk fits into
    are fills drawn in an ink color.
  - The bold-line cartoons' outlines are 2–4 mm wide at print size, and their
    finer lines under 1 mm. On the two scans the manifest's colors are cluster
    centers of printed colors, so print texture and hatching turn into specks
    and short strokes of ink.
- **Ink line F1** compares the ink lines' centerlines (Zhang–Suen thinning) with
  the centers of the lines drawn, and with the centerlines of the ink the page
  prints (from 0.1.28), which is a line of its own.
  - Recall is the share of centerline pixels within 0.5 mm of a drawn line.
  - Precision is the share of drawn-line pixels on or within 0.5 mm of the ink
    lines that lie within 0.5 mm of a centerline. Lines away from the ink, such
    as those between two fills, don't count: a good page draws those too, and
    edge F1 judges them.
  - A page that draws its ink lines where they are scores 1. Before 0.1.28 the
    renderer made a bold ink line a region of its own, a tube, and outlined it
    along both edges, which lie more than 0.5 mm from its middle once it is wider
    than about 1 mm.
  - It no longer carries the plan's target (D-036). Once a page prints the ink
    itself, it compares two skeletons of wide shapes, which branch differently for
    masks that agree pixel for pixel, and the reference's own specks, pinholes
    and pleats put branches and loops into its skeleton: a page printing the ink
    it finds, 97% of the artwork's by pixel, scores 0.58–0.84 on the bold-line
    girl. The printed-ink metrics below carry the target instead.
- **Tubes** are regions at least half of whose pixels lie on ink lines.
- **Ink in shapes** also counts ink lines that became thin parts of bigger
  regions, as when they merge with a fill of the same color: the share of
  ink-line pixels in parts of regions a 5 mm disk doesn't fit into (the opening
  used for slivers). Ink lines along the edge of a wide region, or left out of
  every region, count for neither.
- **Flat colors ΔE00** takes each manifest flat color's CIEDE2000 difference to
  the nearest legend color (exact Lab), and averages them. At Easy the legend
  can have fewer colors than the artwork, so some flat colors have no close
  match.

How the printed-ink metrics are defined:

- From 0.1.28 a line-art page prints its ink (`PageAnalysis.printed_ink`, in the
  gray `.ink_gray`): the ink lines it finds, and patches in the ink's own color
  it takes to be the ink running wider. None of it is in a region. Older
  versions get no value.
- **Ink printed recall** is the share of the artwork's ink-line pixels
  (`source_ink`) within 0.5 mm of a pixel printed: are its lines printed?
- **Ink printed precision** is the share of the pixels printed within 0.5 mm of
  the artwork's ink, in its ink colors however wide (`source_ink_colored`): is
  what the page prints the artwork's ink? A black blob where a bold outline runs
  wider than 5 mm is ink, not a line, and printing it is right.
- As for found ink, precision is only a target where the manifest's colors are
  exact; on the scans the manifest's colors miss much of the line work.

How the found-ink metrics are defined:

- From 0.1.27 the pipeline decides whether a picture is line art and, if it is,
  finds its ink lines without the manifest (`src/tessellatum/core/ink.py`), and
  reports both in its analysis payload (`PageAnalysis.line_art`, `.ink_lines`).
  `case.json` records the decision and the two measures behind it under
  `line_art`. Older versions get no value.
- **Ink found recall** and **precision** compare the pipeline's mask with the
  line-art metrics' own ink lines (`source_ink`), pixel by pixel: recall is the
  share of the artwork's ink-line pixels within 0.5 mm of a pixel found,
  precision the share of the pixels found within 0.5 mm of the artwork's. The
  tolerance forgives a line found a pixel wider or narrower, where an
  anti-aliased edge could go either way.
- Precision is only a target where the manifest's colors are exact
  (`exact_colors`). On a scan they are cluster centers of printed colors, and
  `source_ink` misses much of the line work: on the postcard at preview size it
  finds 4.3% of the page to be ink lines, the pipeline 18%, and the difference
  is wing ribs, hands and outlines drawn in ink the manifest's colors don't
  reach.
- **Stray ink** is ink found on an image whose manifest has no ink colors,
  where every pixel found is a mistake.

How the face metrics are defined:

- They read the image's manifest entry: its `faces`, each with a box and the boxes
  of its `features`, scaled to output size. Images without faces get no value.
- **Face ΔE00** and **face SSIM** average fidelity's ΔE00 and SSIM over the pixels
  inside any face box. A pixel's SSIM comes from the window centered on it, which
  reaches 5 px past the box.
- **Features lost.** A feature survives if the page still shows it, as lines or as a
  shape:
  - lines: at least 30% of the source's edges inside its box (found as for edge F1)
    lie within 0.5 mm of a drawn line. This share is the feature's edge recall;
  - a shape: a drawn region lying at least half inside the box covers at least a
    quarter of it. Edges in a textured box include the texture's: an eye outlined
    as a region of its own can still leave most of the fur's edges around it
    undrawn.
  - On today's pages of the face images, the features that are gone score an edge
    recall of at most 0.182, with no region of their own; those still there score
    at least 0.411, or have a region covering at least 27.7% of their box.
- **Labels on features** counts the numbers whose text box overlaps a feature box. A
  number across two features counts once. Versions before 0.1.10 don't report where
  their numbers are, so the harness rebuilds each box as the renderer placed it: the
  number's text box in the default font, centered on the region's label point and
  kept on the page.

How the text metrics are defined:

- They read the image's manifest entry: its `text` blocks, scaled to output size.
  Images without text get no value.
- **OCR** is RapidOCR on ONNX Runtime, with the PP-OCRv6 recognition model its
  wheel ships (Apache-2.0). Both install with the `dev` extra, and nothing is
  downloaded at run time. `rapidocr` is pinned, because its models are the
  yardstick. Without them, the character error rates are blank and labels on text
  are still counted.
- **Reading a block.** OCR reads a block line by line, as many lines as its
  annotated text has, and recognizes each line without looking for text first:
  - the box is turned upright by its `rotation`, and scaled so that each line is
    48 px tall, whatever the output size;
  - it gets a margin half a line wide, in the median color of its border;
  - it is cut into bands of equal height, one per line, each reaching 0.15 of a
    line into its neighbors;
  - what OCR reads on the lines is joined with spaces.

  So a text box must fit its lines tightly. RapidOCR's own text detection missed
  text in boxes this tight, and most lines 7 px tall.
- **Character error rate** is the edit distance from what OCR read to the annotated
  text (the fewest insertions, deletions and substitutions), summed over the blocks
  and divided by the annotated text's length. 0 means every block reads exactly, 1
  that nothing does; reading extra characters can take it past 1.
  - Before comparing, both texts get Unicode's compatibility forms (NFKC: a
    full-width comma becomes a comma), ASCII quotes and dashes for typographic
    ones, and one space for every run of whitespace, line breaks included.
  - Case and punctuation count.
- **On the source**, OCR doesn't read everything either. At preview size it scores
  0.26 on Times Square, whose smallest signs have lines 7–8 px tall, 0.04 on the
  postcard, and 0.11 on the comics page, whose smallest captions have lines 7 px
  tall. A page can't be expected to read better.
- **Labels on text** counts the numbers whose text box overlaps a text box, as
  labels on features does.

How the report judges these metrics, with their targets and tolerances, is under
"Scorecard and verdict" below.

Agreement metrics, candidate vs. reference (computed by `compare` from the
saved `page.png`, `painted.png` and `regions.npz` of each case):

| metric | meaning |
|---|---|
| page px differing | share of page pixels that differ; 0 means a bit-identical page |
| same partition | whether both produced exactly the same regions |
| boundary F1 | how well region outlines line up, within 2 px (1 = same) |
| painting ΔE00 | mean color difference between the two finished paintings |

## Scorecard and verdict

`compare` judges quality in two separate ways:

- **targets:** what every page should reach, whatever the reference scores;
- **regressions:** where the candidate got worse than the reference by more than
  a metric's tolerance.

### Scorecard

A page does four jobs (`QUALITY_BENCHMARKS.md`): it resembles the image, is
paintable, reads as a clean drawing, and has a palette that works. For each
result set, the scorecard averages each job's metrics over each image category
(an image counts in every category it has) and over all cases. It also counts
the cases meeting all of the job's targets that apply to them. Against a
reference, it averages both sets over the cases both completed, and marks the
means that regressed in bold.

A metric has two tolerances, one for each size it is measured at: **preview**, a
page of 1100 px, and **export**, a page at the image's own size (see
"Regressions" below).

| metric | job | σ preview | σ export | target |
|---|---|---|---|---|
| ΔE00 mean | resembles | 6.9% of the value | 7.3% of the value | – |
| ΔE00 p95 | resembles | 7.3% of the value | 12% of the value | – |
| SSIM | resembles | 0.0066 | 0.0070 | – |
| face ΔE00 | resembles | 7.8% of the value | 8.2% of the value | – |
| face SSIM | resembles | 0.016 | 0.038 | – |
| features lost | resembles | 0.35 | 0.29 | 0 |
| text CER painting | resembles | 0.0096 | 0.017 | – |
| labeled area | paintable | 2.4 points | 0.64 points | – |
| unlabeled | paintable | 69 | 18 | 0 |
| slivers | paintable | 1.5 points | 1.4 points | ≤ 1% |
| labels < 6 pt | paintable | 2.1 points | 2.2 points | 0 |
| labels on lines | paintable | 0 | 0 | 0 |
| overlapping labels | paintable | 0 | 0 | 0 |
| compactness p10 | paintable | 0.023 | 0.020 | – |
| compactness median | paintable | 0.036 | 0.029 | – |
| undersized | paintable | 0 | 0 | – |
| lines per boundary | clean drawing | 0.10 | 0.048 | – |
| lines per boundary (clear) | clean drawing | 0.10 | 0.048 | 1 ± 0.05 |
| unenclosed | clean drawing | 0 | 0 | 0 |
| same-color boundary | clean drawing | 2.8 points | 1.4 points | 0 |
| jaggedness | clean drawing | 0.0085 | 0.011 | ≤ 1.02 |
| edge F1 | clean drawing | 0.023 | 0.020 | – |
| ink line F1 | clean drawing | 0.021 | 0.023 | – |
| ink printed recall | clean drawing | 0.0015 | 0.0034 | ≥ 0.95 |
| ink printed precision | clean drawing | 0.0021 | 0.0038 | ≥ 0.95 on exact colors |
| tubes | clean drawing | 2.6 | 1.9 | 0 |
| ink in shapes < 5 mm | clean drawing | 3.1 points | 2.8 points | – |
| labels on features | clean drawing | 1.6 | 3.6 | – |
| text CER page | clean drawing | 0.017 | 0.018 | ≤ text CER source + 0.1 |
| labels on text | clean drawing | 1.3 | 2.7 | 0 |
| palette min ΔE00 | palette | 1.3 | 0.93 | ≥ 10 |
| color pairs < 10 ΔE00 | palette | 3.2 | 3.1 | – |
| flat colors ΔE00 | palette | 1.2 | 0.81 | – |

Colors, regions, ink and text CER source only inform. The per-case tables add the
number of targets each case misses.

The printed-ink metrics' tolerances come from 48 pairs at each size, the four line-art
images at every preset, measured on 0.1.28. On the same pages ink line F1 moved by 0.0051
and 0.019, under the 0.021 and 0.023 it was given on stroke pages, which it keeps.

Three more metrics score the ink lines the pipeline finds (from 0.1.27; ink found itself
only informs). They score the finding itself rather than a job of the page -- the page's
printed ink has its own two metrics in the clean drawing job -- so they sit outside the
scorecard, and their targets count in the verdict's target misses:

| metric | σ preview | σ export | target |
|---|---|---|---|
| ink found recall | 0.00042 | 0.0046 | ≥ 0.95 |
| ink found precision | 0.0023 | 0.0047 | ≥ 0.95 on exact colors |
| stray ink | 0 | 0 | 0 |

Finding them doesn't depend on the difficulty, so their tolerances come from 12 pairs at
each size, the four line-art images at Easy. Stray ink measured 0 over 21 and 24 pairs: a
picture that isn't line art is decided once, so it has none at any size.

### Targets

A target applies to a case where its metric has a value: the printed ink and tubes on
line art, features lost on faces, and the text targets on images with text. They
spell out the four jobs:
- **paintable:** no slivers, a number on every region, and every number legible
  at print size, with no line and no other number drawn through it;
- **clean drawing:**
  - one smooth line per boundary, and no line between neighbors of the same color;
  - every region enclosed, so no two regions' paint can run together;
  - line art's ink lines printed as the page's lines, with no tubes: at least 95% of
    them printed, and at least 95% of what is printed the artwork's ink where the
    manifest's colors are exact;
  - text still readable, with no numbers on it;
- **resembles:** faces keep their eyes, noses and mouths;
- **palette:** every two colors at least 10 ΔE00 apart.

Three of the targets need explaining:

- **Lines per boundary** carries the target on the count clear of junctions, and
  may be 1 ± 0.05 there, not exactly 1. Near a point where three regions meet, a
  line lies within reach of its neighbors' boundaries too, and no renderer can
  change that: the plain count reaches 1.31 on a page of 778 regions drawn
  strictly once each, against 1.04 on a page of 7 (D-028). The clear count reads
  1.000–1.001 on the same pages, and still reads 2 for a renderer that outlines
  every region, so the doubling it is there to catch is still caught. The plain
  count stays in the report, and both are judged for regressions. Too few lines
  miss the target as much as too many.
- **Jaggedness ≤ 1.02** is met today by the bold-line cartoons (1.007–1.016) and
  `scene.png` (1.002–1.008), whose lines follow smooth shapes. Photos score
  1.05–1.25.
- **Text CER page** counts from the source's, because OCR doesn't read all of the
  source either (0.04–0.26 at preview size, see the text metrics).

The baseline confirmed all three (T1.8): line art scores 1.007–1.017 on jaggedness
at both sizes, a renderer drawing each boundary once scores 1.017–1.047 on the
pages without slivers, and OCR reads the sources at 0.04–0.26 at preview size and
0.05–0.11 at export size.

### Regressions

A metric's tolerance σ is how much one case's value typically changes between
pages that should be equally good. It is the root mean square change between two
sizes of the same image and preset, over the 12 benchmark images at Easy, Medium,
Hard and Max. For ΔE00 mean, ΔE00 p95 and face ΔE00, which range widely across
images, σ is a share of the value.

Each case is judged at its own size, because a page at an image's own size keeps
the source's detail and moves more between renders than a preview does: face SSIM
by 0.038 rather than 0.016, labels on features by 3.6 rather than 1.6.

- **σ preview** comes from 1099, 1100 and 1101 px: 132 pairs, and 36–48 for the
  metrics that need annotations.
- **σ export** comes from 1, 2 and 3 px below each image's own long edge: 144
  pairs, and 48–60. A case counts as an export where its page is within 1% of its
  image's own long edge (`source_size` in `case.json`), or, for result sets from
  before that was recorded, where the size asked for is at least 1375 px.

- **A regression** is a category, or all cases together, whose mean change
  against the reference is worse than 3 standard errors of the mean:
  3 √(Σσ²)/n over the n cases with a value in both sets, which is 3 σ/√n where
  they are all of one size.
  - Chance changes of single cases cancel out in a mean, so a mean over more cases
    may move less: ΔE00 mean may rise 2.9% over 48 cases, 7.0% over 8 and 19.8%
    for one.
  - On the pages rendered 1 px apart, no mean of any metric crossed that line.
- **Cases to look at:** single cases worse by more than 3 σ. They aren't
  regressions: 7 of the 48 cases rendered 1 px apart had one.
- **Direction:** lines per boundary are judged by their distance from 1.
  Undersized regions have σ = 0, so their mean may not rise at all.

The verdict lists regressions, then target misses: each target's miss count on the
reference and the candidate, and the cases that met a target on the reference and
miss it now. Region-count changes over 15% are noted in the per-case tables, but
aren't failures: region count is a difficulty trait, not a quality score.

Lines per boundary (clear) has not been through `noise` yet: it takes the plain
count's tolerances until the next run measures its own (D-028). Unenclosed area
takes 0, as undersized regions do — a page whose lines close has none of it, so
any at all is a regression — and so do labels on lines and overlapping labels,
for the same reason: a page that keeps its numbers clear has none.

`compare --tol METRIC=SIGMA` replaces a metric's tolerance at both sizes, by its
`case.json` key (repeatable), e.g. for a change that trades one metric for another
on purpose. The verdict names the tolerances replaced.

To measure the tolerances again, e.g. after adding or changing a metric, render
the images at three sizes and update `sigma` in `bench_report.METRICS`. `noise`
pairs only sizes up to 1% apart, so result sets that also hold other sizes, such
as exports, can be passed to it as well:

```
.venv\Scripts\python benchmarks\bench.py run WORKTREE=sizes --presets Easy Medium Hard Max --long-edge 1099 1100 1101 --repeats 1 --warmup 0
.venv\Scripts\python benchmarks\bench.py noise sizes
```

`noise` pairs the pages' own sizes, not the sizes asked for, and counts a size
once however many result sets hold it. Pages at an image's own size are left out:
the pipeline never upscales, so every size asked for at or above that renders the
one page, and resizing changes a page by more than chance does. Between an image's
own size and 1 px less, SSIM moves by 0.05 and ΔE00 mean by 17%; between two sizes
below it, by 0.008 and 4.9%.

Every benchmark image is at most 2048 px, so an "export" at 2400 px is the image
at its own size, and `scene.png` (600 px) renders the same page at every size from
600 px up. To measure σ at export size, render each image 1, 2 and 3 px below its
own long edge — one `run --images ... --append` per size, since each image has its
own — and pass that result set alone:

```
.venv\Scripts\python benchmarks\bench.py run WORKTREE=export-sizes --images tests\sample_images\l-photo-lion.jpg --presets Easy Medium Hard Max --long-edge 2044 2045 2046 --repeats 1 --warmup 0 --append
.venv\Scripts\python benchmarks\bench.py noise export-sizes
```

Timings are wall-clock: close heavy apps while benchmarking, and only compare
result sets recorded on the same machine (the report warns when they're not).

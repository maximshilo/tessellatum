# Quality report: v0.1.45 against the baseline

2026-10-04. How today's pages compare with the pages Tessellatum drew before any of the quality work, scored
against the spec in [`QUALITY_BENCHMARKS.md`](QUALITY_BENCHMARKS.md) by the harness in [`README.md`](README.md).

**In short:**
- Every target the spec sets is met on every case but four, all on the cat photo, whose mouth is lost. The
  baseline missed 719 (case, target) pairs; v0.1.45 misses 4.
- Pages are paintable and clean where they were neither: slivers 12.9% → 0.3% of the page, 118 → 0 unnumbered
  regions, lines per boundary 1.80 → 1.00, and white areas leaking into each other on 52% → 0% of the page.
- The palette's colors are now at least 10 ΔE00 apart (6.0 → 11.7 on average).
- Line art prints its own ink and text prints as it looks: tubes 7.9 → 0 a page, OCR on the page 0.98 → 0.12.
- It costs fidelity on photographs and faces, by design. A brush can't paint detail narrower than 3 mm, and two
  colors closer than 10 ΔE00 can't be told apart, so the painting gives both up.
- Previews take about 2.2 times as long: 329 → 738 ms median.

## What was compared

| | version | commit | what it is |
|---|---|---|---|
| **baseline** | 0.1.17 | `36d75ed` | the last version before any change to the pages, which the quality work was measured against (first recorded on 2026-09-15) |
| **now** | 0.1.45 | `4fe9c44` | today's `main`; 0.1.46 changes only documentation |

- **One session.** Both versions ran on the same machine in one session, under today's harness, so every metric
  added since the baseline scores it too, and their timings compare.
- **The benchmark set.** 12 images at Easy, Medium, Hard and Max, each at preview size (1100 px) and at its own
  size (export): 96 cases a version. Timings are the median of 3 runs after 1 warm-up.
- **Checks.** All 96 cases on each side ran, deterministically, and were scored from the pipeline's analysis. The
  baseline drew the very pages it drew on 2026-09-15 (96/96 identical). Today's pages are those benchmarked for
  0.1.45 (96/96 identical, all 79 quality fields equal).
- **Max** is each version's own finest setting. For 0.1.17 that is 40 colors and regions down to 0.0002 of the
  image. For 0.1.45 it is 40 colors and regions down to 30 mm² on paper. So the region counts at Max differ by design.

To reproduce (about half an hour):

```
.venv\Scripts\python benchmarks\bench.py run 36d75ed=baseline 4fe9c44=now --presets Easy Medium Hard Max --long-edge 1100 2400
.venv\Scripts\python benchmarks\bench.py compare baseline now
```

## Targets

Cases missing each target, out of the cases it applies to.

| target | applies to | baseline | now |
|---|---|---|---|
| slivers ≤ 1% | 96 | 88 | **0** |
| gradient slivers ≤ 1% | 16 (images with gradients) | 8 | **0** |
| unlabeled = 0 | 96 | 55 | **0** |
| labels < 6 pt = 0 | 96 | 35 | **0** |
| labels on lines = 0 | 96 | 24 | **0** |
| overlapping labels = 0 | 96 | 0 | **0** |
| lines per boundary (clear) 1 ± 0.05 | 96 | 96 | **0** |
| unenclosed = 0 | 96 | 96 | **0** |
| same-color boundary = 0 | 96 | 76 | **0** |
| jaggedness ≤ 1.02 | 96 | 72 | **0** |
| palette min ΔE00 ≥ 10 | 96 | 82 | **0** |
| ink printed recall ≥ 0.95 | 32 (line art) | no ink printed | **0** |
| ink printed precision ≥ 0.95 | 16 (line art, exact colors) | no ink printed | **0** |
| tubes = 0 | 32 (line art) | 23 | **0** |
| flat colors ΔE00 ≤ best + 2.5 | 24 (exact colors) | 5 | **0** |
| features lost = 0 | 32 (faces) | 14 | **4** |
| text CER page ≤ source + 0.1 | 24 (text) | 24 | **0** |
| labels on text = 0 | 24 (text) | 21 | **0** |

What the pipeline finds, outside the four jobs, meets its targets on every case: ink found (recall ≥ 0.95 on 32,
precision ≥ 0.95 on 16, no stray ink on 64), faces found (recall 1 on 32, no stray face on 96), and text found
(recall ≥ 0.9 on 24, no stray line on 72). The baseline found none of these things.

**The four cases that still lose a feature** are all on the cat photo: its mouth at Easy (both sizes) and at Medium and Hard
(preview), and its nose too at Easy (preview). The mouth is a faint line in pale fur. At Medium and Hard the page
draws 22% and 27% of its edges, just under the 30% a feature needs, where the baseline drew enough of them. In all,
5 of the 128 features scored are lost, against 17 at the baseline.

## Cases meeting every target of a job

| category | images | cases | resembles | paintable | clean drawing | palette |
|---|---|---|---|---|---|---|
| photo | 6 | 48 | 11/16 → 12/16 | 0/48 → 48/48 | 0/48 → 48/48 | 8/48 → 48/48 |
| cartoon | 4 | 32 | 2/8 → 8/8 | 0/32 → 32/32 | 0/32 → 32/32 | 2/32 → 32/32 |
| flat | 1 | 8 | – | 4/8 → 8/8 | 0/8 → 8/8 | 2/8 → 8/8 |
| face | 4 | 32 | 18/32 → 28/32 | 0/32 → 32/32 | 0/32 → 32/32 | 2/32 → 32/32 |
| text | 3 | 24 | – | 0/24 → 24/24 | 0/24 → 24/24 | 2/24 → 24/24 |
| gradient | 2 | 16 | – | 0/16 → 16/16 | 0/16 → 16/16 | 4/16 → 16/16 |
| texture | 2 | 16 | 6/8 → 8/8 | 0/16 → 16/16 | 0/16 → 16/16 | 1/16 → 16/16 |
| **all** | 12 | 96 | 18/32 → 28/32 | 4/96 → 96/96 | 0/96 → 96/96 | 12/96 → 96/96 |

Resembles has one target, features lost, so it applies only to faces. An image counts in every category it has.

## The four jobs in numbers

Means over all 96 cases, baseline → now.

**Resembles**

| metric | baseline → now |
|---|---|
| ΔE00 mean | 7.64 → 7.27 (but +11.3% a case on average; see "What fidelity cost") |
| ΔE00 p95 | 26.6 → 23.3 |
| SSIM | 0.656 → 0.688 |
| face ΔE00 / face SSIM (32 cases) | 7.95 → 8.18 / 0.568 → 0.529 |
| subject ΔE00 / subject SSIM (40 cases) | 9.87 → 10.32 / 0.458 → 0.417 |
| features lost a case (32 cases) | 0.53 → 0.16 |
| text CER painting (24 cases) | 1.00 → 0.29 |

**Paintable**

| metric | baseline → now |
|---|---|
| slivers | 12.9% → 0.3% (worst case 52.7% → 0.8%) |
| gradient slivers (16 cases) | 3.7% → 0.1% |
| unlabeled regions | 118 → 0 |
| labeled area | 94.8% → 100.0% |
| labels < 6 pt | 13.7% → 0% (smallest 3.1 → 6.1 pt) |
| labels on lines | 0.9 → 0 |
| compactness p10 / median | 0.11 → 0.28 / 0.29 → 0.58 |

**Clean drawing**

| metric | baseline → now |
|---|---|
| lines per boundary (clear) | 1.80 → 1.00 (now 1.000 on every case) |
| lines per boundary | 1.80 → 1.08 |
| unenclosed | 52.1% → 0.0% |
| same-color boundary | 5.0% → 0.0% |
| jaggedness | 1.124 → 1.008 (now 1.001-1.012; up to 1.374 at the baseline) |
| edge F1 | 0.50 → 0.54 (photos 0.38 → 0.28; line art 0.61 → 0.86) |
| tubes (32 line-art cases) | 7.9 → 0 |
| ink in shapes < 5 mm (32 line-art cases) | 58.4% → 0.9% |
| ink line F1 (32 line-art cases) | 0.41 → 0.87 |
| ink printed recall / precision (32 line-art cases) | – → 0.984 / 0.905 (precision 0.978-0.989 on the two digital drawings; the scans' manifest colors miss much of their line work) |
| text CER page (24 cases) | 0.98 → 0.12 (at most 0.095 above the source's) |
| labels on text (24 cases) | 7.6 → 0 |

**Palette**

| metric | baseline → now |
|---|---|
| palette min ΔE00 | 6.0 → 11.7 (smallest 10.0) |
| color pairs < 10 ΔE00 | 20.2 → 0 |
| flat colors ΔE00, exact colors (24 cases) | 2.89 → 3.15, at most 2.42 above the best a legend could reach |
| flat colors ΔE00, the scans (16 cases) | 7.23 → 8.30 (reported, not targeted) |

**By difficulty** (both sizes):

| preset | regions | colors on the legend | numbers with a leader |
|---|---|---|---|
| Easy | 22 → 22 | 5.9 → 5.5 | 0 |
| Medium | 61 → 37 | 11.2 → 8.6 | 0.2 |
| Hard | 177 → 74 | 18.3 → 12.3 | 3.1 |
| Max | 866 → 88 | 35.4 → 15.9 | 5.8 |

Regions now have a floor on paper (300, 125 and 40 mm² at Easy, Medium and Hard; 30 mm² at Max), and every region
holds a 3 mm brush. So a page has far fewer regions at Hard and Max, and every one of them can be painted and
carries a number. Colors closer than 10 ΔE00 are merged, so the legend is shorter than the difficulty asks for
wherever the image's colors crowd together.

## What fidelity cost

The harness judges fidelity by the mean of each case's change. It is worse by more than its tolerance in photos,
faces, gradients, textures and over all cases:

| | photo | cartoon | flat | face | text | gradient | texture | all |
|---|---|---|---|---|---|---|---|---|
| ΔE00 mean, change a case | +12.2% | −0.5% | +51.3% | +18.7% | −16.2% | +8.5% | +6.9% | +11.3% |
| SSIM | 0.625 → 0.591 | 0.633 → 0.789 | 0.999 → 0.998 | 0.671 → 0.642 | 0.462 → 0.652 | 0.668 → 0.637 | 0.453 → 0.417 | 0.656 → 0.688 |

The mean ΔE00 over all cases still falls, 7.64 → 7.27, because line art and text gain far more in absolute terms
(cartoon 9.10 → 6.71, text 13.73 → 11.09) than photographs lose (8.34 → 9.06). Judged case by case, the losses
weigh more: a change is a share of each case's own error, and the flat drawing's error, 0.42 → 0.56, is small.

Measured step by step, on each change's own before-and-after pages, nearly all of the loss comes from two
changes. Both were made on purpose, because the page needed them to be paintable and its palette usable:

| version | change | ΔE00 mean, a case | face ΔE00 | other |
|---|---|---|---|---|
| 0.1.20 | parts of a region narrower than the 3 mm brush given away | +39.6% | +20.1% | SSIM −0.038, edge F1 −0.122, features lost +0.63 a case |
| 0.1.24 | palette colors held 10 ΔE00 apart | +8.0% | +4.6% | flat colors +1.28 |
| 0.1.26 | difficulty set in print units | +0.8% | +1.5% | |
| 0.1.28 | line art prints its own ink | −11.9% | −5.5% | SSIM +0.065, edge F1 +0.140 |
| 0.1.29 | line art's palette from its own flat colors | −2.4% | −3.3% | flat colors −1.01 |
| 0.1.31 | a hatched patch as one area | −0.3% | +0.1% | flat colors −0.55 |
| 0.1.34-0.1.36 | faces: half the region floor, marks printed, tones settled | −0.4% | −2.7% | features lost −1.06 a case |
| 0.1.37 | textured edges settled by a vote | −1.7% | −2.2% | |
| 0.1.39 | the subject at half the region floor | −0.5% | 0.0% | subject ΔE00 −1.3% |
| 0.1.41 | lettering printed as it looks | −0.2% | 0.0% | |

The steps don't add up exactly to the total: each is a share of a different page, and each was scored by the
harness of its time. The two costs are the trade the spec allows. A page with no region narrower than the brush
cannot show detail narrower than the brush, and colors a painter can't tell apart aren't worth two numbers. Most
of the flat drawing's +51% and its p95 (0.5 → 1.8) is one effect of the palette margin: k-means had split one flat
color in two, and the margin merges them back into their average.

**Faces.** Face ΔE00 is 7.95 → 8.18 on average and worse on the cat (5.98 → 7.06) and the Vermeer (8.45 → 9.39),
but better on the lion (10.28 → 9.99) and the bold-line girl (7.11 → 6.29). Face SSIM is worse on all three
photographed and painted faces. The face's own steps (0.1.34-0.1.36) won back 2.7 of the 26 percentage points
that the brush, the palette and the presets cost face ΔE00. They also cut the features lost on the three
photographed and painted faces from 39 to 5, without giving the face regions too small to paint.

## Speed and memory

| | baseline | now |
|---|---|---|
| preview median (1100 px) | 329 ms | 738 ms (2.24×) |
| export median (own size) | 752 ms | 1,758 ms (2.34×) |
| slowest preview | 900 ms | 2,276 ms (the complex cartoon at Max) |
| L-size previews over 1 s | 0 | 6 |
| peak memory, median / most | 178 / 383 MB | 595 / 1,070 MB |

By kind of image, at preview size (median):

| | baseline | now |
|---|---|---|
| photographs, the painting and the flat drawing, without text (28 cases) | 331 ms | 685 ms |
| line art (16 cases) | 329 ms | 1,256 ms |
| images with text (12 cases) | 437 ms | 2,042 ms |

Where the time goes, summed over the 96 cases:

| stage | baseline | now |
|---|---|---|
| finding line art's ink | – | 4.0 s |
| finding faces / the subject / text | – | 1.7 s / 7.3 s / 29.0 s |
| quantizing | 49.9 s | 46.2 s |
| building regions | 7.7 s | 10.3 s |
| settling textured edges (the regions rebuilt included) | – | 8.8 s |
| outlining the regions | 3.4 s | 1.7 s |
| drawing and numbering the page | 2.7 s | 12.2 s |
| the rest: line art's and faces' own steps, finding the lettering, resizing, the legend | 0.2 s | 12.7 s |
| **total** | **63.9 s** | **134.0 s** |

The six L-size previews over 1 s are Times Square at every preset (1.5-2.2 s: it has text, so it is looked at twice
for it), the Vermeer at Max (1.17 s) and the Swiss castle at Max (1.09 s). The last two sit at the line: the same
pages read 0.98-1.04 s and 0.93-0.98 s when 0.1.40 to 0.1.44 were benchmarked, in other sessions. Peak memory is highest on images with text (862 MB
median), where the text detector's second, larger look holds most of it while it runs.

## Against the definition of done

The quality plan's definition of done, item by item.

| asks for | status |
|---|---|
| every "How to measure" item implemented, with unit tests | **met**: every metric in the spec is in the harness, and each has unit tests (`tests/test_benchmark_metrics.py`, 89 tests) |
| paintable: slivers ≤ 1%, no unlabeled regions, every number ≥ 6 pt | **met** on 96/96 |
| clean drawing: one line per boundary, every region enclosed, no same-color boundary, jaggedness ≤ 1.02 | **met** on 96/96 |
| palette: smallest ΔE00 ≥ 10 at every preset, its fidelity cost recorded | **met** on 96/96; its cost is the +8.0% above |
| cartoons: page outlines match the ink lines, no tubes | **met**, with the boundary match measured as printed-ink recall and precision ≥ 0.95 (see the spec); no tubes on 32/32 |
| faces: annotated features survive | **4 of 32 cases miss**: the cat's mouth (and once its nose) |
| faces: fidelity inside faces no worse than the baseline | **not met**: face ΔE00 +10.3% a case, face SSIM −0.039, the cost of the brush and the palette above |
| text: OCR on the page within 0.1 of the source, no numbers on text | **met** on 24/24 |
| fidelity: ΔE00 and SSIM within tolerance of the baseline, unless a recorded trade-off | **met through the trade-offs**: regressions in photo, face, gradient, texture and all cases, which the brush (0.1.20) and the palette margin (0.1.24) account for |
| previews stay within the speed guardrail | **per change, yes; overall, no**: each change kept within +25% or was accepted on its own, but the total is +124%, and 6 L-size previews pass 1 s |

## Known gaps and limits

- **The cat's mouth and nose** (above). The mouth is a faint, thin line in pale fur, lighter than the marks a face
  prints and too thin for a region of its own.
- **Face fidelity** is below the baseline's, as above. The baseline's faces scored better partly on detail no brush
  could paint: 17% of their page was slivers.
- **Speed and memory** (above). Finding text is a fifth of the time and most of the peak memory on images with text.
- **Lines per boundary (clear)** borrows the plain count's tolerance; it has not been measured on its own.
- **Not measured** (see the spec): whether the colors print, the lines' width and tone, and the PDF, which the tests
  check against the raster page. The PDF's smoothed ink keeps a pixel-wide stroke or gap at least 0.08 mm wide only
  on pages coarser than about 225 dpi. On a page at 300 dpi a few pinholes ringed by ink touching only at their
  corners print filled (27 of 67,798 on a synthetic test page), and outlining the ink takes about 2 s. No benchmark
  page is finer than 188 dpi.

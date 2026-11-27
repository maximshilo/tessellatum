# Quality and performance

## What a good page is

A page does four jobs, each measured by the benchmark harness (`benchmarks/QUALITY_BENCHMARKS.md` has the full spec):

1. **Resembles** the picture, above all where people look: faces, the subject, text. Measured by CIEDE2000 color error,
   SSIM, whether annotated eyes, noses and mouths survive, and OCR's character error rate on the page.
2. **Paintable:** a brush can paint every region and every region has a legible number. Measured by the share of the
   page in slivers a brush can't paint (target at most 1%), unnumbered regions (0), numbers below the minimum size (0)
   and numbers touching a line (0).
3. **Clean drawing:** one smooth line per boundary, on the picture's real edges. Measured by lines per boundary
   (1 ± 0.05), unenclosed area (0), jaggedness (at most 1.02) and edge F1 against the source's edges.
4. **Palette:** every two colors at least the page's color margin apart; on line art, the artwork's own flat colors.

Thresholds are physical: millimetres and points on the page's sheet (A4 by default, A3 or A2), converted to each
page's pixels. The benchmark runs on A4; `--paper A3` or `A2` makes and measures every page for that sheet instead.

## The benchmark harness

`benchmarks/bench.py` runs any git ref, or the working tree, over the 13 benchmark images (six photographs, two
paintings, two digital cartoons, two scanned drawings and a synthetic scene) at every difficulty and at two sizes:

- each case runs in a fresh Python process with that version's code, so versions never share imports, caches or
  compiled kernels; timings are the median of three runs after a warm-up;
- about 80 quality metrics are computed from what the page is made of (its regions, lines, numbers and printed ink),
  against annotations in `tests/sample_images/manifest.json` (faces and their features, text, flat colors, gradients,
  the subject);
- `compare` reports targets missed and **regressions**: a category's mean that got worse by more than three standard
  errors of the case-to-case noise.

## Results

Against the version before any quality work (0.1.17), on 96 cases (from `benchmarks/REPORT.md`, measured at 0.1.45):

| Measure | Before | After |
|---|---|---|
| Target misses, (case, target) pairs | 719 | 4 |
| Slivers, share of the page | 12.9% | 0.3% |
| Unnumbered regions per page | 118 | 0 |
| Lines per boundary | 1.80 | 1.00 |
| Area where white areas leak together | 52% | 0% |
| Smallest ΔE00 between two legend colors (mean) | 6.0 | 11.7 |
| "Tubes" (ink drawn as regions) per line-art page | 7.9 | 0 |
| OCR character error rate on pages with text | 0.98 | 0.12 |

The four remaining misses are one feature, a cat's mouth. The five difficulty levels ask for more detail than the
targets allow (finer brushes, closer colors), and the spec lists where they miss them, mainly jaggedness on detailed
photographs and numbers without room on the densest pages.

## Platforms and scaling

`docs/check_layout.py` draws the window on each sidebar page, the export dialog and the Preferences' pages:
- on a screen of 1280×720, 1920×1080, 2560×1440 or 3840×2160;
- at an interface scale of 100%, 125%, 150% or 200% (what Preferences > Interface scale and the system's display
  scaling set), so a 1920×1080 screen at 200% is a 960×540 window;
- in the light and the dark theme.

It reports any text cut, by its own width or by the panel around it. The **layout** workflow runs it on Windows,
macOS (Cocoa) and Linux (X11 under Xvfb) for every pull request that touches the interface, and keeps the
screenshots.

In English, nothing is cut at any screen and scale checked (to 3840×2160 on Windows, to 2560×1440 on macOS and
Linux), in either theme. On a narrow window the inspector folds away and the canvas bar moves what doesn't fit into its
menu; where even the view buttons don't fit (a 1280×720 screen at 200%, a 640×360 window), a list to pick the view
from takes their place. The canvas's 100% stays a pixel of the page to a pixel of the screen at any interface scale
(tested at 100% and 150%).

In the pseudo-locale, nothing is cut either: on Windows at 100-200% to 2560×1440, and on macOS and Linux at 100-200% on
screens of 1280×720 and 1920×1080 (the layout workflow). Where a row of choices and buttons doesn't fit the sidebar's
width, it stacks instead: the button under the choices, then the choices one under another, each whole. The sidebar
starts as wide as its rows need side by side, so English at 100% looks as it did.

**The same page on every platform is not a goal.** On one platform, with the same version of the app and its
libraries, a picture and its settings make the same page every time, bit for bit (a test computes it three times
afresh). On another platform it can differ: the lion at Medium has 1,457 regions on Windows, 1,589 on Linux and 1,569
on macOS. Where the difference starts hasn't been traced; the numeric libraries (OpenCV, NumPy, ONNX Runtime) are
built differently for each platform. A page is judged by what it is (every region numbered, no number on ink, regions
large enough to paint), and those guarantees are tested on all three platforms; the quality benchmark runs on Windows.
A project keeps the picture and the settings, not the page, so a project opened on another platform makes a page
that can differ from the one seen where it was saved; the exported PNG or PDF is the page itself.

## Speed

Median preview time and region count over 12 benchmark images (before the dark portrait was added), at each level's working size:

| Level | Median preview | Slowest | Median regions | Most regions |
|---|---|---|---|---|
| Beginner | 0.8 s | 2.4 s | 290 | 826 |
| Easy | 1.1 s | 3.8 s | 417 | 1,199 |
| Medium | 1.5 s | 5.1 s | 552 | 1,887 |
| Hard | 2.2 s | 9.1 s | 750 | 2,632 |
| Realistic | 3.6 s | 16.4 s | 1,292 | 4,124 |

Measured on a 16-thread desktop (November 2026, version 1.0.10), on A4. The slowest are the small cartoons, scaled up
2-3 times to their working size. An export lays the preview's page out at 300 dpi on its paper, about 2 s more at
Beginner and Medium and 4 to 7 s at Realistic, and draws a PNG from that layout in 1 to 2 s at 150 to 600 dpi on A4
(the lion at Realistic: 6 s to lay out, 1.6 s to draw at 600 dpi, 1.1 GB at most; on A2, 37 s, 7.6 s and 3.3 GB).

What makes it fast:

- **Compiled kernels.** Every pixel loop NumPy can't vectorize (union-find labeling, the region merge, the boundary walk,
  geodesic distances, the edge vote, the face cascade, the leader search, the polygon fill) is a Numba kernel, cached
  on disk after the first launch.
- **Equivalence-preserving optimization.** Each optimized stage replaced a straightforward implementation, kept in the
  tests, and must produce pixel-identical output.
- **Algorithmic shortcuts.** One distance transform per class of non-adjacent regions instead of per region; a sparse
  lattice for large bilateral filters; the edge vote only near edges and run by run; contours in each region's own box.
- **Parallelism** across row stripes and regions, on threads that release the GIL.
- **Caching** of resizes, detections, palettes and regions per image, so changing one setting re-runs only what
  depends on it: a change to the lines or numbers draws the page again from its regions (the lion at Medium, 0.9 s
  rather than 1.6 s; line art merges regions for room for the numbers, so it gains less: 3.1 to 2.0 s on the comic).
- **Pages kept** for undo, redo and a setting put back: shown at once rather than made again. On the lion at Medium,
  38 of 38 undos and redos over 20 cycles found their page, in 97 ms at the median and 116 ms at most (a page takes
  2.1 to 2.3 s). The process's memory stayed flat over 50 more cycles (1,219 to 1,226 MB). Under a 200 MB budget, with
  new pages made, it levelled off at about 1,306 MB.
- **Projects stored, not compressed again.** The picture goes into the project as its own bytes, unchanged, so saving
  is a write and a check: 65 ms for the lion (1.4 MB) and 0.48 s for a 24-megapixel JPEG (20 MB), including reading it
  back. Opening takes 29 ms and 0.3 s. From launch, an example's first page is ready in about 5 s: the window in 1.6 s,
  then 3.1 s for a first page with the models loading.
- **Analysis in the background.** The Analysis page measures the lion at Realistic in 2.7 s, off the GUI thread,
  and then shows it in 2 ms; its overlays draw in 6 to 46 ms. It is measured only while the page is open.
- **Inspecting without waiting.** On the lion at Realistic (2,216 × 1,478, 3,758 regions), the coverage and every
  region's area and box are measured in the background in 13 and 127 ms once a page is shown. Picking a region
  measures its widest brush in 0.07 ms at the median (0.9 ms for the largest) and outlines it over its box alone; a
  color's highlight over the whole page is built in 9 ms. The canvas pans in 2 ms a frame at the median at any zoom.
- **Gated inference.** The text detector's fine look runs only where a cheap first look finds text.

## Responsiveness

`benchmarks/ui_bench.py` times the app through its real window (see `benchmarks/README.md`). Before is version 1.0.21,
measured before the workspace was rebuilt; after is 1.1.34. Both on the same 16-thread Windows 11 desktop
(Python 3.12, Qt 6.11.2), a 1920×1080 window at 100%, the lion at Medium (1,583 × 1,056, 1,457 regions) unless said;
median, with the 95th percentile, of 5 to 30 repetitions.

| | Before | After |
|---|---|---|
| Generate Preview, caches empty: lion / pearl / line-art pig | 2.6 s (lion) | 2.58 / 1.94 / 4.58 s |
| Generate Preview again, the stage cache full | 1.91 s | 1.21 / 0.55 / 2.34 s |
| A line width changed, until its page is shown (after the 0.4 s redraw delay) | a full remake, 1.6-2.0 s | 1.21 s (p95 1.24) |
| Undo or redo to a page kept | — (no undo) | 142 ms (p95 146) |
| Undo to a page no longer kept | — | answered in 70 ms; made again in 1.3 s |
| Legend's detail changed | 26 ms | 23 ms |
| Switching a settings page or the analysis | — | 10 ms (p95 12) |
| Switching the view (Page, Completed, Tinted) | 6.2 ms | 8.4 ms (p95 8.9) |
| Zoom frame / pan frame | 1.7 / 1.1 ms (p95 2.3 / 1.6) | 1.0 / 0.5 ms (p95 2.4 / 0.8) |
| The page as it prints, for a zoom past the preview's pixels | 3.30 s | 2.59 s |
| Stop pressed, until the window says it is stopping | — | 2 ms (p95 3) |
| Stop pressed, until the window is idle (lion at Realistic) | 0.65-1.73 s | 1.11 s (p95 1.77) |
| Startup, until the window is shown | 1.11 s | 1.64 s |
| Memory over 50 pages with other settings, then 50 undo/redo steps | — | levels off at 1.14 GB after 16 pages; flat over the undo/redo steps |

The pipeline itself, timed alone the same day on 1.0.21 and 1.1.34: the same regions on every page; a page made from
scratch takes as long (lion at Medium, 2.31 and 2.28 s); a page made again with the stage cache full is 35 to 50%
faster, since the regions are kept (0.94 s rather than 1.64 s for the lion at Medium, 1.97 s rather than 3.19 s for
the pig); exports and stopping take as long as before.

Where it misses the aims it was built to: a page stops at the end of the step it is in, up to 1.8 s on the lion at
Realistic, as the region step doesn't check for a stop inside it (the window says so at once and stays usable). A page
drawn again after a line width takes about a second, as the numbers are placed again on the new lines. Startup is half
a second longer, for the font, the icons and the panels the window builds. What memory grows by is the pages kept
for undo, bounded by the results cache's budget (Preferences > Performance: a quarter of the RAM, at most 2 GB, by
default).

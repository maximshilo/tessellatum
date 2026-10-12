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

Thresholds are physical: millimetres and points on an A4 sheet, converted to each page's pixels.

## The benchmark harness

`benchmarks/bench.py` runs any git ref, or the working tree, over the 12 benchmark images (six photographs, a painting,
two digital cartoons, two scanned drawings and a synthetic scene) at every difficulty and at two sizes:

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

## Speed

Median preview time and region count over the 12 benchmark images, at each level's working size:

| Level | Median preview | Slowest | Median regions | Most regions |
|---|---|---|---|---|
| Beginner | 0.9 s | 3.1 s | 290 | 826 |
| Easy | 1.2 s | 5.1 s | 417 | 1,625 |
| Medium | 1.7 s | 5.6 s | 552 | 1,805 |
| Hard | 2.5 s | 6.7 s | 750 | 2,385 |
| Realistic | 3.2 s | 7.9 s | 1,384 | 4,017 |

Measured on a 16-thread desktop (October 2026, version 0.1.56). Drawing a preview at print size for export adds about
0.5 s at Beginner and Medium.

What makes it fast:

- **Compiled kernels.** Every pixel loop NumPy can't vectorize (union-find labeling, the region merge, the boundary walk,
  geodesic distances, the edge vote, the face cascade, the leader search, the polygon fill) is a Numba kernel, cached
  on disk after the first launch.
- **Equivalence-preserving optimization.** Each optimized stage replaced a straightforward implementation, kept in the
  tests, and must produce pixel-identical output.
- **Algorithmic shortcuts.** One distance transform per class of non-adjacent regions instead of per region; a sparse
  lattice for large bilateral filters; the edge vote only near edges and run by run; contours in each region's own box.
- **Parallelism** across row stripes and regions, on threads that release the GIL.
- **Caching** of resizes, detections and palettes per image, so changing one setting re-runs only what depends on it.
- **Gated inference.** The text detector's fine look runs only where a cheap first look finds text.

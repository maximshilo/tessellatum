# Techniques

Tessellatum uses machine learning for **perception** (what the picture contains) and classical, deterministic image
analysis for **construction** (what the page is). The detectors only steer the construction, a region-size multiplier
here and a printed-text mask there, so a missed detection degrades a page gracefully rather than breaking it.

## Machine learning

All four models are pre-trained, ship with the app (about 15 MB in all) and run on the CPU with no network access.

| Model | Task | Runtime | How the page uses it |
|---|---|---|---|
| **YuNet** | Face detection (lightweight CNN; boxes, scores, 5 landmarks) | OpenCV DNN | Faces get regions up to 4× smaller; their tones and dark marks are settled |
| **lbpcascade_animeface** | Drawn-face detection (boosted cascade of multi-block LBP features) | Re-implemented from OpenCV 4, compiled with Numba | As above, for cartoons and paintings YuNet misses |
| **U²-Net-p** | Salient object segmentation (nested U-structure) | OpenCV DNN | The subject gets regions up to 4× smaller |
| **PP-OCRv6-small (det)** | Scene text detection (Differentiable Binarization) | ONNX Runtime | Lettering is printed; no number goes on it |
| **k-means++** | Unsupervised clustering in CIELAB | OpenCV | The palette, before perceptual merging; also measures how flat a picture is |

Each network is fed exactly as its reference implementation feeds it (input size, scaling, ImageNet normalization), and
its outputs are post-processed in Tessellatum: score thresholds and non-maximum suppression for faces, a probability
threshold for the subject, and the Differentiable Binarization post-processing (thresholding, minimum-area rectangles,
box scoring and unclipping) for text, with shape filters on paper.

## Image analysis

| Technique | Used for |
|---|---|
| Bilateral filtering | Smoothing grain before quantization while keeping edges |
| CIELAB and CIEDE2000 | Perceptual color distances: palette margin, color tests, tone settling |
| Morphological closing and opening | Paper behind ink lines; ground behind lettering; thin-ink detection |
| Black-hat transform | Thin dark marks in faces |
| Directional closings with line segments | Closing small breaks in ink lines |
| Exact Euclidean distance transforms | Where a brush fits (the brush rule); room for numbers; label points |
| Distance transforms with labels (discrete Voronoi) | Giving each unpaintable pixel to the nearest paintable core |
| Chamfer geodesic distances | Nearest core without crossing printed ink |
| Connected-component labeling (union-find) | Regions, white areas, pockets, marks |
| Otsu's method | Whether a line of text has enough contrast to print |
| Histogram equalization, integral images, LBP | The drawn-face cascade |
| Contour following and Ramer–Douglas–Peucker | Region outlines; lettering and ink outlines |
| Bicubic interpolation on a 4× grid | Sub-pixel letter outlines |
| Supersampled anti-aliasing | Lines, leaders and dots thinner than a pixel |

## Geometry and algorithms

| Algorithm | Used for |
|---|---|
| Heap-ordered region merging | Smallest-first merge of regions below the minimum area |
| Greedy graph coloring | Measuring every region's width with a handful of distance transforms |
| Crack-graph traversal | One line per boundary between two regions, each drawn exactly once |
| Constrained Laplacian smoothing | Lines smoothed by at most a pixel; ink outlines that keep their topology |
| Color-weighted Gaussian voting | Settling texture edges while keeping the picture's own |
| Even-odd scan-line filling | Drawing a page again at print size from its outlines |
| Greedy agglomerative merging | Palette merge; face tone settling |
| Longest-processing-time scheduling | Balancing per-region work across threads |

## Why construction is not learned

The constraints a paint-by-numbers page must meet are geometric and physical: a region must admit a disk of a given
diameter in millimetres, two colors must be a given ΔE00 apart, a number must not touch any ink. Constructive methods
meet them exactly, respond directly to every slider, are deterministic, and can be tested pixel for pixel. Learning is
used where it is strongest: deciding where the faces, the subject and the text are.

# What a good paint-by-numbers page does, and how it is measured

A good page does four jobs:

1. **Resembles:** the finished painting looks like the image, above all where people look: faces, the subject, text.
2. **Paintable:** a brush can paint every region, and every region carries a number you can read.
3. **Clean drawing:** the unpainted page reads as a drawing, with one smooth line per boundary, placed on the
   image's real edges, so the subject is recognizable before any paint goes on.
4. **Palette:** the colors are clearly distinguishable from each other.

This is the spec. For each job, and for each kind of image, it says what a good page does and which benchmark
metric checks it, with the metric's target. The harness that computes the metrics is described in
[`README.md`](README.md), with every definition in full and every tolerance. How far today's pages are from the
first measured baseline is in [`REPORT.md`](REPORT.md).

## How a page is judged

- **On paper.** Paintability is physical, so thresholds are set in millimeters and points, not in pixels or as a
  fraction of the image. The print model (`src/tessellatum/core/print_size.py`) converts them to each page's pixels.
  The page fills an A4 sheet inside 10 mm margins, turned to landscape when the image is wider than tall.

  Each page is held to its own settings: the brush and the colors' margin its difficulty asks for, the smallest
  number its style allows.

  | threshold | on paper, at the app's levels (Beginner to Realistic) | at Max |
  |---|---|---|
  | paintable width (the brush) | 1 mm to 0.5 mm | 3 mm |
  | colors' margin | 4 ΔE00 | 10 ΔE00 |
  | smallest number | 3 pt | 3 pt |
  | line | 0.2 mm | 0.2 mm |

  Until 0.1.51 every page had the 3 mm brush, 10 ΔE00, 6 pt numbers and 0.3 mm lines.

- **The painting** is the page with every region filled in its legend color. What the page prints (ink, detail
  marks, lettering) is kept as printed. Fidelity compares it with the image at the page's size.
- **Targets** are what every page must reach, at every difficulty and at both sizes. A metric without a target is
  still judged against a reference result set. They hold at Max, the setting every version is compared at. The app's
  five levels (from 0.1.52, chosen by the user from their own pages) ask for more detail than they allow -- a brush of
  1 mm down to 0.5 mm, colors 4 ΔE00 apart, numbers down to 3 pt -- and miss some by design (see "At the levels"
  below); so may any setting the user moves past them. A category's mean (or the mean over all cases) that gets worse by
  more than 3 standard errors of the case-to-case noise is a **regression**. A few metrics only inform.
- **The benchmark set** is 12 images: six photographs, a painting, two digital bold-line cartoons, two scanned
  drawings and a flat drawing. Each is scored at the app's five levels, Beginner, Easy, Medium, Hard and Realistic,
  and at Max (40 colors, 30 mm² regions, no smoothing, the 3 mm brush and 10 ΔE00: the setting every version is
  compared at, whatever its levels), at preview size (1100 px) and at export size (the image's own size). Each image's manifest entry marks what the metrics need: its faces and their features,
  its text, its flat and ink colors, its gradients and its subject (`tests/sample_images/manifest.json`).

## 1. Resembles

A good page keeps the image's colors and structure. Where it has to simplify, it simplifies the background rather
than the faces or the subject, and it never loses an eye, a nose or a mouth.

| metric | what it measures | target |
|---|---|---|
| ΔE00 mean, ΔE00 p95 | CIEDE2000 color error between the painting and the image | regression |
| SSIM | structural similarity of their lightness | regression |
| face ΔE00, face SSIM | the same, inside the annotated faces | regression |
| subject ΔE00, subject SSIM | the same, inside the annotated subject's outline | regression |
| features lost | annotated eyes, noses and mouths the page no longer shows. A feature survives if at least 30% of its edges lie within 0.5 mm of a line or of printed ink, or if a region of its own covers a quarter of its box | 0 |
| text CER painting | OCR's character error rate on the painting's text | regression |

Fidelity has no absolute target. A paintable page has to give up detail narrower than the brush, so how closely it
can resemble the image depends on the image. Fidelity is judged against a reference instead. Fidelity spent to make
the page paintable is recorded as an intended trade-off.

## 2. Paintable

- Every region is wide enough for a brush at print size. Width matters, not just area: a 2-pixel sliver can still
  pass an area threshold. A corner's point is the exception: a round brush stops short of it, but a painter fills it
  with the brush's tip, so a page may keep a triangle's corners (an opt-in setting) rather than round them to the
  brush.
- Every region carries a legible number, never on a line or on another number. A region too small to hold its number
  gets it just outside, with a leader line, rather than going without one.
- There are few slivers anywhere, and none where a gradient would break into thin bands.

| metric | what it measures | target |
|---|---|---|
| slivers | share of the page a round brush as wide as the page's own can't paint without crossing into another region (the regions' morphological opening by it), but for the points of corners down to 20° that stand apart, which it paints with its tip | ≤ 1% |
| gradient slivers | the same, inside the annotated gradients (sky, calm water) | ≤ 1% |
| unlabeled | regions without a number | 0 |
| labels too small | share of numbers printing smaller than the page's smallest number | 0 |
| labels on lines | numbers with any ink of a line, or of a leader, in their box | 0 |
| overlapping labels | numbers whose boxes overlap | 0 |
| labeled area | share of the area to paint that lies in numbered regions | regression |
| compactness p10, median | 4πA/P² over the regions: 1 for a disk, lower for ragged or stretched ones | regression |
| undersized | regions still smaller than the difficulty allows that have a neighbor to merge into | regression, with no tolerance |
| leaders | numbers written outside their region | inform |
| bands | regions shaped like the bands a gradient breaks into: no brush twice the page's own fits, and at least 4 times as long as wide | inform |

## 3. Clean drawing

- One line per boundary between two regions: never two side by side, and never none.
- No line between neighbors of the same color.
- Every region enclosed, with no gap where two regions' paint can run together.
- Smooth lines, not the pixel grid's staircase.
- Lines on the image's real edges.
- Thin lines, gray rather than black, so they disappear under the paint: 0.2 mm by default. Numbers are a step
  lighter than the lines and small, so they read as part of the page and not as writing in the image.

| metric | what it measures | target |
|---|---|---|
| lines per boundary (clear) | lines drawn along each boundary between two regions, more than 2.5 px from where three meet | 1 ± 0.05 |
| lines per boundary | the same, junctions included | regression |
| unenclosed | share of the page in a white area that covers more than one region | 0 |
| same-color boundary | share of the boundary length that lies between two regions of the same color | 0 |
| jaggedness | length of the drawn lines over their length with wiggles under 0.5 mm smoothed away | ≤ 1.02 |
| edge F1 | how well the region boundaries and the image's edges line up, within 0.5 mm | regression |

Near a junction, a line lies within reach of the other boundaries that meet there, so no renderer could count 1
there. The target is therefore set clear of junctions.

## 4. Palette

The painter can tell every two colors apart, with the margin the difficulty asks for: 4 ΔE00 at every level (about
1 is the smallest difference an eye sees side by side), 10 at Max. On line art, the colors are the artwork's own flat
colors.

| metric | what it measures | target |
|---|---|---|
| palette min ΔE00 | smallest CIEDE2000 difference between two legend colors | ≥ the page's color margin |
| close color pairs | pairs of legend colors closer than the page's color margin | regression |
| flat colors ΔE00 | on line art, mean distance from each of the artwork's flat colors to the nearest legend color | at most 2.5 above the best a legend of that many colors could reach, where the manifest's colors are the file's exact values |

## By kind of image

### Photos and paintings

What went wrong before this work, on the photo of the lion at Hard (then 20 colors and 40 mm² regions):
- Fur and grass became hundreds of jagged, spiky regions with staircase outlines.
- The face couldn't be recognized in the outlines.
- The 20 palette colors were mostly near-identical browns.

What good looks like:
- Texture is simplified into a handful of patches that follow the form, with smooth boundaries.
- Shapes can keep their corners: with the sharpest corner setting on, a roof, a spire or a triangle keeps its point.
- Gradients (sky, shading) become a few broad bands, not thin concentric slivers.
- More detail goes to the subject, less to the background.
- Palette colors are at least the color margin apart.

Measured by slivers, jaggedness, compactness and palette min ΔE00, as for every image. On top of those:

| metric | what it measures | target |
|---|---|---|
| gradient slivers, bands | see Paintable | ≤ 1%; inform |
| subject ΔE00, subject SSIM | see Resembles | regression |
| subject detail | how many times denser the subject's regions are than the background's, per area on paper | inform: a subject in front of a plain wall is denser on any page |

### Cartoons and comics (bold lines)

What went wrong before this work, on a cartoon lion:
- The black ink lines became their own color to paint, drawn as hollow double-outlined "tubes" with tiny "1" labels.
- The mane's shading strokes became unlabeled slivers.

What good looks like:
- The artwork's ink lines are found and printed as the page's lines: solid, pre-inked, not something to paint.
- Each flat fill becomes one region, and shading strokes or hatching don't break it into slivers.
- The palette is the artwork's actual flat colors, with no extra colors from anti-aliased edge pixels.
- Small gaps in the lines get closed.

| metric | what it measures | target |
|---|---|---|
| ink found recall, precision | the ink lines the pipeline finds, against the artwork's (from the manifest's ink colors), within 0.5 mm | ≥ 0.95; precision where the manifest's colors are exact |
| stray ink | ink lines found on an image that isn't line art | 0 |
| ink printed recall | share of the artwork's ink lines the page prints, within 0.5 mm | ≥ 0.95 |
| ink printed precision | share of what the page prints that is the artwork's ink, within 0.5 mm | ≥ 0.95 where the manifest's colors are exact |
| tubes | regions at least half made of the artwork's ink lines | 0 |
| ink in shapes < 5 mm | share of the ink lines lying in parts of regions narrower than 5 mm: ink to paint instead of print | regression |
| ink line F1 | how well the drawn lines and the printed ink's centerlines follow the ink lines' centerlines | regression |
| flat colors ΔE00 | see Palette | see Palette |

The boundary match score this spec first asked for (target 0.9) became the two printed-ink metrics. Once a page
prints the artwork's ink itself, its lines are that ink. Ink line F1 compares the skeletons of two wide shapes, and
those branch differently even for shapes that agree pixel for pixel. So the target moved to how much of the ink the
page prints, and how much of what it prints is ink. Gaps in the lines are judged by unenclosed area, as on every
page. On the two scans the manifest's colors are cluster centers of the printed colors rather than exact values, so
precision and flat colors are reported there, not targeted.

### Faces and other high-detail areas

What went wrong before this work: one global minimum region size either merged the eyes, nose and mouth away (the
photo lion's face was gone) or left them as tiny unlabeled bits. People notice a wrong face far more than a wrong
patch of grass.

What good looks like:
- Adaptive detail: a smaller region limit inside faces and the subject, coarser detail in the background.
- Printed detail lines: features too small to paint (pupils, eyelid lines, whisker dots) are printed on the page
  rather than made into regions.
- Skin and fur: a few large, smooth tones rather than blotches.

| metric | what it measures | target |
|---|---|---|
| face found recall | share of the annotated faces the pipeline finds | 1 |
| stray faces | faces found on no annotated face | 0 |
| features lost | see Resembles | 0 |
| face ΔE00, face SSIM | see Resembles | regression |
| labels on features | numbers overlapping a feature's box | regression |
| face regions | regions lying mostly inside the face boxes | inform: fewer is better while fidelity and features hold |
| background regions | regions touching no face box, where a face's extra detail should add nothing | inform |

### Text

What went wrong before this work: the "Some Generic Text" watermark became one blank box labeled "2", so the text
vanished. More generally:
- letters break into unpaintable fragments;
- the holes in letters like "o" and "e" get merged away;
- digits in the image get confused with region numbers.

What good looks like:
- Signs, titles and captions are found and their letters printed, solid, without the box's ground: a light sign's
  letters print as ink as a dark caption's do. Lettering taller than 15 mm on paper is big enough to paint as
  shapes.
- No region number goes on text or next to it.
- Region numbers are styled so they can't be mistaken for the image's text: small and gray.

| metric | what it measures | target |
|---|---|---|
| text found recall | share of the annotated text that lies inside the lines of text found | ≥ 0.9 |
| stray text lines | lines of text found on an image without annotated text | 0 |
| text CER page | OCR's character error rate on the unpainted page, against the annotated text | at most 0.1 above the CER on the source |
| text CER painting | the same, on the painting | regression |
| labels on text | numbers overlapping a text box | 0 |

OCR doesn't read every source perfectly either (CER 0.04 to 0.26 at preview size), so the page's target counts from
the source's.

## At the levels

The five levels (0.1.52) give pages 4 to 20 times as many regions as the presets before them (median preview: 308 at
Beginner, 614 at Medium, 1,413 at Realistic, against 21 to 73), painted much closer to the picture (median ΔE00 5.4
at Beginner to 4.1 at Realistic, against 7.3 to 8.9). They miss these targets, measured over the 12 images at preview
and export size (`T5-candidate`, median / worst page):

| target | where it is missed |
|---|---|
| jaggedness ≤ 1.02 | the nine photographs, paintings and detailed drawings at every level: 1.03 / 1.06 at Beginner to 1.06 / 1.14 at Realistic. Holding edges to the picture within 10 ΔE00 rather than 1 lowers it a little (the lion at Medium 1.083 to 1.053); the rest is the fineness, outlines bending at less than the 0.5 mm the metric smooths |
| labels on lines 0 | numbers that find no room off the lines: on previews, which draw no number under 10 px, on those nine images at every level (2 / 134 at Beginner, 476 / 6,564 at Realistic); on exports, line art from Beginner and every photograph from Hard (85 / 2,155 at Realistic) |
| overlapping labels 0 | from Medium on, the same numbers (preview worst 75 at Medium, 5,060 at Realistic) |
| slivers ≤ 1% | the lion from Easy on, by its own brush: up to 1.6% |
| tubes 0 | the three cartoons and the comic: up to 9 regions on a page, 37 on one preview at Realistic |
| lines per boundary (clear) 1 ± 0.05 | previews only: the bold-line girl's from Medium on (1.06 to 1.14), and five photographs' at Realistic (up to 1.21) |
| same-color boundary 0 | the complex cartoon and the comic, at most 0.23% of the boundary |
| labels on text 0 | the comic at every level, Times Square from Medium on: up to 57 numbers |
| text CER page | Times Square's preview, as at Max; the comic's from Easy on, and the complex cartoon's preview at Hard |

## Not measured

- **Printable colors.** The palette is 8-bit sRGB, so it always displays. Which of its colors paper and ink can
  reproduce depends on the printer, and checking that needs the printer's color profile.
- **Line width, line tone and the numbers' style.** The page draws them as set (0.2 mm and gray by default), and no
  metric reads them.
- **The PDF.** The harness scores the raster page. The tests check the vector PDF against it
  (`tests/test_export.py`).
- **Each flat fill as one region**, directly. Tubes, slivers and flat colors catch the ways it fails.
- **Texture areas** are marked in the manifest, but no metric reads them. Texture is judged on the whole page by
  slivers, jaggedness and compactness.

# What a good paint-by-numbers page needs to do, and how to tell whether it does

A good page does four jobs:

1. The finished painting resembles the image, above all in the areas people look at.
2. The page is physically paintable.
   - Every region is wide enough for a brush at print size. Width matters, not just area: a 2-pixel sliver can still pass an area threshold.
   - Every region carries a legible number.
   - There are few slivers.
3. The unpainted page reads as a clean drawing. One smooth line per boundary, placed on real edges, so the subject is recognizable before any paint goes on.
4. The palette works. Colors are clearly distinguishable from each other and printable.


## (1) Photos

### Example of a bad rendition of a photo
What I saw on the photo lion at Hard:
- Fur and grass became hundreds of jagged, spiky regions with staircase outlines.
- The face can't be recognized in the outlines.
- The 20 palette colors are mostly near-identical browns.

### What good looks like:
- Texture is simplified into a handful of patches that follow the form, with smooth boundaries.
- Gradients (sky, shading) become a few broad bands, not thin concentric slivers.
- More detail goes to the subject, less to the background.
- Palette colors differ from each other by a clear margin, e.g. a minimum color distance of around 10 ΔE00.

### How to measure:
- share of page area in regions narrower than the paintable width;
- outline jaggedness, i.e. outline length relative to a smoothed version of it;
- region compactness;
- the smallest color distance between any two palette colors.

## (2) Cartoons and comics (bold lines)

### Example of a bad rendition of a cartoon
What I saw on the cartoon lion:
- The black ink lines became their own color to paint, drawn as hollow double-outlined "tubes" with tiny "1" labels.
- The mane's shading strokes became unlabeled slivers.

### What good looks like:
- Detect the artwork's ink lines and print them as the page's outlines: solid, pre-inked, not something to paint.
- Each flat fill becomes one region.
- The palette is the artwork's actual flat colors, with no extra colors from anti-aliased edge pixels.
- Small gaps in the lines get closed.

### How to measure:
- how closely page outlines match the source's dark strokes (boundary match score);
- the number of thin, dark, tube-shaped regions (should be 0);
- how closely the palette matches the artwork's dominant flat colors.

## (3) High-detail areas like a face

The failure: one global minimum region size either merges the eyes, nose and mouth away (the photo lion's face is gone) or leaves them as tiny unlabeled bits. People notice a wrong face far more than a wrong patch of grass.

### What good looks like:
- Adaptive detail: a smaller region limit and finer color steps inside faces or other salient areas, coarser detail in the background.
- Printed detail lines: features too small to paint (pupils, eyelid lines, whisker dots) are drawn on the page rather than made into regions.
- Skin: a few large, smooth tones rather than blotches.

### How to measure:
- color error and structural similarity scored separately inside a detected face or salient area;
- a check that key features survive as their own regions or lines.

## (4) Text

### Example of a bad rendition of a text
What I saw: the "Some Generic Text" watermark became one blank box labeled "2", so the text vanished. More generally:
- letters break into unpaintable fragments;
- the holes in letters like "o" and "e" get merged away;
- digits in the image get confused with region numbers.

### What good looks like:
- Detect text and print it as ink, or as letter outlines if they're big enough to paint.
- Never place region numbers on or next to text.
- Style region numbers so they can't be mistaken for image text, e.g. small, gray, or circled.

### How to measure:
run OCR on the page and the source, and compare the character error rate. This one is easy to automate.

## (5) Applies to every image

- One line per shared boundary.
- No lines between neighbors of the same color. 
- Thin lines, possibly gray, so they disappear under paint.
- Tiny regions still get labels, using leader lines instead of silently dropping the number. At the finest setting only 55% of the photo's page area is labeled.
- Thresholds set in print units (millimeters and points), not as a fraction of image pixels. A starting point: at least ~3 mm paintable width and ~6 pt numbers.
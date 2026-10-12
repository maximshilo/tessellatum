# User guide

The panel on the left holds every setting; **Generate Preview** draws the page with them, and **Export…** draws that
same page at print resolution: the regions, colors and legend the preview shows, its lines smoothed as they are
there, and its numbers placed again, as small as the style lets them print. If a setting changed since the preview,
the export makes the page with it first. Every setting has a tooltip; sizes are on the printed A4 page; the panel keeps
them between runs, and **Reset** puts them all back.

## The three versions

Every preview is drawn in three versions, and the bar above it switches between them, keeping the zoom and the scroll:

- **Page**: the coloring page as it prints.
- **Completed**: the page painted in. Every region is filled in its legend color, which covers the lines and numbers,
  as paint covers their grays; what the page prints of the picture itself (line art's ink, the marks in a face, the
  letters of a sign) shows on top, and paper the page leaves bare stays white.
- **Tinted**: the page under a faint wash of its colors, each 40% of the way from white, with the lines, numbers and
  ink showing through as through a translucent paint: a guide to paint from.

## Zoom

The preview zooms about the point under the cursor, with the mouse wheel, Ctrl+wheel, a touchpad's pinch, Ctrl+plus
and Ctrl+minus, or the buttons above it, and the page is dragged to move it. **Fit** (Ctrl+0) shows the whole page,
**100%** a pixel of the page as it prints on a pixel of the screen, and **Print size** the page as large as on paper,
by the size the screen reports. Further in than the preview's own pixels, the zoom shows the page as it prints: the
export's own drawing, with its numbers where and as small as they print. It is drawn in the background the first time
and kept, so the export saves it without drawing it again. A new preview of the same picture keeps the zoom and the
place, to compare settings there.

## Difficulty

Beginner, Easy, Medium (the default), Hard and Realistic fill in the settings below them; moving any of them picks
**Custom**. The five levels differ in four settings:

| Level | Colors | Smallest region | Brush | Smoothing |
|---|---|---|---|---|
| Beginner | 16 | 10 mm² | 1.0 mm | 1.2 |
| Easy | 20 | 8 mm² | 0.8 mm | 1.0 |
| Medium | 24 | 6 mm² | 0.7 mm | 0.9 |
| Hard | 28 | 4 mm² | 0.6 mm | 0.8 |
| Realistic | 32 | 2 mm² | 0.5 mm | 0.7 |

Every level keeps colors 4 ΔE00 apart, settles ragged edges three times as far as the brush alone would (with the edge
hold at 1 ΔE00, which keeps them closest to the picture's), lets a region in a face or the subject be a quarter of the
smallest, and rounds corners to the brush. Region sizes are areas on paper, so a page's regions are the same size on
paper whatever its pixel count, and a long, narrow picture, which prints smaller, gets fewer regions rather than
smaller ones.

## Settings

**Regions and colors**
- **Colors**: how many colors to look for. Colors closer than the color margin are merged, so a page can keep fewer.
- **Color margin**: how different every two palette colors must be (CIEDE2000; about 1 is the smallest difference an
  eye sees side by side). Lower keeps more, subtler colors.
- **Smallest region**: the smallest area a region may have on paper (2 to 500 mm²); smaller ones merge into a neighbor.
- **Brush width**: the narrowest any part of a region may be (0.5 to 6 mm); narrower parts are given to the region
  beside them.
- **Pixels across brush**: how finely a page is made (4 by default, 2 to 8). A 3:2 picture's page is about 1,100 px
  along its long edge at Beginner and 2,200 at Realistic, never fewer than 1,100, nor more than the picture has or
  than 300 dpi. Not part of the difficulty: moving it keeps the preset.
- **Sharpest corner**: corners down to this angle keep their points rather than being rounded to the brush. 180°, the
  default, rounds every corner; 20° keeps a triangle's points. The spikes of fur and foliage are rounded all the same.
- **Corner contrast**: how plainly the picture must show a corner's point for it to be kept.
- **Smoothing**: how much the picture is smoothed, keeping its edges, before its colors are found.
- **Edge settling** and **Edge hold**: how far, and how firmly, the vote that smooths ragged edges in fur, foliage and
  stone reaches.

**Faces and subject**
- **Face and subject detail**: how many times smaller a region may be in the faces and subject found.
- **Dark mark contrast** and **Dark mark length**: how dark and how long a thin mark in a face (a pupil, a lip line)
  must be to be printed rather than painted over.
- **Face tone step**: a region in a face joins its neighbor while that changes its color by less than this, so skin and
  fur paint in a few large tones.

**Lines and numbers**
- **Line width**: 0.1 to 1 mm (0.2 mm by default). **Tone**: Light, Medium or Dark; numbers are always a step lighter
  than the lines, so they read as the page's apparatus rather than as writing in the picture. Light is for pale paints,
  under which darker lines would show.
- **Line smoothing**: how far along a line its pixel staircase is smoothed.
- **Smallest number**: 3 pt by default. A preview draws no number under 10 px, so smaller sizes may show only on an
  export, or zoomed in.
- **Leader reach**: how far outside a small region its number may go, on a leader line.
- **Gap beside text**: how far numbers keep from lettering.

**Line art and text**
- **Paint over ink up to**: on a cartoon or comic, the paint goes over ink lines up to this wide (hatching, fine
  strokes), never over bolder ones.
- **Close line gaps up to**: gaps in ink lines up to this wide are closed, so two areas' paint doesn't run together.
- **Print lettering up to**: lines of text up to this tall are printed; taller lettering is painted as shapes.

**Picture handling** (each on by default; turn one off when it gets a picture wrong)
- **Print line art's own ink**: a cartoon or comic is drawn from its ink lines, which are printed, and the areas they
  enclose; off, it is drawn from its colors like a photograph.
- **More detail on faces and subject**: on a photograph or painting, the faces and subject found get smaller regions,
  a face is painted in a few tones, and its thin dark marks are printed.
- **Print text**: the letters of signs, titles and captions found are printed, their ground left bare, and no number
  goes on them.

**Abort** stops a page in progress.

## Export

The export renders the page at 300 dpi on A4: a long edge of 2,244 to 3,272 px depending on the picture's shape, or the
picture's own size where that is smaller, since nothing is upscaled. **Version** picks which of the three versions to
export, in either format; the legend comes with each.

- **PDF**: two A4 sheets of vector art. The page fills the printable area inside 10 mm margins, centered, on a
  landscape sheet when the picture is wider than tall. Its lines are paths of the width asked for and its numbers text,
  so a printer draws them at its own resolution. Line art's ink, the marks in a face and the letters of signs are
  filled outlines, smoothed off the pixel staircase without joining or breaking anything. The painted versions fill
  each region along a path made of the page's own lines. Nothing on the sheet is an image. The legend gets a portrait
  sheet of its own, its swatches 12 mm squares in their exact colors.
- **PNG**: the page above its legend in one image, which records its resolution, so it prints as large as the PDF's
  page.

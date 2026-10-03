# Tessellatum

Turn any image into a "paint by numbers" style coloring page: the source
image is reduced to flat, outlined regions, each region is numbered, and a
color legend maps each number to a color. Preview the result, tune the
difficulty (more/smaller regions and colors = harder), and export to PNG or
PDF.

Tessellatum is a native desktop app (PySide6/Qt) — no browser, no server,
just double-click and run.

## Running from source

```
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"          # Windows
# ./.venv/bin/pip install -e ".[dev]"          # Linux/macOS

.venv\Scripts\python -m tessellatum.main       # Windows
# ./.venv/bin/python -m tessellatum.main       # Linux/macOS
```

## Building a standalone app

Produces a folder you can copy anywhere and double-click to run — no
installer, no Python required on the target machine.

- Windows: `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1` → `dist\Tessellatum\Tessellatum.exe`
- Linux: `./packaging/build_linux.sh` → `dist/Tessellatum/Tessellatum`
- macOS: `./packaging/build_macos.sh` → `dist/Tessellatum.app`

The Linux and macOS scripts are written to be portable but have only been
built/tested on Windows so far — if something's missing on your distro,
run the binary from a terminal to see the error.

## How it works

1. **Quantize**: the image is smoothed and reduced to a small palette of
   flat colors via k-means clustering in Lab color space. Colors the painter
   could not tell apart or mix — closer than 10 CIEDE2000 — are then merged
   into the color their pixels average to, so a photograph of fur or stone
   comes back with a handful of clearly different browns instead of twenty
   near-identical ones. That is why the legend can be shorter than the
   difficulty asked for.
2. **Regionize**: same-color pixels are grouped into connected regions;
   regions smaller than the difficulty's threshold are merged into their
   largest neighbor, and neighbors left sharing a color by that merge become
   one region, so the page never draws a line between two areas the painter
   fills alike.
3. **Widen**: anything a brush cannot paint is given away. Every part of a
   region narrower than the brush — 3 mm on the printed page — goes to the
   region whose paint reaches it first, and a region thinner than that
   everywhere disappears into its neighbors, so the page asks for no stroke
   too fine to make.
4. **Settle**: where a photograph is textured — fur, foliage, stone — its
   colors alternate faster than a brush is wide, and the edges between its
   regions come out ragged, with bumps and notches the brush can't paint
   into. So every pixel near an edge votes: it takes the color that holds the
   most of the page around it, weighted over about a brush's reach (a
   Gaussian of a third of the brush's width, 1 mm), with each color counting
   less the further it is from the pixel's own color in the picture (e-fold
   per 10 L\*a\*b\* units, the palette's margin). Where the picture has an
   edge of its own the pixels on each side suit their own color far better,
   and the edge stays on it; in fur, where they suit both alike, the edge runs
   smooth. The regions are then rebuilt by steps 2 and 3, so everything they
   promise still holds. See `src/tessellatum/core/texture.py`.
5. **Draw**: the boundaries between regions are traced from the region map
   and each one is drawn once, as a single line its two regions share, so no
   boundary is doubled or left out. Each line is then smoothed along its
   length to take the pixel grid's staircase off it, but never by more than a
   pixel, so it stays on the boundary it draws and the page still closes.
6. **Ink**: the lines go down as a round pen 0.3 mm across — a size on paper,
   so a preview and an export of one image print the same line — laid on a
   grid four times finer than the page and averaged back down, which
   anti-aliases it and lets it be thinner than a pixel. Lines and numbers
   print gray rather than black, so they vanish under the paint meant to
   cover them and a number is not mistaken for writing in the picture. Width
   and tone are `PageStyle` in `src/tessellatum/core/render.py`.
7. **Number**: every region gets its number (matched to a legend swatch),
   never smaller than 6 pt on paper (nor than 10 px), and never where any of
   a line's ink falls, however faint. It goes at the region's most interior
   point if it fits there; otherwise wherever in the region it keeps farthest
   from the lines, made smaller if it has to be. A region too small to hold
   even that has its number written just outside it, with a leader line in
   the numbers' gray running to a dot inside it. See
   `src/tessellatum/core/labels.py`.

Difficulty controls three things: how many colors k-means looks for, how
small a region may be on the printed page before it is merged away, and how
much smoothing is applied before quantizing — see
`src/tessellatum/core/difficulty.py`. Region sizes are areas on paper:
300 mm² at Easy, 125 mm² at Medium, 40 mm² at Hard, and 30–500 mm² in
Custom. So a preview and an export of one image get regions of the same
size, and a long, narrow picture, which prints smaller, gets fewer regions
rather than smaller ones. Custom stops at 30 mm²: with smaller regions, a
detailed page at the finest setting gets so many that a 3 mm brush can't reach
into their corners over more than 1% of an A4 page, or, on a scanned drawing,
that the numbers of its hatched areas find no room off its lines. (Settling a
photograph's edges, step 4, would keep it under 1% down to 15 mm²; the scanned
drawing is what holds the floor.)
The brush width comes from the printed page instead, along with the smallest
region any setting can keep and how wide a line prints — see
`src/tessellatum/core/print_size.py`.

Line art — a cartoon, a comic — is flat fills with dark ink lines between
them, and those lines are printed, not painted. The pipeline tells such a
picture from a photograph and finds its ink lines without being told its
colors: a line is dark against what lies beside it and at most 5 mm wide on
paper, and a picture is line art when its fills are flat and deep lines cover
enough of it (`src/tessellatum/core/ink.py`). Its page is then drawn from its
ink:

- the ink is printed solid, in the artwork's own tone — black for digital
  line art, the dark gray of the printed ink for a scan — and it is the line
  wherever it runs: no line is drawn beside it and no number goes on it. The
  one exception is a number with no room anywhere near but on hatching: the
  hatching of its own region is cleared under it, and a line's width round
  it, so that it reads;
- the colors are the fills' own. A cartoon is painted in the colors the
  artist chose, so the page offers those rather than the means k-means lands
  on, which carry the blends along every fill's edge: the fills' colors are
  counted into a histogram, and the color the most pixels crowd around, the
  farthest from every color taken so far, is taken over and over, never within
  10 CIEDE2000 of one already taken. Nothing is merged afterwards and nothing
  is random, so a drawing keeps colors a merge would have spent. A shade the
  margin leaves off the legend is painted in the color nearest it in L\*a\*b\*,
  the one it looks closest to;
- the regions are the areas the ink encloses; a boundary between two fills
  that no ink divides is drawn as on any page. Hatching encloses nothing: the
  white areas a 3 mm brush fits in are what a painter sees as areas, and the
  regions are built through every thin stroke but those that keep two such
  areas apart, so a hatched patch is one run of its own gaps' colors — the
  policeman's hatched coat is painted blue, not the sky's gray around it.
  Every white area a brush fits in is a region of its own, with its own
  number;
- a brush cannot keep off ink thinner than it, so the paint goes over ink
  thinner than 1.5 mm: a hatching or shading stroke inside one area is
  painted over along with the gaps around it, and a fine line between two
  areas is shared down its middle, as a page's own lines are on any other
  picture — the two areas still never touch, and each keeps its own number.
  Bolder ink, an outline or a black shape, is never painted. The ink is
  printed all the same;
- a shape the ink encloses on its own, a finger or a button, keeps its number
  whatever the difficulty's smallest region, as long as a 3 mm brush fits in
  it; one too small for the brush is left as bare paper — unless it is a gap
  between thin strokes in the color of the area round it, which is painted
  with it — and a small patch in the ink's own color, edged mostly by the ink,
  is printed with it;
- what a brush still can't reach, in pockets walled mostly by ink it may not
  go over — the tips a fill makes against a bold outline, the channels
  between dark blobs of a scan — is left as bare paper, outlined like any
  other area: too narrow to paint;
- a region whose number finds no room anywhere, not even on its own hatching
  cleared, joins the area its white shares an edge with, as a region below the
  difficulty's smallest area does — or, if it is hatching with no room for a
  brush in its white, the region it touches most through its strokes.

Hatching drawn finer than the ink's anti-aliased edge — gaps under about
0.5 mm between strokes — has no pixels of its fill's own color to go by: such
a patch takes its color from the fill nearest it beyond its strokes.

People notice a wrong face far more than a wrong patch of grass, so the
pipeline also looks for faces, to give them more detail than the rest of the
page. Two small detectors ship with the app and run offline: YuNet, a network
for photographed and painted faces (trained on people, it finds a cat's or a
lion's face too), and lbpcascade_animeface, a cascade for drawn ones. A face
narrower than 15 mm on paper is left out: its eyes would be narrower than the
brush. See `src/tessellatum/core/faces.py`; the models' sources and licenses
are in `src/tessellatum/resources/MODELS.md`.

On a photograph or a painting, a region inside a face found may be half the
difficulty's smallest area — each of its pixels there counts twice — so eyes,
a nose or a mouth that would merge into the skin keep regions of their own.
The rest of the page is drawn as before, and the brush is 3 mm everywhere, so
a feature no wider than that would still merge away. Those thin dark marks
inside a face — pupils, the lines of the eyelids and lips, a nose's rim, the
dots whiskers grow from — are printed on the page instead, solid, in their own
tone, as line art's ink is: part of the picture, not something to paint. A mark
is no wider than the brush, at least 12 L* darker than what lies around it,
at least 2 mm long, and no lighter than the paint that would cover it (see
`src/tessellatum/core/marks.py`).

A face's skin or fur should read as a few large tones rather than blotches, so
the regions lying mostly inside a face found are also settled (see
`src/tessellatum/core/tones.py`). Each takes the palette color nearest its own
pixels: a region is otherwise painted the color of the part of it a brush
fits in, whatever has merged into it since. Then a region joins a neighbor
wherever painting it in that neighbor's color would put it less than 1 ΔE00 a
pixel further from the picture, the faintest step first. An eye, a lip or a
nostril is far from the color beside it, and keeps its region.

A page should also spend its detail on what the picture is of — the person,
animal or building — rather than the wall, sky or grass behind it. A third small
network ships with the app for that: U²-Net-p (4.6 MB), which looks at the whole
picture squeezed to 320 px square and says where its one salient object is. On
a photograph or a painting, a region inside that subject may be half the
difficulty's smallest area too, as inside a face; the background is drawn as
before. A picture with no one thing to look at, such as a crowded street, has
little or no subject, and its page is the one it would be without. See
`src/tessellatum/core/subject.py`.

A drawing's faces and subject are its ink's, so line art's page uses neither.

Signs, titles and captions should still read on the page, so the pipeline also
finds the lines of text in a picture, drawn or photographed. A fourth network
ships with the app for that: PP-OCRv6-small, PaddleOCR's text detector (9.9 MB),
run by [ONNX Runtime](https://onnxruntime.ai/). It looks at the picture at half
its preview size, which finds lettering whose lines print about 4 mm tall or
more, and where that finds any, again at twice its preview size from the
source's own pixels, which finds the small print beside it, down to lines about
1.7 mm tall. A picture with no lettering that big is taken to have none, so a
picture without text pays only for the small first look (a few tens of
milliseconds); one with text pays for the second too, about 0.6-0.9 s on its
first preview, after which it is remembered. A line is kept only if it looks
like one: at most 15 mm tall on paper (bigger lettering is shapes to paint) and
at least half again as long as tall, which an eye, a window or a disc mostly
isn't.

The page prints the lettering in each line found as it looks: the line's own
lightness, from bare paper to solid ink, stretched so that its lightest tones
(the 98th percentile) are paper and its darkest (the 2nd) solid ink, in the
ink's tone, anti-aliased. Dark lettering prints as ink on paper; light
lettering — a lit sign, white type on a dark panel — as paper letters in its
dark printed ground. Which of the two a line is can't be told reliably from its
pixels (a sign's panel can shade from light to dark behind its letters), and
printed as it looks it reads either way. Inside a line the lettering takes the
place of a scan's own ink lying in its regions, so the scan doesn't print its
letters a second time, binarized and twice as bold; the lines between regions
still run through it, and ink that keeps two regions apart — bold ink, the
seam down a line two regions share — still prints solid. The regions and
the palette are left as they are: the lettering darker than halfway is printed
ink, painted round, which ends the lines crossing it, and no number goes on any
of its ink. A line whose lettering stands out from its ground by less than 20
L\* prints nothing. See `src/tessellatum/core/text.py`.

### Performance

Previews are meant to be quick enough to tweak difficulty interactively:

- Smoothing samples the bilateral filter's window on a sparse lattice (a few
  hundred taps per pixel instead of thousands) and filters rows in parallel.
- Region labeling, small-region merging, the walk that turns the region
  map into one line per boundary and, on line art, the search that keeps a
  thin part to its own side of the ink run as compiled
  [Numba](https://numba.pydata.org/) kernels whose cost grows roughly
  linearly with image size, and contour extraction works on each region's
  bounding box rather than the whole image. Their output is pixel-identical
  to the original straightforward implementation.
- So do line art's own region steps — looking through hatching, leaving
  pockets as paper, giving every white area its own region — which search
  every area at once, in one pass over the page, rather than one area at a
  time. Their output is pixel-identical to the NumPy code they replaced.
- The vote that settles a photograph's edges looks only at the pixels near
  an edge, and walks each row of its window run by run rather than pixel by
  pixel, in parallel over the page's rows.
- Resizing and quantization results are cached per image, so changing only
  the region size skips straight to the region stages.
- The compiled kernels are built on the very first launch (a few seconds, in
  the background while you pick an image) and cached on disk after that.

## Benchmarks

`benchmarks/` holds a speed + quality benchmark harness that runs any git
ref or the working tree over the sample images and compares versions on
timings and output-quality metrics:

```
.venv\Scripts\python benchmarks\bench.py run main WORKTREE
.venv\Scripts\python benchmarks\bench.py compare main-<commit> worktree-<commit>-dirty
```

See [`benchmarks/README.md`](benchmarks/README.md) for details.

## Tests

```
.venv\Scripts\python -m pytest tests/
```

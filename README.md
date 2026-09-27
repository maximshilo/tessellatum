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
4. **Draw**: the boundaries between regions are traced from the region map
   and each one is drawn once, as a single line its two regions share, so no
   boundary is doubled or left out. Each line is then smoothed along its
   length to take the pixel grid's staircase off it, but never by more than a
   pixel, so it stays on the boundary it draws and the page still closes.
5. **Ink**: the lines go down as a round pen 0.3 mm across — a size on paper,
   so a preview and an export of one image print the same line — laid on a
   grid four times finer than the page and averaged back down, which
   anti-aliases it and lets it be thinner than a pixel. Lines and numbers
   print gray rather than black, so they vanish under the paint meant to
   cover them and a number is not mistaken for writing in the picture. Width
   and tone are `PageStyle` in `src/tessellatum/core/render.py`.
6. **Number**: every region gets its number (matched to a legend swatch),
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
detailed photograph at the finest setting gets so many that a 3 mm brush
can't reach into their corners over more than 1% of an A4 page.
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

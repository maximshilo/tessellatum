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
rather than smaller ones. Custom stops at 30 mm² because smaller regions
mean so many that a 3 mm brush could no longer reach into their corners on
an A4 page.
The brush width comes from the printed page instead, along with the smallest
region any setting can keep and how wide a line prints — see
`src/tessellatum/core/print_size.py`.

### Performance

Previews are meant to be quick enough to tweak difficulty interactively:

- Smoothing samples the bilateral filter's window on a sparse lattice (a few
  hundred taps per pixel instead of thousands) and filters rows in parallel.
- Region labeling, small-region merging and the walk that turns the region
  map into one line per boundary run as compiled
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

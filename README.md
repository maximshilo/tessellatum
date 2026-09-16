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
   flat colors via k-means clustering in Lab color space.
2. **Regionize**: same-color pixels are grouped into connected regions;
   regions smaller than the difficulty's threshold are merged into their
   largest neighbor, and neighbors left sharing a color by that merge become
   one region, so the page never draws a line between two areas the painter
   fills alike.
3. **Outline + number**: each surviving region gets a black outline and a
   number (matched to a legend swatch) placed at its most interior point.

Difficulty controls three things: how many colors are used, how small a
region is allowed to get before being merged away, and how much smoothing
is applied before quantizing — see `src/tessellatum/core/difficulty.py`.

### Performance

Previews are meant to be quick enough to tweak difficulty interactively:

- Smoothing samples the bilateral filter's window on a sparse lattice (a few
  hundred taps per pixel instead of thousands) and filters rows in parallel.
- Region labeling and small-region merging run as compiled
  [Numba](https://numba.pydata.org/) kernels whose cost grows roughly
  linearly with image size, and contour extraction and outline drawing work
  on each region's bounding box rather than the whole image. Their output is
  pixel-identical to the original straightforward implementation.
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

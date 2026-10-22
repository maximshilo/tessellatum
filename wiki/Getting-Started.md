# Getting started

## Download (Windows)

1. Download `Tessellatum-<version>-windows-x64.zip` from the
   [latest release](https://github.com/maximshilo/tessellatum/releases/latest).
2. Unzip it anywhere and run `Tessellatum\Tessellatum.exe`. No installer and no Python are needed.

The first launch compiles the image kernels in the background (a few seconds, while you pick an image); later launches
reuse them.

## Run from source

Python 3.10 or newer.

```
git clone https://github.com/maximshilo/tessellatum.git
cd tessellatum
py -3 -m venv .venv                            # Windows
# python3 -m venv .venv                        # Linux/macOS
.venv\Scripts\pip install -e ".[dev]"          # Windows
# ./.venv/bin/pip install -e ".[dev]"          # Linux/macOS

.venv\Scripts\python -m tessellatum.main       # Windows
# ./.venv/bin/python -m tessellatum.main       # Linux/macOS
```

The `dev` extra adds pytest, PyInstaller and RapidOCR (used only by the benchmark's text metrics).

## Build a standalone app

Produces a folder you can copy anywhere and double-click to run.

- Windows: `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1` → `dist\Tessellatum\Tessellatum.exe`
- Linux: `./packaging/build_linux.sh` → `dist/Tessellatum/Tessellatum`
- macOS: `./packaging/build_macos.sh` → `dist/Tessellatum.app`

The real output is `dist/Tessellatum/`; PyInstaller's `build/` folder is scratch space. The test suite runs on
Windows, Linux and macOS, but the Linux and macOS standalone builds have not been tried yet; if something is missing on
your system, run the binary from a terminal to see the error.

## A first page

1. **Open Image…** and pick a picture.
2. Choose a difficulty (Medium is the default) and click **Generate Preview**.
3. Switch between **Page**, **Completed** and **Tinted** above the preview; zoom with the mouse wheel.
4. **Export…** to PNG or PDF.

The [user guide](User-Guide) describes every setting.

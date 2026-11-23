# Getting started

## Download

Each [release](https://github.com/maximshilo/tessellatum/releases/latest) has the app for Windows, macOS and Linux. No
installer and no Python are needed: unpack it anywhere and run it.

- **Windows** (64-bit): `Tessellatum-<version>-windows-x64.zip`. Unzip it and run `Tessellatum\Tessellatum.exe`.
- **macOS** (Apple silicon): `Tessellatum-<version>-macos-arm64.zip`. Unzip it and move `Tessellatum.app` into
  Applications. The app isn't notarized by Apple, so macOS blocks it the first time: open it once, then click
  **Open Anyway** in System Settings > Privacy & Security. Or, in a terminal, once:
  `xattr -dr com.apple.quarantine /Applications/Tessellatum.app`.
- **Linux** (x86-64, glibc 2.35 or later, such as Ubuntu 22.04 and later): `Tessellatum-<version>-linux-x64.tar.gz`.
  Unpack it with `tar -xzf` and run `Tessellatum/Tessellatum`. It uses the system's OpenGL, xkbcommon and fontconfig
  libraries, which desktops have; if it doesn't start, run it from a terminal to see which is missing.

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

The real output is `dist/Tessellatum/` (`dist/Tessellatum.app` on macOS); PyInstaller's `build/` folder is scratch
space. `Tessellatum --self-test DIR` checks a build: it makes a page of a picture it draws itself and exports it into
`DIR`, as the release workflow does with every build it publishes. If something is missing on your system, run the
binary from a terminal to see the error.

## A first page

1. **Open Image…** and pick a picture.
2. Choose a difficulty (Medium is the default) and click **Generate Preview**.
3. Switch between **Page**, **Completed** and **Tinted** above the preview; zoom with the mouse wheel.
4. **Export…** to PNG or PDF.

The [user guide](User-Guide) describes every setting.

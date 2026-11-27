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
reuse them. The app shows its own icon in the taskbar or the Dock, and follows the system's light or dark appearance
(View > Appearance picks one for good). It keeps its settings and window layout in the platform's settings store, and
recovery snapshots of unsaved work in its data folder (see [Recovery](User-Guide#recovery)).

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

1. The window opens on a welcome page: **Open Picture…**, drop a picture anywhere on the window, paste one
   (Ctrl+V), or click one of the examples. A project saved before opens the same way, or from **Recent projects**.
2. The **Generation** page (left) sets the difficulty: Medium is the default. Click **Generate Preview** (F5) on the
   toolbar; the status bar says what it is doing, and Cancel (Esc) stops it.
3. Above the page, switch between **Page**, **Completed**, **Tinted** and the **Original**; zoom with the mouse wheel,
   pan by dragging. The inspector (right) lists the page's colors: pick one to see where it goes.
4. Change a setting: the lines, numbers and tone redraw by themselves; anything else marks the preview **Out of date**
   until the next Generate Preview. Ctrl+Z undoes it, and a page made before comes back at once.
5. **Export…** (Ctrl+E) opens the export dialog: PNG, PDF or a palette file, which version, at what resolution, with
   what it will write said before it is written. **Save Project** (Ctrl+S) keeps the picture and its settings together.

The [user guide](User-Guide) describes the workspace and every feature; the [settings reference](Settings-Reference)
lists every setting.

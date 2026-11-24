"""Render the app's icons for every platform from the logo's SVG masters (see ``make_logo.py``):

    .venv/Scripts/python packaging/icons/make_icons.py

Writes into ``src/tessellatum/resources/``:

- ``app-icon/<size>.png`` at the freedesktop hicolor sizes 16-512: the window icon at every size (the app loads them
  all, see ``main.window_icon``), and what a Linux desktop entry installs;
- ``icon.png``: 256 px, for anything that wants one file;
- ``icon.ico``: Windows, 16-256 px, which the PyInstaller spec gives the executable;
- ``icon.icns``: macOS, 16-1024 px, which the spec gives the app bundle.

16, 20 and 24 px are drawn from ``logo-small.svg``, whose fewer regions and thicker lines stay legible there; larger
sizes from ``logo.svg``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

RESOURCES = Path(__file__).resolve().parents[2] / "src" / "tessellatum" / "resources"
SMALL_UP_TO = 24
HICOLOR = (16, 20, 24, 32, 48, 64, 128, 256, 512)
ICO = (16, 20, 24, 32, 40, 48, 64, 128, 256)
ICNS = (16, 32, 64, 128, 256, 512, 1024)


def render(size: int) -> Image.Image:
    """The logo at ``size`` pixels square, from the master drawn for that size."""
    master = RESOURCES / ("logo-small.svg" if size <= SMALL_UP_TO else "logo.svg")
    renderer = QSvgRenderer(str(master))
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def main() -> None:
    _app = QGuiApplication.instance() or QGuiApplication([])
    folder = RESOURCES / "app-icon"
    folder.mkdir(exist_ok=True)
    rendered = {size: render(size) for size in sorted({*HICOLOR, *ICO, *ICNS})}
    for size in HICOLOR:
        rendered[size].save(folder / f"{size}.png", optimize=True)
    rendered[256].save(RESOURCES / "icon.png", optimize=True)
    # Pillow writes each ICO size from the largest image unless given them: hand it every size drawn for itself.
    largest = rendered[max(ICO)]
    largest.save(RESOURCES / "icon.ico", format="ICO", sizes=[(s, s) for s in ICO],
                 append_images=[rendered[s] for s in ICO if s != max(ICO)])
    rendered[max(ICNS)].save(RESOURCES / "icon.icns", format="ICNS",
                             append_images=[rendered[s] for s in ICNS if s != max(ICNS)])
    print(f"written app-icon/ ({len(HICOLOR)} sizes), icon.png, icon.ico ({len(ICO)} sizes), icon.icns ({len(ICNS)})")


if __name__ == "__main__":
    main()

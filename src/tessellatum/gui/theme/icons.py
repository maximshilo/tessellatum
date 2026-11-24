"""The interface's icons: Lucide outlines (ISC, see ``resources/icons/LICENSE``), drawn in the theme's colors.

Each SVG is drawn with ``currentColor``, which is replaced by a token's color as it is loaded, so one file serves both
themes and every state. ``icon`` gives a ``QIcon`` with the normal, active, selected and disabled colors at the sizes
the interface uses; ``svg_file`` writes a recolored copy for a stylesheet, which can only name files.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from tessellatum.gui.theme.tokens import ICON_SIZE, ICON_STROKE, Colors

ICONS_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"

# The commands' icons, by what they do, so that a command never names a file.
NAMES = {
    "open": "folder-open", "save": "save", "import": "image-plus", "paste": "clipboard-paste", "export": "download",
    "undo": "undo-2", "redo": "redo-2", "generate": "wand-sparkles", "cancel": "square", "zoom_in": "zoom-in",
    "zoom_out": "zoom-out", "fit": "scan", "actual_size": "file-image", "focus": "maximize", "reset": "rotate-ccw",
    "source": "image", "generation": "sliders-horizontal", "colors": "palette", "document": "file-text",
    "analysis": "chart-column", "search": "search", "settings": "settings", "show": "eye", "compare": "columns-2",
    "layers": "layers", "ruler": "ruler", "grid": "grid-3x3", "left_panel": "panel-left", "right_panel": "panel-right",
    "info": "info", "warning": "triangle-alert", "error": "circle-x", "success": "circle-check", "done": "check",
    "close": "x", "add": "plus", "remove": "minus", "busy": "loader-circle", "window": "app-window", "quit": "log-out",
    "shortcuts": "keyboard", "expand": "chevron-down", "collapse": "chevron-right", "up": "chevron-up",
    "down": "chevron-down",
}


def svg_text(name: str, color: str, stroke: float = ICON_STROKE) -> str:
    """Icon ``name`` (a file in ``resources/icons``, or a key of ``NAMES``) drawn in ``color``."""
    path = ICONS_DIR / f"{NAMES.get(name, name)}.svg"
    text = path.read_text(encoding="utf-8").replace("currentColor", color)
    return text.replace('stroke-width="2"', f'stroke-width="{stroke:g}"')


@lru_cache(maxsize=1024)
def pixmap(name: str, color: str, size: int, ratio: float = 1.0) -> QPixmap:
    """Icon ``name`` drawn in ``color``, ``size`` pixels square at a device pixel ratio of ``ratio``."""
    renderer = QSvgRenderer(QByteArray(svg_text(name, color).encode("utf-8")))
    side = max(1, round(size * ratio))
    image = QImage(side, side, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, side, side))
    painter.end()
    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(ratio)
    return result


def icon(name: str, colors: Colors, role: str = "text") -> QIcon:
    """A ``QIcon`` of ``name`` in the token ``role``'s color, the accent when selected, and the disabled text color when
    disabled, at every icon size the interface uses and at 1x and 2x pixel ratios."""
    result = QIcon()
    normal = getattr(colors, role)
    for size in ICON_SIZE.values():
        for ratio in (1.0, 2.0):
            result.addPixmap(pixmap(name, normal, size, ratio), QIcon.Normal, QIcon.Off)
            result.addPixmap(pixmap(name, normal, size, ratio), QIcon.Active, QIcon.Off)
            result.addPixmap(pixmap(name, colors.accent, size, ratio), QIcon.Normal, QIcon.On)
            result.addPixmap(pixmap(name, colors.text_disabled, size, ratio), QIcon.Disabled, QIcon.Off)
    return result


_files_dir: Path | None = None


def svg_file(name: str, color: str) -> str:
    """A file holding icon ``name`` in ``color``, for a stylesheet's ``image: url(...)``, as a path with forward
    slashes. The files live in a folder of the app's own, removed when it exits."""
    global _files_dir
    if _files_dir is None:
        _files_dir = Path(tempfile.mkdtemp(prefix="tessellatum-icons-"))
        atexit.register(shutil.rmtree, _files_dir, True)
    path = _files_dir / f"{NAMES.get(name, name)}-{color.lstrip('#')}.svg"
    if not path.exists():
        path.write_text(svg_text(name, color, stroke=2), encoding="utf-8")
    return path.as_posix()


def missing() -> list[str]:
    """The icons ``NAMES`` maps to that aren't in ``resources/icons`` (none in a whole install or build)."""
    return sorted({file for file in NAMES.values() if not (ICONS_DIR / f"{file}.svg").exists()})

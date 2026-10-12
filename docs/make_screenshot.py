"""Regenerate ``docs/assets/screenshot.png``: the app with a page previewed, as the README shows it.

Run from the repository root on a desktop session (Qt's offscreen platform has no fonts, so this uses the native one;
the window is rendered off screen and never appears):

    .venv\\Scripts\\python docs\\make_screenshot.py

The panel starts from the default settings and keeps nothing between runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402

PICTURE = ROOT / "tests" / "sample_images" / "l-photo-lion.jpg"
SHOWN = Version.PAGE
SIZE = (1440, 900)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()  # no store: the user's saved settings are neither read nor changed
    window.setAttribute(Qt.WA_DontShowOnScreen)
    window.resize(*SIZE)
    window.show()
    window.current_image_bgr = pipeline.load_image_bgr(PICTURE)
    window.controls.set_thumbnail(main_window._bgr_to_pixmap(window.current_image_bgr))
    window.version_bar.buttons[SHOWN].setChecked(True)
    window.generate_preview()
    deadline = time.monotonic() + 300
    while window.current_page is None or (window._worker is not None and window._worker.isRunning()):
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("the preview didn't finish")
        time.sleep(0.01)
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
    out = ROOT / "docs" / "assets" / "screenshot.png"
    window.grab().save(str(out))
    print("written", out.relative_to(ROOT), window.statusBar().currentMessage())
    window.close()


if __name__ == "__main__":
    main()

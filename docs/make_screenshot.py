"""Regenerate ``docs/assets/screenshot.png``: the app with a page previewed, as the README shows it; or a set of
screenshots of every main view, and compare a set with reference images.

Run from the repository root on a desktop session (Qt's offscreen platform has no fonts, so this uses the native one;
the window is rendered off screen and never appears):

    .venv\\Scripts\\python docs\\make_screenshot.py                      # the README's screenshot
    .venv\\Scripts\\python docs\\make_screenshot.py --set OUT_DIR        # the set: see VIEWS
    .venv\\Scripts\\python docs\\make_screenshot.py --compare REF_DIR OUT_DIR

The set is the welcome, the Generation and Colors pages with a color picked, a region selected, the analysis with its
overlays, two comparisons, the export dialog and Preferences, in the light and dark themes, at 1280×720 and 1920×1080
(``{theme}-{width}x{height}-{nn}-{view}.png``). ``--compare`` matches each of OUT_DIR's PNGs with REF_DIR's of the
same name and fails when more than ``--tolerance`` of a shot's pixels differ by more than a few levels: fonts and
anti-aliasing differ by platform, so a set is compared only with references from the same platform and display.

The panel starts from the default settings and keeps nothing between runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PICTURE = ROOT / "tests" / "sample_images" / "l-photo-lion.jpg"
SIZE = (1440, 900)
SIZES = ((1280, 720), (1920, 1080))
THEMES = ("light", "dark")
VIEWS = ("welcome", "generation", "colors", "region", "analysis", "compare-original", "split-tinted", "export",
         "preferences")
LEVELS = 24  # a pixel differs when a channel is off by more than this many levels (of 255)
TOLERANCE = 0.01  # the share of a shot's pixels that may differ


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _settle(app, rounds: int = 20) -> None:
    for _ in range(rounds):
        app.processEvents()
        time.sleep(0.01)


def _wait(app, window, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while window.current_page is None or not window.jobs.is_idle():
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("the preview didn't finish")
        time.sleep(0.01)
    _settle(app)


def readme() -> None:
    from PySide6.QtCore import Qt

    from tessellatum.core import pipeline
    from tessellatum.core.painting import Version
    from tessellatum.gui import main_window, theme

    app = _app()
    theme.install(app).set_mode(theme.LIGHT_MODE)  # as the app looks, in its light theme whatever the system's
    window = main_window.MainWindow()  # no store: the user's saved settings are neither read nor changed
    window.setAttribute(Qt.WA_DontShowOnScreen)
    window.resize(*SIZE)
    window.show()
    window.current_image_bgr = pipeline.load_image_bgr(PICTURE)
    window.controls.set_thumbnail(main_window._bgr_to_pixmap(window.current_image_bgr))
    height, width = window.current_image_bgr.shape[:2]
    window.controls.set_picture_size((width, height), PICTURE.name)
    window.version_bar.buttons[Version.PAGE].setChecked(True)
    window.generate_preview()
    _wait(app, window)
    _settle(app, 50)
    out = ROOT / "docs" / "assets" / "screenshot.png"
    window.grab().save(str(out))
    print("written", out.relative_to(ROOT), window.status.text(), window.status.page_label.text())
    window.close()


def screenshot_set(out: Path) -> None:
    from PySide6.QtCore import Qt

    from tessellatum.core.painting import Version
    from tessellatum.gui import main_window, preferences, theme
    from tessellatum.gui.preview_widget import COMPARE_OFF, ORIGINAL, SPLIT_VERTICAL

    out.mkdir(parents=True, exist_ok=True)
    app = _app()
    manager = theme.install(app)
    window = main_window.MainWindow()  # no store: the user's saved settings are neither read nor changed
    for widget in (window, window.export_dialog):
        widget.setAttribute(Qt.WA_DontShowOnScreen)
    window.preferences_dialog = preferences.PreferencesDialog(window.preferences, window.commands, window)
    window.preferences_dialog.setAttribute(Qt.WA_DontShowOnScreen)
    window.show()
    written = []

    def each(view: str, take) -> None:
        """``take(theme, size)`` grabs the shot for every theme and size."""
        number = VIEWS.index(view) + 1
        for mode in THEMES:
            manager.set_mode(theme.LIGHT_MODE if mode == "light" else theme.DARK_MODE)
            for width, height in SIZES:
                window.resize(width, height)
                _settle(app)
                image = take()
                name = f"{mode}-{width}x{height}-{number:02d}-{view}.png"
                image.save(str(out / name))
                written.append(name)

    each("welcome", window.grab)
    window.open_path(PICTURE)
    window.generate_preview()
    _wait(app, window)
    window.show_activity("Generation")
    each("generation", window.grab)
    window.show_activity("Colors")
    window.inspector.select(2)
    each("colors", window.grab)
    window.inspector.select(None)
    window.version_bar.buttons[Version.COMPLETED].click()
    page = window.current_page
    region = int(page.model.region_id_map[page.model.region_id_map.shape[0] // 3,
                                          page.model.region_id_map.shape[1] // 2])
    window.select_region(region, center=False)
    each("region", window.grab)
    window.select_region(None)
    window.version_bar.buttons[Version.PAGE].click()
    window.show_activity("Analysis")
    _wait(app, window)
    each("analysis", window.grab)
    window.show_activity("Generation")
    window.set_compare_with(ORIGINAL)
    each("compare-original", window.grab)
    window.set_compare_with(Version.TINTED)
    window.set_compare(SPLIT_VERTICAL)
    each("split-tinted", window.grab)
    window.set_compare(COMPARE_OFF)

    def dialog(show, widget):
        def take():
            show()
            _settle(app)
            image = widget.grab()
            widget.hide()
            return image
        return take

    each("export", dialog(window.commands["export"].trigger, window.export_dialog))
    each("preferences", dialog(window.open_preferences, window.preferences_dialog))
    (out / "set.json").write_text(json.dumps({"platform": app.platformName(), "shots": written}, indent=1),
                                  encoding="utf-8")
    print(f"written {len(written)} shots to {out}")
    window.close()


def compare(reference: Path, out: Path, tolerance: float) -> int:
    import numpy as np
    from PIL import Image

    failed, rows = 0, []
    for ref in sorted(reference.glob("*.png")):
        new = out / ref.name
        if not new.exists():
            rows.append((ref.name, "missing", ""))
            failed += 1
            continue
        a = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(new).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            rows.append((ref.name, "size", f"{a.shape[1]}x{a.shape[0]} -> {b.shape[1]}x{b.shape[0]}"))
            failed += 1
            continue
        share = float((np.abs(a - b).max(axis=2) > LEVELS).mean())
        ok = share <= tolerance
        failed += not ok
        rows.append((ref.name, "ok" if ok else "differs", f"{share:.2%}"))
    for name, verdict, detail in rows:
        print(f"{verdict:8} {detail:>16}  {name}")
    print(f"{len(rows) - failed} of {len(rows)} within {tolerance:.1%} of their pixels")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", type=Path, metavar="OUT_DIR", help="write the set of screenshots there")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REF_DIR", "OUT_DIR"))
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare, args.tolerance)
    if args.set:
        screenshot_set(args.set)
    else:
        readme()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Render the app at window sizes and interface scales, and report text that doesn't fit: the platform and scaling
check (the wiki's Quality and performance page).

Run from the repository root on a desktop session (Qt's offscreen platform has no fonts; on Linux CI, under xvfb):

    .venv\\Scripts\\python docs\\check_layout.py OUT_DIR
    .venv\\Scripts\\python docs\\check_layout.py OUT_DIR --scales 1 1.25 1.5 2 --sizes 1280x720 1920x1080 --pseudo

For each interface scale (``QT_SCALE_FACTOR``, as Preferences > Interface scale sets it), a process of its own opens
the window off screen, makes the lion's page, and at each window size (in logical pixels at that scale) grabs:
- the window on each of the sidebar's pages, in the light and the dark theme;
- the export dialog and the Preferences' pages.
In each, a label or button narrower than the least its text needs (its ``minimumSizeHint``), or cut at the side by
the panel it is in, is reported: its text is cut. A window size smaller than the window's own minimum there is reported too (the window can't be made that small,
or its panels are cut).
``--pseudo`` runs in the pseudo-locale (``gui.i18n``), every string about 40% longer. The PNGs and ``layout.md``, the
results by cell, go to OUT_DIR; the exit status is 1 if any text is cut.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICTURE = ROOT / "tests" / "sample_images" / "l-photo-lion.jpg"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--sizes", nargs="+", default=["1280x720", "1920x1080"])
    parser.add_argument("--themes", nargs="+", default=["light", "dark"])
    parser.add_argument("--pseudo", action="store_true")
    parser.add_argument("--one", type=float, help=argparse.SUPPRESS)  # a child: one scale
    return parser.parse_args()


def _cut(root) -> list[dict]:
    """Every visible label and button in ``root`` narrower than its text needs."""
    from PySide6.QtWidgets import QAbstractButton, QLabel

    found = []
    for widget in [*root.findChildren(QLabel), *root.findChildren(QAbstractButton)]:
        if not widget.isVisible() or not widget.text().strip():
            continue
        if isinstance(widget, QLabel) and widget.wordWrap():
            continue
        need = widget.minimumSizeHint().width()
        shown = widget.visibleRegion().boundingRect().width()  # what its parents leave of it, side to side
        if need > widget.width() + 1 or 0 < shown < min(need, widget.width()) - 1:
            found.append({"widget": type(widget).__name__, "name": widget.objectName(),
                          "text": widget.text().replace("&", "")[:60], "width": min(widget.width(), shown or 10**6),
                          "needs": need})
    return found


def _settle(app, rounds: int = 30) -> None:
    for _ in range(rounds):
        app.processEvents()
        time.sleep(0.01)


def child(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from tessellatum.core import pipeline, settings
    from tessellatum.gui import i18n, main_window, preferences, theme

    app = QApplication.instance() or QApplication([])
    i18n.install(app)
    manager = theme.install(app)
    main_window._warm_up_pipeline = lambda: None
    window = main_window.MainWindow()  # no store: the user's saved settings are neither read nor changed
    window.setAttribute(Qt.WA_DontShowOnScreen)
    window.export_dialog.setAttribute(Qt.WA_DontShowOnScreen)
    window.show()
    window.current_image_bgr = pipeline.load_image_bgr(PICTURE)
    window.generate_preview()
    deadline = time.monotonic() + 300
    while window.current_page is None or not window.jobs.is_idle():
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("the preview didn't finish")
        time.sleep(0.01)
    scale = f"{args.one:g}"
    cells = []
    for size in args.sizes:
        width, height = (int(n) for n in size.lower().split("x"))
        logical = (round(width / args.one), round(height / args.one))  # the window fills a screen of that size
        for mode in args.themes:
            manager.set_mode(theme.LIGHT_MODE if mode == "light" else theme.DARK_MODE)
            window.resize(*logical)
            _settle(app)
            minimum = window.minimumSizeHint()  # once the panels have adapted to the width (the inspector folds away)
            cut: list[dict] = []
            shots = []
            for category in (*settings.CATEGORIES, main_window.ANALYSIS):
                window.show_activity(category)
                _settle(app, 10)
                name = f"window_{scale}x_{size}_{mode}_{category.lower()}.png"
                window.grab().save(str(args.out / name))
                shots.append(name)
                cut += [{**c, "where": category} for c in _cut(window)]
            window.commands["export"].trigger()
            _settle(app, 10)
            name = f"export_{scale}x_{size}_{mode}.png"
            window.export_dialog.grab().save(str(args.out / name))
            shots.append(name)
            cut += [{**c, "where": "export"} for c in _cut(window.export_dialog)]
            window.export_dialog.reject()
            if window.preferences_dialog is None:  # made here, off screen, before the window shows it
                window.preferences_dialog = preferences.PreferencesDialog(window.preferences, window.commands, window)
                window.preferences_dialog.setAttribute(Qt.WA_DontShowOnScreen)
            window.open_preferences()
            dialog = window.preferences_dialog
            for index in range(dialog.pages.count()):
                dialog.pages.setCurrentRow(index)
                _settle(app, 5)
                name = f"preferences{index}_{scale}x_{size}_{mode}.png"
                dialog.grab().save(str(args.out / name))
                shots.append(name)
                cut += [{**c, "where": f"preferences {index}"} for c in _cut(dialog)]
            dialog.close()
            cells.append({
                "scale": args.one, "size": size, "theme": mode, "logical": logical,
                "window": [window.width(), window.height()], "minimum": [minimum.width(), minimum.height()],
                "fits": minimum.width() <= logical[0] and minimum.height() <= logical[1],
                "cut": cut, "shots": shots, "platform": QApplication.platformName(),
                "screen": [app.primaryScreen().size().width(), app.primaryScreen().size().height(),
                           app.primaryScreen().devicePixelRatio()],
            })
    (args.out / f"layout_{scale}.json").write_text(json.dumps(cells, indent=1, ensure_ascii=False), encoding="utf-8")
    window.close()
    return 0


def main() -> int:
    args = _parse()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.one is not None:
        code = child(args)
        sys.stdout.flush()
        os._exit(code)  # the results are written; Qt's teardown at exit can crash with the page's workers about
    cells = []
    for scale in args.scales:
        env = {**os.environ, "QT_SCALE_FACTOR": f"{scale:g}"}
        if args.pseudo:
            env["TESSELLATUM_PSEUDO_LOCALE"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), str(args.out), "--one", f"{scale:g}",
                   "--sizes", *args.sizes, "--themes", *args.themes]
        subprocess.run(command, env=env, check=True, timeout=1200)
        cells += json.loads((args.out / f"layout_{scale:g}.json").read_text(encoding="utf-8"))
    lines = ["| scale | window | theme | fits | text cut |", "|---|---|---|---|---|"]
    for cell in cells:
        cut = "; ".join(f"{c['where']}: {c['text']} ({c['width']} of {c['needs']} px)" for c in cell["cut"]) or "none"
        fits = "yes" if cell["fits"] else f"no (at least {cell['minimum'][0]}×{cell['minimum'][1]})"
        lines.append(f"| {cell['scale']:g}× | {cell['size']} | {cell['theme']} | {fits} | {cut} |")
    platform = cells[0]["platform"] if cells else "?"
    report = f"Platform: {platform}, {sys.platform}; pseudo-locale: {'yes' if args.pseudo else 'no'}\n\n" + \
        "\n".join(lines) + "\n"
    (args.out / "layout.md").write_text(report, encoding="utf-8")
    sys.stdout.buffer.write(report.encode("utf-8"))
    return 1 if any(cell["cut"] for cell in cells) else 0


if __name__ == "__main__":
    sys.exit(main())

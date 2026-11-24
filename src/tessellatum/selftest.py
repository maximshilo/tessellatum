"""A check that a build of the app can make and export a page: ``Tessellatum --self-test OUTPUT_DIR``.

The release workflow runs it on every standalone build it publishes, on Windows, Linux and macOS: launching the app
alone wouldn't show a model, a compiled kernel or QtPdf missing from the bundle, as they load only once a page is made.
It makes a page of a picture drawn here (so it needs no file), through every network the pipeline runs on a picture,
saves its PDF and a PNG of each version into ``OUTPUT_DIR``, and exits 0. On a failure it writes the traceback to
``OUTPUT_DIR/self-test-error.txt`` too, since a windowed build has no console to print it to, and exits 1.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

ERROR_FILE = "self-test-error.txt"


def picture() -> np.ndarray:
    """A 640 x 420 BGR picture of soft color fields, a few shapes and a dark bar: something of every kind to make
    regions of, with no file to read."""
    h, w = 420, 640
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    image = np.stack(
        [
            120 + 100 * np.sin(x / 90.0),
            110 + 90 * np.cos(y / 70.0),
            130 + 80 * np.sin((x + y) / 110.0),
        ],
        axis=2,
    )
    image[60:180, 80:240] = (40, 160, 220)
    image[np.hypot(x - 450, y - 260) < 90] = (200, 90, 60)
    image[330:360, 60:580] = (25, 25, 25)
    return np.clip(image, 0, 255).astype(np.uint8)


def check_interface_assets() -> None:
    """Raise if the build lacks the interface's font or icons, or their licenses: a window would open without them,
    in the platform's font and with blank buttons, so only a check shows them missing."""
    from tessellatum.gui import theme
    from tessellatum.gui.theme import icons, tokens

    lacking = [name for name in (*tokens.FONT_FILES, "OFL.txt") if not (theme.FONTS_DIR / name).is_file()]
    lacking += [f"{name}.svg" for name in icons.missing()]
    if not (icons.ICONS_DIR / "LICENSE").is_file():
        lacking.append("icons/LICENSE")
    if lacking:
        raise RuntimeError(f"the interface's assets are missing: {', '.join(lacking)}")


def run(out_dir: Path) -> int:
    """Make a page and export it into ``out_dir``; 0 if every file was written, else 1 (see the module)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from tessellatum import __version__
        from tessellatum.core import difficulty, export, pipeline
        from tessellatum.core.painting import Version

        started = time.perf_counter()
        image = picture()
        params = difficulty.params_for_preset("Medium")
        page = pipeline.generate(image, params, pipeline.export_long_edge(image), vector=True)
        if page.drawing is None or page.num_regions < 2:
            raise RuntimeError(f"the page has {page.num_regions} regions and no drawing")
        written = [out_dir / "self-test.pdf"]
        export.save_pdf(page.drawing, page.palette_rgb, written[0], painting=page.painting)
        for version in Version:
            path = out_dir / f"self-test-{version.name.lower()}.png"
            export.export_png(page.drawing, page.palette_rgb, path, 150, version, page.painting)
            written.append(path)
        missing = [p.name for p in written if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"not written: {', '.join(missing)}")
        check_interface_assets()
        print(
            f"Tessellatum {__version__} self-test passed in {time.perf_counter() - started:.1f} s: "
            f"{page.num_regions} regions, {page.num_colors_used} colors; wrote {', '.join(p.name for p in written)}"
        )
        return 0
    except Exception:  # noqa: BLE001 -- any failure fails the check, with its traceback
        message = traceback.format_exc()
        (out_dir / ERROR_FILE).write_text(message, encoding="utf-8")
        print(message, file=sys.stderr)
        return 1

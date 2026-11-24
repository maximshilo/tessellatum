"""The logo and the app's icons: present at every size each platform asks for, each size drawn for itself."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum import main  # noqa: E402

_app = QApplication.instance() or QApplication([])

RESOURCES = main.RESOURCES_DIR


@pytest.mark.parametrize("name", ["logo.svg", "logo-small.svg"])
def test_the_logo_masters_are_plain_svg(name):
    renderer = QSvgRenderer(str(RESOURCES / name))
    assert renderer.isValid() and renderer.defaultSize() == QSize(256, 256)
    text = (RESOURCES / name).read_text(encoding="utf-8")
    assert "<script" not in text and "href" not in text  # nothing external, nothing run


def test_the_window_icon_has_every_size():
    sizes = {(s.width(), s.height()) for s in main.window_icon().availableSizes()}
    assert sizes == {(s, s) for s in main.APP_ICON_SIZES}


def test_each_platform_icon_carries_its_sizes():
    ico = Image.open(RESOURCES / "icon.ico")
    assert {16, 24, 32, 48, 256} <= {w for w, _h in ico.ico.sizes()}
    icns = Image.open(RESOURCES / "icon.icns")
    assert {(16, 16, 2), (512, 512, 2)} <= set(icns.info["sizes"])
    for size in main.APP_ICON_SIZES:
        png = Image.open(main.APP_ICON_DIR / f"{size}.png")
        assert png.size == (size, size) and png.mode == "RGBA" and not png.info.get("exif")


def test_small_sizes_are_their_own_drawing():
    """16 px is the simplified logo, not the large one shrunk: it differs clearly from 512 px scaled down."""
    small = np.asarray(Image.open(main.APP_ICON_DIR / "16.png").convert("RGBA"), float)
    shrunk = np.asarray(Image.open(main.APP_ICON_DIR / "512.png").convert("RGBA").resize((16, 16), Image.LANCZOS), float)
    assert np.abs(small - shrunk).mean() > 8


def test_the_icons_are_rendered_from_the_masters(tmp_path, monkeypatch):
    """The icons committed are what the masters render now: an edited master needs make_icons.py run again."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_icons", Path(__file__).parents[1] / "packaging" / "icons" / "make_icons.py")
    make_icons = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_icons)
    for size in (16, 48, 256):
        rendered = np.asarray(make_icons.render(size), int)
        committed = np.asarray(Image.open(main.APP_ICON_DIR / f"{size}.png").convert("RGBA"), int)
        assert np.abs(rendered - committed).max() <= 2, size

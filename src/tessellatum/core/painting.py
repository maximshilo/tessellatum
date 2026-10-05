"""The page painted in: the versions of a page the app shows and exports besides the page itself.

A page is printed to be painted. Two more versions show it painted, drawn
from the same regions, palette and ink as the page:

- **Completed**: the finished painting. Every region is filled in its legend
  color, which covers the page's lines and numbers, since their grays are
  meant to vanish under the paint. What the page prints of the picture itself
  -- line art's ink, the marks in a face, the letters of a sign -- shows on
  top, and so does bare paper wherever the page has no region and no ink.
  It is the painting the benchmarks score (``benchmarks/QUALITY_BENCHMARKS.md``),
  but for the letters' own tones round their outline, laid over the paint
  rather than over white.
- **Tinted**: the page with its colors laid on faintly, as a wash
  ``TINT_OPACITY`` of the way from white to the paint, and the lines, numbers
  and ink showing through it as they would through a translucent paint: the
  page multiplied by the wash. A guide to paint from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from tessellatum.core.render import PAPER

# How far the tinted version's wash goes from white to the paint. A wash multiplies the page, which keeps the ratio of
# every gray on it to the paper round it, so the lines and numbers stand out from a wash as they do from paper, only
# darker; at this, black washes to a 60% gray, its numbers about as dark as the page's lines on paper.
TINT_OPACITY = 0.4


class Version(Enum):
    """The versions of a page there are to look at and to export."""

    PAGE = "Page"
    COMPLETED = "Completed"
    TINTED = "Tinted"

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS = {
    Version.PAGE: "The coloring page as it prints: lines and numbers, ready to paint.",
    Version.COMPLETED: "The page painted in: every region in its legend color, the lines and numbers covered, "
    "the picture's printed ink on top.",
    Version.TINTED: "The page with its colors laid on faintly, the lines and numbers showing through: "
    "a guide to paint from.",
}


@dataclass
class Painting:
    """What a page is painted with: which paint goes where, and the ink the paint leaves showing.

    ``region_color`` gives each region id of ``region_id_map`` (-1: in none, left bare) its paint, an index into
    ``palette_rgb``, whose first colors are the legend's, in legend order, so paint ``i`` is numbered ``i + 1``. A
    region too thin to have a number keeps its own color, which may lie past the legend's. ``ink`` is the ink the page
    prints of the picture itself, 0 bare paper to 255 solid in ``ink_gray`` (see ``render.RenderedPage.picture_ink``).
    """

    region_id_map: np.ndarray
    region_color: np.ndarray
    palette_rgb: list[tuple[int, int, int]]
    ink: np.ndarray
    ink_gray: int

    def paint(self) -> np.ndarray:
        """HxWx3 uint8: every region in its paint, white paper in none."""
        return self.fill(self.palette_rgb)

    def fill(self, colors) -> np.ndarray:
        """HxWx3 uint8: every region in its paint's entry of ``colors`` (one per paint, RGB), white paper in none."""
        lut = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[np.asarray(self.region_color, dtype=np.int64)]
        lut = np.vstack([lut, np.full((1, 3), PAPER, dtype=np.uint8)])  # id -1, in no region, takes the last: paper
        return lut[self.region_id_map]


def tint_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """The wash the tinted version lays down for a paint of ``rgb``, on white paper."""
    return tuple(int(v) for v in np.rint(PAPER + TINT_OPACITY * (np.asarray(rgb, dtype=np.float64) - PAPER)))


def completed(painting: Painting) -> Image.Image:
    """The finished painting: every region in its paint, the picture's ink laid over it (see the module's docstring)."""
    painted = painting.paint()
    inked = painting.ink > 0
    if inked.any():
        coverage = painting.ink[inked].astype(np.float64)[:, None] / PAPER
        under = painted[inked].astype(np.float64)
        painted[inked] = np.rint(under + (painting.ink_gray - under) * coverage).astype(np.uint8)
    return Image.fromarray(painted, "RGB")


def tinted(page: Image.Image, painting: Painting) -> Image.Image:
    """``page`` under a wash of its paint: multiplied by each paint taken ``TINT_OPACITY`` of the way from white."""
    wash = painting.fill([tint_rgb(rgb) for rgb in painting.palette_rgb]).astype(np.uint16)
    # Rounded to the nearest level: a product of two levels over 255 never falls halfway between two.
    out = (np.asarray(page.convert("RGB"), dtype=np.uint16) * wash + PAPER // 2) // PAPER
    return Image.fromarray(out.astype(np.uint8), "RGB")


def render_version(version: Version, page: Image.Image, painting: Painting | None) -> Image.Image:
    """The image of ``version`` of a page: ``page`` itself, or it painted with ``painting``."""
    if version is Version.PAGE:
        return page
    if painting is None:
        raise ValueError(f"the {version.value.lower()} version needs what the page is painted with")
    return completed(painting) if version is Version.COMPLETED else tinted(page, painting)

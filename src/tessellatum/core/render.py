"""Render the final coloring page: lines + numbers on a white canvas."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.boundaries import trace_boundaries
from tessellatum.core.regions import Region

MIN_LABEL_RADIUS_PX = 9.0
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 40
FONT_SIZE_RADIUS_RATIO = 0.85
OUTLINE_WIDTH = 2  # px across the whole line, which straddles the crack it is drawn on

# Lines run along pixel cracks, so their coordinates are half-integers; a shift
# of one bit carries them exactly.
_SUBPIXEL_BITS = 1
# Room for a line on the page edge, and for the widening below, before cropping.
_CANVAS_MARGIN_PX = OUTLINE_WIDTH


@dataclass
class Label:
    """A region's number as drawn on the page."""

    region_id: int
    text: str
    font_size: int  # px: the font's em size
    box: tuple[float, float, float, float]  # (x0, y0, x1, y1): the text's bounding box on the page


@dataclass
class RenderedPage:
    image: Image.Image  # RGB: outlines + numbers
    outlines: Image.Image  # "L": the outlines alone, 0 = black line, 255 = paper
    labels: list[Label]  # every number on the page, in drawing order
    strokes: list[np.ndarray]  # every line drawn, in drawing order: Nx2 float64 (x, y); a closed one returns to its first point


def render_page(size: tuple[int, int], regions: list[Region], region_id_map: np.ndarray) -> RenderedPage:
    """Draw the boundaries of ``region_id_map`` + numbers for ``regions`` onto a white ``size`` canvas.

    Every boundary between two regions is drawn once, as the line its two
    regions share (see ``boundaries.trace_boundaries``), rather than as part
    of an outline around each of them.

    Returns the page, plus what it was built from: the lines on their own, the
    geometry each was drawn from, and where each number went.
    """
    strokes = trace_boundaries(region_id_map)
    outlines = _draw_lines(size, strokes)

    page = outlines.convert("RGB")
    draw = ImageDraw.Draw(page)
    labels: list[Label] = []
    for region in regions:
        if region.interior_radius < MIN_LABEL_RADIUS_PX:
            continue
        font_size = int(
            max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO))
        )
        text = str(region.color_index + 1)
        bbox = _text_bbox(text, font_size)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = region.interior_point
        left = min(max(x - tw / 2, 0), size[0] - tw)
        top = min(max(y - th / 2, 0), size[1] - th)
        draw.text((left - bbox[0], top - bbox[1]), text, fill="black", font=_font(font_size))
        labels.append(Label(region.region_id, text, font_size, (left, top, left + tw, top + th)))

    return RenderedPage(image=page, outlines=outlines, labels=labels, strokes=strokes)


def _draw_lines(size: tuple[int, int], strokes: list[np.ndarray]) -> Image.Image:
    """The line layer: ``strokes`` drawn ``OUTLINE_WIDTH`` px wide, 0 = line, 255 = paper.

    A line sits on the crack between two pixels, so it cannot be centered on a
    pixel. It is rasterized one pixel wide -- which puts it on the pixel right
    of or below the crack -- and then widened up and left, so that it covers
    the pixels on both sides of the crack evenly. The canvas has a margin so
    that a line on the page edge, whose other half falls off the page, is
    drawn rather than clipped away.
    """
    width, height = size
    margin = _CANVAS_MARGIN_PX
    canvas = np.zeros((height + 2 * margin, width + 2 * margin), dtype=np.uint8)
    paths = [np.rint((stroke + margin) * (1 << _SUBPIXEL_BITS)).astype(np.int32) for stroke in strokes]
    if paths:
        cv2.polylines(canvas, paths, False, 255, thickness=1, lineType=cv2.LINE_8, shift=_SUBPIXEL_BITS)
        # The anchor decides which way the widening goes: at (0, 0) a 2x2
        # element spreads a pixel up and left, which is the side of the crack
        # the one-pixel line missed.
        spread_up_left = (OUTLINE_WIDTH - 1) // 2
        cv2.dilate(
            canvas,
            np.ones((OUTLINE_WIDTH, OUTLINE_WIDTH), np.uint8),
            dst=canvas,
            anchor=(spread_up_left, spread_up_left),
        )
    return Image.fromarray(np.invert(canvas[margin : margin + height, margin : margin + width]), "L")


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


_MEASURING_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@lru_cache(maxsize=4096)
def _text_bbox(text: str, font_size: int) -> tuple[int, int, int, int]:
    """``textbbox`` of ``text`` at the origin, as measured on an RGB page."""
    return _MEASURING_DRAW.textbbox((0, 0), text, font=_font(font_size))

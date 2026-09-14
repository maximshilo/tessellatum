"""Render the final coloring page: outlines + numbers on a white canvas."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.regions import Region

MIN_LABEL_RADIUS_PX = 9.0
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 40
FONT_SIZE_RADIUS_RATIO = 0.85
OUTLINE_WIDTH = 2

# Pillow never draws a polygon outline outside the polygon itself (it masks
# wide strokes to the fill); keep a little extra room around it anyway.
_OUTLINE_MARGIN_PX = OUTLINE_WIDTH + 1
# Outlines that fit inside one tile of this size are drawn together on that tile.
_TILE_PX = 256


def render_page(size: tuple[int, int], regions: list[Region]) -> Image.Image:
    """Draw outlines + numbers for ``regions`` onto a white ``size`` canvas."""
    width, height = size
    # Pillow draws a wide polygon outline through a scratch mask as big as the
    # image it draws on, so drawing straight onto the page would cost a
    # full-page allocation per region. Instead, draw each outline on a small
    # crop -- shared by all outlines that fit in the same tile -- and paste it
    # back. Same pixels: strokes are pure black on white, so the page is just
    # their union, in any order. A 1-byte canvas keeps the copying cheap.
    outlines = Image.new("L", size, 255)
    tiles: dict[tuple[int, int], list[np.ndarray]] = {}

    for region in regions:
        contour = region.contour
        if len(contour) == 1:
            ImageDraw.Draw(outlines).point((int(contour[0, 0, 0]), int(contour[0, 0, 1])), fill=0)
            continue
        if len(contour) < 2:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        box = (
            max(x - _OUTLINE_MARGIN_PX, 0),
            max(y - _OUTLINE_MARGIN_PX, 0),
            min(x + w + _OUTLINE_MARGIN_PX, width),
            min(y + h + _OUTLINE_MARGIN_PX, height),
        )
        tile = (box[0] // _TILE_PX, box[1] // _TILE_PX)
        if ((box[2] - 1) // _TILE_PX, (box[3] - 1) // _TILE_PX) == tile:
            tiles.setdefault(tile, []).append(contour)
        else:
            _draw_outlines(outlines, box, [contour])

    for (tx, ty), contours in tiles.items():
        box = (tx * _TILE_PX, ty * _TILE_PX, min((tx + 1) * _TILE_PX, width), min((ty + 1) * _TILE_PX, height))
        _draw_outlines(outlines, box, contours)

    page = outlines.convert("RGB")
    draw = ImageDraw.Draw(page)
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
        draw_x = min(max(x - tw / 2, 0), size[0] - tw) - bbox[0]
        draw_y = min(max(y - th / 2, 0), size[1] - th) - bbox[1]
        draw.text((draw_x, draw_y), text, fill="black", font=_font(font_size))

    return page


def _draw_outlines(canvas: Image.Image, box: tuple[int, int, int, int], contours: list[np.ndarray]) -> None:
    """Draw polygon outlines lying entirely inside ``box`` onto ``canvas``."""
    crop = canvas.crop(box)
    draw = ImageDraw.Draw(crop)
    for contour in contours:
        local_points = (contour.reshape(-1, 2) - (box[0], box[1])).reshape(-1).tolist()
        draw.polygon(local_points, outline=0, width=OUTLINE_WIDTH)
    canvas.paste(crop, box)


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


_MEASURING_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@lru_cache(maxsize=4096)
def _text_bbox(text: str, font_size: int) -> tuple[int, int, int, int]:
    """``textbbox`` of ``text`` at the origin, as measured on an RGB page."""
    return _MEASURING_DRAW.textbbox((0, 0), text, font=_font(font_size))

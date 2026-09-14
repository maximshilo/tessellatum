"""Build the color-index -> swatch legend shown alongside the coloring page."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SWATCH_SIZE = 48
SWATCH_GAP = 12
SWATCH_BORDER = 2
LABEL_FONT_RATIO = 0.42


def render_legend(palette_bgr: np.ndarray, width: int) -> Image.Image:
    """Render a wrapped grid of numbered color swatches, ``width`` pixels wide."""
    cell_w = SWATCH_SIZE + SWATCH_GAP
    cols = max(1, width // cell_w)
    rows = (len(palette_bgr) + cols - 1) // cols
    height = rows * (SWATCH_SIZE + SWATCH_GAP) + SWATCH_GAP

    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default(size=int(SWATCH_SIZE * LABEL_FONT_RATIO))

    for index, bgr in enumerate(palette_bgr):
        row, col = divmod(index, cols)
        x0 = SWATCH_GAP + col * cell_w
        y0 = SWATCH_GAP + row * (SWATCH_SIZE + SWATCH_GAP)
        x1, y1 = x0 + SWATCH_SIZE, y0 + SWATCH_SIZE
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        draw.rectangle([x0, y0, x1, y1], fill=rgb, outline="black", width=SWATCH_BORDER)

        text = str(index + 1)
        text_color = "white" if _luminance(rgb) < 140 else "black"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + SWATCH_SIZE / 2 - tw / 2 - bbox[0], y0 + SWATCH_SIZE / 2 - th / 2 - bbox[1]),
            text,
            fill=text_color,
            font=font,
        )

    return legend


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

"""Build the color-index -> swatch legend shown alongside the coloring page."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A swatch's size on paper, so that a legend prints as large on an export as on a preview of the same picture, and a
# number on it stays legible at 300 dpi: 12 mm swatches 3 mm apart, framed 0.5 mm wide, numbered 0.42 of a swatch high
# (14 pt). On an 1100 px preview that is about the 48 px swatch the legend used to have.
SWATCH_MM = 12.0
SWATCH_GAP_MM = 3.0
SWATCH_BORDER_MM = 0.5
LABEL_FONT_RATIO = 0.42


def render_legend(palette_bgr: np.ndarray, width: int, px_per_mm: float) -> Image.Image:
    """Render a wrapped grid of numbered color swatches, ``width`` pixels wide, at ``px_per_mm`` pixels a millimeter."""
    swatch = max(8, round(SWATCH_MM * px_per_mm))
    gap = max(2, round(SWATCH_GAP_MM * px_per_mm))
    border = max(1, round(SWATCH_BORDER_MM * px_per_mm))
    cell_w = swatch + gap
    cols = max(1, width // cell_w)
    rows = (len(palette_bgr) + cols - 1) // cols
    height = rows * (swatch + gap) + gap

    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default(size=round(swatch * LABEL_FONT_RATIO))

    for index, bgr in enumerate(palette_bgr):
        row, col = divmod(index, cols)
        x0 = gap + col * cell_w
        y0 = gap + row * (swatch + gap)
        x1, y1 = x0 + swatch, y0 + swatch
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        draw.rectangle([x0, y0, x1, y1], fill=rgb, outline="black", width=border)

        text = str(index + 1)
        text_color = number_fill(rgb)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + swatch / 2 - tw / 2 - bbox[0], y0 + swatch / 2 - th / 2 - bbox[1]),
            text,
            fill=text_color,
            font=font,
        )

    return legend


def number_fill(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """What a swatch of ``rgb`` has its number written in: white on a dark swatch, black on a light one."""
    return (255, 255, 255) if _luminance(rgb) < 140 else (0, 0, 0)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

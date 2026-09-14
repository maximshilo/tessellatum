"""Render the final coloring page: outlines + numbers on a white canvas."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from tessellatum.core.regions import Region

MIN_LABEL_RADIUS_PX = 9.0
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 40
FONT_SIZE_RADIUS_RATIO = 0.85
OUTLINE_WIDTH = 2


def render_page(size: tuple[int, int], regions: list[Region]) -> Image.Image:
    """Draw outlines + numbers for ``regions`` onto a white ``size`` canvas."""
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)

    for region in regions:
        points = [(int(p[0][0]), int(p[0][1])) for p in region.contour]
        if len(points) >= 2:
            draw.polygon(points, outline="black", width=OUTLINE_WIDTH)
        elif len(points) == 1:
            draw.point(points[0], fill="black")

    for region in regions:
        if region.interior_radius < MIN_LABEL_RADIUS_PX:
            continue
        font_size = int(
            max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, region.interior_radius * FONT_SIZE_RADIUS_RATIO))
        )
        font = ImageFont.load_default(size=font_size)
        text = str(region.color_index + 1)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = region.interior_point
        draw_x = min(max(x - tw / 2, 0), size[0] - tw) - bbox[0]
        draw_y = min(max(y - th / 2, 0), size[1] - th) - bbox[1]
        draw.text((draw_x, draw_y), text, fill="black", font=font)

    return page

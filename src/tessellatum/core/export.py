"""Save a generated coloring page + legend to PNG or PDF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

LEGEND_GAP_PX = 24


def save_png(page: Image.Image, legend: Image.Image, path: Path) -> None:
    """Stack the page and legend into a single PNG (PNG has no page concept)."""
    width = max(page.width, legend.width)
    height = page.height + LEGEND_GAP_PX + legend.height
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(page, ((width - page.width) // 2, 0))
    combined.paste(legend, ((width - legend.width) // 2, page.height + LEGEND_GAP_PX))
    combined.save(path, "PNG")


def save_pdf(page: Image.Image, legend: Image.Image, path: Path, dpi: int = 300) -> None:
    """Save as a two-page PDF: the coloring page, then the color legend."""
    page.save(path, "PDF", save_all=True, append_images=[legend], resolution=dpi)

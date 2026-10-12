"""Regenerate the images the README, the wiki and the repository's social preview show, with the current pipeline.

Run from the repository root after a change that visibly alters the pages (and before a release):

    .venv\\Scripts\\python docs\\make_assets.py

It writes:

- ``docs/assets/banner.png``: the README's banner;
- ``docs/assets/social-preview.png``: 1280 x 640, for the repository's Settings > Social preview (uploaded by hand:
  GitHub has no API for it);
- ``docs/gallery/<name>.jpg``: one strip per sample image -- the original, the coloring page and the page painted in.

Every picture comes from ``tests/sample_images/`` (CC0 or public domain, see its ``SOURCES.md``), so the outputs are
safe to publish. Pillow writes no metadata into them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tessellatum.core import difficulty, pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402

SAMPLES = ROOT / "tests" / "sample_images"
ASSETS = ROOT / "docs" / "assets"
GALLERY = ROOT / "docs" / "gallery"

# Gallery entries: output name, sample image, difficulty preset. The README shows the first few (see its Gallery).
GALLERY_ITEMS = [
    ("photo-lion", "l-photo-lion.jpg", "Medium"),
    ("painting-portrait", "l-painting-face-girl-with-a-pearl-earring.jpg", "Medium"),
    ("cartoon-line-art", "m-cartoon-bold-lines-girl.png", "Medium"),
    ("comic-text", "m-comics-upside-downs-writing-pig.jpg", "Medium"),
    ("photo-castle", "l-photo-switzerland-castle.jpg", "Medium"),
    ("photo-cat", "l-photo-cats-face.jpg", "Medium"),
    ("photo-architecture", "l-photo-palermo-castle.jpg", "Hard"),
    ("photo-sunset", "l-photo-sunset-silhouette-reflections.jpg", "Medium"),
    ("illustration-scan", "m-cartoon-complex.jpg", "Medium"),
    ("cartoon-grays", "m-cartoon-bold-lines-reaper.png", "Medium"),
]

PAPER = (255, 255, 255)
BACKDROP = (246, 245, 241)
INK = (27, 27, 26)
ACCENT = (31, 78, 121)
MUTED = (98, 97, 92)
PANEL_HEIGHT = 560
GAP = 14
LABEL_BAR = 46


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Segoe UI where it exists (Windows), else Pillow's own scalable font."""
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_page(sample: str, preset: str):
    image = pipeline.load_image_bgr(SAMPLES / sample)
    params = difficulty.params_for_preset(preset)
    pipeline.clear_cache()
    page = pipeline.generate(image, params, pipeline.working_long_edge(image, params))
    source = Image.fromarray(np.ascontiguousarray(pipeline.resize_to_long_edge(image, max(page.page.size))[:, :, ::-1]))
    return source.resize(page.page.size, Image.LANCZOS), page


def fit_height(img: Image.Image, height: int) -> Image.Image:
    return img.convert("RGB").resize((round(img.width * height / img.height), height), Image.LANCZOS)


def strip(panels: list[Image.Image], labels: list[str]) -> Image.Image:
    """Panels side by side at one height, each labeled underneath."""
    sized = [fit_height(p, PANEL_HEIGHT) for p in panels]
    width = sum(p.width for p in sized) + GAP * (len(sized) - 1)
    out = Image.new("RGB", (width, PANEL_HEIGHT + LABEL_BAR), PAPER)
    draw = ImageDraw.Draw(out)
    label_font = font(24)
    x = 0
    for panel, label in zip(sized, labels):
        out.paste(panel, (x, 0))
        draw.rectangle([x, 0, x + panel.width - 1, PANEL_HEIGHT - 1], outline=(214, 213, 207), width=1)
        draw.text((x + panel.width / 2, PANEL_HEIGHT + LABEL_BAR / 2), label, fill=MUTED, font=label_font, anchor="mm")
        x += panel.width + GAP
    return out


def blend_across(left: Image.Image, right: Image.Image, start: float, end: float) -> Image.Image:
    """``left`` turning into ``right`` from ``start`` to ``end`` of the width (a smooth step)."""
    width, height = left.size
    t = np.clip((np.arange(width) / width - start) / (end - start), 0, 1)
    t = t * t * (3 - 2 * t)
    mask = Image.fromarray(np.tile((t * 255).astype(np.uint8), (height, 1)), "L")
    return Image.composite(right, left, mask)


def hero(source: Image.Image, page, size: tuple[int, int], crop: tuple[float, float, float, float]) -> Image.Image:
    """The lion: the photograph, then its page, then the page painted in, across the picture."""
    painted = page.image(Version.COMPLETED).convert("RGB")
    lines = page.image(Version.PAGE).convert("RGB")
    w, h = lines.size
    box = (round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h))
    parts = [im.crop(box).resize(size, Image.LANCZOS) for im in (source, lines, painted)]
    first = blend_across(parts[0], parts[1], 0.16, 0.30)
    return blend_across(first, parts[2], 0.62, 0.76)


def banner(width: int, height: int, art: Image.Image, title_size: int, subtitle_size: int, small_size: int) -> Image.Image:
    out = Image.new("RGB", (width, height), BACKDROP)
    art_width = round(width * 0.58)
    out.paste(art.resize((art_width, height), Image.LANCZOS), (width - art_width, 0))
    # A soft edge where the backdrop meets the picture.
    fade = round(width * 0.06)
    x0 = width - art_width
    region = out.crop((x0, 0, x0 + fade, height))
    backdrop = Image.new("RGB", region.size, BACKDROP)
    ramp = Image.fromarray(np.tile(np.linspace(255, 0, fade).astype(np.uint8), (height, 1)), "L")
    out.paste(Image.composite(backdrop, region, ramp), (x0, 0))
    draw = ImageDraw.Draw(out)
    left = round(width * 0.05)
    y = round(height * 0.24)
    draw.text((left, y), "Tessellatum", fill=INK, font=font(title_size, bold=True))
    y += round(title_size * 1.3)
    for line in ("Any picture into a printable", "paint-by-numbers page"):
        draw.text((left, y), line, fill=ACCENT, font=font(subtitle_size))
        y += round(subtitle_size * 1.25)
    y += round(subtitle_size * 0.7)
    draw.rectangle([left, y, left + round(width * 0.05), y + 4], fill=ACCENT)
    y += round(small_size * 1.2)
    for line in ("Computer vision  ·  Color science", "On-device neural networks"):
        draw.text((left, y), line, fill=MUTED, font=font(small_size))
        y += round(small_size * 1.35)
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    GALLERY.mkdir(parents=True, exist_ok=True)
    pages = {}
    for name, sample, preset in GALLERY_ITEMS:
        source, page = make_page(sample, preset)
        pages[name] = (source, page)
        labels = ["Original", f"Coloring page ({preset}: {page.num_colors_used} colors, {page.num_regions:,} regions)",
                  "Painted"]
        panels = [source, page.image(Version.PAGE), page.image(Version.COMPLETED)]
        strip(panels, labels).save(GALLERY / f"{name}.jpg", quality=86, optimize=True, progressive=True)
        print(f"{name}: {page.num_colors_used} colors, {page.num_regions} regions")

    source, page = pages["photo-lion"]
    art = hero(source, page, (1400, 1000), (0.17, 0.0, 0.83, 0.94))
    banner(1600, 560, art, 96, 40, 28).save(ASSETS / "banner.png", optimize=True)
    banner(1280, 640, art, 82, 36, 26).save(ASSETS / "social-preview.png", optimize=True)
    print("banner and social preview written")


if __name__ == "__main__":
    main()

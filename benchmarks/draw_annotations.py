#!/usr/bin/env python
"""Draw manifest annotations onto copies of the benchmark images, to check them by eye.

  python benchmarks/draw_annotations.py                       # every annotated image
  python benchmarks/draw_annotations.py tests/sample_images/scene.png --out some/dir

Faces are red, features yellow, text green (with its ground truth), gradient
areas cyan, texture areas magenta; flat and ink colors are swatches under the image.
Writes ``<image stem>.annotated.png`` into --out (default: the git-ignored
benchmarks/results/annotations).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import bench_manifest

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = BENCH_DIR.parent / "tests" / "sample_images"
DEFAULT_OUT_DIR = BENCH_DIR / "results" / "annotations"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

FACE_COLOR = (255, 48, 48)
FEATURE_COLOR = (255, 214, 0)
TEXT_COLOR = (0, 220, 110)
AREA_COLORS = {"gradient": (0, 200, 255), "texture": (255, 0, 220)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, default=[DEFAULT_IMAGES_DIR], help="image files and/or directories")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="where to write the annotated copies")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path in _collect(args.images):
        info = bench_manifest.find_image(image_path)
        if info is None:
            print(f"no manifest entry: {image_path.name}", file=sys.stderr)
            continue
        out_path = args.out / f"{image_path.stem}.annotated.png"
        draw(image_path, info).save(out_path)
        print(out_path)
        written += 1
    return 0 if written else 1


def draw(image_path: Path, info: bench_manifest.ImageInfo) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != info.size:
        print(f"warning: {image_path.name} is {image.size}, the manifest says {info.size}", file=sys.stderr)

    long_edge = max(image.size)
    width = max(2, round(long_edge / 600))
    font = ImageFont.load_default(size=max(12, round(long_edge / 90)))
    small = ImageFont.load_default(size=max(10, round(long_edge / 140)))
    canvas = ImageDraw.Draw(image)

    for area in info.areas:
        color = AREA_COLORS[area.kind]
        _rect(canvas, area.box, color, width)
        _label(canvas, (area.box.x + width, area.box.y + width), area.kind, color, font)
    for face in info.faces:
        _rect(canvas, face.box, FACE_COLOR, width * 2)
        _label(canvas, (face.box.x, face.box.y1 + width), f"face ({face.kind})", FACE_COLOR, font)
        for feature in face.features:
            _rect(canvas, feature.box, FEATURE_COLOR, width)
            _label(canvas, (feature.box.x, feature.box.y1 + 1), feature.part, FEATURE_COLOR, small)
    for block in info.text:
        _rect(canvas, block.box, TEXT_COLOR, width)
        first_line = block.string.splitlines()[0]
        more = " …" if "\n" in block.string else ""
        turned = f" (turned {block.rotation}°)" if block.rotation else ""
        _label(canvas, (block.box.x, block.box.y1 + 1), f"{first_line}{more}{turned}", TEXT_COLOR, small)

    swatches = [(rgb, "") for rgb in info.flat_colors] + [(rgb, " ink") for rgb in info.ink_colors]
    if swatches:
        image = _with_swatches(image, swatches, font)
    return image


def _with_swatches(image: Image.Image, swatches, font) -> Image.Image:
    per_row = 8
    swatch_w = image.width // per_row
    swatch_h = max(40, swatch_w // 3)
    rows = -(-len(swatches) // per_row)
    out = Image.new("RGB", (image.width, image.height + rows * swatch_h), (255, 255, 255))
    out.paste(image, (0, 0))
    canvas = ImageDraw.Draw(out)
    for i, (rgb, suffix) in enumerate(swatches):
        x0 = (i % per_row) * swatch_w
        y0 = image.height + (i // per_row) * swatch_h
        canvas.rectangle((x0, y0, x0 + swatch_w - 1, y0 + swatch_h - 1), fill=rgb, outline=(128, 128, 128))
        luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ink = (0, 0, 0) if luma > 128 else (255, 255, 255)
        canvas.text((x0 + 6, y0 + 6), "#{:02x}{:02x}{:02x}".format(*rgb) + suffix, fill=ink, font=font)
    return out


def _rect(canvas: ImageDraw.ImageDraw, box: bench_manifest.Box, color, width: int) -> None:
    # The outline is drawn just outside the box, so it never hides what the box marks.
    canvas.rectangle((box.x - width, box.y - width, box.x1 - 1 + width, box.y1 - 1 + width), outline=color, width=width)


def _label(canvas: ImageDraw.ImageDraw, xy, text: str, color, font) -> None:
    left, top, right, bottom = canvas.textbbox(xy, text, font=font)
    canvas.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(0, 0, 0))
    canvas.text(xy, text, fill=color, font=font)


def _collect(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            images.append(path)
        else:
            print(f"warning: {path} does not exist, skipping", file=sys.stderr)
    return images


if __name__ == "__main__":
    sys.exit(main())

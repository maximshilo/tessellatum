"""The benchmark image manifest: what each image is and where its important parts are.

``manifest.json`` sits next to the images it describes (``tests/sample_images/``).
An optional, git-ignored ``manifest.local.json`` beside it covers images that
only exist in a local checkout; its entries are added to the committed ones and
replace any of the same name. The schema is documented in benchmarks/README.md.

Coordinates are in pixels of the source file. ``ImageInfo.scaled_to`` converts
them to an output size. Deliberately independent of the tessellatum package,
like ``bench_metrics``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
LOCAL_MANIFEST_NAME = "manifest.local.json"

CATEGORIES = ("photo", "cartoon", "flat", "face", "text", "gradient", "texture")
UNCATEGORIZED = "uncategorized"
FACE_KINDS = ("human", "animal", "cartoon")
FEATURE_PARTS = ("eye", "nose", "mouth")
AREA_KINDS = ("gradient", "texture")
TEXT_ROTATIONS = (0, 90, 180, 270)  # quarter turns: slanted lettering isn't annotated

_TOP_KEYS = {"schema", "images"}
_IMAGE_KEYS = {"size", "categories", "notes", "faces", "text", "flat_colors", "ink_colors", "exact_colors", "areas"}
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class ManifestError(ValueError):
    """A manifest that doesn't follow the schema."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in pixels: left, top, width, height."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def slices(self) -> tuple[slice, slice]:
        """``image[box.slices]`` selects the box from an image-shaped array."""
        return slice(self.y, self.y1), slice(self.x, self.x1)

    def contains(self, other: Box) -> bool:
        return self.x <= other.x and self.y <= other.y and other.x1 <= self.x1 and other.y1 <= self.y1

    def scaled(self, sx: float, sy: float, bounds: tuple[int, int]) -> Box:
        """This box in a resized image, rounded outward and clipped to ``bounds`` (w, h)."""
        eps = 1e-9
        x0 = max(0, math.floor(self.x * sx + eps))
        y0 = max(0, math.floor(self.y * sy + eps))
        x1 = min(bounds[0], max(x0 + 1, math.ceil(self.x1 * sx - eps)))
        y1 = min(bounds[1], max(y0 + 1, math.ceil(self.y1 * sy - eps)))
        return Box(x0, y0, x1 - x0, y1 - y0)


@dataclass(frozen=True)
class Feature:
    part: str  # one of FEATURE_PARTS
    box: Box


@dataclass(frozen=True)
class Face:
    kind: str  # one of FACE_KINDS
    box: Box
    features: tuple[Feature, ...] = ()


@dataclass(frozen=True)
class TextBlock:
    box: Box
    string: str  # ground truth; lines separated by "\n"
    rotation: int = 0  # degrees counterclockwise the text is turned from upright (90: reads bottom to top)


@dataclass(frozen=True)
class Area:
    kind: str  # one of AREA_KINDS
    box: Box


@dataclass(frozen=True)
class ImageInfo:
    name: str
    size: tuple[int, int]  # (width, height) of the source file
    categories: tuple[str, ...]  # the first one is the image's primary category
    faces: tuple[Face, ...] = ()
    text: tuple[TextBlock, ...] = ()
    flat_colors: tuple[tuple[int, int, int], ...] = ()  # sRGB fill colors
    ink_colors: tuple[tuple[int, int, int], ...] = ()  # sRGB line-art ink, not among flat_colors
    # The flat and ink colors are the file's own pixel values, as in digital artwork, not cluster centers of printed colors.
    exact_colors: bool = False
    areas: tuple[Area, ...] = ()
    notes: str = ""

    @property
    def primary_category(self) -> str:
        return self.categories[0]

    def areas_of(self, kind: str) -> tuple[Area, ...]:
        return tuple(a for a in self.areas if a.kind == kind)

    def missing_annotations(self) -> list[str]:
        """What this image's categories need that its entry doesn't provide."""
        missing = []
        if "face" in self.categories:
            if not self.faces:
                missing.append("face: no faces")
            missing += [f"face: faces[{i}] has no features" for i, face in enumerate(self.faces) if not face.features]
        if "text" in self.categories and not self.text:
            missing.append("text: no text blocks")
        for category in ("cartoon", "flat"):
            if category in self.categories and len(self.flat_colors) < 2:
                missing.append(f"{category}: fewer than 2 flat colors")
        if "cartoon" in self.categories and not self.ink_colors:
            missing.append("cartoon: no ink colors")
        for kind in AREA_KINDS:
            if kind in self.categories and not self.areas_of(kind):
                missing.append(f"{kind}: no {kind} areas")
        return missing

    def scaled_to(self, size: tuple[int, int]) -> ImageInfo:
        """The same annotations for this image resized to ``size`` (width, height)."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]

        def box(b: Box) -> Box:
            return b.scaled(sx, sy, size)

        return ImageInfo(
            name=self.name,
            size=tuple(size),
            categories=self.categories,
            faces=tuple(
                Face(f.kind, box(f.box), tuple(Feature(p.part, box(p.box)) for p in f.features)) for f in self.faces
            ),
            text=tuple(TextBlock(box(t.box), t.string, t.rotation) for t in self.text),
            flat_colors=self.flat_colors,
            ink_colors=self.ink_colors,
            exact_colors=self.exact_colors,
            areas=tuple(Area(a.kind, box(a.box)) for a in self.areas),
            notes=self.notes,
        )


def load_manifest(path: Path) -> dict[str, ImageInfo]:
    """Parse and validate one manifest file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON: {exc}") from exc
    except ValueError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    return parse_manifest(data, source=str(path))


def parse_manifest(data: object, source: str = "manifest") -> dict[str, ImageInfo]:
    """Validate an already-decoded manifest."""
    if not isinstance(data, dict):
        raise ManifestError(f"{source}: expected a JSON object")
    _check_keys(data, _TOP_KEYS, source)
    if data.get("schema") != SCHEMA_VERSION:
        raise ManifestError(f"{source}: schema must be {SCHEMA_VERSION}, got {data.get('schema')!r}")
    images = data.get("images")
    if not isinstance(images, dict):
        raise ManifestError(f"{source}: 'images' must be an object keyed by file name")
    return {name: _parse_image(name, entry, f"{source}: {name}") for name, entry in images.items()}


def load_directory(directory: Path) -> dict[str, ImageInfo]:
    """The manifest for images in ``directory``: the committed file plus the local one (local wins)."""
    entries: dict[str, ImageInfo] = {}
    for name in (MANIFEST_NAME, LOCAL_MANIFEST_NAME):
        path = Path(directory) / name
        if path.is_file():
            entries.update(load_manifest(path))
    return entries


def find_image(image_path: Path) -> ImageInfo | None:
    """The manifest entry for one image file, if its directory's manifest has one."""
    image_path = Path(image_path)
    return load_directory(image_path.parent).get(image_path.name)


# -- parsing helpers ----------------------------------------------------------
def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _value in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate key(s): {', '.join(duplicates)}")
    return dict(pairs)


def _check_keys(obj: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")


def _list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ManifestError(f"{where}: expected a list")
    return value


def _object(value: object, where: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise ManifestError(f"{where}: expected an object")
    _check_keys(value, required | optional, where)
    missing = sorted(required - set(value))
    if missing:
        raise ManifestError(f"{where}: missing key(s) {', '.join(missing)}")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _choice(value: object, choices: tuple[str, ...], where: str) -> str:
    if value not in choices:
        raise ManifestError(f"{where}: {value!r} is not one of {', '.join(choices)}")
    return value


def _parse_box(value: object, size: tuple[int, int], where: str) -> Box:
    if not (isinstance(value, list) and len(value) == 4 and all(_is_int(v) for v in value)):
        raise ManifestError(f"{where}: a box is [x, y, width, height] in integer pixels")
    box = Box(*value)
    if box.x < 0 or box.y < 0 or box.w <= 0 or box.h <= 0 or box.x1 > size[0] or box.y1 > size[1]:
        raise ManifestError(f"{where}: box {value} is empty or outside the {size[0]}x{size[1]} image")
    return box


def _parse_colors(value: object, where: str) -> tuple[tuple[int, int, int], ...]:
    colors = []
    for i, color in enumerate(_list(value, where)):
        if not (isinstance(color, str) and _HEX_COLOR.match(color)):
            raise ManifestError(f"{where}[{i}]: expected '#rrggbb', got {color!r}")
        colors.append(tuple(int(color[k : k + 2], 16) for k in (1, 3, 5)))
    if len(set(colors)) != len(colors):
        raise ManifestError(f"{where}: colors must not repeat")
    return tuple(colors)


def _parse_image(name: str, entry: object, where: str) -> ImageInfo:
    entry = _object(entry, where, required={"size", "categories"}, optional=_IMAGE_KEYS)

    size = entry["size"]
    if not (isinstance(size, list) and len(size) == 2 and all(_is_int(v) and v > 0 for v in size)):
        raise ManifestError(f"{where}: size is [width, height] in pixels")
    size = (size[0], size[1])

    categories = _list(entry["categories"], f"{where}: categories")
    if not categories:
        raise ManifestError(f"{where}: categories must not be empty")
    for i, category in enumerate(categories):
        _choice(category, CATEGORIES, f"{where}: categories[{i}]")
    if len(set(categories)) != len(categories):
        raise ManifestError(f"{where}: categories must not repeat")

    faces = []
    for i, raw in enumerate(_list(entry.get("faces", []), f"{where}: faces")):
        at = f"{where}: faces[{i}]"
        raw = _object(raw, at, required={"kind", "box"}, optional={"features"})
        face_box = _parse_box(raw["box"], size, f"{at}.box")
        features = []
        for j, feature in enumerate(_list(raw.get("features", []), f"{at}.features")):
            fat = f"{at}.features[{j}]"
            feature = _object(feature, fat, required={"part", "box"})
            feature_box = _parse_box(feature["box"], size, f"{fat}.box")
            if not face_box.contains(feature_box):
                raise ManifestError(f"{fat}.box: {feature['box']} is not inside the face box {raw['box']}")
            features.append(Feature(_choice(feature["part"], FEATURE_PARTS, f"{fat}.part"), feature_box))
        faces.append(Face(_choice(raw["kind"], FACE_KINDS, f"{at}.kind"), face_box, tuple(features)))

    text = []
    for i, raw in enumerate(_list(entry.get("text", []), f"{where}: text")):
        at = f"{where}: text[{i}]"
        raw = _object(raw, at, required={"box", "string"}, optional={"rotation"})
        string, rotation = raw["string"], raw.get("rotation", 0)
        if not isinstance(string, str) or not string.strip():
            raise ManifestError(f"{at}.string: expected non-empty text")
        if not (_is_int(rotation) and rotation in TEXT_ROTATIONS):
            raise ManifestError(f"{at}.rotation: expected one of {', '.join(map(str, TEXT_ROTATIONS))} degrees")
        text.append(TextBlock(_parse_box(raw["box"], size, f"{at}.box"), string, rotation))

    flat_colors = _parse_colors(entry.get("flat_colors", []), f"{where}: flat_colors")
    ink_colors = _parse_colors(entry.get("ink_colors", []), f"{where}: ink_colors")
    if set(flat_colors) & set(ink_colors):
        raise ManifestError(f"{where}: a color can't be both a flat color and an ink color")
    exact_colors = entry.get("exact_colors", False)
    if not isinstance(exact_colors, bool):
        raise ManifestError(f"{where}: exact_colors must be true or false")
    if exact_colors and not (flat_colors or ink_colors):
        raise ManifestError(f"{where}: exact_colors says the colors are exact, but there are none")

    areas = []
    for i, raw in enumerate(_list(entry.get("areas", []), f"{where}: areas")):
        at = f"{where}: areas[{i}]"
        raw = _object(raw, at, required={"kind", "box"})
        areas.append(Area(_choice(raw["kind"], AREA_KINDS, f"{at}.kind"), _parse_box(raw["box"], size, f"{at}.box")))

    notes = entry.get("notes", "")
    if not isinstance(notes, str):
        raise ManifestError(f"{where}: notes must be text")

    return ImageInfo(
        name=name,
        size=size,
        categories=tuple(categories),
        faces=tuple(faces),
        text=tuple(text),
        flat_colors=flat_colors,
        ink_colors=ink_colors,
        exact_colors=exact_colors,
        areas=tuple(areas),
        notes=notes,
    )

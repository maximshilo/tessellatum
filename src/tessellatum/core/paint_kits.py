"""Sample kits: a few generic sets of paints or pencils to start a list of one's own from.

Each kit is named for what it is, not for a maker: its paints are named by their pigments or hues, and each color is a
representative sRGB value, approximate -- real tubes and pencils of the same name differ from maker to maker, and from
a swatch painted thick to one painted thin. A kit is a starting point: replace a color with one's own (picked from a
photo of one's swatches, say) where it differs.

The colors of a kit are far enough apart that the legend can keep every one (the presets' color margin, 4 CIEDE2000).
"""

from __future__ import annotations

from dataclasses import dataclass

from tessellatum.core.paints import Paint


@dataclass(frozen=True)
class Kit:
    """A sample kit: ``key``, a short name without spaces (the benchmark's ``-Kit<key>`` presets); ``name``, as the
    menu says it; and its ``paints``, (name, RGB) each."""

    key: str
    name: str
    paints: tuple[Paint, ...]


def _gray(lightness: float) -> tuple[int, int, int]:
    """The sRGB gray of CIE lightness ``lightness`` (0-100)."""
    fy = (lightness + 16) / 116
    luminance = fy**3 if fy**3 > 216 / 24389 else (116 * fy - 16) * 27 / 24389
    encoded = 12.92 * luminance if luminance <= 0.0031308 else 1.055 * luminance ** (1 / 2.4) - 0.055
    value = int(round(min(1.0, max(0.0, encoded)) * 255))
    return (value, value, value)


KITS: tuple[Kit, ...] = (
    Kit(
        "Acrylics",
        "Basic acrylics (12)",
        (
            ("Titanium white", (246, 246, 241)),
            ("Mars black", (34, 32, 32)),
            ("Lemon yellow", (250, 232, 40)),
            ("Cadmium yellow medium", (252, 186, 3)),
            ("Yellow ochre", (198, 145, 58)),
            ("Cadmium red medium", (204, 36, 40)),
            ("Alizarin crimson", (140, 22, 46)),
            ("Ultramarine blue", (36, 44, 150)),
            ("Phthalo blue", (8, 62, 125)),
            ("Sap green", (78, 108, 40)),
            ("Burnt sienna", (138, 68, 38)),
            ("Burnt umber", (86, 56, 40)),
        ),
    ),
    Kit(
        "Primaries",
        "Primaries (5)",
        (
            ("Titanium white", (246, 246, 241)),
            ("Mars black", (34, 32, 32)),
            ("Primary yellow", (252, 210, 0)),
            ("Primary red", (210, 32, 48)),
            ("Primary blue", (0, 84, 160)),
        ),
    ),
    Kit(
        "Pencils",
        "Colored pencils (24)",
        (
            ("White", (250, 250, 248)),
            ("Lemon yellow", (250, 238, 80)),
            ("Yellow", (252, 200, 20)),
            ("Orange", (245, 130, 30)),
            ("Vermilion", (232, 72, 40)),
            ("Scarlet", (200, 30, 40)),
            ("Crimson", (150, 20, 50)),
            ("Pink", (240, 150, 175)),
            ("Magenta", (190, 40, 130)),
            ("Violet", (110, 50, 140)),
            ("Ultramarine", (40, 50, 150)),
            ("Blue", (30, 100, 190)),
            ("Sky blue", (110, 180, 230)),
            ("Turquoise", (0, 150, 160)),
            ("Dark green", (20, 90, 50)),
            ("Green", (40, 150, 60)),
            ("Light green", (150, 200, 70)),
            ("Olive green", (110, 115, 45)),
            ("Ochre", (200, 150, 60)),
            ("Light brown", (165, 110, 70)),
            ("Burnt sienna", (140, 65, 40)),
            ("Dark brown", (80, 50, 35)),
            ("Gray", (140, 140, 140)),
            ("Black", (30, 30, 30)),
        ),
    ),
    Kit(
        "Earth",
        "Earth palette (8)",
        (
            ("Titanium white", (246, 246, 241)),
            ("Ivory black", (40, 38, 37)),
            ("Yellow ochre", (198, 145, 58)),
            ("Raw sienna", (178, 108, 48)),
            ("Burnt sienna", (138, 68, 38)),
            ("Raw umber", (112, 90, 62)),
            ("Burnt umber", (86, 56, 40)),
            ("Green earth", (112, 124, 92)),
        ),
    ),
    Kit(
        "Grays",
        "Grays (6)",
        tuple((name, _gray(lightness)) for name, lightness in (
            ("Black", 0), ("Dark gray", 20), ("Gray 40", 40), ("Gray 60", 60), ("Light gray", 80), ("White", 100),
        )),
    ),
)


def kit(key: str) -> Kit:
    """The kit whose ``key`` is ``key`` (any case). Raises KeyError for none."""
    for found in KITS:
        if found.key.lower() == key.lower():
            return found
    raise KeyError(key)

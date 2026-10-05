"""Difficulty presets: map a difficulty choice to pipeline parameters.

Harder difficulty -> more colors, smaller regions, less smoothing (so more of
the source image's detail survives into separate regions).

Sizes are set on the printed page, in square millimeters and millimeters (see
``print_size``): a page prints at the same physical size whatever its pixel
count, so a preview and an export of one image are held to the same limits,
and a long, narrow picture, which prints smaller, gets fewer regions rather
than smaller ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tessellatum.core.color import MIN_PALETTE_DE00
from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM


# How many times smaller a region may be in the faces and the subject found, by default (see ``DifficultyParams``).
DETAIL_WEIGHT = 2


@dataclass(frozen=True)
class DifficultyParams:
    """Parameters that control how granular the generated coloring page is.

    Attributes:
        num_colors: how many colors the palette looks for. The page keeps at
            most that many, every two at least ``palette_margin_de00`` apart
            (see ``quantize``), so a picture whose colors crowd together keeps
            fewer.
        min_region_area_mm2: the smallest region on the printed page, in square
            millimeters; smaller regions are merged into a neighbor. Never
            below the brush's footprint, a disk ``min_width_mm`` across.
            Inside the subject or a face found in a photograph or painting, a
            region may be ``detail_weight`` times smaller (see ``subject``,
            ``faces`` and ``regions.build_regions``).
        blur_sigma: bilateral-filter smoothing strength applied before
            quantization. Higher = smoother/simpler source, fewer stray
            regions.
        min_width_mm: the narrowest any part of a region may be on the printed
            page, in millimeters -- the brush the page is painted with; 0 sets
            no width at all, and then settles no edges either, since the vote
            reaches as far as the brush says (see ``edge_settling``).
        palette_margin_de00: the least CIEDE2000 difference between two of the
            palette's colors: closer ones are merged into one (see
            ``quantize``). 0 keeps every color found.
        edge_settling: how far the vote that settles the regions' edges
            reaches, as a multiple of its reach at this brush (a third of the
            brush to one sigma, see ``texture``); 0 settles nothing.
        edge_color_step_de00: how far a color may lie from a pixel's own, in
            L*a*b*, before it counts e times less in that vote: lower holds the
            edges closer to the picture's own, higher smooths them more.
        detail_weight: how many times smaller a region may be in the faces and
            the subject found than elsewhere (each pixel there counts that many
            times towards ``min_region_area_mm2``); 1 spends no more detail on
            them.
    """

    num_colors: int
    min_region_area_mm2: float
    blur_sigma: float
    min_width_mm: float = MIN_PAINTABLE_WIDTH_MM
    palette_margin_de00: float = MIN_PALETTE_DE00
    edge_settling: float = 1.0
    edge_color_step_de00: float = MIN_PALETTE_DE00
    detail_weight: int = DETAIL_WEIGHT


# Every preset paints with the same 3 mm brush; they differ in how many regions
# they ask for. Median regions over the benchmark images (preview / export):
# Easy 13.5 / 13, Medium 26.5 / 23.5, Hard 69.5 / 53.
PRESETS: dict[str, DifficultyParams] = {
    "Easy": DifficultyParams(num_colors=6, min_region_area_mm2=300.0, blur_sigma=9.0),
    "Medium": DifficultyParams(num_colors=12, min_region_area_mm2=125.0, blur_sigma=5.0),
    "Hard": DifficultyParams(num_colors=20, min_region_area_mm2=40.0, blur_sigma=2.5),
}

# Bounds of the Custom settings, by field. The app's sliders and ``custom_params`` keep to them.
#
# At the 3 mm brush and 10 ΔE00, the finest region the sliders offered was 30 mm²: a round brush can't reach into a
# region's corners, and with smaller regions a detailed photograph had more than 1% of an A4 page in paint the brush
# can't put down without crossing a line (25 mm² left one page at 1.01%), and the scanned postcard's hatched areas got
# numbers with no room off its lines (20 mm²). That is still the benchmark's finest page (``finest_params``). The
# sliders now reach well past it -- smaller regions, a finer brush, closer colors -- for pages with more detail than
# those limits allow; how paintable such a page is, is the user's choice.
#
# They stop where a page still comes out in seconds: with every one at its finest, a detailed photograph's preview
# has 1,000-2,700 regions and takes 4-12 s, most of it placing their numbers. A 0.5 mm brush and 2 mm² regions gave
# the lion and the Palermo castle over 7,000 regions and 100 s.
CUSTOM_RANGES: dict[str, tuple[float, float]] = {
    "num_colors": (2, 64),
    "min_region_area_mm2": (5.0, 500.0),
    "blur_sigma": (0.0, 12.0),
    "min_width_mm": (1.0, 6.0),
    "palette_margin_de00": (0.0, 20.0),
    "edge_settling": (0.0, 3.0),
    "edge_color_step_de00": (1.0, 40.0),
    "detail_weight": (1, 4),
}
# The fields that take whole numbers.
_INTEGER_FIELDS = frozenset({"num_colors", "detail_weight"})
CUSTOM_COLORS_RANGE = CUSTOM_RANGES["num_colors"]
CUSTOM_MIN_REGION_AREA_MM2_RANGE = CUSTOM_RANGES["min_region_area_mm2"]
CUSTOM_BLUR_RANGE = CUSTOM_RANGES["blur_sigma"]

DEFAULT_PRESET = "Medium"


def preset_names() -> list[str]:
    return [*PRESETS.keys(), "Custom"]


def params_for_preset(name: str) -> DifficultyParams:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty preset: {name!r}") from exc


def custom_params(num_colors: int, min_region_area_mm2: float, blur_sigma: float, **more: float) -> DifficultyParams:
    """The parameters asked for, each held within its range in ``CUSTOM_RANGES``.

    ``more`` sets any of ``DifficultyParams``' other fields by name; those not
    given keep their defaults. Whole-number fields are rounded. Raises
    TypeError for a name that is not a field.
    """
    values = dict(num_colors=num_colors, min_region_area_mm2=min_region_area_mm2, blur_sigma=blur_sigma, **more)
    unknown = set(values) - set(CUSTOM_RANGES)
    if unknown:
        raise TypeError(f"not a difficulty setting: {', '.join(sorted(unknown))}")
    return DifficultyParams(**{name: clamp(name, value) for name, value in values.items()})


def clamp(name: str, value: float) -> float:
    """``value`` held within the range of the field ``name`` in ``CUSTOM_RANGES``, rounded if it takes whole numbers."""
    low, high = CUSTOM_RANGES[name]
    value = max(low, min(high, value))
    return int(round(value)) if name in _INTEGER_FIELDS else float(value)


def finest_params() -> DifficultyParams:
    """The benchmark's Max: 40 colors, regions of 30 mm², no smoothing, every other setting at its default.

    It was the finest page the Custom sliders could ask for until they reached
    past the 3 mm brush and 10 ΔE00. It is kept as it was, so that result sets
    of every version compare at it.
    """
    return DifficultyParams(num_colors=40, min_region_area_mm2=30.0, blur_sigma=0.0)


def describe(params: DifficultyParams) -> str:
    """The settings in words, in the printed page's units, e.g. for a tooltip."""
    side_mm = math.sqrt(params.min_region_area_mm2)
    detail = {1: "as much", 2: "half that"}.get(params.detail_weight, f"1/{params.detail_weight} of that")
    return (
        f"Up to {params.num_colors} colors, at least {params.palette_margin_de00:g} ΔE00 apart. "
        f"Regions of at least {params.min_region_area_mm2:.0f} mm² "
        f"(about {side_mm:.0f} × {side_mm:.0f} mm; {detail} on a photograph's or painting's subject and faces) "
        f"and {params.min_width_mm:g} mm wide on the printed A4 page."
    )

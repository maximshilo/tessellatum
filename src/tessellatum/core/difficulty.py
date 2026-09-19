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

from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM


@dataclass(frozen=True)
class DifficultyParams:
    """Parameters that control how granular the generated coloring page is.

    Attributes:
        num_colors: how many colors k-means looks for. The page keeps at most
            that many: colors too close to tell apart are merged (see
            ``quantize``), so a picture whose colors crowd together keeps fewer.
        min_region_area_mm2: the smallest region on the printed page, in square
            millimeters; smaller regions are merged into a neighbor. Never
            below ``print_size.MIN_REGION_AREA_MM2``, the brush's footprint.
        blur_sigma: bilateral-filter smoothing strength applied before
            quantization. Higher = smoother/simpler source, fewer stray
            regions.
        min_width_mm: the narrowest any part of a region may be on the printed
            page, in millimeters -- the brush the page is painted with. Never
            below ``print_size.MIN_PAINTABLE_WIDTH_MM``.
    """

    num_colors: int
    min_region_area_mm2: float
    blur_sigma: float
    min_width_mm: float = MIN_PAINTABLE_WIDTH_MM


# Every preset paints with the same 3 mm brush; they differ in how many regions
# they ask for. Median regions over the benchmark images (preview / export):
# Easy 13.5 / 13, Medium 26.5 / 23.5, Hard 69.5 / 53.
PRESETS: dict[str, DifficultyParams] = {
    "Easy": DifficultyParams(num_colors=6, min_region_area_mm2=300.0, blur_sigma=9.0),
    "Medium": DifficultyParams(num_colors=12, min_region_area_mm2=125.0, blur_sigma=5.0),
    "Hard": DifficultyParams(num_colors=20, min_region_area_mm2=40.0, blur_sigma=2.5),
}

# Bounds used by the "Custom" UI sliders.
CUSTOM_COLORS_RANGE = (4, 40)
# A round brush can't reach into a region's corners, and every region has
# some: with the smallest regions under 30 mm², a detailed photograph gets so
# many that more than 1% of an A4 page is paint the 3 mm brush can't put down
# without crossing a line. Of 10, 15, 20, 25 and 30 mm², only 30 keeps every
# benchmark page, preview or export, under 1% at the Custom sliders' finest
# setting (25 leaves one at 1.01%). That bounds the finest setting and the
# presets, not every mix of the sliders: slivers don't simply fall as the
# settings coarsen, and at 30 mm² a few other mixes (fewer colors, or a little
# smoothing) leave the most textured photograph at 1.03-1.10%.
CUSTOM_MIN_REGION_AREA_MM2_RANGE = (30.0, 500.0)
CUSTOM_BLUR_RANGE = (0.0, 12.0)

DEFAULT_PRESET = "Medium"


def preset_names() -> list[str]:
    return [*PRESETS.keys(), "Custom"]


def params_for_preset(name: str) -> DifficultyParams:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty preset: {name!r}") from exc


def custom_params(num_colors: int, min_region_area_mm2: float, blur_sigma: float) -> DifficultyParams:
    return DifficultyParams(
        num_colors=max(CUSTOM_COLORS_RANGE[0], min(CUSTOM_COLORS_RANGE[1], num_colors)),
        min_region_area_mm2=max(
            CUSTOM_MIN_REGION_AREA_MM2_RANGE[0], min(CUSTOM_MIN_REGION_AREA_MM2_RANGE[1], min_region_area_mm2)
        ),
        blur_sigma=max(CUSTOM_BLUR_RANGE[0], min(CUSTOM_BLUR_RANGE[1], blur_sigma)),
    )


def finest_params() -> DifficultyParams:
    """The most granular page the Custom sliders can ask for: the most colors, the smallest regions, no smoothing."""
    return custom_params(CUSTOM_COLORS_RANGE[1], CUSTOM_MIN_REGION_AREA_MM2_RANGE[0], CUSTOM_BLUR_RANGE[0])


def describe(params: DifficultyParams) -> str:
    """The settings in words, in the printed page's units, e.g. for a tooltip."""
    side_mm = math.sqrt(params.min_region_area_mm2)
    return (
        f"Up to {params.num_colors} colors. Regions of at least {params.min_region_area_mm2:.0f} mm² "
        f"(about {side_mm:.0f} × {side_mm:.0f} mm) and {params.min_width_mm:g} mm wide on the printed A4 page."
    )

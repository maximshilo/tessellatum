"""Difficulty presets: map a difficulty choice to pipeline parameters.

Harder difficulty -> more colors, smaller minimum region size, less smoothing
(so more of the source image's detail survives into separate regions).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyParams:
    """Parameters that control how granular the generated coloring page is.

    Attributes:
        num_colors: number of distinct colors/regions in the palette.
        min_region_fraction: minimum region size, as a fraction of total
            image area, below which a region gets merged into a neighbor.
        blur_sigma: bilateral-filter smoothing strength applied before
            quantization. Higher = smoother/simpler source, fewer stray
            regions.
    """

    num_colors: int
    min_region_fraction: float
    blur_sigma: float


PRESETS: dict[str, DifficultyParams] = {
    "Easy": DifficultyParams(num_colors=6, min_region_fraction=0.006, blur_sigma=9.0),
    "Medium": DifficultyParams(num_colors=12, min_region_fraction=0.0025, blur_sigma=5.0),
    "Hard": DifficultyParams(num_colors=20, min_region_fraction=0.0008, blur_sigma=2.5),
}

# Bounds used by the "Custom" UI sliders.
CUSTOM_COLORS_RANGE = (4, 40)
CUSTOM_MIN_REGION_RANGE = (0.0002, 0.01)  # fraction of image area
CUSTOM_BLUR_RANGE = (0.0, 12.0)

DEFAULT_PRESET = "Medium"


def preset_names() -> list[str]:
    return [*PRESETS.keys(), "Custom"]


def params_for_preset(name: str) -> DifficultyParams:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty preset: {name!r}") from exc


def custom_params(num_colors: int, min_region_fraction: float, blur_sigma: float) -> DifficultyParams:
    return DifficultyParams(
        num_colors=max(CUSTOM_COLORS_RANGE[0], min(CUSTOM_COLORS_RANGE[1], num_colors)),
        min_region_fraction=max(
            CUSTOM_MIN_REGION_RANGE[0], min(CUSTOM_MIN_REGION_RANGE[1], min_region_fraction)
        ),
        blur_sigma=max(CUSTOM_BLUR_RANGE[0], min(CUSTOM_BLUR_RANGE[1], blur_sigma)),
    )

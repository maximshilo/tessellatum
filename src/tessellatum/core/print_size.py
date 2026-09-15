"""Print-size model: how large a generated page comes out on paper.

Paintability thresholds (brush-wide regions, legible numbers, thin lines) are
physical sizes, so they are set in millimeters and points and converted to
pixels for a given output image.

A page prints on a sheet of its own (the legend gets another), scaled to fill
the sheet's printable area inside the margins with its aspect ratio kept: on a
landscape sheet when the image is wider than tall, on a portrait one otherwise.
Its physical size therefore depends only on its shape. A preview and an export
of the same image print equally large, and a threshold in millimeters covers
proportionally more pixels on the export.

Standard library only, so the benchmark harness can load this file directly
without importing the package (see ``bench_metrics``).
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0


@dataclass(frozen=True)
class PaperFormat:
    """A sheet of paper, sized in portrait orientation, with the same margin on every side."""

    name: str
    width_mm: float
    height_mm: float
    margin_mm: float

    def __post_init__(self) -> None:
        if not 0 < self.width_mm <= self.height_mm:
            raise ValueError(f"{self.name}: give the portrait size, with 0 < width <= height")
        if not 0 <= 2 * self.margin_mm < self.width_mm:
            raise ValueError(f"{self.name}: the margins leave no printable area")

    def printable_mm(self, landscape: bool = False) -> tuple[float, float]:
        """(width, height) of the area inside the margins."""
        width = self.width_mm - 2 * self.margin_mm
        height = self.height_mm - 2 * self.margin_mm
        return (height, width) if landscape else (width, height)


# The print target: A4 at 300 dpi, with 10 mm margins.
A4 = PaperFormat("A4", width_mm=210.0, height_mm=297.0, margin_mm=10.0)
PRINT_DPI = 300

# Paintability thresholds at print size; benchmarks/README.md ("Print scale") explains them.
MIN_PAINTABLE_WIDTH_MM = 3.0  # narrowest a region, or any part of one, can be and still take a brush
MIN_LABEL_SIZE_PT = 6.0  # smallest region number, as the font's em size
OUTLINE_WIDTH_MM = 0.3


@dataclass(frozen=True)
class PrintScale:
    """How an output image of ``size_px`` (width, height) maps onto paper."""

    size_px: tuple[int, int]
    landscape: bool  # printed on a landscape sheet
    px_per_mm: float

    @property
    def printed_size_mm(self) -> tuple[float, float]:
        return (self.size_px[0] / self.px_per_mm, self.size_px[1] / self.px_per_mm)

    @property
    def px_per_pt(self) -> float:
        return self.px_per_mm * MM_PER_INCH / PT_PER_INCH

    @property
    def dpi(self) -> float:
        """The resolution the output image effectively prints at."""
        return self.px_per_mm * MM_PER_INCH

    def mm_to_px(self, mm: float) -> float:
        return mm * self.px_per_mm

    def px_to_mm(self, px: float) -> float:
        return px / self.px_per_mm

    def pt_to_px(self, pt: float) -> float:
        return pt * self.px_per_pt

    def px_to_pt(self, px: float) -> float:
        return px / self.px_per_pt

    def mm2_to_px(self, mm2: float) -> float:
        """An area in square millimeters, in pixels."""
        return mm2 * self.px_per_mm**2


def print_scale(size_px: tuple[int, int], paper: PaperFormat = A4) -> PrintScale:
    """The print scale of an output image of ``size_px`` (width, height) fitted onto ``paper``."""
    width, height = size_px
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {size_px}")
    landscape = width > height
    area_width, area_height = paper.printable_mm(landscape)
    mm_per_px = min(area_width / width, area_height / height)
    return PrintScale(size_px=(width, height), landscape=landscape, px_per_mm=1 / mm_per_px)

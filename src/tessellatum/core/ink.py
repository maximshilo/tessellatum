"""Ink lines: the dark lines line art is drawn with.

A cartoon, a comic or a coloring-book page is flat fills with dark lines drawn
between them. Those lines are something to print, not something to paint, so
the page has to know where they are. This module finds them without being told
the artwork's colors, and decides whether a picture is line art at all.

A line is dark against what lies around it, and thin: a dark area that a disk
``MAX_LINE_WIDTH_MM`` wide fits into is a fill drawn in a dark color, not a
line. The paper around a pixel is the image's morphological closing by that
disk -- the dark parts too narrow for the disk are filled in with the lighter
colors beside them -- and a pixel is ink where it is darker than halfway from
that paper to black, the level an anti-aliased edge half covered by ink sits
at. A line running along a dark fill only has the fill for paper, so it is
held to the fill: black on dark navy still counts. Short breaks where a faint
or anti-aliased stretch of line falls under halfway are closed, up to
``GAP_MM``, but never across a gap the artist left white.

A photograph has thin dark shapes too -- a window in a sunlit wall, the night
between two lit signs -- and locally they look just like ink. What they lack
is line art's flat fills. So a picture is taken for line art when, away from
its lines, a handful of flat colors match it closely, and deep lines -- dark
against light -- cover enough of it. Either alone is not enough: a portrait's
dark background is flat, a castle's windows are deep lines. The decision is
about the picture, not the page, so the pipeline makes it on the picture at
preview size, and a preview and an export always agree.

Sizes are on the printed page (see ``print_size``), so a line is judged by how
wide it prints, whatever the page's pixel count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tessellatum.core.print_size import print_scale

# The widest a line can print. A dark area wider than this is a fill in a dark
# color. The benchmark harness finds the artwork's own ink lines the same way.
MAX_LINE_WIDTH_MM = 5.0
# How much darker than the paper around it, in CIE L*, a pixel must be to be
# ink at all, however dark the paper is.
MIN_LINE_DEPTH = 15.0
# Breaks in a line up to this long are closed where the break is at least
# GAP_MIN_DEPTH darker than the paper around it: a faint stretch of the line,
# not a gap the artist left.
GAP_MM = 0.5
GAP_MIN_DEPTH = 5.0

# Deciding whether a picture is line art. Its lines are deep where at least
# DEEP_LINE_DEPTH darker than the paper around them, as ink on a light fill
# is; line art has them over at least MIN_DEEP_LINE_SHARE of the picture. Its
# flatness is how far the pixels off the lines, after a blur of FLAT_BLUR_MM
# that evens out print texture, lie from the nearest of FLAT_COLORS colors
# k-means finds for them (the median, in CIE Lab units); line art is at most
# MAX_FLATNESS. Pixels within LINE_MARGIN_MM of a line are left out, which
# keeps the lines' anti-aliased edges out of the fills.
DEEP_LINE_DEPTH = 40.0
MIN_DEEP_LINE_SHARE = 0.016
FLAT_BLUR_MM = 0.25
FLAT_COLORS = 16
MAX_FLATNESS = 3.4
LINE_MARGIN_MM = 0.5
# Flatness is measured on at most this many pixels, spread evenly over the picture.
FLATNESS_SAMPLES = 20_000

# CIE L* runs 0-100; OpenCV's 8-bit Lab stores it as 0-255.
_L_UNITS_PER_LSTAR = 255 / 100
_WHITE = 255
_KMEANS_ITERATIONS = 20
_KMEANS_EPSILON = 0.5
# One k-means++ run leaves a textured picture's flatness varying by up to 0.9 with the seed (the postcard 2.55-3.48
# over five seeds); keeping the best of three holds it within 0.3.
_KMEANS_ATTEMPTS = 3


@dataclass(frozen=True)
class LineArt:
    """Whether a picture is line art, and the two measures that decide it."""

    is_line_art: bool
    flatness: float  # median CIE Lab distance off the lines to the nearest of FLAT_COLORS flat colors
    deep_line_share: float  # share of the picture in deep lines


def find_ink(image_bgr: np.ndarray, picture_bgr: np.ndarray | None = None) -> tuple[LineArt, np.ndarray]:
    """The ink lines of ``image_bgr``, if the picture is line art.

    ``picture_bgr`` is the picture the decision is made on, at preview size,
    so that every size of one picture gets the same answer; without it, the
    decision is made on ``image_bgr`` itself. Returns the decision and an HxW
    bool mask of ``image_bgr``'s ink lines, all False unless the picture is
    line art.
    """
    decision = line_art(image_bgr if picture_bgr is None else picture_bgr)
    if not decision.is_line_art:
        return decision, np.zeros(image_bgr.shape[:2], dtype=bool)
    return decision, ink_lines(image_bgr)


def ink_lines(image_bgr: np.ndarray) -> np.ndarray:
    """HxW bool: the pixels of ``image_bgr`` on dark lines at most ``MAX_LINE_WIDTH_MM`` wide on paper, gaps closed.

    Finds lines whether or not the picture is line art; ``find_ink`` decides
    whether to use them.
    """
    scale = print_scale((image_bgr.shape[1], image_bgr.shape[0]))
    lightness, depth = _lightness_and_depth(image_bgr, scale.mm_to_px(MAX_LINE_WIDTH_MM))
    ink = _ink(lightness, depth)
    return ink | (_close_breaks(ink, scale.mm_to_px(GAP_MM)) & (depth > GAP_MIN_DEPTH))


def line_art(image_bgr: np.ndarray) -> LineArt:
    """Whether ``image_bgr`` is line art: flat fills and deep lines (see the module's docstring)."""
    scale = print_scale((image_bgr.shape[1], image_bgr.shape[0]))
    lightness, depth = _lightness_and_depth(image_bgr, scale.mm_to_px(MAX_LINE_WIDTH_MM))
    ink = _ink(lightness, depth)
    deep_line_share = float(np.count_nonzero(ink & (depth >= DEEP_LINE_DEPTH))) / ink.size
    near_lines = cv2.dilate(ink.view(np.uint8), _disk(max(1.0, scale.mm_to_px(LINE_MARGIN_MM)))).view(bool)
    flatness = _flatness(image_bgr, ~near_lines, scale.mm_to_px(FLAT_BLUR_MM))
    is_line_art = flatness <= MAX_FLATNESS and deep_line_share >= MIN_DEEP_LINE_SHARE
    return LineArt(is_line_art=is_line_art, flatness=flatness, deep_line_share=deep_line_share)


def _lightness_and_depth(image_bgr: np.ndarray, max_width_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's lightness and how much darker than the paper around it it is, both in CIE L* (HxW float32).

    The paper around a pixel is the lightness closed by a disk ``max_width_px``
    wide. Off the page counts as white paper, so a dark band along the page's
    edge is a line if it is thin enough.
    """
    lightness = cv2.cvtColor(np.ascontiguousarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2Lab)[:, :, 0]
    disk = _disk(max_width_px / 2)
    pad = disk.shape[0] // 2 + 1
    padded = cv2.copyMakeBorder(lightness, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=_WHITE)
    paper = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, disk)[pad:-pad, pad:-pad]
    to_lstar = np.float32(1 / _L_UNITS_PER_LSTAR)
    return lightness * to_lstar, (paper.astype(np.float32) - lightness) * to_lstar


def _ink(lightness: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pixels darker than halfway from the paper around them to black, and at least MIN_LINE_DEPTH darker."""
    paper = lightness + depth
    return (depth > MIN_LINE_DEPTH) & (2 * lightness < paper)


def _flatness(image_bgr: np.ndarray, off_lines: np.ndarray, blur_px: float) -> float:
    """Median CIE Lab distance from the ``off_lines`` pixels, blurred, to the nearest of FLAT_COLORS k-means colors.

    Measured on a grid of about FLATNESS_SAMPLES pixels.
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), max(blur_px, 0.5))
    step = max(1, int(math.sqrt(off_lines.size / FLATNESS_SAMPLES)))
    grid = np.ascontiguousarray(blurred[::step, ::step])
    samples = cv2.cvtColor(grid.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[off_lines[::step, ::step]]
    if len(samples) == 0:
        return 0.0
    colors = min(FLAT_COLORS, len(samples))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_ITERATIONS, _KMEANS_EPSILON)
    _compactness, _labels, centers = cv2.kmeans(
        np.ascontiguousarray(samples), colors, None, criteria, _KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    distance = np.sqrt(((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(np.median(distance))


def _disk(radius: float) -> np.ndarray:
    """The pixels within ``radius`` of a center pixel, as a structuring element.

    The brush the benchmark harness measures widths with, so that a line
    ``MAX_LINE_WIDTH_MM`` wide is judged the same way here and there.
    """
    n = int(np.floor(radius))
    y, x = np.mgrid[-n : n + 1, -n : n + 1]
    return ((x * x + y * y) <= radius * radius).astype(np.uint8)


def _close_breaks(mask: np.ndarray, gap_px: float) -> np.ndarray:
    """``mask`` with every break up to ``gap_px`` long filled, along rows, columns and both diagonals.

    A break is a run of pixels outside the mask with mask pixels at both ends
    of it, and its length is measured along the run: a run of n pixels along a
    diagonal is n * sqrt(2) long. Closing by a line segment of n + 1 pixels
    fills exactly the runs of up to n pixels along it. A break of one pixel is
    always closed, however small ``gap_px`` is. A disk would not do: one
    small enough to stop at ``gap_px`` fits through the break in a thin
    diagonal line beside it, where no pixel is in the mask.
    """
    straight = max(1, math.floor(gap_px)) + 1
    diagonal = max(1, math.floor(gap_px / math.sqrt(2))) + 1
    eye = np.eye(diagonal, dtype=np.uint8)
    segments = (np.ones((1, straight), np.uint8), np.ones((straight, 1), np.uint8), eye, np.ascontiguousarray(eye[:, ::-1]))
    # Off the page is outside the mask. OpenCV erodes as if it were inside, which would fill the page's corners.
    pad = straight
    source = cv2.copyMakeBorder(mask.view(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    closed = mask.copy()
    for segment in segments:
        # OpenCV's MORPH_CLOSE erodes by the kernel itself, not by its reflection, which is only a closing for a
        # kernel symmetric about its anchor; a segment of an even number of pixels has no such anchor.
        h, w = segment.shape
        dilated = cv2.dilate(source, segment, anchor=(0, 0))
        eroded = cv2.erode(dilated, np.ascontiguousarray(segment[::-1, ::-1]), anchor=(w - 1, h - 1))
        closed |= eroded[pad:-pad, pad:-pad].view(bool)
    return closed

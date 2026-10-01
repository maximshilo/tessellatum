"""Text: where a picture has lettering, for the stages that keep it readable.

A sign, a title or a caption should still read on the page, and no number
should be written on it. This module finds the lines of text; nothing on the
page uses them yet.

They are found by **PP-OCRv6-small**, the text detection network of
PaddleOCR (9.9 MB), run by ONNX Runtime, shipped with the app and run offline
(``resources/MODELS.md`` has its source and license). The network says how
likely each pixel is to lie in the core of a line of text -- a word, or a run
of words -- and each patch of likely pixels, grown back out to the line's
edge, is a line found (Differentiable Binarization, as PaddleOCR reads it).

It looks twice. First at the picture as it is, at preview size, which finds
lettering whose lines print at least about 2 mm tall. Where that finds text,
it looks again at the picture twice that size, taken from the source's own
pixels where it has them, which finds the smaller print beside it: captions,
small signs. A picture with no lettering big enough to see at preview size is
taken to have none, which keeps the finer look from taking window rows,
shutters or fur for text.

A line found must look like one: at most ``MAX_BOX_HEIGHT_MM`` tall on the
printed page -- lettering taller than that is big enough to paint as shapes of
its own -- and at least ``MIN_ELONGATION`` times as long as it is tall, as a
run of two letters or more is. The things the network takes for text that
aren't -- an eye, a disc, a window -- are mostly as tall as they are wide.

Text is found once per picture, so a preview and an export always agree.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import cv2
import numpy as np

from tessellatum.core import parallel
from tessellatum.core.faces import RESOURCES_DIR
from tessellatum.core.print_size import print_scale

MODEL = "PP-OCRv6_small_det.onnx"

# The first look sees the picture at preview size, the second at twice that, each side rounded to a multiple of
# SIDE_MULTIPLE pixels, as the network needs.
FINE_SCALE = 2.0
SIDE_MULTIPLE = 32
# The picture is fed as PaddleOCR feeds it: its BGR values scaled to [0, 1] and standardized channel by channel by these
# (ImageNet's RGB statistics, applied to BGR as PaddleOCR applies them).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# A line's core is where the network's probability is above THRESHOLD; the line is kept if its mean probability there is
# at least MIN_SCORE, and grown out on every side by its area times UNCLIP over its perimeter, back to the line's edge.
# PaddleOCR's own setting for this network grows a line by 1.4, which at twice the preview's size leaves the words of a
# small caption apart.
THRESHOLD = 0.2
MIN_SCORE = 0.4
UNCLIP = 2.0
# A line's box, grown out to the line's edge, is at most this tall on the printed page, and at least this many times as
# long as it is tall.
MAX_BOX_HEIGHT_MM = 15.0
MIN_ELONGATION = 1.5
# Patches of the map no wider than this many of its pixels, before and after growing, are noise; and only this many
# patches are read.
_MIN_CORE_PX = 3
_MIN_BOX_PX = 5
_MAX_CANDIDATES = 1000

_lock = threading.Lock()
_session = None


@dataclass(frozen=True)
class TextLine:
    """A line of text found: its box, and how sure the network is."""

    # The box's four corners in order round it, (x, y) in pixels of the picture or page it is given for, a pixel's corner
    # at whole numbers. A rectangle turned to the line (all but: the network sees the picture with its sides rounded to
    # SIDE_MULTIPLE, a hair off its shape); it may reach past the picture's edge.
    quad: tuple[tuple[float, float], ...]
    score: float  # the network's mean probability of text over the line's core, 0-1

    @property
    def sides(self) -> tuple[float, float]:
        """The box's height and length, in pixels: its shorter and its longer side."""
        p = np.asarray(self.quad, dtype=np.float64)
        a, b = float(np.hypot(*(p[1] - p[0]))), float(np.hypot(*(p[2] - p[1])))
        return min(a, b), max(a, b)


def find_text(picture_bgr: np.ndarray, source_bgr: np.ndarray | None = None) -> list[TextLine]:
    """The lines of text in ``picture_bgr``, in its pixels, top to bottom.

    ``picture_bgr`` is an HxWx3 uint8 picture at preview size, and
    ``source_bgr`` the same picture at its own size, from which the second,
    finer look takes its pixels (``picture_bgr`` itself if None).
    """
    picture = _checked(picture_bgr)
    source = picture if source_bgr is None else _checked(source_bgr)
    height, width = picture.shape[:2]
    size = (width, height)
    if not _lines(_view(picture, picture, 1.0), size):
        return []
    return sorted(_lines(_view(picture, source, FINE_SCALE), size), key=_reading_order)


def scaled(lines: list[TextLine], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[TextLine]:
    """``lines`` found on a picture of ``from_size`` (width, height), given for the same picture at ``to_size``."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return [replace(line, quad=tuple((x * sx, y * sy) for x, y in line.quad)) for line in lines]


def mask(lines: list[TextLine], size: tuple[int, int]) -> np.ndarray:
    """HxW bool for a picture of ``size`` (width, height): the pixels whose middle lies in the box of a line.

    A middle on a box's edge is in it.
    """
    width, height = size
    covered = np.zeros((height, width), dtype=bool)
    for line in lines:
        quad = np.asarray(line.quad, dtype=np.float64)
        x0, x1 = max(int(np.floor(quad[:, 0].min() - 0.5)), 0), min(int(np.ceil(quad[:, 0].max() - 0.5)) + 1, width)
        y0, y1 = max(int(np.floor(quad[:, 1].min() - 0.5)), 0), min(int(np.ceil(quad[:, 1].max() - 0.5)) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        # Inside a convex polygon: on the same side of every edge (or on it).
        sides = [
            (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) for a, b in zip(quad, np.roll(quad, -1, axis=0))
        ]
        inside = np.logical_and.reduce([s >= 0 for s in sides]) | np.logical_and.reduce([s <= 0 for s in sides])
        covered[y0:y1, x0:x1] |= inside
    return covered


def _checked(picture_bgr: np.ndarray) -> np.ndarray:
    picture = np.asarray(picture_bgr)
    if picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3 or 0 in picture.shape[:2]:
        raise ValueError(f"expected an HxWx3 uint8 picture, got {picture.dtype} {picture.shape}")
    return picture


def _input_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    """(width, height) the network sees a picture of ``size`` at ``scale``: each side rounded to ``SIDE_MULTIPLE``."""
    return tuple(max(SIDE_MULTIPLE, int(round(side * scale / SIDE_MULTIPLE)) * SIDE_MULTIPLE) for side in size)


def _view(picture: np.ndarray, source: np.ndarray, scale: float) -> np.ndarray:
    """The picture as the network sees it at ``scale`` times ``picture``'s size: from ``source``, shrunk or stretched."""
    width, height = _input_size(picture.shape[1::-1], scale)
    if source is picture:
        return cv2.resize(picture, (width, height))
    shrink = source.shape[1] >= width
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def _probability(view: np.ndarray) -> np.ndarray:
    """The network's map for ``view``: how likely each of its pixels is to lie in the core of a line of text."""
    global _session
    blob = ((view.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()
    with _lock:
        if _session is None:
            import onnxruntime  # here, so that importing the pipeline doesn't pay for it

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = parallel.worker_count()
            options.log_severity_level = 3  # errors only
            _session = onnxruntime.InferenceSession(
                (RESOURCES_DIR / MODEL).read_bytes(), options, providers=["CPUExecutionProvider"]
            )
        session = _session
    # One session runs from several threads at once; its result doesn't depend on how many threads it uses.
    probability = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return probability.reshape(probability.shape[-2:])


def _lines(view: np.ndarray, size: tuple[int, int]) -> list[TextLine]:
    """The lines the network finds in ``view``, given in the pixels of the picture of ``size`` (width, height) it shows.

    Only lines that look like one are kept (see ``MAX_BOX_HEIGHT_MM``).
    """
    probability = _probability(view)
    map_height, map_width = probability.shape
    sx, sy = size[0] / map_width, size[1] / map_height
    max_height_px = print_scale(size).mm_to_px(MAX_BOX_HEIGHT_MM)
    # Grown by a pixel right and down, so that a core's outline, through its pixels' middles, spans its pixels' corners.
    cores = cv2.dilate((probability > THRESHOLD).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(cores, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours[:_MAX_CANDIDATES]:
        center, (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) < _MIN_CORE_PX:
            continue
        score = _mean_inside(probability, cv2.boxPoints((center, (w, h), angle)))
        if score < MIN_SCORE:
            continue
        grow = w * h * UNCLIP / (2 * (w + h))
        w, h = w + 2 * grow, h + 2 * grow
        if min(w, h) < _MIN_BOX_PX:
            continue
        corners = cv2.boxPoints((center, (w, h), angle)).astype(np.float64) * (sx, sy)
        line = TextLine(quad=tuple((float(x), float(y)) for x, y in corners), score=score)
        height, length = line.sides
        if height <= max_height_px and length >= MIN_ELONGATION * height:
            lines.append(line)
    return lines


def _mean_inside(probability: np.ndarray, corners: np.ndarray) -> float:
    """The mean of ``probability`` over the box with these corners, filled as PaddleOCR fills it (corners truncated)."""
    height, width = probability.shape
    x0 = int(np.clip(np.floor(corners[:, 0].min()), 0, width - 1))
    x1 = int(np.clip(np.ceil(corners[:, 0].max()), 0, width - 1))
    y0 = int(np.clip(np.floor(corners[:, 1].min()), 0, height - 1))
    y1 = int(np.clip(np.ceil(corners[:, 1].max()), 0, height - 1))
    inside = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillPoly(inside, [(corners - (x0, y0)).astype(np.int32)], 1)
    return float(cv2.mean(probability[y0 : y1 + 1, x0 : x1 + 1], inside)[0])


def _reading_order(line: TextLine) -> tuple[float, float]:
    quad = np.asarray(line.quad)
    return float(quad[:, 1].mean()), float(quad[:, 0].mean())

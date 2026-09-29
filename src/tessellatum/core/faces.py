"""Faces: where a picture has them, for the stages that give them more detail.

People notice a wrong face far more than a wrong patch of grass, so a page
should spend its detail there. This module finds the faces; the region stage
lets a region be smaller inside them (see ``regions.build_regions``).

Two detectors look, both shipped with the app and run offline
(``resources/MODELS.md`` has their sources and licenses):

- **YuNet**, a small network OpenCV's own DNN module runs, for photographed
  and painted faces. It was trained on people, but it finds cats and lions
  too;
- **lbpcascade_animeface**, a cascade of LBP features for drawn faces, which
  YuNet misses. OpenCV 5 no longer runs cascades, so ``detect_cascade`` does,
  window for window as OpenCV 4's ``CascadeClassifier.detectMultiScale``. It
  frames a drawn head, hair and all, rather than the face alone.

A face both find is listed by each. Faces are found once per picture, on it
at preview size, so a preview and an export always agree.

A face narrower than ``MIN_FACE_WIDTH_MM`` on the printed page is left out:
at that width an eye, about a fifth of a face across, is as wide as the
brush, so no stage can paint the features of a smaller one.
"""

from __future__ import annotations

import math
import threading
import xml.etree.ElementTree as ElementTree
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM, print_scale

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
YUNET_MODEL = "face_detection_yunet_2026may.onnx"
CASCADE_MODEL = "lbpcascade_animeface.xml"

# Five brush widths: an eye is about a fifth of a face across.
MIN_FACE_WIDTH_MM = 5 * MIN_PAINTABLE_WIDTH_MM

# YuNet looks at the picture with its long edge at most this many pixels, where the benchmark's faces sit best in the
# range it was trained on (faces of about 10 to 300 px), and keeps faces it scores at least YUNET_MIN_SCORE, dropping
# any that overlap a better one by more than YUNET_MAX_OVERLAP (intersection over union).
YUNET_LONG_EDGE = 640
YUNET_MIN_SCORE = 0.5
YUNET_MAX_OVERLAP = 0.3

# The cascade looks at the picture in gray, its histogram equalized, with its long edge at most CASCADE_LONG_EDGE
# pixels, and padded all round by CASCADE_PAD of that edge with its own border pixels, so that a head filling the
# picture still has windows to fall in. Windows grow by CASCADE_SCALE_STEP from one size to the next, and a face is a
# group of more than CASCADE_MIN_NEIGHBORS windows that found it.
CASCADE_LONG_EDGE = 480
CASCADE_PAD = 0.15
CASCADE_SCALE_STEP = 1.1
CASCADE_MIN_NEIGHBORS = 3

# How far apart two windows may lie and be grouped as one face, as a share of their size (OpenCV's GROUP_EPS).
GROUP_EPS = 0.2
# OpenCV lowers every stage threshold by this much when it loads a cascade.
_THRESHOLD_EPS = np.float32(1e-5)


@dataclass(frozen=True)
class Face:
    """A face found: its box, how sure the detector is, and which detector found it."""

    box: tuple[float, float, float, float]  # x, y, width, height, in pixels of the picture or page it is given for
    # YuNet's score, 0-1; for the cascade, how many windows found the face.
    score: float
    detector: str  # "yunet" or "cascade"
    # YuNet's five points: the right eye, the left eye, the tip of the nose, and the right and left corners of the
    # mouth (the face's own right and left); none from the cascade.
    landmarks: tuple[tuple[float, float], ...] = ()


def find_faces(picture_bgr: np.ndarray) -> list[Face]:
    """The faces in ``picture_bgr``, in its pixels: YuNet's, then the cascade's, each best first.

    Boxes are clipped to the picture, and faces narrower than
    ``MIN_FACE_WIDTH_MM`` on its printed page are left out.
    """
    h, w = picture_bgr.shape[:2]
    min_width_px = print_scale((w, h)).mm_to_px(MIN_FACE_WIDTH_MM)
    found = [_clipped(face, (w, h)) for face in _yunet_faces(picture_bgr) + _cascade_faces(picture_bgr)]
    return [face for face in found if face.box[2] >= min_width_px]


def scaled(faces: list[Face], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[Face]:
    """``faces`` found on a picture of ``from_size`` (width, height), given for the same picture at ``to_size``."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return [
        replace(
            face,
            box=(face.box[0] * sx, face.box[1] * sy, face.box[2] * sx, face.box[3] * sy),
            landmarks=tuple((x * sx, y * sy) for x, y in face.landmarks),
        )
        for face in faces
    ]


def mask(faces: list[Face], size: tuple[int, int]) -> np.ndarray:
    """HxW bool for a picture of ``size`` (width, height): the pixels whose middle lies in a box of ``faces``.

    A box's corner (x, y) is the corner of pixel (x, y), as in OpenCV, so a
    box of whole pixels covers exactly the pixels it spans.
    """
    width, height = size
    covered = np.zeros((height, width), dtype=bool)
    for face in faces:
        x, y, w, h = face.box
        # Pixel c's middle is at c + 0.5: inside when x <= c + 0.5 < x + w.
        x0, x1 = max(math.ceil(x - 0.5), 0), min(math.ceil(x + w - 0.5), width)
        y0, y1 = max(math.ceil(y - 0.5), 0), min(math.ceil(y + h - 0.5), height)
        covered[y0:y1, x0:x1] = True
    return covered


def _clipped(face: Face, size: tuple[int, int]) -> Face:
    x, y, w, h = face.box
    x0, y0 = min(max(x, 0.0), size[0]), min(max(y, 0.0), size[1])
    x1, y1 = min(max(x + w, 0.0), size[0]), min(max(y + h, 0.0), size[1])
    return replace(face, box=(x0, y0, x1 - x0, y1 - y0))


def _shrunk(picture_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = picture_bgr.shape[:2]
    if max(h, w) <= long_edge:
        return picture_bgr
    scale = long_edge / max(h, w)
    return cv2.resize(picture_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


# --- YuNet ---------------------------------------------------------------------------------------------------------

_yunet_lock = threading.Lock()
_yunet = None


def _yunet_faces(picture_bgr: np.ndarray) -> list[Face]:
    global _yunet
    small = _shrunk(picture_bgr, YUNET_LONG_EDGE)
    size = (small.shape[1], small.shape[0])
    with _yunet_lock:  # one detector, which isn't safe to run from two threads at once
        if _yunet is None:
            model = np.frombuffer((RESOURCES_DIR / YUNET_MODEL).read_bytes(), dtype=np.uint8)
            with _quiet_opencv():  # OpenCV 5 warns that its new DNN engine takes no target device
                _yunet = cv2.FaceDetectorYN.create(
                    "onnx", model, np.empty(0, dtype=np.uint8), size, YUNET_MIN_SCORE, YUNET_MAX_OVERLAP
                )
        _yunet.setInputSize(size)
        _, rows = _yunet.detect(small)
    if rows is None:
        return []
    sx, sy = picture_bgr.shape[1] / size[0], picture_bgr.shape[0] / size[1]
    faces = [
        Face(
            box=(float(r[0]) * sx, float(r[1]) * sy, float(r[2]) * sx, float(r[3]) * sy),
            score=float(r[14]),
            detector="yunet",
            landmarks=tuple((float(r[4 + 2 * k]) * sx, float(r[5 + 2 * k]) * sy) for k in range(5)),
        )
        for r in rows
    ]
    return sorted(faces, key=lambda face: -face.score)


@contextmanager
def _quiet_opencv() -> Iterator[None]:
    """Hold OpenCV's log to errors within a ``with`` block."""
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging is None:
        yield
        return
    level = logging.getLogLevel()
    logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        logging.setLogLevel(level)


# --- The cascade ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cascade:
    """A boosted cascade of LBP stumps, in OpenCV's XML format, as arrays for ``kernels.lbp_cascade_rows``."""

    window: tuple[int, int]  # width, height of the window it was trained on
    stage_ends: np.ndarray  # int32: stage s ends at weak classifier stage_ends[s]
    thresholds: np.ndarray  # float32: a window passes stage s with a sum of at least thresholds[s]
    features: np.ndarray  # int32: the feature each weak classifier reads
    subsets: np.ndarray  # uint32, (classifiers, 8): the codes that take leaves[i, 0], as a 256-bit set
    leaves: np.ndarray  # float32, (classifiers, 2)
    fx: np.ndarray  # int32, (features, 4): the column edges of each feature's 3 x 3 blocks, from the window's left
    fy: np.ndarray  # int32, (features, 4): their row edges, from the window's top


def load_cascade(path: Path) -> Cascade:
    """Read an LBP cascade saved by OpenCV's ``opencv_traincascade`` (``stageType`` BOOST, ``featureType`` LBP).

    Numbers are read as doubles and kept in single precision, and stage
    thresholds lowered by 1e-5 in single precision, as OpenCV does when it
    loads one. A cascade whose features reach outside its window, or whose
    stumps read a feature it doesn't have, is refused: the window search
    doesn't check where it reads.
    """
    root = ElementTree.parse(path).getroot().find("cascade")
    if root is None or root.findtext("featureType") != "LBP" or root.findtext("stageType") != "BOOST":
        raise ValueError(f"{path}: not a boosted LBP cascade")
    subset_size = (int(root.findtext("featureParams/maxCatCount")) + 31) // 32
    stage_ends, thresholds, features, subsets, leaves = [], [], [], [], []
    for stage in root.find("stages"):
        thresholds.append(np.float32(float(stage.findtext("stageThreshold"))) - _THRESHOLD_EPS)
        for weak in stage.find("weakClassifiers"):
            nodes = [int(value) for value in weak.findtext("internalNodes").split()]
            if len(nodes) != 3 + subset_size:
                raise ValueError(f"{path}: only stumps (one split per weak classifier) are supported")
            features.append(nodes[2])
            subsets.append(nodes[3:])
            leaves.append([float(value) for value in weak.findtext("leafValues").split()])
            if len(leaves[-1]) != 2:
                raise ValueError(f"{path}: a stump has two leaves")
        stage_ends.append(len(features))
    rects = np.array(
        [[int(value) for value in feature.findtext("rect").split()] for feature in root.find("features")], dtype=np.int32
    ).reshape(-1, 4)
    window = (int(root.findtext("width")), int(root.findtext("height")))
    x, y, w, h = rects.T
    if ((w < 1) | (h < 1) | (x < 0) | (y < 0) | (x + 3 * w > window[0]) | (y + 3 * h > window[1])).any():
        raise ValueError(f"{path}: a feature reaches outside the {window[0]} x {window[1]} window")
    if any(not 0 <= feature < len(rects) for feature in features):
        raise ValueError(f"{path}: a stump reads a feature the cascade doesn't have")
    steps = np.arange(4, dtype=np.int32)
    return Cascade(
        window=window,
        stage_ends=np.array(stage_ends, dtype=np.int32),
        thresholds=np.array(thresholds, dtype=np.float32),
        features=np.array(features, dtype=np.int32),
        subsets=np.array(subsets, dtype=np.int64).astype(np.uint32),  # the file writes them as signed 32-bit
        leaves=np.array(leaves, dtype=np.float32),
        fx=np.ascontiguousarray(rects[:, :1] + rects[:, 2:3] * steps),
        fy=np.ascontiguousarray(rects[:, 1:2] + rects[:, 3:4] * steps),
    )


@lru_cache(maxsize=1)
def _bundled_cascade() -> Cascade:
    return load_cascade(RESOURCES_DIR / CASCADE_MODEL)


def _cascade_faces(picture_bgr: np.ndarray) -> list[Face]:
    small = _shrunk(picture_bgr, CASCADE_LONG_EDGE)
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    pad = int(round(CASCADE_PAD * max(gray.shape)))
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sx, sy = picture_bgr.shape[1] / small.shape[1], picture_bgr.shape[0] / small.shape[0]
    found = detect_cascade(padded, _bundled_cascade(), CASCADE_SCALE_STEP, CASCADE_MIN_NEIGHBORS)
    faces = [
        Face(box=((x - pad) * sx, (y - pad) * sy, w * sx, h * sy), score=float(count), detector="cascade")
        for (x, y, w, h), count in found
    ]
    return sorted(faces, key=lambda face: -face.score)


def detect_cascade(
    gray: np.ndarray,
    cascade: Cascade,
    scale_step: float = 1.1,
    min_neighbors: int = 3,
    min_size: tuple[int, int] = (0, 0),
) -> list[tuple[tuple[int, int, int, int], int]]:
    """``cascade``'s detections in the 8-bit ``gray``, as OpenCV 4's ``CascadeClassifier.detectMultiScale`` gives them.

    Each is an (x, y, width, height) box, clipped to the image, and how many
    windows it groups; with ``min_neighbors`` 0, every window that passed, each
    counted once. Windows start at the cascade's own size and grow by
    ``scale_step`` while they fit in the image, skipping those smaller than
    ``min_size`` -- unless that is all of them, when the size nearest
    ``min_size`` is searched. The image is shrunk to each size instead of the
    window grown, and the windows are one pixel apart on images shrunk at least
    twofold, two pixels apart on the others. As OpenCV splits the rows of
    windows into stripes, a count that rounds down, a two-pixel step can leave
    the last row unsearched (see ``_searched_rows``).

    The image's integral must fit in 32 bits, as OpenCV's does: at most
    8,421,504 pixels.
    """
    h, w = gray.shape
    if 255 * gray.size >= 2**31:
        raise ValueError(f"a {w} x {h} image is too big for a 32-bit integral")
    win_w, win_h = cascade.window
    every = []  # every size that fits, as OpenCV lists them: the growing factor in double, kept in single precision
    factor = 1.0
    while round(win_w * factor) <= w and round(win_h * factor) <= h:
        every.append(np.float32(factor))
        factor *= scale_step

    def window(scale: np.float32) -> tuple[int, int]:
        return int(np.rint(np.float32(win_w) * scale)), int(np.rint(np.float32(win_h) * scale))

    scales = []
    for scale in every:
        if window(scale)[0] > w or window(scale)[1] > h:
            break
        if window(scale)[0] >= min_size[0] and window(scale)[1] >= min_size[1]:
            scales.append(scale)
    if not scales and every:
        distances = [(min_size[0] - ww) ** 2 + (min_size[1] - wh) ** 2 for ww, wh in map(window, every)]
        scales = [every[distances.index(min(distances))]]

    levels = []  # per size: its scale, integral image, step, rows searched, first row among all sizes', and hits
    rows_before = 0
    stripes = None
    for scale in scales:
        size = (int(np.rint(np.float32(w) / scale)), int(np.rint(np.float32(h) / scale)))
        shrunk = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR_EXACT)
        integral = cv2.integral(shrunk, sdepth=cv2.CV_32S)
        step = 1 if scale >= 2 else 2
        if stripes is None:  # set by the first size's row of windows
            stripes = -(-max(integral.shape[1] - win_w, 0) // 32)
        rows = _searched_rows(max(integral.shape[0] - win_h, 0), step, stripes)
        levels.append((scale, integral, step, rows, rows_before, np.zeros(integral.shape, dtype=np.uint8)))
        rows_before += rows

    def search(start: int, stop: int) -> None:
        for _scale, integral, step, rows, first, hits in levels:
            row_from, row_to = max(start - first, 0), min(stop - first, rows)
            if row_from < row_to:
                kernels.lbp_cascade_rows(
                    integral, win_w, win_h, step, row_from, row_to, cascade.fx, cascade.fy, cascade.stage_ends,
                    cascade.thresholds, cascade.features, cascade.subsets, cascade.leaves, hits,
                )

    parallel.for_each_stripe(search, rows_before)

    windows = []
    for scale, _integral, _step, _rows, _first, hits in levels:
        box_w, box_h = int(np.rint(np.float32(win_w) * scale)), int(np.rint(np.float32(win_h) * scale))
        ys, xs = np.nonzero(hits)
        windows += [
            (int(np.rint(np.float32(x) * scale)), int(np.rint(np.float32(y) * scale)), box_w, box_h)
            for y, x in zip(ys.tolist(), xs.tolist())
        ]
    grouped = group_windows(windows, min_neighbors)
    return [(box, count) for box, count in ((_clip_box(box, (w, h)), count) for box, count in grouped) if box[2] > 0 < box[3]]


def _searched_rows(positions: int, step: int, stripes: int) -> int:
    """How many rows of windows OpenCV 4 searches, of the rows at y = 0, step, ... below ``positions``.

    It splits them into ``stripes`` stripes (one per 32 windows along the
    first size's row) of whole steps, ``positions // step`` rows shared out,
    rounded up, and at least one step each. Where that division comes out
    exact, and a row starts at the last position, the stripes stop short of
    it.
    """
    stripe = max((positions // step + stripes - 1) // stripes, 1) * step
    return len(range(0, min(stripes * stripe, positions), step))


def group_windows(
    windows: list[tuple[int, int, int, int]], min_neighbors: int, eps: float = GROUP_EPS
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Group overlapping detection windows into faces, as OpenCV's ``groupRectangles`` does, with each group's size.

    Windows join a group where their four edges all lie within ``eps`` of
    their mean size of each other (and so on, transitively). A group of more
    than ``min_neighbors`` windows becomes their mean box, unless it lies
    inside a bigger group's box (give or take ``eps`` of that box's size) that
    has more windows than it and than 3, or it has fewer than 3. With
    ``min_neighbors`` 0 or less, every window is returned alone. Groups are
    listed in the order of their first window.
    """
    if min_neighbors <= 0 or not windows:
        return [(tuple(window), 1) for window in windows]
    boxes = np.array(windows, dtype=np.int64)
    x0, y0, bw, bh = boxes.T
    x1, y1 = x0 + bw, y0 + bh
    delta = eps * (np.minimum.outer(bw, bw) + np.minimum.outer(bh, bh)) * 0.5
    similar = (
        (np.abs(np.subtract.outer(x0, x0)) <= delta)
        & (np.abs(np.subtract.outer(y0, y0)) <= delta)
        & (np.abs(np.subtract.outer(x1, x1)) <= delta)
        & (np.abs(np.subtract.outer(y1, y1)) <= delta)
    )
    parent = list(range(len(windows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(similar, 1))):
        a, b = root(int(i)), root(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)

    group_of: dict[int, int] = {}
    sums: list[list[int]] = []
    for i in range(len(windows)):
        r = root(i)
        if r not in group_of:
            group_of[r] = len(sums)
            sums.append([0, 0, 0, 0, 0])
        total = sums[group_of[r]]
        for k in range(4):
            total[k] += windows[i][k]
        total[4] += 1
    means = []
    for *edges, count in sums:
        share = np.float32(1) / np.float32(count)
        means.append((tuple(int(np.rint(np.float32(edge) * share)) for edge in edges), count))

    kept = []
    for i, (box, count) in enumerate(means):
        if count <= min_neighbors:
            continue
        inside_bigger = False
        for j, (other, other_count) in enumerate(means):
            if j == i or other_count <= min_neighbors:
                continue
            dx, dy = int(np.rint(other[2] * eps)), int(np.rint(other[3] * eps))
            if (
                box[0] >= other[0] - dx
                and box[1] >= other[1] - dy
                and box[0] + box[2] <= other[0] + other[2] + dx
                and box[1] + box[3] <= other[1] + other[3] + dy
                and (other_count > max(3, count) or count < 3)
            ):
                inside_bigger = True
                break
        if not inside_bigger:
            kept.append((box, count))
    return kept


def _clip_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0 = max(box[0], 0), max(box[1], 0)
    x1, y1 = min(box[0] + box[2], size[0]), min(box[1] + box[3], size[1])
    return (x0, y0, x1 - x0, y1 - y0)

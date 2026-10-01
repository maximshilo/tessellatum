"""The subject: what a picture is of, for the stages that give it more detail.

A page should spend its detail where people look: on the subject -- a person,
an animal, a building -- rather than on the wall, the grass or the sky behind
it. This module finds the subject; the region stage lets a region inside it be
half as large, as inside a face (see ``regions.build_regions``).

It is found by **U²-Net-p**, a small salient object detection network
(4.6 MB) that OpenCV's own DNN module runs, shipped with the app and run
offline (``resources/MODELS.md`` has its source and license). The network
looks at the whole picture squeezed to ``INPUT_SIZE`` pixels square, the size
it was trained at, and says how likely each of its pixels is to be part of
the picture's salient object; the subject is where that is at least
``MIN_PROBABILITY``. A picture with no one thing to look at -- a crowded
street, a landscape -- has little or no subject.

The subject is found once per picture, on it at preview size, so a preview
and an export always agree.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from tessellatum.core.faces import RESOURCES_DIR, _quiet_opencv

MODEL = "u2netp.onnx"

# The network sees the picture squeezed to this many pixels square, its shape not kept, as it was trained.
INPUT_SIZE = 320
# The subject is where the network gives a pixel at least this probability of being the salient object.
MIN_PROBABILITY = 0.5
# The picture, scaled to [0, 1] by its brightest value as U²-Net's own code does, is standardized by the ImageNet
# statistics the network was trained with (RGB).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_net = None


def find_subject(picture_bgr: np.ndarray) -> np.ndarray:
    """How likely each pixel of the network's view of ``picture_bgr`` is to be its subject.

    ``picture_bgr`` is an HxWx3 uint8 picture. Returns an
    ``INPUT_SIZE`` x ``INPUT_SIZE`` float32 map in [0, 1] over the whole
    picture, squeezed to a square as the network saw it; ``mask`` stretches it
    onto a page.
    """
    global _net
    picture = np.asarray(picture_bgr)
    if picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3 or 0 in picture.shape[:2]:
        raise ValueError(f"expected an HxWx3 uint8 picture, got {picture.dtype} {picture.shape}")
    rgb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    brightest = small.max()
    if brightest > 0:
        small /= brightest
    blob = ((small - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()
    with _lock:  # one network, which isn't safe to run from two threads at once
        if _net is None:
            model = np.frombuffer((RESOURCES_DIR / MODEL).read_bytes(), dtype=np.uint8)
            with _quiet_opencv():  # OpenCV 5 warns that its new DNN engine takes no target device
                _net = cv2.dnn.readNetFromONNX(model)
        _net.setInput(blob)
        probability = _net.forward()  # the first output: the network's fused map
    return np.clip(probability[0, 0], 0.0, 1.0).astype(np.float32)


def mask(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The subject's pixels on a page of ``size`` (width, height), where ``probability`` is at least ``MIN_PROBABILITY``.

    ``probability`` is ``find_subject``'s map, which covers the whole
    picture, so it is stretched over the page, to the page's own shape.
    """
    probability = np.asarray(probability, dtype=np.float32)
    width, height = size
    if probability.ndim != 2 or 0 in probability.shape or width < 1 or height < 1:
        raise ValueError(f"expected a 2-D map and a page size, got {probability.shape} and {size}")
    return cv2.resize(probability, (int(width), int(height)), interpolation=cv2.INTER_LINEAR) >= MIN_PROBABILITY

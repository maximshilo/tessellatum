"""Color-vision simulation: the canvas shown as a person with a color-vision deficiency, or without color, sees it.

A view filter on the screen only: the page, its versions and every export keep their colors. It answers a painter's
question the app can't measure for them: whether two colors of the page that a viewer must tell apart still look
different to someone who sees fewer of them, and whether the page reads in values alone (Grayscale).

The simulations are Machado, Oliveira and Fernandes' (2009) at full severity, a 3×3 matrix on linear RGB each:
dichromacy, the strongest form of each deficiency. Their tritanopia is the least exact of the three (they model it as
they do the others, from a shift of the cone's sensitivity). Grayscale is the relative luminance (Rec. 709 weights).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QT_TRANSLATE_NOOP
from PySide6.QtGui import QImage

NORMAL, PROTANOPIA, DEUTERANOPIA, TRITANOPIA, GRAYSCALE = "normal", "protanopia", "deuteranopia", "tritanopia", "grayscale"
MODES = (NORMAL, PROTANOPIA, DEUTERANOPIA, TRITANOPIA, GRAYSCALE)
NAMES = {
    NORMAL: QT_TRANSLATE_NOOP("vision", "Normal"),
    PROTANOPIA: QT_TRANSLATE_NOOP("vision", "Protanopia"),
    DEUTERANOPIA: QT_TRANSLATE_NOOP("vision", "Deuteranopia"),
    TRITANOPIA: QT_TRANSLATE_NOOP("vision", "Tritanopia"),
    GRAYSCALE: QT_TRANSLATE_NOOP("vision", "Grayscale"),
}  # translated where shown, under the context "vision"
DESCRIPTIONS = {
    NORMAL: QT_TRANSLATE_NOOP("vision", "The page's own colors"),
    PROTANOPIA: QT_TRANSLATE_NOOP("vision", "As a person without red-sensitive cones sees it (about 1 in 100 men)"),
    DEUTERANOPIA: QT_TRANSLATE_NOOP("vision", "As a person without green-sensitive cones sees it (about 1 in 100 "
                                              "men)"),
    TRITANOPIA: QT_TRANSLATE_NOOP("vision", "As a person without blue-sensitive cones sees it (rare)"),
    GRAYSCALE: QT_TRANSLATE_NOOP("vision", "Without color: whether the page reads in light and dark alone"),
}

_MATRICES = {
    PROTANOPIA: ((0.152286, 1.052583, -0.204868),
                 (0.114503, 0.786281, 0.099216),
                 (-0.003882, -0.048116, 1.051998)),
    DEUTERANOPIA: ((0.367322, 0.860646, -0.227968),
                   (0.280085, 0.672501, 0.047413),
                   (-0.011820, 0.042940, 0.968881)),
    TRITANOPIA: ((1.255528, -0.076749, -0.178779),
                 (-0.078411, 0.930809, 0.147602),
                 (0.004733, 0.691367, 0.303900)),
    GRAYSCALE: ((0.2126, 0.7152, 0.0722),) * 3,
}

# sRGB to linear for each of the 256 values, and back from linear quantized to 4096 steps (finer than 8 bits in the
# darks, where sRGB's steps are smallest in linear terms).
_codes = np.arange(256, dtype=np.float64) / 255
_TO_LINEAR = np.where(_codes <= 0.04045, _codes / 12.92, ((_codes + 0.055) / 1.055) ** 2.4).astype(np.float32)
_STEPS = 4096
_levels = np.arange(_STEPS, dtype=np.float64) / (_STEPS - 1)
_FROM_LINEAR = np.round(
    255 * np.where(_levels <= 0.0031308, _levels * 12.92, 1.055 * _levels ** (1 / 2.4) - 0.055)
).astype(np.uint8)


def simulate_rgb(rgb: np.ndarray, mode: str) -> np.ndarray:
    """``rgb`` (H×W×3 uint8, sRGB) as seen under ``mode``: a new array, or ``rgb`` itself for Normal."""
    if mode == NORMAL:
        return rgb
    matrix = np.asarray(_MATRICES[mode], dtype=np.float32)
    linear = _TO_LINEAR[rgb]
    seen = linear @ matrix.T
    np.clip(seen, 0.0, 1.0, out=seen)
    return _FROM_LINEAR[(seen * (_STEPS - 1) + 0.5).astype(np.int32)]


def simulate(image: QImage, mode: str) -> QImage:
    """``image`` as seen under ``mode``; ``image`` itself for Normal. Transparency is kept."""
    if mode == NORMAL or image.isNull():
        return image
    alpha = image.hasAlphaChannel()
    fmt = QImage.Format_RGBA8888 if alpha else QImage.Format_RGB888
    source = image.convertToFormat(fmt)
    channels = 4 if alpha else 3
    width, height = source.width(), source.height()
    rows = np.frombuffer(source.constBits(), np.uint8).reshape(height, source.bytesPerLine())
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    seen = np.empty_like(pixels)
    seen[..., :3] = simulate_rgb(np.ascontiguousarray(pixels[..., :3]), mode)
    if alpha:
        seen[..., 3] = pixels[..., 3]
    result = QImage(seen.data, width, height, width * channels, fmt)
    return result.copy()  # its own pixels, not the array's

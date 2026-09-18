"""How far apart two colors look: CIE Lab and CIEDE2000.

A palette is only useful if the painter can tell its colors apart and mix
them. ``benchmarks/QUALITY_BENCHMARKS.md`` puts that at a clear margin of
about 10 CIEDE2000 between any two colors on the legend, which is what
``MIN_PALETTE_DE00`` holds.

Distances are measured on the palette's final 8-bit sRGB colors, converted to
Lab exactly as the sRGB and CIELAB standards define it. OpenCV's conversion
interpolates lookup tables, which puts a color up to about 0.5 CIEDE2000 from
its true value: too much for a handful of colors judged against a threshold,
though it averages out over a whole image. Only a few dozen colors are ever
compared, so the exact conversion costs nothing.
"""

from __future__ import annotations

import numpy as np

# The clear margin a palette's colors keep from each other, in CIEDE2000.
MIN_PALETTE_DE00 = 10.0

# sRGB (IEC 61966-2-1) primaries to CIE XYZ. Rows are X, Y, Z; each row's sum
# is the white point's coordinate, which the conversion divides by.
_SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def bgr_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    """uint8 sRGB colors (BGR order, ...x3) -> float64 CIE Lab (D65), with L in 0..100.

    Relative to the matrix's own white point, so grays come out with no chroma.
    """
    rgb = np.asarray(colors_bgr, dtype=np.float64)[..., ::-1] / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T / _SRGB_TO_XYZ.sum(axis=1)
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz * (29 / 6) ** 2 / 3 + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Per-element CIEDE2000 color difference (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c_bar7 = ((np.hypot(a1, b1) + np.hypot(a2, b2)) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma_zero = (c1p * c2p) == 0

    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(chroma_zero, 0.0, dh)
    d_big_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(np.abs(h1p - h2p) > 180, np.where(h_sum < 360, h_sum + 360, h_sum - 360), h_sum) / 2
    h_bar = np.where(chroma_zero, h_sum, h_bar)

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7))
    l50 = (l_bar - 50) ** 2
    s_l = 1 + 0.015 * l50 / np.sqrt(20 + l50)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c

    return np.sqrt((dl / s_l) ** 2 + (dc / s_c) ** 2 + (d_big_h / s_h) ** 2 + r_t * (dc / s_c) * (d_big_h / s_h))


def pairwise_de00(colors_bgr: np.ndarray) -> np.ndarray:
    """CIEDE2000 between every two of ``colors_bgr`` (Kx3 uint8, BGR), with infinity down the diagonal.

    A color is never its own nearest neighbor, so the smallest value in the
    matrix is the palette's closest pair.
    """
    colors = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 3)
    lab = bgr_to_lab(colors)
    distance = ciede2000(lab[:, None, :], lab[None, :, :])
    np.fill_diagonal(distance, np.inf)
    return distance

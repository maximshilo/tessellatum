"""Color quantization: reduce an image to a small palette of flat colors.

K-means in Lab space, then a merge of the colors a painter could neither tell
apart nor mix -- closer than ``color.MIN_PALETTE_DE00`` -- so the palette comes
back with every color clearly different from every other.
"""

from __future__ import annotations

import cv2
import numpy as np

from tessellatum.core import kernels, parallel
from tessellatum.core.color import MIN_PALETTE_DE00, pairwise_de00

# The sparse bilateral filter samples its window every sigma_space /
# _TAPS_PER_SIGMA pixels along each axis.
_TAPS_PER_SIGMA = 6

# One k-means++ run over every pixel. On the sample images, three attempts
# took 3x as long for a finished painting that was on average no closer to
# the source (worst case: flat-color artwork at Easy, +3.7% color error, or
# 0.18 CIEDE2000). Fitting on a sample of pixels instead -- even half of them
# -- occasionally lost a distinct color entirely.
_KMEANS_ATTEMPTS = 1
_KMEANS_MAX_ITERATIONS = 30
_KMEANS_EPSILON = 0.5


def quantize(
    image_bgr: np.ndarray,
    num_colors: int,
    blur_sigma: float,
    seed: int = 0,
    min_de00: float = MIN_PALETTE_DE00,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``image_bgr`` to at most ``num_colors`` flat, clearly different colors.

    K-means minimizes distance in Lab, where equal distances are not equally
    visible: a photograph of fur or stone fills its clusters with browns a
    painter can neither tell apart nor mix. So colors closer than ``min_de00``
    CIEDE2000 are merged afterwards (``_merge_close_colors``), which is why
    the palette can come back shorter than asked for.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order (OpenCV convention).
        num_colors: how many clusters k-means looks for.
        blur_sigma: bilateral-filter smoothing strength (0 disables smoothing).
        seed: RNG seed so results are reproducible for the same inputs.
        min_de00: the clear margin every two palette colors keep, in CIEDE2000.
            0 leaves k-means' colors as they are.

    Returns:
        (label_map, palette_bgr):
            label_map: HxW int32 array, each pixel's color index.
            palette_bgr: Kx3 uint8 array of BGR colors, K <= num_colors,
                ordered from darkest to lightest (by perceptual lightness)
                so numbering reads naturally.
    """
    smoothed = _smooth(image_bgr, blur_sigma)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITERATIONS, _KMEANS_EPSILON)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers_lab = cv2.kmeans(
        samples, num_colors, None, criteria, attempts=_KMEANS_ATTEMPTS, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(lab.shape[:2]).astype(np.int32)
    pixels_per_color = np.bincount(labels.reshape(-1), minlength=len(centers_lab))
    centers_lab, group = _merge_close_colors(centers_lab, pixels_per_color, min_de00)

    kept = np.unique(group)  # the colors that survived the merge, in cluster order
    kept_centers = centers_lab[kept]
    order = np.argsort(kept_centers[:, 0])  # sort by L (lightness), darkest first
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)

    final_index = np.zeros(len(centers_lab), dtype=np.int32)  # cluster -> its color's place on the legend
    final_index[kept] = rank
    labels = final_index[group][labels].astype(np.int32)
    palette_bgr = _lab_centers_to_bgr(kept_centers)[order]

    return labels, palette_bgr


def _lab_centers_to_bgr(centers_lab: np.ndarray) -> np.ndarray:
    """Cluster centers in OpenCV's 8-bit Lab scale -> the Kx3 uint8 BGR colors the page is painted in."""
    centers_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(centers_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def _merge_close_colors(
    centers_lab: np.ndarray, pixels_per_color: np.ndarray, min_de00: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge the two closest colors over and over, until every two differ by ``min_de00``.

    Distance is CIEDE2000 between the colors as they will be printed and
    painted -- the 8-bit sRGB the centers convert to -- so it is the same
    number the finished legend is judged on.

    Two merged colors become the one their pixels average to, which is the
    color the painter would have mixed for both of them. Which of the two
    keeps the slot makes no difference: the merged color is the same either
    way, and the palette is sorted by lightness afterwards. Merging the
    *closest* pair first, rather than the pair that costs the least error, is
    what keeps the most colors and the closest painting: measured against five
    other mechanisms before it was chosen.

    Args:
        centers_lab: Kx3 cluster centers in OpenCV's 8-bit Lab scale.
        pixels_per_color: how many pixels each cluster has.
        min_de00: the margin to reach; 0 or less leaves the centers alone.

    Returns:
        (centers_lab, group): the centers, with every survivor moved to its
        merged color, and for each original cluster the index of the center it
        ended up in (``group[i] == i`` for a color that survived).
    """
    centers = np.array(centers_lab, dtype=np.float64)
    weights = np.asarray(pixels_per_color, dtype=np.float64).copy()
    group = np.arange(len(centers), dtype=np.int32)
    if min_de00 <= 0:
        return centers, group

    alive = list(range(len(centers)))
    while len(alive) > 1:
        distance = pairwise_de00(_lab_centers_to_bgr(centers[alive]))
        first, second = divmod(int(np.argmin(distance)), len(alive))
        if distance[first, second] >= min_de00:
            break
        survivor, absorbed = alive[first], alive[second]
        total = weights[survivor] + weights[absorbed]
        if total > 0:
            centers[survivor] = (
                centers[survivor] * weights[survivor] + centers[absorbed] * weights[absorbed]
            ) / total
        weights[survivor] = total
        group[group == absorbed] = survivor
        alive.remove(absorbed)

    return centers, group


def _smooth(image_bgr: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma <= 0:
        return image_bgr
    return bilateral_filter(image_bgr, sigma_color=blur_sigma * 8, sigma_space=blur_sigma * 4)


def bilateral_filter(image_bgr: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Edge-preserving smoothing: ``cv2.bilateralFilter(image, 0, sigma_color, sigma_space)``, sampled sparsely.

    The exact filter visits every pixel within radius 1.5 * sigma_space, so
    its cost grows with sigma_space squared: ~9,000 taps per pixel at the Easy
    preset. Here the same window and weights are sampled on a lattice about
    sigma_space / 6 pixels apart -- a few hundred taps -- with rows filtered in
    parallel. The result differs from the exact filter by a small fraction of
    a just-noticeable color difference. When the lattice would be every pixel
    anyway, OpenCV's filter is used as is.
    """
    spacing = max(1, round(sigma_space / _TAPS_PER_SIGMA))
    if spacing == 1:
        return cv2.bilateralFilter(image_bgr, 0, sigma_color, sigma_space)
    return _sparse_bilateral(image_bgr, sigma_color, sigma_space, spacing)


def _sparse_bilateral(image_bgr: np.ndarray, sigma_color: float, sigma_space: float, spacing: int) -> np.ndarray:
    """Bilateral filter using window taps ``spacing`` pixels apart (1 = every pixel)."""
    h, w = image_bgr.shape[:2]
    radius = max(1, round(sigma_space * 1.5))

    steps = np.arange(-(radius // spacing), radius // spacing + 1) * spacing
    dy, dx = np.meshgrid(steps, steps, indexing="ij")
    inside = dy * dy + dx * dx <= radius * radius
    dy, dx = dy[inside].astype(np.int64), dx[inside].astype(np.int64)
    space_weights = np.exp(-0.5 * (dy * dy + dx * dx) / (sigma_space * sigma_space)).astype(np.float32)
    color_weights = np.exp(-0.5 * (np.arange(3 * 255 + 1) / sigma_color) ** 2).astype(np.float32)
    offsets = (dy * (w + 2 * radius) + dx) * 3

    padded = cv2.copyMakeBorder(image_bgr, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    padded_flat = np.ascontiguousarray(padded).reshape(-1)
    out = np.empty((h, w, 3), dtype=np.uint8)
    out_flat = out.reshape(-1)

    def filter_rows(y_start: int, y_stop: int) -> None:
        kernels.bilateral_rows(
            padded_flat, out_flat, y_start, y_stop, w, radius, offsets, space_weights, color_weights
        )

    parallel.for_each_stripe(filter_rows, h)
    return out

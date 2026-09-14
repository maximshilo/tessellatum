"""Sanity checks for the benchmark harness's quality metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402

# Reference pairs from Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference formula".
SHARMA_PAIRS = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def test_ciede2000_matches_published_reference_values():
    lab1 = np.array([pair[0] for pair in SHARMA_PAIRS])
    lab2 = np.array([pair[1] for pair in SHARMA_PAIRS])
    expected = [pair[2] for pair in SHARMA_PAIRS]

    assert bm.ciede2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_ssim_is_one_for_identical_images_and_drops_with_noise():
    rng = np.random.default_rng(0)
    gradient = np.linspace(40, 200, 64, dtype=np.float64)
    image = np.repeat(np.tile(gradient, (64, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    noisy = np.clip(image.astype(np.int16) + rng.integers(-40, 40, size=image.shape), 0, 255).astype(np.uint8)

    assert bm.ssim_gray(image, image) == pytest.approx(1.0)
    assert bm.ssim_gray(image, noisy) < 0.9


def test_paint_fills_regions_with_their_palette_color():
    region_id_map = np.array([[0, 1], [1, 2]])
    region_color = np.array([2, 0, 1])
    palette_bgr = np.array([[0, 0, 0], [10, 20, 30], [255, 255, 255]], dtype=np.uint8)

    painted = bm.paint(region_id_map, region_color, palette_bgr)

    assert painted[0, 0].tolist() == [255, 255, 255]
    assert painted[0, 1].tolist() == [0, 0, 0]
    assert painted[1, 1].tolist() == [10, 20, 30]


def test_partition_identical_ignores_region_ids():
    a = np.array([[0, 0, 1], [2, 2, 1]])
    relabeled = np.array([[5, 5, 3], [4, 4, 3]])
    different = relabeled.copy()
    different[1, 0] = 5

    assert bm.partition_identical(a, relabeled)
    assert not bm.partition_identical(a, different)


def test_boundary_f1_tolerates_only_small_shifts():
    a = np.zeros((40, 40), dtype=np.int32)
    a[:, 20:] = 1
    shifted_1px = np.zeros_like(a)
    shifted_1px[:, 21:] = 1
    shifted_6px = np.zeros_like(a)
    shifted_6px[:, 26:] = 1

    assert bm.boundary_f1(a, a) == 1.0
    assert bm.boundary_f1(a, shifted_1px) == 1.0
    assert bm.boundary_f1(a, shifted_6px) == 0.0

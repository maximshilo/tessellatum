"""The color math the palette's margin is measured with."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402
from tessellatum.core import color  # noqa: E402

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

    assert color.ciede2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric_and_zero_for_one_color():
    lab1 = np.array([pair[0] for pair in SHARMA_PAIRS])
    lab2 = np.array([pair[1] for pair in SHARMA_PAIRS])

    assert color.ciede2000(lab2, lab1) == pytest.approx(color.ciede2000(lab1, lab2))
    assert color.ciede2000(lab1, lab1) == pytest.approx(np.zeros(len(lab1)))


def test_grays_convert_to_their_analytic_lightness_with_no_chroma():
    grays = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)

    lab = color.bgr_to_lab(grays)

    linear = np.where(
        np.arange(256) / 255 <= 0.04045,
        np.arange(256) / 255 / 12.92,
        ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4,
    )
    f = np.where(linear > (6 / 29) ** 3, np.cbrt(linear), linear * (29 / 6) ** 2 / 3 + 4 / 29)
    assert lab[:, 0] == pytest.approx(116 * f - 16, abs=1e-9)
    assert np.abs(lab[:, 1:]).max() < 1e-9


def test_the_app_measures_color_distance_exactly_as_the_benchmark_does():
    # The page has to clear a margin the harness measures with its own copy of
    # this math, so the two must not drift apart even in the last bits.
    rng = np.random.default_rng(7)
    colors = rng.integers(0, 256, size=(4000, 3), dtype=np.uint8)

    lab = color.bgr_to_lab(colors)
    expected_lab = bm.bgr_to_lab_exact(colors)

    assert np.abs(lab - expected_lab).max() == 0.0
    assert np.abs(color.ciede2000(lab[:2000], lab[2000:]) - bm.ciede2000(expected_lab[:2000], expected_lab[2000:])).max() == 0.0


def test_pairwise_de00_puts_infinity_down_the_diagonal():
    colors = np.array([[10, 20, 30], [200, 40, 60], [11, 21, 31]], dtype=np.uint8)

    distance = color.pairwise_de00(colors)

    assert np.isinf(np.diag(distance)).all()
    assert distance[0, 2] == pytest.approx(distance[2, 0])
    assert np.argmin(distance) == 2  # the near-identical pair (0, 2) is the closest
    assert distance[0, 2] < distance[0, 1]

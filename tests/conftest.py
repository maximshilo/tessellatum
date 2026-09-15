import numpy as np
import pytest


@pytest.fixture
def sample_image_bgr() -> np.ndarray:
    """A synthetic 200x200 image with a few flat-colored blocks plus noise."""
    rng = np.random.default_rng(42)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100, :100] = (200, 30, 30)     # blue-ish block (BGR)
    img[:100, 100:] = (30, 180, 30)     # green block
    img[100:, :100] = (30, 30, 200)     # red block
    img[100:, 100:] = (200, 200, 30)    # cyan-ish block
    noise = rng.integers(-10, 10, size=img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def speckled_image_bgr() -> np.ndarray:
    """Dark and light halves sprinkled with single gray pixels.

    Quantized to 3 colors, the specks get a color of their own, but merging
    folds them into their neighbors, so that color never reaches the legend.
    """
    img = np.full((60, 80, 3), 230, dtype=np.uint8)
    img[:, :40] = 20
    img[5::10, 5::10] = 128
    return img

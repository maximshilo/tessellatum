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

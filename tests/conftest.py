import sys
import threading

import numpy as np
import pytest

try:  # numbers shown and typed in the fields follow the locale: the tests' is US English, whatever the machine's
    from PySide6.QtCore import QLocale

    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _no_warm_up_left_running():
    """A window starts warming the pipeline up in a thread of its own, which can outlive the test that opened it (on a
    cold machine it loads the models first). It calls ``pipeline.generate``: left running, it would call the next
    test's fake instead. Each test starts once any such thread is done."""
    for thread in threading.enumerate():
        if thread.name == "pipeline-warm-up":
            thread.join(timeout=300)
    yield


@pytest.fixture(autouse=True)
def _unsaved_changes_let_go(monkeypatch):
    """A window closed with unsaved changes asks what to do in a modal dialog, which would wait for ever in a test:
    every test lets them go, unless it answers itself (``MainWindow._ask_unsaved``)."""
    window = sys.modules.get("tessellatum.gui.main_window")
    if window is not None:
        monkeypatch.setattr(window.MainWindow, "_ask_unsaved", lambda self, name: window.DISCARD_CHANGES)
    yield


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

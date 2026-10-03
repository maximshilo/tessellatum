"""The headless GUI drive: open, preview, cancel and export through the window, its dialogs answered by stubs."""

import os
import re
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty, pipeline  # noqa: E402
from tessellatum.core.pipeline import Handling, PipelineCancelled  # noqa: E402
from tessellatum.core.print_size import print_scale  # noqa: E402
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

# One application for the whole module, as in test_controls_panel.
_app = QApplication.instance() or QApplication([])

TIMEOUT_S = 300


def _blocks(size: tuple[int, int]) -> np.ndarray:
    """A picture of four flat blocks, BGR."""
    width, height = size
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[: height // 2, : width // 2] = (200, 40, 40)
    picture[: height // 2, width // 2 :] = (40, 170, 40)
    picture[height // 2 :, : width // 2] = (40, 40, 200)
    picture[height // 2 :, width // 2 :] = (210, 210, 60)
    return picture


class _Drive:
    """A window whose dialogs are stubs: what they were shown, and the file names they answer with."""

    def __init__(self, monkeypatch):
        self.dialogs: list[tuple[str, str, str]] = []
        self.open_path = ""
        self.save_path = ""
        self.calls: list[tuple[tuple, dict]] = []
        drive = self

        class MessageBox:
            @staticmethod
            def information(parent, title, text):
                drive.dialogs.append(("information", title, text))

            @staticmethod
            def warning(parent, title, text):
                drive.dialogs.append(("warning", title, text))

            @staticmethod
            def critical(parent, title, text):
                drive.dialogs.append(("critical", title, text))

        class FileDialog:
            @staticmethod
            def getOpenFileName(*args, **kwargs):
                return drive.open_path, ""

            @staticmethod
            def getSaveFileName(*args, **kwargs):
                return drive.save_path, ""

        monkeypatch.setattr(main_window, "QMessageBox", MessageBox)
        monkeypatch.setattr(main_window, "QFileDialog", FileDialog)
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)  # no second thread in the pipeline
        real = pipeline.generate
        monkeypatch.setattr(pipeline, "generate", lambda *args, **kwargs: self.calls.append((args, kwargs)) or real(*args, **kwargs))
        self.window = MainWindow()

    def wait(self) -> None:
        """Until the window's worker is done, and its last signals have reached the window."""
        deadline = time.monotonic() + TIMEOUT_S
        while self.window._worker is not None and self.window._worker.isRunning():
            assert time.monotonic() < deadline, "the pipeline didn't finish"
            _app.processEvents()
            time.sleep(0.005)
        _app.processEvents()

    def status(self) -> str:
        return self.window.statusBar().currentMessage()


@pytest.fixture
def drive(monkeypatch):
    drive = _Drive(monkeypatch)
    yield drive
    drive.window.close()
    drive.window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_an_opened_picture_is_previewed_with_the_panel_s_settings(drive, tmp_path):
    Image.fromarray(_blocks((600, 400))[:, :, ::-1]).save(tmp_path / "blocks.png")
    drive.open_path = str(tmp_path / "blocks.png")
    drive.window.open_image()
    assert drive.window.current_image_bgr.shape == (400, 600, 3)
    assert "blocks.png" in drive.status()

    controls = drive.window.controls
    controls.preset_combo.setCurrentText("Hard")
    controls.line_width_slider.setValue(controls.line_width_slider.maximum())
    controls.tone_combo.setCurrentText("Dark")
    controls.text_check.setChecked(False)
    drive.window.generate_preview()
    assert not controls.generate_button.isEnabled()  # busy while it runs
    drive.wait()

    (image, params, long_edge), kwargs = drive.calls[-1]
    assert image is drive.window.current_image_bgr
    assert params == difficulty.params_for_preset("Hard") and long_edge == pipeline.PREVIEW_LONG_EDGE
    assert kwargs["style"] == PageStyle.from_settings(0.5, "Dark")
    assert kwargs["handling"] == Handling(text=False)
    page = drive.window.current_page
    assert page is not None and page.page.size == (600, 400)  # never upscaled
    assert controls.generate_button.isEnabled() and controls.export_button.isEnabled()
    assert drive.window.preview._pixmap_item is not None
    assert drive.status() == f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions."
    assert drive.dialogs == []


def test_a_png_and_a_pdf_are_exported_at_300_dpi_on_a4(drive, tmp_path):
    # D-052 (Q36): a picture larger than its 300 dpi size is exported at that size, on A4.
    drive.window.current_image_bgr = _blocks((4000, 2000))
    controls = drive.window.controls
    drive.save_path = str(tmp_path / "page")  # the suffix is added for the format
    drive.window.export_page()
    drive.wait()

    (_image, _params, long_edge), kwargs = drive.calls[-1]
    assert long_edge == 3272 and kwargs["style"] == PageStyle() and kwargs["handling"] == Handling()
    with Image.open(tmp_path / "page.png") as png:
        assert png.width == 3272 and png.info["dpi"] == pytest.approx((300, 300), abs=0.2)
    assert drive.dialogs == [("information", "Export complete", f"Saved to:\n{tmp_path / 'page.png'}\n\n3272 × 1636 px, 300 dpi on A4")]
    assert drive.status().endswith("(3272 × 1636 px, 300 dpi on A4)")

    controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "page.pdf")
    drive.window.export_page()
    drive.wait()
    data = (tmp_path / "page.pdf").read_bytes()
    assert data.startswith(b"%PDF-1.4\n") and data.endswith(b"%%EOF\n")
    boxes = re.findall(rb"/MediaBox \[0 0 (\S+) (\S+)\]", data)
    assert boxes == [(b"841.8898", b"595.2756"), (b"595.2756", b"841.8898")]  # A4 landscape, then the legend's portrait
    assert b"/Width 3272 /Height 1636" in data
    assert print_scale((3272, 1636)).dpi == pytest.approx(300, abs=0.2)
    assert len(drive.dialogs) == 2 and drive.dialogs[-1][1] == "Export complete"


def test_a_running_generation_can_be_cancelled(drive, monkeypatch):
    def until_cancelled(*args, should_cancel, **kwargs):
        deadline = time.monotonic() + TIMEOUT_S
        while not should_cancel():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        raise PipelineCancelled()

    monkeypatch.setattr(pipeline, "generate", until_cancelled)
    drive.window.current_image_bgr = _blocks((300, 200))
    drive.window.generate_preview()
    assert drive.window._worker.isRunning()
    drive.window.abort_generation()
    assert drive.status() == "Cancelling…"
    drive.wait()

    assert drive.status() == "Cancelled."
    assert drive.window.current_page is None and drive.window.controls.generate_button.isEnabled()
    assert drive.dialogs == []


def test_a_failure_is_reported_and_the_window_is_ready_again(drive, monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError("no luck")

    monkeypatch.setattr(pipeline, "generate", failing)
    drive.window.current_image_bgr = _blocks((300, 200))
    drive.window.generate_preview()
    drive.wait()

    assert drive.dialogs == [("critical", "Generation failed", "no luck")]
    assert drive.status() == "Generation failed." and drive.window.controls.generate_button.isEnabled()


def test_nothing_is_generated_or_exported_without_a_picture(drive):
    drive.window.generate_preview()
    drive.window.export_page()

    assert drive.dialogs == [("warning", "No image", "Open an image first.")] * 2
    assert drive.calls == []

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
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.pipeline import Handling, PipelineCancelled  # noqa: E402
from tessellatum.core.print_size import print_scale  # noqa: E402
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

import pdf_reading  # noqa: E402

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
        self.save_names: list[str] = []  # the file names the save dialog was opened with
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
            def getSaveFileName(parent, caption, name, file_filter):
                drive.save_names.append(name)
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
    # The vector page (T7.3): drawn in the pixels of the page rendered at 300 dpi, mapped onto the sheet at 300 dpi.
    page_sheet = pdf_reading.sheets(data)[0]
    assert "0 0 3272 1636 re W n" in page_sheet["content"]
    _left, _top, mm_per_px, _ = pdf_reading.page_transform(page_sheet["content"])
    assert 25.4 / mm_per_px == pytest.approx(300, abs=0.2)
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


def test_on_a_short_window_the_panel_scrolls_and_its_scroll_bar_covers_none_of_it(drive):
    window = drive.window
    window.controls.preset_combo.setCurrentText("Custom")
    window.resize(900, 500)
    splitter = window.centralWidget()
    splitter.setSizes([1, 899])  # as narrow as it goes without collapsing
    window.show()
    _app.processEvents()
    scroll = splitter.widget(0)

    assert scroll.verticalScrollBar().maximum() > 0  # it scrolls
    assert scroll.viewport().width() >= window.controls.minimumSizeHint().width()
    window.hide()


def _pixmap_pixel(drive, x: int, y: int) -> tuple[int, int, int]:
    color = drive.window.preview._pixmap_item.pixmap().toImage().pixelColor(x, y)
    return color.red(), color.green(), color.blue()


def test_every_version_of_a_preview_is_drawn_with_it_and_the_bar_switches_between_them(drive):
    window = drive.window
    window.current_image_bgr = _blocks((600, 400))
    bar = window.version_bar
    assert not bar.isEnabled() and bar.version() is Version.PAGE  # nothing to show yet
    window.generate_preview()
    drive.wait()

    page = window.current_page
    assert bar.isEnabled()
    assert set(page._images) == set(Version)  # drawn by the worker, with the page
    middle = (150, 100)  # the middle of the top left block, in the page's pixels
    shown = {}
    for version in Version:
        bar.buttons[version].click()
        assert bar.version() is version
        shown[version] = _pixmap_pixel(drive, *middle)
        assert shown[version] == page.image(version).getpixel(middle)
    assert shown[Version.PAGE] == (255, 255, 255)  # bare paper, to paint
    assert shown[Version.COMPLETED] in page.palette_rgb  # painted
    assert shown[Version.PAGE] != shown[Version.TINTED] != shown[Version.COMPLETED]

    # Switching keeps the view where it is; a new preview keeps the version shown.
    window.preview.scale(2, 2)
    transform = window.preview.transform()
    bar.buttons[Version.TINTED].click()
    assert window.preview.transform() == transform
    window.generate_preview()
    drive.wait()
    assert bar.version() is Version.TINTED
    assert _pixmap_pixel(drive, *middle) == window.current_page.image(Version.TINTED).getpixel(middle)


def test_the_version_asked_for_is_exported(drive, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((600, 400))
    controls = window.controls
    controls.version_combo.setCurrentIndex(controls.version_combo.findData(Version.COMPLETED))
    drive.save_path = str(tmp_path / "completed")
    window.export_page()
    drive.wait()

    assert drive.save_names == ["coloring_page_completed.png"]
    (image, params, long_edge), kwargs = drive.calls[-1]
    expected = pipeline.generate(image, params, long_edge, style=kwargs["style"], handling=kwargs["handling"])
    with Image.open(tmp_path / "completed.png") as png:
        top = png.convert("RGB").crop((0, 0, 600, 400))
        assert top.tobytes() == expected.image(Version.COMPLETED).tobytes()
    assert drive.dialogs[-1][1] == "Export complete"

    controls.version_combo.setCurrentIndex(controls.version_combo.findData(Version.TINTED))
    controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "tinted.pdf")
    window.export_page()
    drive.wait()
    assert drive.save_names[-1] == "coloring_page_tinted.pdf"
    page_sheet = pdf_reading.sheets((tmp_path / "tinted.pdf").read_bytes())[0]
    assert page_sheet["content"].endswith("q /Mu gs /Pg Do Q\nQ")  # the page laid over its wash
    assert drive.dialogs[-1][1] == "Export complete" and len(drive.dialogs) == 2


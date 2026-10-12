"""The headless GUI drive: open, preview, cancel and export through the window, its dialogs answered by stubs."""

import dataclasses
import os
import re
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import difficulty, pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.pipeline import Handling, PipelineCancelled  # noqa: E402
from tessellatum.core.print_size import print_scale  # noqa: E402
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui import controls_panel, main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.worker import PipelineWorker  # noqa: E402

import pdf_reading  # noqa: E402
from difficulty_levels import COARSE  # noqa: E402

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
        """Until the window's workers are done -- making or exporting a page, drawing it at print size, and any stopped
        -- and their last signals have reached the window."""
        deadline = time.monotonic() + TIMEOUT_S
        window = self.window

        def running() -> bool:
            return any(
                worker is not None and worker.isRunning()
                for worker in (window._worker, window._print_worker, *window._retired)
            )

        while True:
            # A finished worker's signals are queued by the time it stops running, and may start another.
            idle = not running()
            _app.processEvents()
            if idle and not running():
                return
            assert time.monotonic() < deadline, "the pipeline didn't finish"
            time.sleep(0.005)

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
    for name, value in (("min_width_mm", 2.0), ("line_width_mm", 0.5), ("ink_gap_mm", 1.0)):
        control = controls.controls[name]
        control.slider.setValue(controls_panel.slider_position(control.setting, value))
    controls.tone_combo.setCurrentText("Dark")
    controls.text_check.setChecked(False)
    drive.window.generate_preview()
    assert not controls.generate_button.isEnabled()  # busy while it runs
    drive.wait()

    (image, params, long_edge), kwargs = drive.calls[-1]
    assert image is drive.window.current_image_bgr
    # Hard with a 2 mm brush: a Custom difficulty.
    assert controls.preset_combo.currentText() == "Custom"
    assert params == dataclasses.replace(difficulty.params_for_preset("Hard"), min_width_mm=2.0)
    # At its working size (D-012): here the picture's own, smaller than a preview.
    assert long_edge == pipeline.working_long_edge(image, params, kwargs["handling"]) == 600
    assert kwargs["style"] == PageStyle.from_settings(0.5, "Dark")
    assert kwargs["handling"] == Handling(text=False, ink_gap_mm=1.0)
    page = drive.window.current_page
    assert page is not None and page.page.size == (600, 400)  # never upscaled
    assert controls.generate_button.isEnabled() and controls.export_button.isEnabled()
    assert drive.window.preview.shown_image() is not None
    assert drive.status() == (
        f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions, made at 600 × 400 px."
    )
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


def test_an_export_is_its_preview_s_page_drawn_at_print_size(drive, tmp_path, monkeypatch):
    # D-006: the export draws the page the preview shows, unless the settings changed since.
    drawn = []
    real = pipeline.draw_page
    monkeypatch.setattr(pipeline, "draw_page", lambda page, *args, **kwargs: drawn.append(page) or real(page, *args, **kwargs))
    drive.window.current_image_bgr = _blocks((4000, 2000))
    drive.window.generate_preview()
    drive.wait()
    preview = drive.window.current_page
    calls = len(drive.calls)

    drive.save_path = str(tmp_path / "page")
    drive.window.export_page()
    drive.wait()
    assert len(drive.calls) == calls and drawn == [preview]  # drawn from the preview, not made again
    with Image.open(tmp_path / "page.png") as png:
        assert png.width == 3272

    drive.window.controls.preset_combo.setCurrentText("Hard")  # a setting the preview wasn't made with
    drive.window.export_page()
    drive.wait()
    (_image, params, long_edge), _kwargs = drive.calls[-1]
    assert len(drive.calls) == calls + 1 and long_edge == 3272 and params == difficulty.params_for_preset("Hard")


def test_a_preview_made_at_print_size_is_the_export(drive, drawn, tmp_path):
    # D-012 (T8): Realistic's brush asks for 2216 px, more than this picture's 2000, so its preview is made at the
    # picture's own size, which is the export's: the export saves it, neither made again nor drawn.
    window = drive.window
    window.current_image_bgr = _blocks((2000, 1000))
    window.controls.preset_combo.setCurrentText("Realistic")
    window.generate_preview()
    drive.wait()
    preview = window.current_page
    assert preview.page.size == (2000, 1000) and drive.calls[-1][0][2] == 2000
    calls = len(drive.calls)

    drive.save_path = str(tmp_path / "page")
    window.export_page()
    drive.wait()
    assert len(drive.calls) == calls and drawn == []
    with Image.open(tmp_path / "page.png") as png:
        assert np.asarray(png.convert("RGB"))[:1000].tobytes() == preview.image(Version.PAGE).tobytes()
    assert drive.dialogs[-1][1] == "Export complete" and not window.controls.progress_row.isVisible()


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
    color = drive.window.preview.shown_image().pixelColor(x, y)
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
    assert len(set(shown.values())) == len(Version)  # each version its own

    # Switching keeps the view where it is; a new preview of the picture keeps the version shown, and the view.
    window.preview.zoom_by(2)
    transform = window.preview.transform()
    bar.buttons[Version.TINTED].click()
    assert window.preview.transform() == transform
    window.generate_preview()
    drive.wait()
    assert bar.version() is Version.TINTED and window.preview.transform() == transform
    assert _pixmap_pixel(drive, *middle) == window.current_page.image(Version.TINTED).getpixel(middle)


def test_opening_another_picture_leaves_the_last_one_s_versions_out_of_reach(drive, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((600, 400))
    window.generate_preview()
    drive.wait()
    assert window.version_bar.isEnabled()

    Image.fromarray(_blocks((300, 200))[:, :, ::-1]).save(tmp_path / "other.png")
    drive.open_path = str(tmp_path / "other.png")
    window.open_image()

    assert not window.version_bar.isEnabled() and not window.controls.export_button.isEnabled()


def test_cancelling_while_the_versions_are_drawn_cancels_the_preview(monkeypatch):
    drawn = []

    class Page:
        def image(self, version):
            drawn.append(version)

    def finishing_as_cancelled(*args, should_cancel, **kwargs):
        """A pipeline that finishes just as Abort is clicked: done, it no longer looks."""
        deadline = time.monotonic() + TIMEOUT_S
        while not should_cancel():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        return Page()

    monkeypatch.setattr(pipeline, "generate", finishing_as_cancelled)
    worker = PipelineWorker(_blocks((60, 40)), COARSE, 60, versions=tuple(Version))
    outcomes = []
    worker.succeeded.connect(lambda page: outcomes.append("succeeded"))
    worker.cancelled.connect(lambda: outcomes.append("cancelled"))
    worker.start()
    worker.requestInterruption()
    assert worker.wait(TIMEOUT_S * 1000)
    _app.processEvents()

    assert outcomes == ["cancelled"] and drawn == []


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


def _array(image) -> np.ndarray:
    """A QImage's or a PIL image's RGB pixels."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    image = image.convertToFormat(QImage.Format_RGB888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, : image.width() * 3].reshape(image.height(), image.width(), 3).copy()


@pytest.fixture
def drawn(monkeypatch):
    """The pages ``pipeline.draw_page`` is asked to draw at print size."""
    pages = []
    real = pipeline.draw_page
    monkeypatch.setattr(
        pipeline, "draw_page", lambda page, *args, **kwargs: pages.append(page) or real(page, *args, **kwargs)
    )
    return pages


def test_zoomed_in_the_preview_shows_its_page_as_it_prints_and_the_export_saves_that_page(drive, drawn, tmp_path):
    # T5: past the preview's own pixels the zoom shows the page drawn at print size, once, and a tile of it at 100% is
    # the export's pixels there.
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    # Made at Realistic's working size, where its 0.5 mm brush is 4 px across (D-012): a page whose sides both scale to
    # the print's by the same factor, so that at 100% a pixel of the screen is one of the print on both axes.
    window.controls.preset_combo.setCurrentText("Realistic")
    window.generate_preview()
    drive.wait()
    preview = window.preview
    assert window.current_page.page.size == (2216, 1108) and window._print_worker is None  # nothing drawn yet

    preview.zoom_to(1.5)
    assert window._print_worker.isRunning() and drive.status() == "Drawing the page as it prints, for the zoom…"
    drive.wait()
    printed = window._print_page
    assert drawn == [window.current_page] and printed.page.size == (3272, 1636)
    assert set(printed._images) == set(Version)  # every version drawn with it
    assert drive.status() == window._ready_message
    assert (_array(preview.shown_image()) == _array(printed.image(Version.PAGE))).all()

    preview.zoom_to_printed_pixels()
    window.version_bar.buttons[Version.COMPLETED].click()
    preview._sharpen()  # as the view does a moment after the zoom settles
    tile = _array(preview.viewport().grab().toImage())
    inverse, _ = preview.viewportTransform().inverted()
    corner = inverse.map(QPointF(0, 0)) * (3272 / 2216)
    left, top = round(corner.x()), round(corner.y())

    window.controls.version_combo.setCurrentIndex(window.controls.version_combo.findData(Version.COMPLETED))
    drive.save_path = str(tmp_path / "page")
    window.export_page()
    drive.wait()
    assert drawn == [window.current_page]  # the export is the page drawn for the zoom, not drawn again
    assert drive.dialogs[-1][1] == "Export complete" and not window.controls.progress_row.isVisible()
    with Image.open(tmp_path / "page.png") as png:
        exported = np.asarray(png.convert("RGB"))
    assert (exported[: 1636] == _array(printed.image(Version.COMPLETED))).all()
    assert (tile == exported[top : top + tile.shape[0], left : left + tile.shape[1]]).all()


def test_an_export_asked_for_while_the_zoom_draws_the_page_waits_for_it(drive, drawn, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    window.preview.zoom_to(1.5)
    assert window._print_worker.isRunning()

    drive.save_path = str(tmp_path / "page.pdf")
    window.controls.pdf_radio.setChecked(True)
    window.export_page()
    assert drive.status() == "Rendering the PDF at print resolution…"
    assert not window.controls.generate_button.isEnabled()
    window.generate_preview()  # not while an export waits
    drive.wait()

    assert drawn == [window.current_page] and len(drive.calls) == 1
    saved = f"Saved to:\n{tmp_path / 'page.pdf'}\n\n3272 × 1636 px, 300 dpi on A4"
    assert drive.dialogs == [("information", "Export complete", saved)]
    assert window.controls.generate_button.isEnabled()


def test_an_export_waiting_for_the_page_drawn_at_print_size_can_be_cancelled(drive, monkeypatch, tmp_path):
    def until_cancelled(page, *args, should_cancel, **kwargs):
        deadline = time.monotonic() + TIMEOUT_S
        while not should_cancel():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        raise PipelineCancelled()

    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    monkeypatch.setattr(pipeline, "draw_page", until_cancelled)
    drive.save_path = str(tmp_path / "page")
    window.export_page()
    assert window._print_worker.isRunning()
    window.abort_generation()
    drive.wait()

    assert drive.status() == "Cancelled." and window.controls.generate_button.isEnabled()
    assert window._pending_export_path is None and not (tmp_path / "page.png").exists() and drive.dialogs == []


def test_a_new_preview_drops_the_page_drawn_for_the_last_one(drive, drawn):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    first = window.current_page
    window.preview.zoom_to(1.5)
    on_paper = window.preview.printed_zoom()
    stopped = window._print_worker
    assert stopped.isRunning()

    window.controls.preset_combo.setCurrentText("Hard")
    window.generate_preview()  # while the zoom's page is still being drawn
    assert stopped.isInterruptionRequested() and window._print_worker is None
    drive.wait()

    # The same picture: the new page is shown as far in on paper, though Hard's is made at more pixels than Medium's,
    # so the zoom asks for its page as it prints, and gets it.
    second = window.current_page
    assert second.page.size == (1847, 924) and first.page.size == (1583, 792)
    assert second is not first and window.preview.printed_zoom() == pytest.approx(on_paper)
    assert window.preview.zoom() == pytest.approx(1.5 * 1583 / 1847)
    assert drawn[-1] is second and window._print_page.model is second.model
    assert (_array(window.preview.shown_image()) == _array(window._print_page.image(Version.PAGE))).all()


def test_a_page_no_larger_than_its_preview_is_shown_as_it_is_however_far_in(drive, drawn):
    window = drive.window
    window.current_image_bgr = _blocks((600, 400))
    window.generate_preview()
    drive.wait()
    window.preview.zoom_to(3.0)
    drive.wait()

    assert drawn == [] and window._print_worker is None
    assert window.preview.printed_zoom() == pytest.approx(3.0)
    assert (_array(window.preview.shown_image()) == _array(window.current_page.image(Version.PAGE))).all()


def test_a_zoom_while_a_preview_is_made_draws_the_new_page_not_the_last(drive, drawn):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    first = window.current_page
    window.controls.preset_combo.setCurrentText("Hard")
    window.generate_preview()
    window.preview.zoom_to(1.5)  # while the new page is made
    assert window._print_worker is None
    drive.wait()

    second = window.current_page
    assert second is not first and drawn == [second] and window._print_page.model is second.model


def _until(condition) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline
        _app.processEvents()
        time.sleep(0.005)


def _held_back(real, release: threading.Event, started=None, done=None, progress=None):
    """``real``, once ``release`` is set, or cancelled before: as it waits it reports ``progress`` and sets
    ``started``; what it gives is put in ``done``."""

    def held_back(*args, should_cancel, progress_callback=None, **kwargs):
        if progress is not None:
            progress_callback(progress)
        if started is not None:
            started.set()
        while not release.wait(0.005):
            if should_cancel():
                raise PipelineCancelled()
        page = real(*args, should_cancel=should_cancel, progress_callback=progress_callback, **kwargs)
        if done is not None:
            done.append(page)
        return page

    return held_back


def test_a_page_drawn_for_the_zoom_while_an_export_with_other_settings_is_made_is_not_what_it_saves(
    drive, monkeypatch, tmp_path
):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    window.preview.zoom_to(1.5)  # the page as it prints, drawn for the zoom
    release, made = threading.Event(), []
    monkeypatch.setattr(pipeline, "generate", _held_back(pipeline.generate, release, done=made))
    window.controls.tone_combo.setCurrentText("Dark")  # lines of another tone
    drive.save_path = str(tmp_path / "page")
    window.export_page()  # another page, made while the zoom's is drawn
    _until(lambda: window._print_page is not None)

    assert drive.dialogs == [] and window._pending_export_path is not None  # the export waits for its own page
    assert drive.status() == "Rendering the PNG at print resolution…"
    release.set()
    drive.wait()

    assert drive.dialogs[-1][1] == "Export complete"
    with Image.open(tmp_path / "page.png") as png:
        exported = np.asarray(png.convert("RGB"))
    assert (exported[: 1636] == _array(made[0].image(Version.PAGE))).all()
    assert not (exported[: 1636] == _array(window._print_page.image(Version.PAGE))).all()


def test_abort_cancels_an_export_with_other_settings_not_the_zoom_s_drawing(drive, monkeypatch, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    drawing, drawn_release = threading.Event(), threading.Event()
    held_back = _held_back(pipeline.draw_page, drawn_release, started=drawing, progress=77)
    monkeypatch.setattr(pipeline, "draw_page", held_back)
    window.preview.zoom_to(1.5)
    monkeypatch.setattr(pipeline, "generate", _held_back(pipeline.generate, threading.Event()))
    window.controls.preset_combo.setCurrentText("Hard")
    drive.save_path = str(tmp_path / "page")
    window.export_page()
    assert drawing.wait(TIMEOUT_S)
    _app.processEvents()
    assert window.controls.progress_bar.value() != 77  # the zoom's drawing isn't the export's progress

    window.abort_generation()
    _until(lambda: drive.status() == "Cancelled.")
    assert window._print_worker.isRunning()  # the zoom still wants its page
    assert window._worker.wait(TIMEOUT_S * 1000)
    drawn_release.set()
    drive.wait()

    assert drive.dialogs == [] and not (tmp_path / "page.png").exists()
    assert window._print_page is not None and window.controls.generate_button.isEnabled()


def test_a_zoom_while_an_export_is_made_with_other_settings_draws_the_preview_s_page_after_it(drive, drawn, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    shown = window.current_page
    window.controls.preset_combo.setCurrentText("Hard")
    drive.save_path = str(tmp_path / "page")
    window.export_page()  # another page: made at the preview's size and drawn at print size
    window.preview.zoom_to(1.5)
    assert window._print_worker is None
    drive.wait()

    assert drive.dialogs[-1][1] == "Export complete" and len(drawn) == 2
    assert drawn[-1] is shown and window._print_page.model is shown.model  # the page the preview shows

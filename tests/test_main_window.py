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

from tessellatum.core import difficulty, export, pipeline, print_size  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.pipeline import Handling, PipelineCancelled  # noqa: E402
from tessellatum.core.print_size import print_scale  # noqa: E402
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui import controls_panel, main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.app import jobs  # noqa: E402
from tessellatum.app.jobs import EXPORT, PREVIEW, PRINT_LAYOUT, JobManager, page_task  # noqa: E402

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
    """A window whose file dialogs are stubs answering with the file names given, and what it told: its notifications
    (the level and the text) and any message box."""

    def __init__(self, monkeypatch):
        self.dialogs: list[tuple[str, ...]] = []
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
        # The tests' PNGs at 300 dpi, a quarter of the default's pixels: the default and others are tested on their own.
        self.window.controls.set_export_dpi(300)
        self.window.notification.notified.connect(lambda level, text: self.dialogs.append((level, text)))

    def wait(self) -> None:
        """Until the window's jobs are done -- making or exporting a page, drawing it at print size, writing the file,
        and any stopped -- and their outcomes have reached the window."""
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            # An outcome delivered may start another job.
            _app.processEvents()
            if self.window.jobs.is_idle():
                _app.processEvents()
                if self.window.jobs.is_idle():
                    return
            assert time.monotonic() < deadline, "the pipeline didn't finish"
            time.sleep(0.005)

    def status(self) -> str:
        return self.window.status.text()

    def exported(self) -> bool:
        """Whether the last thing the window told was an export done."""
        return bool(self.dialogs) and self.dialogs[-1][0] == "success" and self.dialogs[-1][1].startswith("Exported ")


@pytest.fixture
def drive(monkeypatch):
    drive = _Drive(monkeypatch)
    yield drive
    drive.window.close()
    drive.window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_a_preview_is_made_at_its_paper_s_working_size(drive, tmp_path):
    # T3a: a larger sheet makes the page at more pixels, as many as its brush asks for there.
    Image.fromarray(_blocks((2400, 1600))[:, :, ::-1]).save(tmp_path / "blocks.png")
    drive.open_path = str(tmp_path / "blocks.png")
    drive.window.open_image()
    controls = drive.window.controls
    controls.preset_combo.setCurrentText("Beginner")
    controls.paper_combo.setCurrentText("A2")
    drive.window.generate_preview()
    drive.wait()
    (image, params, long_edge), kwargs = drive.calls[-1]
    assert kwargs["style"].paper == "A2"
    assert long_edge == pipeline.working_long_edge(image, params, kwargs["handling"], print_size.A2)
    assert long_edge > pipeline.working_long_edge(image, params, kwargs["handling"])


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
    # At its working size (D-012), the picture scaled up to it (D-009).
    assert long_edge == pipeline.working_long_edge(image, params, kwargs["handling"]) == 1100
    assert kwargs["style"] == PageStyle.from_settings(0.5, "Dark")
    assert kwargs["handling"] == Handling(text=False, ink_gap_mm=1.0)
    page = drive.window.current_page
    assert page is not None and page.page.size == (1100, 733)
    assert controls.generate_button.isEnabled() and controls.export_button.isEnabled()
    assert drive.window.preview.shown_image() is not None
    assert drive.status() == "Preview ready."
    status = drive.window.status
    assert status.page_label.text() == f"{page.num_colors_used} colors · {page.num_regions:,} regions"
    assert status.size_label.text().startswith("1,100 × 733 px · ")
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
    assert drive.dialogs == [("success", "Exported page.png: 3,272 × 1,636 px, 300 dpi on A4.")]
    assert drive.status().endswith("(3,272 × 1,636 px, 300 dpi on A4)")

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
    assert len(drive.dialogs) == 2 and drive.exported()


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


def test_every_export_is_drawn_from_the_preview_s_print_layout(drive, drawn, tmp_path):
    # T3b: Realistic's brush asks for 2216 px, more than this picture's 2000, so its preview is made of the picture
    # scaled up (D-009). Each export is that page laid out once at 300 dpi on A4, drawn from it, not made again: the PNG
    # as the PDF's vectors drawn in pixels, the PDF from the same layout, not laid out again.
    window = drive.window
    window.current_image_bgr = _blocks((2000, 1000))
    window.controls.preset_combo.setCurrentText("Realistic")
    window.generate_preview()
    drive.wait()
    preview = window.current_page
    assert preview.page.size == (2216, 1108) and drive.calls[-1][0][2] == 2216
    calls = len(drive.calls)

    drive.save_path = str(tmp_path / "page")
    window.export_page()
    drive.wait()
    layout = window._print_page
    assert len(drive.calls) == calls and drawn == [preview] and layout.page.size == (3272, 1636)
    expected = export.rasterize(layout.drawing, 300)
    with Image.open(tmp_path / "page.png") as png:
        assert png.width == 3272 and png.info["dpi"] == pytest.approx((300, 300), abs=0.2)
        assert np.asarray(png.convert("RGB"))[: expected.height].tobytes() == expected.tobytes()
    assert drive.exported() and not window.controls.progress_row.isVisible()

    window.controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "page.pdf")
    window.export_page()
    drive.wait()
    assert len(drive.calls) == calls and drawn == [preview]
    assert "0 0 3272 1636 re W n" in pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[0]["content"]

    # Fewer pixels across the brush: the preview is another setting's, so the export is made again, laid out to print.
    control = window.controls.controls["brush_px"]
    control.slider.setValue(controls_panel.slider_position(control.setting, 2.0))
    window.export_page()
    drive.wait()
    (_image, _params, long_edge), kwargs = drive.calls[-1]
    assert len(drive.calls) == calls + 1 and long_edge == 3272 and kwargs["handling"].brush_px == 2.0
    assert kwargs["vector"] and drive.exported()


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
    assert drive.window.jobs.is_running(PREVIEW)
    drive.window.abort_generation()
    assert drive.status() == "Cancelling: it stops after the current step…"
    drive.wait()

    assert drive.status() == "Cancelled."
    assert drive.window.current_page is None and drive.window.controls.generate_button.isEnabled()
    assert drive.dialogs == []


def test_the_x_beside_the_progress_bar_cancels_the_generation(drive, monkeypatch):
    def until_cancelled(*args, should_cancel, **kwargs):
        deadline = time.monotonic() + TIMEOUT_S
        while not should_cancel():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        raise PipelineCancelled()

    monkeypatch.setattr(pipeline, "generate", until_cancelled)
    drive.window.current_image_bgr = _blocks((300, 200))
    drive.window.generate_preview()
    assert drive.window.jobs.is_running(PREVIEW) and drive.window.controls.abort_button.isEnabled()
    drive.window.controls.abort_button.click()
    assert drive.status().startswith("Cancelling") and not drive.window.controls.abort_button.isEnabled()
    drive.wait()
    assert drive.status() == "Cancelled." and drive.window.current_page is None


def test_a_failure_is_reported_and_the_window_is_ready_again(drive, monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError("no luck")

    monkeypatch.setattr(pipeline, "generate", failing)
    drive.window.current_image_bgr = _blocks((300, 200))
    drive.window.generate_preview()
    drive.wait()

    assert drive.dialogs == [("error", "The page couldn't be made: no luck")]
    assert drive.status() == "Generation failed." and drive.window.controls.generate_button.isEnabled()


def test_nothing_is_generated_or_exported_without_a_picture(drive):
    drive.window.generate_preview()
    drive.window.export_page()

    assert drive.dialogs == [("info", "Open a picture first.")] * 2
    assert drive.calls == []


def test_on_a_short_window_the_panel_scrolls_and_its_scroll_bar_covers_none_of_it(drive):
    window = drive.window
    window.controls.preset_combo.setCurrentText("Custom")
    window.controls.sections[controls_panel.advanced_key(window.controls.category())].set_expanded(True)  # its tallest
    window.resize(900, 500)
    window.splitter.setSizes([1, 899, 0])  # as narrow as it goes
    window.show()
    _app.processEvents()
    scroll = window.controls.scrolls[window.controls.category()]

    assert scroll.verticalScrollBar().maximum() > 0  # it scrolls
    assert scroll.viewport().width() >= window.controls.pages[window.controls.category()].minimumSizeHint().width()
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
    manager = JobManager()
    outcomes = []
    manager.succeeded.connect(lambda job, page: outcomes.append("succeeded"))
    manager.cancelled.connect(lambda job: outcomes.append("cancelled"))
    task = page_task(_blocks((60, 40)), COARSE, 60, versions=tuple(Version))
    seen = []
    # The task itself stops before drawing a version once asked to, whatever the manager makes of its outcome.
    job = manager.start(PREVIEW, lambda progress, should_cancel: seen.append(_run_to_cancel(task, progress, should_cancel)))
    manager.cancel(PREVIEW)
    _until(manager.is_idle)
    manager.deleteLater()

    assert outcomes == ["cancelled"] and drawn == [] and seen == ["cancelled"] and job.state == jobs.CANCELLED


def _run_to_cancel(task, progress, should_cancel) -> str:
    try:
        task(progress, should_cancel)
    except PipelineCancelled:
        return "cancelled"
    return "finished"


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
    assert long_edge == 3272 and kwargs["vector"]
    page = pipeline.generate(image, params, long_edge, style=kwargs["style"], handling=kwargs["handling"], vector=True)
    expected = export.rasterize(page.drawing, 300, Version.COMPLETED, page.painting)
    with Image.open(tmp_path / "completed.png") as png:
        top = png.convert("RGB").crop((0, 0, *expected.size))
        assert top.tobytes() == expected.tobytes()
    assert drive.exported()

    controls.version_combo.setCurrentIndex(controls.version_combo.findData(Version.TINTED))
    controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "tinted.pdf")
    window.export_page()
    drive.wait()
    assert drive.save_names[-1] == "coloring_page_tinted.pdf"
    page_sheet = pdf_reading.sheets((tmp_path / "tinted.pdf").read_bytes())[0]
    assert page_sheet["content"].endswith("q /Mu gs /Pg Do Q\nQ")  # the page laid over its wash
    assert drive.exported() and len(drive.dialogs) == 2


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
    assert window.current_page.page.size == (2216, 1108) and window.jobs.current(PRINT_LAYOUT) is None  # nothing drawn yet

    preview.zoom_to(1.5)
    assert window.jobs.is_running(PRINT_LAYOUT) and drive.status() == "Drawing the page as it prints, for the zoom…"
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
    assert drive.exported() and not window.controls.progress_row.isVisible()
    with Image.open(tmp_path / "page.png") as png:
        exported = np.asarray(png.convert("RGB"))
    # The PNG is the layout's vectors drawn in pixels (T3b); the zoom, the layout's own drawing of them: the same page,
    # to the anti-aliasing.
    drawn_png = export.rasterize(printed.drawing, 300, Version.COMPLETED, printed.painting)
    assert (exported[: 1636] == np.asarray(drawn_png)).all()
    there = exported[top : top + tile.shape[0], left : left + tile.shape[1]].astype(np.int16)
    assert np.abs(tile.astype(np.int16) - there).mean() < 4


def test_an_export_asked_for_while_the_zoom_draws_the_page_waits_for_it(drive, drawn, monkeypatch, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    release = threading.Event()  # the zoom's page is held until the export is asked for, so that it is still drawn then
    monkeypatch.setattr(pipeline, "draw_page", _held_until(pipeline.draw_page, release))
    window.preview.zoom_to(1.5)
    assert window.jobs.is_running(PRINT_LAYOUT)

    # A PNG: a PDF places its numbers for itself, on a page drawn for it (T2a).
    drive.save_path = str(tmp_path / "page.png")
    window.export_page()
    assert drive.status() == "Rendering the PNG at print resolution…"
    assert not window.controls.generate_button.isEnabled()
    window.generate_preview()  # not while an export waits
    release.set()
    drive.wait()

    assert drawn == [window.current_page] and len(drive.calls) == 1
    assert drive.dialogs == [("success", "Exported page.png: 3,272 × 1,636 px, 300 dpi on A4.")]
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
    assert window.jobs.is_running(PRINT_LAYOUT)
    window.abort_generation()
    drive.wait()

    assert drive.status() == "Cancelled." and window.controls.generate_button.isEnabled()
    assert window._pending_export_path is None and not (tmp_path / "page.png").exists() and drive.dialogs == []


def test_a_new_preview_drops_the_page_drawn_for_the_last_one(drive, drawn, monkeypatch):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    first = window.current_page
    # The zoom's page is held where its drawing starts until the new preview has asked it to stop: a worker that has
    # stopped reads isInterruptionRequested() as False, and one free to stop could do so before it is checked.
    release = threading.Event()
    monkeypatch.setattr(pipeline, "draw_page", _held_until(pipeline.draw_page, release))
    window.preview.zoom_to(1.5)
    on_paper = window.preview.printed_zoom()
    stopped = window.jobs.current(PRINT_LAYOUT)
    assert stopped.state == jobs.RUNNING

    window.controls.preset_combo.setCurrentText("Hard")
    window.generate_preview()  # while the zoom's page is still being drawn
    assert window.jobs.current(PRINT_LAYOUT) is None and stopped.state == jobs.RUNNING  # dropped, still stopping
    release.set()
    drive.wait()

    # The same picture: the new page is shown as far in on paper, though Hard's is made at more pixels than Medium's,
    # so the zoom asks for its page as it prints, and gets it.
    second = window.current_page
    assert second.page.size == (1847, 924) and first.page.size == (1583, 792)
    assert second is not first and window.preview.printed_zoom() == pytest.approx(on_paper)
    assert window.preview.zoom() == pytest.approx(1.5 * 1583 / 1847)
    assert drawn[-1] is second and window._print_page.model is second.model
    assert (_array(window.preview.shown_image()) == _array(window._print_page.image(Version.PAGE))).all()


def test_a_small_picture_s_page_is_shown_as_it_prints_far_in(drive, drawn):
    # T3b: every page is laid out at 300 dpi to print, however few pixels its picture has, and the zoom shows that.
    window = drive.window
    window.current_image_bgr = _blocks((600, 400))
    window.generate_preview()
    drive.wait()
    page = window.current_page
    window.preview.zoom_to(3.0)
    drive.wait()

    assert drawn == [page] and window._print_page.page.size == (3272, 2181)
    assert (_array(window.preview.shown_image()) == _array(window._print_page.image(Version.PAGE))).all()


def test_a_zoom_while_a_preview_is_made_draws_the_new_page_not_the_last(drive, drawn):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    first = window.current_page
    window.controls.preset_combo.setCurrentText("Hard")
    window.generate_preview()
    window.preview.zoom_to(1.5)  # while the new page is made
    assert window.jobs.current(PRINT_LAYOUT) is None
    drive.wait()

    second = window.current_page
    assert second is not first and drawn == [second] and window._print_page.model is second.model


def _until(condition) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline
        _app.processEvents()
        time.sleep(0.005)


def _held_until(real, release: threading.Event):
    """``real``, once ``release`` is set. Unlike ``_held_back`` it doesn't stop for a cancel while it waits, so its
    worker is surely still running until then: ``isInterruptionRequested()`` reads True only while a worker runs."""

    def held(*args, **kwargs):
        assert release.wait(TIMEOUT_S)
        return real(*args, **kwargs)

    return held


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

    assert drive.exported()
    with Image.open(tmp_path / "page.png") as png:
        exported = np.asarray(png.convert("RGB"))
    assert (exported[: 1636] == np.asarray(export.rasterize(made[0].drawing, 300))).all()
    assert not (exported[: 1636] == np.asarray(export.rasterize(window._print_page.drawing, 300))).all()


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
    assert window.jobs.is_running(PRINT_LAYOUT)  # the zoom still wants its page
    assert not window.jobs.is_running(EXPORT)
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
    assert window.jobs.current(PRINT_LAYOUT) is None
    drive.wait()

    assert drive.exported() and len(drawn) == 2
    assert drawn[-1] is shown and window._print_page.model is shown.model  # the page the preview shows


def test_a_png_is_drawn_at_the_resolution_asked_for(drive, tmp_path):
    # T3c: in dots per inch, or in pixels along its long edge; the legend drawn at it too, the file recording it.
    window = drive.window
    window.current_image_bgr = _blocks((2000, 1000))
    window.generate_preview()
    drive.wait()
    controls = window.controls
    controls.set_export_dpi(150)
    drive.save_path = str(tmp_path / "at150.png")
    window.export_page()
    drive.wait()
    layout = window._print_page
    expected = export.rasterize(layout.drawing, 150)
    with Image.open(tmp_path / "at150.png") as png:
        assert png.width == expected.width == 1636 and png.info["dpi"] == pytest.approx((150, 150), abs=0.2)
        assert np.asarray(png.convert("RGB"))[: expected.height].tobytes() == expected.tobytes()
        assert png.height > expected.height  # its legend below
    assert drive.exported() and "1,636 × 818 px, 150 dpi on A4" in drive.dialogs[-1][1]

    controls.resolution_unit.setCurrentIndex(controls.resolution_unit.findData(controls_panel.PIXELS))
    controls.resolution_spin.setValue(5000)
    drive.save_path = str(tmp_path / "wide.png")
    window.export_page()
    drive.wait()
    with Image.open(tmp_path / "wide.png") as png:
        assert png.width == 5000
    assert window._print_page is layout  # every resolution drawn from the one layout


def test_the_legend_s_detail_changes_the_preview_s_legend_and_the_export_without_making_the_page_again(drive, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    calls = len(drive.calls)
    detailed = window.preview._legend
    # Drawn finer than the page, at 300 dpi, and shown at the page's scale.
    page_px_per_mm = print_scale(window.current_page.page.size).px_per_mm
    assert detailed.width() == pytest.approx(window.current_page.page.width * 300 / 25.4 / page_px_per_mm, abs=1)
    assert window.preview._legend_size[0] == pytest.approx(window.current_page.page.width, abs=0.5)

    window.controls.legend_combo.setCurrentText("Number only")
    numbered = window.preview._legend
    assert len(drive.calls) == calls and numbered != detailed  # the names and codes gone, the page not made again
    assert window.preview._legend_size[0] == pytest.approx(window.current_page.page.width, abs=0.5)

    window.controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "page.pdf")
    window.export_page()
    drive.wait()
    assert len(drive.calls) == calls  # drawn from the preview, not made again
    legend_sheet = pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[1]["content"]
    assert "RGB" not in legend_sheet
    window.controls.legend_combo.setCurrentText("Name, hex and RGB")
    window.export_page()
    drive.wait()
    assert len(drive.calls) == calls
    assert "RGB" in pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[1]["content"]


@pytest.mark.parametrize("suffix", [".swatches", ".ase", ".gpl"])
def test_the_palette_is_exported_from_the_preview_s_legend_without_drawing_the_page(drive, tmp_path, suffix):
    from tessellatum.core import swatches

    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    calls = len(drive.calls)
    controls = window.controls
    controls.palette_radio.setChecked(True)
    controls.palette_combo.setCurrentIndex(controls.palette_combo.findData(suffix))
    drive.save_path = str(tmp_path / "colors")  # the suffix is added for the format
    window.export_page()
    drive.wait()
    path = tmp_path / f"colors{suffix}"
    assert len(drive.calls) == calls and window._print_page is None  # nothing made or drawn
    expected = tmp_path / f"expected{suffix}"
    swatches.save_palette(expected, window.current_page.palette_rgb, title="colors")
    assert path.read_bytes() == expected.read_bytes()
    assert drive.dialogs[-1] == ("success", f"Exported {path.name}.")


def test_a_page_made_from_the_paints_listed_is_painted_in_them_and_its_legend_calls_them_by_name(drive):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    paints = (
        ("Ultramarine", (40, 40, 200)), ("Sap green", (40, 170, 40)), ("Cadmium red", (200, 40, 40)),
        ("Lemon", (60, 210, 210)), ("Titanium white", (255, 255, 255)), ("Ivory black", (20, 20, 22)),
    )
    window.controls.set_paints(paints)
    window.generate_preview()
    drive.wait()
    _args, kwargs = drive.calls[-1]
    assert kwargs["handling"].paints == paints
    page = window.current_page
    owned = {rgb: name for name, rgb in paints}
    assert sorted(page.color_names) == ["Cadmium red", "Lemon", "Sap green", "Ultramarine"]
    assert page.color_names == [owned[rgb] for rgb in page.palette_rgb]


def test_the_legend_s_recipes_mix_each_color_from_the_paints_listed_and_follow_them(drive, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    window.generate_preview()
    drive.wait()
    calls = len(drive.calls)
    controls = window.controls
    controls.legend_combo.setCurrentText("Name and recipe")
    coded = window.preview._legend  # no paints listed: the codes
    controls.set_paints((("Titanium white", (255, 255, 255)), ("Ultramarine", (25, 0, 89)), ("Lemon", (250, 240, 60))))
    assert controls.colors_from_combo.currentText() == "The picture"  # listed for the recipes, the colors the picture's
    assert window.jobs.is_running(jobs.RECIPES)  # mixed off the GUI thread, then the legend drawn again with them
    drive.wait()
    with_recipes = window.preview._legend
    assert with_recipes != coded and len(drive.calls) == calls  # the legend drawn again, the page not made again

    controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "page.pdf")
    window.export_page()
    drive.wait()
    assert len(drive.calls) == calls
    legend_sheet = pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[1]["content"]
    assert "Titanium white" in legend_sheet and "RGB" not in legend_sheet
    controls.set_paints(())
    assert window.preview._legend == coded


def test_mixed_colors_come_after_the_paints_on_the_preview_s_legend_the_pdf_and_the_palette_file(drive, tmp_path):
    window = drive.window
    window.current_image_bgr = _blocks((4000, 2000))
    controls = window.controls
    # Blue, white and yellow: the green and the cyan block are mixes.
    controls.set_paints((("Blue", (40, 40, 200)), ("Titanium white", (250, 250, 250)), ("Yellow", (250, 220, 20))))
    controls.mix_check.setChecked(True)
    window.generate_preview()
    drive.wait()
    _args, kwargs = drive.calls[-1]
    assert kwargs["handling"].mix_paints
    page = window.current_page
    start = next(i for i, lines in enumerate(page.mixes) if lines is not None)
    assert start >= 1 and all(lines is not None for lines in page.mixes[start:])
    assert all(name in ("Blue", "Titanium white", "Yellow") for name in page.color_names[:start])

    controls.palette_radio.setChecked(True)
    controls.palette_combo.setCurrentIndex(controls.palette_combo.findData(".gpl"))
    drive.save_path = str(tmp_path / "colors.gpl")
    window.export_page()
    drive.wait()
    text = (tmp_path / "colors.gpl").read_text(encoding="utf-8")
    for index in range(start, len(page.palette_rgb)):
        recipe = " + ".join(page.mixes[index])
        assert f"{index + 1} {page.color_names[index]} ({recipe})" in text

    controls.pdf_radio.setChecked(True)
    drive.save_path = str(tmp_path / "page.pdf")
    window.export_page()
    drive.wait()
    legend_sheet = pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[1]["content"]
    assert "(Your paints)" in legend_sheet and all(f"({line})" in legend_sheet for line in page.mixes[start])


def test_a_preview_of_a_picture_replaced_while_it_was_made_is_never_shown(drive, monkeypatch):
    # G1 (R1): a result is delivered only if it is of the picture the document still holds.
    window = drive.window
    release, started = threading.Event(), threading.Event()
    monkeypatch.setattr(pipeline, "generate", _held_back(pipeline.generate, release, started=started))
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    assert started.wait(TIMEOUT_S)
    window.current_image_bgr = _blocks((200, 300))  # a new picture, as a drop or a paste will bring one (W1, H5)
    release.set()
    drive.wait()

    assert window.current_page is None and window.preview.shown_image() is None
    assert window.controls.generate_button.isEnabled() and drive.dialogs == []


def test_a_preview_made_while_a_setting_changed_is_shown_as_made_with_the_settings_before(drive, monkeypatch):
    # D-011: the page is of the current picture, kept with the settings it was made with, so it is never taken for
    # one made with the new ones.
    window = drive.window
    release, started = threading.Event(), threading.Event()
    monkeypatch.setattr(pipeline, "generate", _held_back(pipeline.generate, release, started=started))
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    assert started.wait(TIMEOUT_S)
    made_with = window.document.settings()
    # A change that makes the page again, and so waits for Generate (the tone would draw it again by itself, G2).
    window.controls.controls["num_colors"].step(-1)
    release.set()
    drive.wait()

    assert window.current_page is not None and window.document.result_settings() == made_with
    assert not window.document.result_is_current()


def test_closing_the_window_while_a_page_is_made_returns_at_once(monkeypatch):
    # G1 (R3): the window no longer waits on its threads; the application does, bounded, when it quits.
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    release = threading.Event()
    monkeypatch.setattr(pipeline, "generate", _held_until(pipeline.generate, release))
    window = MainWindow()
    window.current_image_bgr = _blocks((300, 200))
    window.generate_preview()
    began = time.monotonic()
    window.close()
    window.deleteLater()
    _app.processEvents()
    assert time.monotonic() - began < 1.0 and jobs.live_threads() >= 1
    release.set()
    assert jobs.wait_for_all(TIMEOUT_S)
    _until(lambda: jobs.live_threads() == 0)

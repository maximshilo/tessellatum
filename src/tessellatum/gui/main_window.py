"""Top-level application window."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import export, pipeline
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import GeneratedPage, PREVIEW_LONG_EDGE
from tessellatum.core.print_size import print_scale
from tessellatum.gui.controls_panel import ControlsPanel
from tessellatum.gui.preview_widget import PreviewWidget, VersionBar, ZoomBar
from tessellatum.gui.worker import PipelineWorker

OPEN_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)"


class MainWindow(QMainWindow):
    def __init__(self, store: QSettings | None = None):
        """``store``, if given, is where the panel keeps its settings between runs (see ``ControlsPanel``)."""
        super().__init__()
        self.setWindowTitle("Tessellatum")

        self.controls = ControlsPanel(store=store)
        self.preview = PreviewWidget()
        # Every version of a page is drawn with it, so the bar switches between them at once.
        self.version_bar = VersionBar()
        self.version_bar.setEnabled(False)
        self.zoom_bar = ZoomBar(self.preview)
        bars = QHBoxLayout()
        bars.addWidget(self.version_bar, 1)
        bars.addWidget(self.zoom_bar)
        preview_pane = QWidget()
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(bars)
        preview_layout.addWidget(self.preview)

        # The panel scrolls rather than holding the window taller than a small screen.
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(self.controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # As wide as the panel needs with the scroll bar beside it, so that nothing is cut off when it shows.
        controls_scroll.setMinimumWidth(
            self.controls.minimumSizeHint().width() + controls_scroll.verticalScrollBar().sizeHint().width()
        )

        splitter = QSplitter()
        splitter.addWidget(controls_scroll)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 880])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open an image to get started.")

        self.current_image_bgr: np.ndarray | None = None
        self.current_page: GeneratedPage | None = None
        self._previewed_image: np.ndarray | None = None  # the picture current_page is of
        self._worker: PipelineWorker | None = None
        self._pending_export_path: Path | None = None
        self._pending_export_format: str | None = None
        self._pending_export_version: Version | None = None
        self._pending_export_drawn = False  # whether the export waits for the preview's page drawn at print size
        self._ready_message = ""  # the status of the preview shown
        # Whether _worker is making a page whose end the window hasn't handled yet. (Its last signal can come while Qt
        # still reports it running.)
        self._making = False
        # The preview's page drawn at print size: what the zoom shows past the preview's own pixels, and what the
        # export saves. Drawn in the background when either first asks for it, and dropped with the preview.
        self._print_page: GeneratedPage | None = None
        self._print_worker: PipelineWorker | None = None
        self._retired: list[PipelineWorker] = []  # workers stopped but maybe not finished yet: kept until they are

        self.controls.open_image_requested.connect(self.open_image)
        self.controls.generate_requested.connect(self.generate_preview)
        self.controls.export_requested.connect(self.export_page)
        self.controls.abort_requested.connect(self.abort_generation)
        self.version_bar.version_changed.connect(self.preview.show_version)
        self.preview.detail_wanted.connect(self._draw_print_page)

        self.resize(1280, 840)

        # Get one-time costs (loading/compiling the compiled kernels, first
        # OpenCV/Pillow calls) out of the way while the user picks an image,
        # so the first preview is as fast as later ones.
        threading.Thread(target=_warm_up_pipeline, name="pipeline-warm-up", daemon=True).start()

    # -- Open -----------------------------------------------------------
    def open_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Image", "", OPEN_IMAGE_FILTER)
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.current_image_bgr = pipeline.load_image_bgr(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open image", str(exc))
            return

        self.controls.set_thumbnail(_bgr_to_pixmap(self.current_image_bgr))
        self.current_page = None
        self._drop_print_page()
        self.controls.set_export_enabled(False)
        self.version_bar.setEnabled(False)  # the preview left showing is the last picture's, until the next one
        self.statusBar().showMessage(f"Loaded {path.name}. Click \"Generate Preview\".")

    # -- Preview ----------------------------------------------------------
    def generate_preview(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if (self._worker is not None and self._worker.isRunning()) or self._pending_export_path is not None:
            return

        self._drop_print_page()  # the zoom asks for the new page's
        self._making = True
        self.controls.set_busy(True)
        self.statusBar().showMessage("Generating preview…")

        self._worker = self._make_worker(PREVIEW_LONG_EDGE, versions=tuple(Version))
        self._worker.succeeded.connect(self._on_preview_ready)
        self._worker.start()

    def abort_generation(self) -> None:
        # An export of the preview's page waits for it to be drawn at print size.
        worker = self._print_worker if self._pending_export_drawn else self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            self.statusBar().showMessage("Cancelling…")

    def _on_worker_cancelled(self) -> None:
        self._making = False
        self.controls.set_busy(False)
        self._clear_pending_export()
        self.statusBar().showMessage("Cancelled.")

    def _on_preview_ready(self, page: GeneratedPage) -> None:
        self._making = False
        # A new page of the picture shown is shown as far in and in the same place, to compare the two.
        same_picture = self.current_page is not None and self._previewed_image is self.current_image_bgr
        self._drop_print_page()  # any drawn of the last page, for a zoom while this one was made
        self.current_page = page
        self._previewed_image = self.current_image_bgr
        self.controls.set_busy(False)
        self.controls.set_export_enabled(True)
        self._ready_message = f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions."
        self.statusBar().showMessage(self._ready_message)
        versions = {version: page.image(version) for version in Version}
        long_edge = self._print_long_edge()
        print_scale = 1.0 if long_edge is None else long_edge / max(page.page.size)
        self.preview.show_page(versions, page.legend, self.version_bar.version(), print_scale, keep_view=same_picture)
        self.version_bar.setEnabled(True)

    def _on_worker_failed(self, message: str) -> None:
        self._making = False
        self.controls.set_busy(False)
        self._clear_pending_export()
        QMessageBox.critical(self, "Generation failed", message)
        self.statusBar().showMessage("Generation failed.")

    # -- Export -----------------------------------------------------------
    def export_page(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        fmt = self.controls.get_output_format()
        version = self.controls.get_export_version()
        suffix = ".png" if fmt == "PNG" else ".pdf"
        file_filter = "PNG image (*.png)" if fmt == "PNG" else "PDF document (*.pdf)"
        name = "coloring_page" if version is Version.PAGE else f"coloring_page_{version.value.lower()}"
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Coloring Page", f"{name}{suffix}", file_filter)
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)

        self._pending_export_path = path
        self._pending_export_format = fmt
        self._pending_export_version = version
        self.controls.set_busy(True)
        self.statusBar().showMessage(f"Rendering the {fmt} at print resolution…")

        # 300 dpi on A4, or the picture's own size where that is less: the pipeline never upscales. The export is the
        # preview's page drawn at that size, so that it has the regions and numbering the preview shows (drawn once for
        # the zoom and the export alike); made first if the settings changed since. A PNG's image is drawn with the
        # page; a PDF draws the version again from what the page is made of.
        long_edge = pipeline.export_long_edge(self.current_image_bgr)
        if self._preview_to_draw(long_edge) is not None:
            if self._print_page is not None:
                self._on_export_ready(self._print_page)
            else:
                self._pending_export_drawn = True
                self._draw_print_page()  # saved once drawn (see _on_print_page_ready)
            return
        self._making = True
        self._worker = self._make_worker(long_edge, versions=(version,) if fmt == "PNG" else ())
        self._worker.succeeded.connect(self._on_export_ready)
        self._worker.start()

    def _on_export_ready(self, page: GeneratedPage) -> None:
        self._making = False
        self.controls.set_busy(False)
        path = self._pending_export_path
        fmt = self._pending_export_format
        version = self._pending_export_version
        self._clear_pending_export()
        if path is None or fmt is None or version is None:
            return

        try:
            if fmt == "PNG":
                export.save_png(page.image(version), page.legend, path)
            else:
                export.save_pdf(page.drawing, page.palette_rgb, path, version=version, painting=page.painting)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            self.statusBar().showMessage("Export failed.")
            return

        if self.preview.wants_detail():
            self._draw_print_page()  # asked for by a zoom while the export was made
        width, height = page.page.size
        dpi = print_scale(page.page.size).dpi
        self.statusBar().showMessage(f"Exported to {path} ({width} × {height} px, {dpi:.0f} dpi on A4)")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}\n\n{width} × {height} px, {dpi:.0f} dpi on A4")

    def _clear_pending_export(self) -> None:
        self._pending_export_path = None
        self._pending_export_format = None
        self._pending_export_version = None
        self._pending_export_drawn = False

    def _preview_to_draw(self, long_edge: int) -> GeneratedPage | None:
        """The page previewed, if an export at ``long_edge`` is drawn from it: it is smaller, and made with the panel's
        settings as they are now. None if not, and the export is made of the picture itself."""
        page = self.current_page
        if page is None or page.model is None or long_edge <= max(page.page.size):
            return None
        model = page.model
        current = (self.controls.get_difficulty_params(), self.controls.get_page_style(), self.controls.get_handling())
        return page if (model.params, model.style, model.handling) == current else None

    def _make_worker(self, long_edge: int, versions: tuple[Version, ...] = ()) -> PipelineWorker:
        """A worker generating the current image at ``long_edge`` with the panel's settings, wired to its progress, and
        drawing ``versions`` of the page too."""
        worker = PipelineWorker(
            self.current_image_bgr,
            self.controls.get_difficulty_params(),
            long_edge,
            style=self.controls.get_page_style(),
            handling=self.controls.get_handling(),
            versions=versions,
        )
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.progress.connect(self.controls.set_progress)
        return worker

    # -- The preview's page at print size -------------------------------------
    def _print_long_edge(self) -> int | None:
        """The long edge the preview's page prints at, if it is drawn larger to print; None if the preview is the page
        as it prints, or there is none."""
        page = self.current_page
        if page is None or page.model is None or self._previewed_image is None:
            return None
        long_edge = pipeline.export_long_edge(self._previewed_image)
        return long_edge if long_edge > max(page.page.size) else None

    def _draw_print_page(self) -> None:
        """Draw the preview's page at print size, in the background, unless it is drawn or being drawn: for the zoom,
        which shows it past the preview's own pixels, and for the export, which saves it. Not while a page is being
        made: a preview, which will replace this one, or an export with other settings."""
        long_edge = self._print_long_edge()
        running = self._print_worker is not None and self._print_worker.isRunning()
        if long_edge is None or self._print_page is not None or running or self._making:
            return
        page = self.current_page
        model = page.model
        worker = PipelineWorker(
            self._previewed_image,
            model.params,
            long_edge,
            style=model.style,
            handling=model.handling,
            versions=tuple(Version),
            drawn_from=page,
        )
        worker.succeeded.connect(self._on_print_page_ready)
        worker.failed.connect(self._on_print_page_failed)
        worker.cancelled.connect(self._on_print_page_cancelled)
        worker.progress.connect(self._on_print_page_progress)
        self._print_worker = worker
        if self._pending_export_path is None:
            self.statusBar().showMessage("Drawing the page as it prints, for the zoom…")
        worker.start()

    def _drop_print_page(self) -> None:
        """Forget the preview's page at print size, and stop drawing it: the preview is about to change."""
        self._print_page = None
        worker = self._print_worker
        self._print_worker = None
        self._retired = [retired for retired in self._retired if not retired.isFinished()]
        if worker is not None and not worker.isFinished():
            worker.requestInterruption()
            self._retired.append(worker)

    # Each of these is the export's, if it waits for the page; else the zoom's alone, telling nothing over an export
    # with other settings that may be under way.
    def _on_print_page_progress(self, percent: int) -> None:
        if self.sender() is self._print_worker and self._pending_export_drawn:
            self.controls.set_progress(percent)

    def _on_print_page_ready(self, drawn: GeneratedPage) -> None:
        page = self.current_page
        if self.sender() is not self._print_worker or page is None or drawn.model is not page.model:
            return  # drawn from a preview since replaced
        self._print_page = drawn
        self.preview.show_detail({version: drawn.image(version) for version in Version})
        if self._pending_export_drawn:
            self._on_export_ready(drawn)
        elif self._pending_export_path is None:
            self.statusBar().showMessage(self._ready_message)

    def _on_print_page_failed(self, message: str) -> None:
        if self.sender() is not self._print_worker:
            return
        if self._pending_export_drawn:
            self._on_worker_failed(message)
        elif self._pending_export_path is None:
            self.statusBar().showMessage(f"The page could not be drawn as it prints: {message}")

    def _on_print_page_cancelled(self) -> None:
        if self.sender() is not self._print_worker:
            return
        if self._pending_export_drawn:
            self._on_worker_cancelled()
        elif self._pending_export_path is None:
            self.statusBar().showMessage(self._ready_message)

    def closeEvent(self, event) -> None:
        # Stop what is still running, rather than leave a thread to be destroyed with the window.
        for worker in (self._worker, self._print_worker, *self._retired):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait()
        super().closeEvent(event)


def _warm_up_pipeline() -> None:
    try:
        pipeline.warm_up()
    except Exception:  # noqa: BLE001 - best effort; a real failure resurfaces on Generate
        pass


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = image_bgr[:, :, ::-1].copy()
    h, w, _ = rgb.shape
    qimage = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

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
from tessellatum.gui.preview_widget import PreviewWidget, VersionBar
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
        preview_pane = QWidget()
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.version_bar)
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
        self._worker: PipelineWorker | None = None
        self._pending_export_path: Path | None = None
        self._pending_export_format: str | None = None
        self._pending_export_version: Version | None = None

        self.controls.open_image_requested.connect(self.open_image)
        self.controls.generate_requested.connect(self.generate_preview)
        self.controls.export_requested.connect(self.export_page)
        self.controls.abort_requested.connect(self.abort_generation)
        self.version_bar.version_changed.connect(self.preview.show_version)

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
        self.controls.set_export_enabled(False)
        self.version_bar.setEnabled(False)  # the preview left showing is the last picture's, until the next one
        self.statusBar().showMessage(f"Loaded {path.name}. Click \"Generate Preview\".")

    # -- Preview ----------------------------------------------------------
    def generate_preview(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self.controls.set_busy(True)
        self.statusBar().showMessage("Generating preview…")

        self._worker = self._make_worker(PREVIEW_LONG_EDGE, versions=tuple(Version))
        self._worker.succeeded.connect(self._on_preview_ready)
        self._worker.start()

    def abort_generation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self.statusBar().showMessage("Cancelling…")

    def _on_worker_cancelled(self) -> None:
        self.controls.set_busy(False)
        self._clear_pending_export()
        self.statusBar().showMessage("Cancelled.")

    def _on_preview_ready(self, page: GeneratedPage) -> None:
        self.current_page = page
        self.controls.set_busy(False)
        self.controls.set_export_enabled(True)
        versions = {version: page.image(version) for version in Version}
        self.preview.show_page(versions, page.legend, self.version_bar.version())
        self.version_bar.setEnabled(True)
        self.statusBar().showMessage(
            f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions."
        )

    def _on_worker_failed(self, message: str) -> None:
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

        # 300 dpi on A4, or the picture's own size where that is less: the pipeline never upscales. A PNG's image is
        # drawn with the page; a PDF draws the version again from what the page is made of.
        self._worker = self._make_worker(
            pipeline.export_long_edge(self.current_image_bgr), versions=(version,) if fmt == "PNG" else ()
        )
        self._worker.succeeded.connect(self._on_export_ready)
        self._worker.start()

    def _on_export_ready(self, page: GeneratedPage) -> None:
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

        width, height = page.page.size
        dpi = print_scale(page.page.size).dpi
        self.statusBar().showMessage(f"Exported to {path} ({width} × {height} px, {dpi:.0f} dpi on A4)")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}\n\n{width} × {height} px, {dpi:.0f} dpi on A4")

    def _clear_pending_export(self) -> None:
        self._pending_export_path = None
        self._pending_export_format = None
        self._pending_export_version = None

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

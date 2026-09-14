"""Top-level application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QSplitter, QWidget

from tessellatum.core import export, pipeline
from tessellatum.core.pipeline import GeneratedPage, PREVIEW_LONG_EDGE, EXPORT_LONG_EDGE
from tessellatum.gui.controls_panel import ControlsPanel
from tessellatum.gui.preview_widget import PreviewWidget
from tessellatum.gui.worker import PipelineWorker

OPEN_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tessellatum")

        self.controls = ControlsPanel()
        self.preview = PreviewWidget()

        splitter = QSplitter()
        splitter.addWidget(self.controls)
        splitter.addWidget(self.preview)
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

        self.controls.open_image_requested.connect(self.open_image)
        self.controls.generate_requested.connect(self.generate_preview)
        self.controls.export_requested.connect(self.export_page)

        self.resize(1280, 840)

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
        self.statusBar().showMessage(f"Loaded {path.name}. Click \"Generate Preview\".")

    # -- Preview ----------------------------------------------------------
    def generate_preview(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        params = self.controls.get_difficulty_params()
        self.controls.set_busy(True)
        self.statusBar().showMessage("Generating preview…")

        self._worker = PipelineWorker(self.current_image_bgr, params, PREVIEW_LONG_EDGE)
        self._worker.succeeded.connect(self._on_preview_ready)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_preview_ready(self, page: GeneratedPage) -> None:
        self.current_page = page
        self.controls.set_busy(False)
        self.controls.set_export_enabled(True)
        self.preview.show_page(page.page, page.legend)
        self.statusBar().showMessage(
            f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions."
        )

    def _on_worker_failed(self, message: str) -> None:
        self.controls.set_busy(False)
        self._pending_export_path = None
        self._pending_export_format = None
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
        suffix = ".png" if fmt == "PNG" else ".pdf"
        file_filter = "PNG image (*.png)" if fmt == "PNG" else "PDF document (*.pdf)"
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Coloring Page", f"coloring_page{suffix}", file_filter)
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)

        params = self.controls.get_difficulty_params()
        self._pending_export_path = path
        self._pending_export_format = fmt
        self.controls.set_busy(True)
        self.statusBar().showMessage(f"Rendering high-resolution {fmt} for export…")

        self._worker = PipelineWorker(self.current_image_bgr, params, EXPORT_LONG_EDGE)
        self._worker.succeeded.connect(self._on_export_ready)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_export_ready(self, page: GeneratedPage) -> None:
        self.controls.set_busy(False)
        path = self._pending_export_path
        fmt = self._pending_export_format
        self._pending_export_path = None
        self._pending_export_format = None
        if path is None or fmt is None:
            return

        try:
            if fmt == "PNG":
                export.save_png(page.page, page.legend, path)
            else:
                export.save_pdf(page.page, page.legend, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            self.statusBar().showMessage("Export failed.")
            return

        self.statusBar().showMessage(f"Exported to {path}")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}")


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = image_bgr[:, :, ::-1].copy()
    h, w, _ = rgb.shape
    qimage = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

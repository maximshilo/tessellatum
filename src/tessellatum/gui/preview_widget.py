"""Zoomable preview of the generated coloring page + legend."""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    qimage = QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class PreviewWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._placeholder = self._scene.addText("Open an image and click \"Generate Preview\"")

    def show_page(self, page: Image.Image, legend: Image.Image) -> None:
        gap = 24
        width = max(page.width, legend.width)
        height = page.height + gap + legend.height
        combined = Image.new("RGB", (width, height), "white")
        combined.paste(page, ((width - page.width) // 2, 0))
        combined.paste(legend, ((width - legend.width) // 2, page.height + gap))

        pixmap = pil_to_pixmap(combined)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_to_window()

"""Zoomable preview of the generated coloring page + legend, in any of its versions, and the bar that picks one."""

from __future__ import annotations

from typing import Mapping

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from tessellatum.core.painting import Version


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
        self._pixmaps: dict[Version, QPixmap] = {}
        self._placeholder = self._scene.addText("Open an image and click \"Generate Preview\"")

    def show_page(
        self, versions: Mapping[Version, Image.Image], legend: Image.Image, shown: Version = Version.PAGE
    ) -> None:
        """Show a new page: each of its ``versions`` above the legend, ``shown`` of them, fitted to the window."""
        self._pixmaps = {version: pil_to_pixmap(_above_legend(image, legend)) for version, image in versions.items()}
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(self._pixmaps[shown])
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()

    def show_version(self, version: Version) -> None:
        """Show ``version`` of the page in place of the one shown, as far in and in the same place."""
        if self._pixmap_item is not None and version in self._pixmaps:
            self._pixmap_item.setPixmap(self._pixmaps[version])

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


class VersionBar(QWidget):
    """A row of buttons, one per version of the page, of which one is down: the version the preview shows."""

    version_changed = Signal(object)  # Version

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.buttons: dict[Version, QPushButton] = {}
        self._group = QButtonGroup(self)  # exclusive: one button down at a time
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Show"))
        for version in Version:
            button = QPushButton(version.value)
            button.setCheckable(True)
            button.setToolTip(version.description)
            self._group.addButton(button)
            self.buttons[version] = button
            layout.addWidget(button)
            button.toggled.connect(lambda down, version=version: down and self.version_changed.emit(version))
        layout.addStretch(1)
        self.buttons[Version.PAGE].setChecked(True)

    def version(self) -> Version:
        return next(version for version, button in self.buttons.items() if button.isChecked())


def _above_legend(page: Image.Image, legend: Image.Image) -> Image.Image:
    gap = 24
    width = max(page.width, legend.width)
    height = page.height + gap + legend.height
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(page, ((width - page.width) // 2, 0))
    combined.paste(legend, ((width - legend.width) // 2, page.height + gap))
    return combined

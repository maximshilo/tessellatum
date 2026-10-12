"""Zoomable preview of the generated coloring page + legend, in any of its versions, and the bars that pick one and
zoom into it."""

from __future__ import annotations

from typing import Mapping

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut, QTransform, QWheelEvent
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
from tessellatum.core.print_size import MM_PER_INCH, print_scale

# One notch of a mouse wheel, or one press of + or -, zooms in or out this far.
ZOOM_STEP = 1.25
# The furthest in: this many pixels of the screen to a pixel of the page as it prints.
MAX_ZOOM_PRINTED = 4.0
# The legend's distance below the page, in the preview's pixels.
LEGEND_GAP = 24
# How long after the last step of a zoom the page is drawn again at the screen's resolution, in ms. Until then the
# view scales what it showed.
SHARPEN_DELAY_MS = 120


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888).copy()


class PreviewWidget(QGraphicsView):
    """The page above its legend, zoomed in and out about the point under the cursor, and dragged to pan.

    How far in it is, ``zoom``, counts the screen's pixels (its device pixels) to a pixel of the preview. Up to the
    preview's own pixels the preview is shown; past them, the page as it prints (``show_detail``), once it is drawn,
    since the preview has nothing finer to show. Either is drawn at the screen's resolution a moment after the zoom
    settles, scaled down smoothly, so its lines stay whole; further in than the page as it prints, its pixels are
    scaled up. Until the page as it prints is there, the preview is scaled up, and the widget asks for it
    (``detail_wanted``).
    """

    detail_wanted = Signal()
    zoom_changed = Signal(float)  # how far in, as ``printed_zoom`` gives it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # The zoom keeps its own point in place (see ``zoom_to``), and a resize keeps the top left corner.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self._page_item: QGraphicsPixmapItem | None = None
        self._legend_item: QGraphicsPixmapItem | None = None
        self._page_size: tuple[int, int] | None = None
        self._previews: dict[Version, QImage] = {}
        self._details: dict[Version, QImage] = {}
        self._legend: QImage | None = None
        self._version = Version.PAGE
        self._print_scale = 1.0
        self._fitted = True
        # The last zoom's anchor, the point of the scene held there, and the view it left: the next zoom about the same
        # anchor of the same view holds the same point, so that steps of a zoom don't add up their rounding.
        self._held: tuple[QPointF, QPointF, QTransform] | None = None
        self._shown: dict[str, tuple] = {}  # what each item shows: its image and the size it is drawn at
        self._sharpen_timer = QTimer(self)
        self._sharpen_timer.setSingleShot(True)
        self._sharpen_timer.setInterval(SHARPEN_DELAY_MS)
        self._sharpen_timer.timeout.connect(self._sharpen)
        self._placeholder = self._scene.addText("Open an image and click \"Generate Preview\"")
        for keys, action in (
            ((QKeySequence.ZoomIn, "Ctrl+="), self.zoom_in),
            ((QKeySequence.ZoomOut,), self.zoom_out),
            (("Ctrl+0",), self.fit_to_window),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.setContext(Qt.WindowShortcut)
                shortcut.activated.connect(action)

    def show_page(
        self,
        versions: Mapping[Version, Image.Image],
        legend: Image.Image,
        shown: Version = Version.PAGE,
        print_scale: float = 1.0,
        keep_view: bool = False,
    ) -> None:
        """Show a new page: each of its ``versions`` above the legend, ``shown`` of them, fitted to the window.

        ``print_scale`` is how many times larger the page prints than it is here: 1 if the page is as it prints, else
        zoomed in past its pixels the widget asks for it as it prints (see ``show_detail``). With ``keep_view``, a page
        of the picture shown is shown as far in and in the same place, unless the view is fitted to the window; made at
        another size (another level's, see ``pipeline.working_long_edge``), as far in on paper, the same point of the
        picture in the middle.
        """
        size = next(iter(versions.values())).size
        keep = keep_view and not self._fitted and self._page_size is not None
        held = None
        if keep and size != self._page_size:
            ratio = size[0] / self._page_size[0]
            keep = abs(size[1] - self._page_size[1] * ratio) <= 1  # the same shape, to the pixel it was rounded to
            if keep:
                middle = QRectF(self.viewport().rect()).center()
                inverse, _invertible = self.viewportTransform().inverted()
                held = (middle, inverse.map(middle) * ratio, self.zoom() / ratio)
        self._previews = {version: pil_to_qimage(image) for version, image in versions.items()}
        self._details = {}
        self._legend = pil_to_qimage(legend)
        self._version = shown
        self._page_size = size
        self._print_scale = max(1.0, print_scale)
        self._shown = {}
        if self._page_item is None:
            self._scene.clear()
            self._page_item = self._scene.addPixmap(QPixmap())
            self._legend_item = self._scene.addPixmap(QPixmap())
            for item in (self._page_item, self._legend_item):
                item.setTransformationMode(Qt.SmoothTransformation)
        width, height = size
        self._legend_item.setPos((width - legend.width) / 2, height + LEGEND_GAP)
        legend_rect = QRectF(self._legend_item.pos().x(), self._legend_item.pos().y(), legend.width, legend.height)
        self._scene.setSceneRect(QRectF(0, 0, width, height).united(legend_rect))
        if held is not None:
            middle, point, zoom = held
            self._held = (middle, point, self.viewportTransform())  # the zoom holds that point there
            self.zoom_to(zoom, middle)
        elif keep:
            self._zoomed()
        else:
            self.fit_to_window()
        self._sharpen()

    def show_version(self, version: Version) -> None:
        """Show ``version`` of the page in place of the one shown, as far in and in the same place."""
        if version in self._previews:
            self._version = version
            self._sharpen()

    def show_detail(self, versions: Mapping[Version, Image.Image]) -> None:
        """``versions`` of the page shown, as it prints (see ``pipeline.draw_page``): what is shown from the preview's
        own pixels in. They are ``print_scale`` times its size (to the pixel)."""
        self._details = {version: pil_to_qimage(image) for version, image in versions.items()}
        self._sharpen()

    def shown_image(self) -> QImage | None:
        """The page shown, at its own size: the version shown, of the preview or, as far in as it is shown, of the page
        as it prints."""
        if self._page_size is None:
            return None
        detail = self._details.get(self._version) if self.zoom() > 1 + 1e-6 else None
        return detail if detail is not None else self._previews[self._version]

    def zoom(self) -> float:
        """How far in the page is shown: the screen's pixels to a pixel of the preview."""
        return self.transform().m11() * self.devicePixelRatioF()

    def printed_zoom(self) -> float:
        """How far in the page is shown, by the page as it prints: 1 is a pixel of the screen to a pixel of the page as
        it prints (on a page no larger than the preview, of the preview)."""
        return self.zoom() / self._print_scale

    def zoom_range(self) -> tuple[float, float]:
        """The zooms the page can be shown at: out to fitting the window (or the preview's own pixels, if it fits at
        more), in to ``MAX_ZOOM_PRINTED`` pixels of the screen to one of the page as it prints."""
        fit = self._fit_zoom()
        return min(fit, 1.0), max(fit, MAX_ZOOM_PRINTED * self._print_scale)

    def zoom_to(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Show the page at ``zoom`` (within ``zoom_range``), the point at ``anchor`` in the viewport (its middle by
        default) staying where it is, to the pixel, wherever the page reaches past it."""
        if self._page_size is None:
            return
        low, high = self.zoom_range()
        zoom = min(max(zoom, low), high)
        if anchor is None:
            anchor = QRectF(self.viewport().rect()).center()
        if self._held is not None and self._held[0] == anchor and self._held[2] == self.viewportTransform():
            point = self._held[1]
        else:
            inverse, _invertible = self.viewportTransform().inverted()
            point = inverse.map(anchor)
        scale = zoom / self.devicePixelRatioF()
        self.setTransform(QTransform.fromScale(scale, scale))
        moved = self.viewportTransform().map(point) - anchor
        for bar, by in ((self.horizontalScrollBar(), moved.x()), (self.verticalScrollBar(), moved.y())):
            bar.setValue(bar.value() + round(by))
        # Where the page doesn't reach past the anchor (fitted, say), it is placed in the middle rather than scrolled:
        # what is held there is the point it puts there.
        missed = self.viewportTransform().map(point) - anchor
        inverse, _invertible = self.viewportTransform().inverted()
        there = inverse.map(anchor)
        point = QPointF(
            point.x() if abs(missed.x()) <= 0.5 else there.x(), point.y() if abs(missed.y()) <= 0.5 else there.y()
        )
        self._held = (QPointF(anchor), point, self.viewportTransform())
        self._fitted = False
        self._zoomed()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        self.zoom_to(self.zoom() * factor, anchor)

    def zoom_in(self) -> None:
        self.zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1 / ZOOM_STEP)

    def fit_to_window(self) -> None:
        """The whole page and its legend in the window, and kept so as it is resized."""
        if self._page_size is not None:
            self.zoom_to(self._fit_zoom())
            self._fitted = True

    def zoom_to_printed_pixels(self) -> None:
        """A pixel of the page as it prints to a pixel of the screen."""
        self.zoom_to(self._print_scale)

    def zoom_to_print_size(self) -> None:
        """The page as large as it prints on paper, by the size the screen says it is."""
        if self._page_size is not None:
            width_mm = print_scale(self._page_size).printed_size_mm[0]
            per_inch = self.screen().physicalDotsPerInch()  # the screen's logical pixels to an inch
            self.zoom_to(width_mm / MM_PER_INCH * per_inch / self._page_size[0] * self.devicePixelRatioF())

    def wants_detail(self) -> bool:
        """Whether the page is shown further in than the preview's own pixels, with no page as it prints to show."""
        return self._print_scale > 1 and not self._details and self.zoom() > 1 + 1e-6

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120
        # A touchpad's scrolling (which only some platforms tell apart, by its phases) pans; its pinch zooms.
        panning = event.phase() != Qt.NoScrollPhase and not event.modifiers() & Qt.ControlModifier
        if self._page_size is None or steps == 0 or panning:
            super().wheelEvent(event)
            return
        self.zoom_by(ZOOM_STEP**steps, event.position())
        event.accept()

    def viewportEvent(self, event: QEvent) -> bool:
        # A touchpad's pinch, on the platforms that report it as one.
        if (
            event.type() == QEvent.NativeGesture
            and event.gestureType() == Qt.ZoomNativeGesture
            and self._page_size is not None
        ):
            self.zoom_by(1 + event.value(), event.position())
            return True
        return super().viewportEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit_to_window()

    def _fit_zoom(self) -> float:
        rect = self.sceneRect()
        size = self.maximumViewportSize()  # the viewport without its scroll bars
        margin = 4  # as ``fitInView`` leaves
        fit = min((size.width() - margin) / rect.width(), (size.height() - margin) / rect.height())
        return max(fit, 1e-3) * self.devicePixelRatioF()

    def _zoomed(self) -> None:
        self._sharpen_timer.start()
        self.zoom_changed.emit(self.printed_zoom())
        if self.wants_detail():
            self.detail_wanted.emit()

    def _sharpen(self) -> None:
        """Draw the page and its legend at the screen's resolution for the zoom: the preview, or past its pixels the
        page as it prints if there is one, scaled down smoothly to the pixels it takes on the screen."""
        self._sharpen_timer.stop()
        if self._page_size is None:
            return
        zoom = self.zoom()
        _show_at(self._page_item, self.shown_image(), self._page_size, zoom, self._shown, "page")
        legend_size = (self._legend.width(), self._legend.height())
        _show_at(self._legend_item, self._legend, legend_size, zoom, self._shown, "legend")


def _show_at(
    item: QGraphicsPixmapItem, image: QImage, size: tuple[int, int], zoom: float, shown: dict[str, tuple], name: str
) -> None:
    """Show ``image`` in ``item``, ``size`` pixels of the scene large, drawn for ``zoom`` screen pixels to one of the
    scene: scaled down smoothly to that many if it has more, else as it is."""
    target = (max(1, round(size[0] * zoom)), max(1, round(size[1] * zoom)))
    if target[0] >= image.width():
        target = (image.width(), image.height())
    key = (image.cacheKey(), target)
    if shown.get(name) == key:
        return
    shown[name] = key
    if target != (image.width(), image.height()):
        image = image.scaled(target[0], target[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    item.setPixmap(QPixmap.fromImage(image))
    item.setTransform(QTransform.fromScale(size[0] / image.width(), size[1] / image.height()))


class ZoomBar(QWidget):
    """Buttons that zoom the preview, and how far in it is: 100% is a pixel of the page as it prints to a pixel of the
    screen."""

    def __init__(self, preview: PreviewWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.preview = preview
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.out_button = QPushButton("−")
        self.out_button.setToolTip("Zoom out (Ctrl+-, or the mouse wheel)")
        self.zoom_label = QLabel()
        self.zoom_label.setToolTip("How far in: 100% shows a pixel of the page as it prints on a pixel of the screen")
        self.in_button = QPushButton("+")
        self.in_button.setToolTip("Zoom in (Ctrl++, or the mouse wheel); drag the page to move it")
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("The whole page in the window (Ctrl+0)")
        self.pixels_button = QPushButton("100%")
        self.pixels_button.setToolTip("A pixel of the page as it prints on a pixel of the screen")
        self.paper_button = QPushButton("Print size")
        self.paper_button.setToolTip("The page as large as on paper, by the size the screen reports")
        for widget in (
            self.out_button, self.zoom_label, self.in_button, self.fit_button, self.pixels_button, self.paper_button
        ):
            layout.addWidget(widget)
        self.out_button.clicked.connect(preview.zoom_out)
        self.in_button.clicked.connect(preview.zoom_in)
        self.fit_button.clicked.connect(preview.fit_to_window)
        self.pixels_button.clicked.connect(preview.zoom_to_printed_pixels)
        self.paper_button.clicked.connect(preview.zoom_to_print_size)
        preview.zoom_changed.connect(self._show_zoom)
        self.setEnabled(False)  # until there is a page to zoom into

    def _show_zoom(self, printed_zoom: float) -> None:
        self.setEnabled(True)
        self.zoom_label.setText(f"{printed_zoom * 100:.0f}%")


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

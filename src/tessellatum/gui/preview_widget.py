"""The canvas: the generated coloring page and its legend, in any of its versions or as the picture itself, zoomed and
panned; and the bars that pick the view and the zoom.

The scene is in layers, each a parent item at its own height (``LAYERS``): the artwork (the page and its legend), the
guides (the grid and the sheet's bounds), the annotations (highlights of colors and regions), the diagnostics (the
analysis's overlays) and the interaction (the selection). A layer is drawn over the ones under it, so a highlight never
draws the artwork again, and nothing but the artwork is ever exported: exports are drawn from the page's own data,
never from the canvas.
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping

from PIL import Image
from PySide6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from tessellatum.core.painting import Version
from tessellatum.core.print_size import A4, MM_PER_INCH, PaperFormat, print_scale

# One notch of a mouse wheel, or one press of + or -, zooms in or out this far.
ZOOM_STEP = 1.25
# The furthest in: this many pixels of the screen to a pixel of the page as it prints.
MAX_ZOOM_PRINTED = 4.0
# The legend's distance below the page, in the preview's pixels.
LEGEND_GAP = 24
# How long after the last step of a zoom the page is drawn again at the screen's resolution, in ms. Until then the
# view scales what it showed.
SHARPEN_DELAY_MS = 120

# The picture itself, at the page's size and in its place: a view beside the page's versions, never exported.
ORIGINAL = "Original"
ORIGINAL_DESCRIPTION = "The picture itself, in the page's place: to compare the page with."
VIEWS: tuple[Hashable, ...] = (*Version, ORIGINAL)


def view_name(view: Hashable) -> str:
    return view.value if isinstance(view, Version) else str(view)


def view_description(view: Hashable) -> str:
    return view.description if isinstance(view, Version) else ORIGINAL_DESCRIPTION


# The scene's layers, bottom to top, and the height each is drawn at.
ARTWORK, GUIDES, ANNOTATION, DIAGNOSTICS, INTERACTION = "artwork", "guides", "annotation", "diagnostics", "interaction"
LAYERS = {ARTWORK: 0.0, GUIDES: 10.0, ANNOTATION: 20.0, DIAGNOSTICS: 30.0, INTERACTION: 40.0}

# The rulers: their breadth in the screen's logical pixels, and the least room between two numbers on them and between
# two ticks. Their numbers are millimeters or inches on paper, from the page's top left corner.
RULER_PX = 20
RULER_LABEL_ROOM_PX = 56
RULER_TICK_ROOM_PX = 5
MM, INCH = "mm", "in"
RULER_STEPS = {
    MM: (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    INCH: (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32),
}
# The grid's spacing on paper, in its unit.
GRID_STEP = {MM: 10.0, INCH: 1.0}

# Zoom presets to pick from in the zoom field, by the page as it prints, and after them the page as large as on paper.
ZOOM_PRESETS = (0.25, 0.5, 1.0, 2.0, 4.0)
PRINT_SIZE = "print size"


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888).copy()


def ruler_steps(px_per_unit: float, unit: str) -> tuple[float, float | None]:
    """The rulers' steps for ``px_per_unit`` screen pixels to one ``unit`` on paper: between two numbers, the smallest
    step leaving ``RULER_LABEL_ROOM_PX`` between them; between two ticks, the largest step under it that divides it
    and leaves ``RULER_TICK_ROOM_PX`` (None: no ticks between the numbers)."""
    steps = RULER_STEPS[unit]
    label = next((step for step in steps if step * px_per_unit >= RULER_LABEL_ROOM_PX), steps[-1])
    tick = None
    for step in steps:
        if step >= label:
            break
        ratio = label / step
        if abs(ratio - round(ratio)) < 1e-9 and step * px_per_unit >= RULER_TICK_ROOM_PX:
            tick = step
            break
    return label, tick


class PreviewWidget(QGraphicsView):
    """The page above its legend, zoomed in and out about the point under the cursor, and dragged to pan.

    How far in it is, ``zoom``, counts the screen's pixels (its device pixels) to a pixel of the preview. Up to the
    preview's own pixels the preview is shown; past them, the page as it prints (``show_detail``), once it is drawn,
    since the preview has nothing finer to show. Either is drawn at the screen's resolution a moment after the zoom
    settles, scaled down smoothly, so its lines stay whole; further in than the page as it prints, its pixels are
    scaled up. Until the page as it prints is there, the preview is scaled up, and the widget asks for it
    (``detail_wanted``).

    The views are the page's versions and the picture itself (``ORIGINAL``), each in the page's place: a view of other
    pixels than the page's (the picture's own) is mapped onto the page's, so a point of the page is the same point in
    every view, and the zoom, the place and anything selected carry from one to another.

    A drag pans with the left button, the middle one, or the left with Space held (the Space only while the canvas has
    the keyboard, so a text field keeps its spaces). A click without a drag says where (``clicked``).
    """

    detail_wanted = Signal()
    zoom_changed = Signal(float)  # how far in, as ``printed_zoom`` gives it
    clicked = Signal(QPointF)  # a click that didn't pan: where, in the page's pixels (the scene's)
    view_moved = Signal()  # the zoom or the place changed: what a view kept in step with this one follows

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.NoDrag)
        # The zoom keeps its own point in place (see ``zoom_to``), and a resize keeps the middle.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)  # Space pans, and the zoom's keys work, while it has the keyboard
        self.setAccessibleName("Canvas")
        self.layers: dict[str, QGraphicsItemGroup] = {}
        for name, height in LAYERS.items():
            layer = QGraphicsItemGroup()
            layer.setZValue(height)
            layer.setHandlesChildEvents(False)
            self._scene.addItem(layer)
            self.layers[name] = layer
        self._page_item = QGraphicsPixmapItem(self.layers[ARTWORK])
        self._legend_item = QGraphicsPixmapItem(self.layers[ARTWORK])
        for item in (self._page_item, self._legend_item):
            item.setTransformationMode(Qt.SmoothTransformation)
            item.hide()
        self._overlays: dict[tuple[str, str], QGraphicsPixmapItem] = {}
        self._grid_item = QGraphicsPathItem(self.layers[GUIDES])
        self._bounds_item = QGraphicsPathItem(self.layers[GUIDES])
        self._margin_item = QGraphicsPathItem(self.layers[GUIDES])
        # The sheet, as paper, under the page: shown with the bounds.
        self._sheet_item = QGraphicsRectItem(self.layers[ARTWORK])
        self._sheet_item.setZValue(-1)
        self._sheet_item.setBrush(QColor(255, 255, 255))
        self._sheet_item.setPen(QPen(Qt.NoPen))
        for item in (self._grid_item, self._bounds_item, self._margin_item, self._sheet_item):
            item.hide()
        self._page_size: tuple[int, int] | None = None
        self._previews: dict[Hashable, QImage] = {}
        self._details: dict[Hashable, QImage] = {}
        self._legend: QImage | None = None
        self._legend_size: tuple[float, float] = (0.0, 0.0)  # the legend's size in the scene, which its image may exceed
        self._version: Hashable = Version.PAGE
        self._print_scale = 1.0
        self._paper = A4
        self._fitted = True
        # The last zoom's anchor, the point of the scene held there, and the view it left: the next zoom about the same
        # anchor of the same view holds the same point, so that steps of a zoom don't add up their rounding.
        self._held: tuple[QPointF, QPointF, QTransform] | None = None
        self._shown: dict[str, tuple] = {}  # what each item shows: its image and the size it is drawn at
        self._sharpen_timer = QTimer(self)
        self._sharpen_timer.setSingleShot(True)
        self._sharpen_timer.setInterval(SHARPEN_DELAY_MS)
        self._sharpen_timer.timeout.connect(self._sharpen)
        # Panning: Space held, and the drag under way (where it was last, and whether it is a pan yet).
        self._space = False
        self._press: QPointF | None = None
        self._panning = False
        self._rulers = False
        self._grid = False
        self._bounds = False
        self._units = MM
        self.rulers = (_Ruler(self, True), _Ruler(self, False))
        self._ruler_corner = _RulerCorner(self)
        for widget in (*self.rulers, self._ruler_corner):
            widget.hide()
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self._moved)
        # Its zoom's keys are the window's View actions (see ``gui.actions``), and the menu's.
        self._placeholder = None
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._placeholder = self._scene.addText("Open an image and click \"Generate Preview\"")
        self._placeholder.setDefaultTextColor(self.palette().color(QPalette.WindowText))

    def has_page(self) -> bool:
        return self._page_size is not None

    def page_size(self) -> tuple[int, int] | None:
        """The page's size in its pixels, the scene's: every view is shown at it."""
        return self._page_size

    def clear_page(self) -> None:
        """Show no page, as at the start: the picture was closed."""
        for item in (self._page_item, self._legend_item):
            item.setPixmap(QPixmap())
            item.hide()
        for key in list(self._overlays):
            self.set_overlay(key[0], key[1], None)
        self._page_size = None
        self._previews, self._details, self._shown = {}, {}, {}
        self._legend = None
        self._legend_size = (0.0, 0.0)
        self._held = None
        self._fitted = True
        if self._placeholder is None:
            self._show_placeholder()
        self._scene.setSceneRect(self._placeholder.boundingRect())
        self._show_guides()
        self.resetTransform()

    def show_page(
        self,
        versions: Mapping[Hashable, Image.Image],
        legend: Image.Image,
        shown: Hashable = Version.PAGE,
        print_scale: float = 1.0,
        keep_view: bool = False,
        paper: PaperFormat = A4,
        legend_scale: float = 1.0,
    ) -> None:
        """Show a new page: each of its ``versions`` above the legend, ``shown`` of them, fitted to the window.

        ``versions`` are views (see ``VIEWS``), the page's own (``Version.PAGE``) among them, whose size is the
        page's; another of other pixels, the picture itself, is mapped onto them. ``print_scale`` is how many times
        larger the page prints than it is here: 1 if the page is as it prints, else zoomed in past its pixels the
        widget asks for it as it prints (see ``show_detail``). With ``keep_view``, a page of the picture shown is shown
        as far in and in the same place, unless the view is fitted to the window; made at another size (another
        level's, see ``pipeline.working_long_edge``), as far in on paper, the same point of the picture in the middle.
        ``paper`` is the sheet the page is made for. ``legend`` may be drawn finer than the page, ``legend_scale`` of
        its pixels to one of the page's, so that its small print stays sharp zoomed in.
        """
        page = versions[Version.PAGE] if Version.PAGE in versions else next(iter(versions.values()))
        size = page.size
        keep = keep_view and not self._fitted and self._page_size is not None
        held = None
        if keep and size != self._page_size:
            ratio = size[0] / self._page_size[0]
            keep = abs(size[1] - self._page_size[1] * ratio) <= 1  # the same shape, to the pixel it was rounded to
            if keep:
                middle = QRectF(self.viewport().rect()).center()
                inverse, _invertible = self.viewportTransform().inverted()
                held = (middle, inverse.map(middle) * ratio, self.zoom() / ratio)
        self._previews = {view: pil_to_qimage(image) for view, image in versions.items()}
        self._details = {}
        self._version = shown if shown in self._previews else Version.PAGE
        self._page_size = size
        self._print_scale = max(1.0, print_scale)
        self._paper = paper
        self._shown = {}
        if self._placeholder is not None:
            self._scene.removeItem(self._placeholder)
            self._placeholder = None
        for item in (self._page_item, self._legend_item):
            item.show()
        self._place_legend(legend, legend_scale)
        if held is not None:
            middle, point, zoom = held
            self._held = (middle, point, self.viewportTransform())  # the zoom holds that point there
            self.zoom_to(zoom, middle)
        elif keep:
            self._zoomed()
        else:
            self.fit_to_window()
        self._sharpen()

    def views(self) -> list[Hashable]:
        """The views there are of the page shown."""
        return list(self._previews)

    def view(self) -> Hashable:
        """The view shown."""
        return self._version

    def show_version(self, version: Hashable) -> None:
        """Show ``version`` of the page (any view, see ``VIEWS``) in place of the one shown, as far in and in the same
        place."""
        if version in self._previews:
            self._version = version
            self._sharpen()

    def show_detail(self, versions: Mapping[Hashable, Image.Image]) -> None:
        """``versions`` of the page shown, as it prints (see ``pipeline.draw_page``): what is shown from the preview's
        own pixels in. They are ``print_scale`` times its size (to the pixel)."""
        self._details = {version: pil_to_qimage(image) for version, image in versions.items()}
        self._sharpen()

    def view_image(self, view: Hashable) -> QImage | None:
        """``view`` of the page at the preview's size, or of its own pixels for one that isn't the page's (the
        picture's); None if there is none."""
        return self._previews.get(view)

    def shown_image(self) -> QImage | None:
        """The page shown, at its own size: the version shown, of the preview or, as far in as it is shown, of the page
        as it prints."""
        if self._page_size is None:
            return None
        detail = self._details.get(self._version) if self.zoom() > 1 + 1e-6 else None
        return detail if detail is not None else self._previews[self._version]

    # -- The zoom ---------------------------------------------------------------------------------
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

    def zoom_to_printed(self, printed_zoom: float) -> None:
        """Show the page ``printed_zoom`` far in, by the page as it prints (1: 100%), about the middle."""
        self.zoom_to(printed_zoom * self._print_scale)

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
            width_mm = print_scale(self._page_size, self._paper).printed_size_mm[0]
            per_inch = self.screen().physicalDotsPerInch()  # the screen's logical pixels to an inch
            self.zoom_to(width_mm / MM_PER_INCH * per_inch / self._page_size[0] * self.devicePixelRatioF())

    def wants_detail(self) -> bool:
        """Whether the page is shown further in than the preview's own pixels, with no page as it prints to show."""
        return self._print_scale > 1 and not self._details and self.zoom() > 1 + 1e-6

    def center_on_rect(self, rect: QRectF) -> None:
        """Put the middle of ``rect`` (in the page's pixels) in the middle of the window, as far in as it is."""
        self.centerOn(rect.center())
        self._held = None

    def view_state(self) -> tuple[float, int, int]:
        """The zoom and where the page is scrolled to: what ``set_view_state`` puts another view at."""
        return self.zoom(), self.horizontalScrollBar().value(), self.verticalScrollBar().value()

    def set_view_state(self, state: tuple[float, int, int]) -> None:
        """Show the page as ``state`` (``view_state``, of a view of the same page as large) says: as far in and scrolled
        to the same place."""
        zoom, x, y = state
        if self._page_size is None:
            return
        scale = zoom / self.devicePixelRatioF()
        if abs(self.transform().m11() - scale) > 1e-12:
            self.setTransform(QTransform.fromScale(scale, scale))
            self._fitted = False
        self.horizontalScrollBar().setValue(x)
        self.verticalScrollBar().setValue(y)
        self._held = None
        self._sharpen_timer.start()
        self.zoom_changed.emit(self.printed_zoom())

    # -- Where things are --------------------------------------------------------------------------
    def page_point(self, viewport_point: QPointF) -> QPointF:
        """The point of the page (its pixels, the scene's) at ``viewport_point``."""
        inverse, _invertible = self.viewportTransform().inverted()
        return inverse.map(QPointF(viewport_point))

    def page_pixel(self, viewport_point: QPointF) -> tuple[int, int] | None:
        """The page's pixel (x, y) at ``viewport_point``, or None off the page."""
        if self._page_size is None:
            return None
        point = self.page_point(viewport_point)
        x, y = math.floor(point.x()), math.floor(point.y())
        if 0 <= x < self._page_size[0] and 0 <= y < self._page_size[1]:
            return x, y
        return None

    def px_per_mm(self) -> float | None:
        """The page's pixels to a millimeter on paper, or None with no page."""
        if self._page_size is None:
            return None
        return print_scale(self._page_size, self._paper).px_per_mm

    # -- The input ---------------------------------------------------------------------------------
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.modifiers():
            if not event.isAutoRepeat():
                self._space = True
                self._show_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = False
            self._show_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        self._space = False  # its release goes elsewhere
        self._show_cursor()
        super().focusOutEvent(event)

    def space_held(self) -> bool:
        return self._space

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._page_size is not None and (
            event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space)
        ):
            self._press, self._panning = event.position(), True
            self._show_cursor()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._press, self._panning = event.position(), False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press is not None:
            moved = event.position() - self._press
            if not self._panning and self._page_size is not None and \
                    moved.manhattanLength() >= QApplication.startDragDistance():
                self._panning = True
                self._show_cursor()
            if self._panning:
                self._press = event.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(moved.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(moved.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is not None and event.button() in (Qt.LeftButton, Qt.MiddleButton):
            clicked = not self._panning and event.button() == Qt.LeftButton
            self._press, self._panning = None, False
            self._show_cursor()
            if clicked and self._page_size is not None:
                self.clicked.emit(self.page_point(event.position()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _show_cursor(self) -> None:
        if self._panning:
            self.viewport().setCursor(QCursor(Qt.ClosedHandCursor))
        elif self._space:
            self.viewport().setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.viewport().unsetCursor()

    def changeEvent(self, event: QEvent) -> None:
        # The scene's text isn't a widget: it takes the theme's colors here, when they change.
        if event.type() == QEvent.PaletteChange:
            if self._placeholder is not None:
                self._placeholder.setDefaultTextColor(self.palette().color(QPalette.WindowText))
            self._show_guides()
        super().changeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_rulers()
        if self._fitted:
            self.fit_to_window()

    def _fit_zoom(self) -> float:
        rect = self._fit_rect()
        size = self.maximumViewportSize()  # the viewport without its scroll bars
        margin = 4  # as ``fitInView`` leaves
        fit = min((size.width() - margin) / rect.width(), (size.height() - margin) / rect.height())
        return max(fit, 1e-3) * self.devicePixelRatioF()

    def _fit_rect(self) -> QRectF:
        """What a fit fits in the window: the page and its legend."""
        if self._page_size is None:
            return self.sceneRect()
        width, height = self._page_size
        legend = QRectF(self._legend_item.pos().x(), self._legend_item.pos().y(), *self._legend_size)
        rect = QRectF(0, 0, width, height).united(legend)
        return rect.united(self.sheet_rect()) if self._bounds else rect

    # -- The legend ---------------------------------------------------------------------------------
    def set_legend(self, legend: Image.Image, legend_scale: float = 1.0) -> None:
        """Show ``legend`` under the page shown, in place of its own (``legend_scale`` as ``show_page`` takes it)."""
        if self._page_size is None:
            return
        self._place_legend(legend, legend_scale)
        if self._fitted:
            self.fit_to_window()  # the legend may be taller or shorter than it was
        self._sharpen()

    def _place_legend(self, legend: Image.Image, legend_scale: float) -> None:
        self._legend = pil_to_qimage(legend)
        self._legend_size = (legend.width / legend_scale, legend.height / legend_scale)
        self._shown.pop("legend", None)
        self._position_legend()

    def _position_legend(self) -> None:
        """The legend under the page, or with the sheet's bounds shown under the sheet, which it prints apart from."""
        width, height = self._page_size
        top = height + LEGEND_GAP
        if self._bounds:
            top = max(top, self.sheet_rect().bottom() + LEGEND_GAP)
        self._legend_item.setPos((width - self._legend_size[0]) / 2, top)
        self._update_scene_rect()
        self._show_guides()

    def _update_scene_rect(self) -> None:
        rect = self._fit_rect()
        if self._bounds and self._page_size is not None:
            rect = rect.united(self.sheet_rect())
        self._scene.setSceneRect(rect)

    # -- The layers ----------------------------------------------------------------------------------
    def set_overlay(self, layer: str, key: str, image: QImage | None, smooth: bool = False) -> None:
        """Show ``image`` over the page, in ``layer`` (see ``LAYERS``), as ``key`` there: stretched over the page, of
        whatever size it is; None takes it away. The page under it isn't drawn again."""
        item = self._overlays.get((layer, key))
        if image is None or image.isNull() or self._page_size is None:
            if item is not None:
                self._scene.removeItem(item)
                del self._overlays[(layer, key)]
            return
        if item is None:
            item = QGraphicsPixmapItem(self.layers[layer])
            self._overlays[(layer, key)] = item
        item.setTransformationMode(Qt.SmoothTransformation if smooth else Qt.FastTransformation)
        item.setPixmap(QPixmap.fromImage(image))
        width, height = self._page_size
        item.setTransform(QTransform.fromScale(width / image.width(), height / image.height()))

    def overlay(self, layer: str, key: str) -> QGraphicsPixmapItem | None:
        return self._overlays.get((layer, key))

    def page_pixmap_key(self) -> int:
        """Which pixmap the page's item shows, to tell that it wasn't drawn again."""
        return self._page_item.pixmap().cacheKey()

    # -- The guides: rulers, grid, the sheet's bounds -----------------------------------------------
    def set_units(self, units: str) -> None:
        """The unit of the rulers and the grid: ``MM`` or ``INCH``."""
        self._units = units if units in RULER_STEPS else MM
        self._show_guides()
        self._update_rulers()

    def units(self) -> str:
        return self._units

    def set_rulers(self, shown: bool) -> None:
        self._rulers = bool(shown)
        self.setViewportMargins(RULER_PX if shown else 0, RULER_PX if shown else 0, 0, 0)
        for widget in (*self.rulers, self._ruler_corner):
            widget.setVisible(self._rulers)
        self._place_rulers()
        if self._fitted:
            self.fit_to_window()
        self._update_rulers()

    def rulers_shown(self) -> bool:
        return self._rulers

    def set_grid(self, shown: bool) -> None:
        self._grid = bool(shown)
        self._show_guides()

    def grid_shown(self) -> bool:
        return self._grid

    def set_bounds(self, shown: bool) -> None:
        self._bounds = bool(shown)
        if self._page_size is None:
            self._show_guides()
            return
        self._position_legend()
        if self._fitted:
            self.fit_to_window()

    def bounds_shown(self) -> bool:
        return self._bounds

    def sheet_rect(self) -> QRectF:
        """The sheet the page prints on, in the page's pixels: the page in the middle of the area inside its margins,
        as the export lays it out."""
        scale = print_scale(self._page_size, self._paper)
        sheet = (self._paper.height_mm, self._paper.width_mm) if scale.landscape else \
            (self._paper.width_mm, self._paper.height_mm)
        printed = scale.printed_size_mm
        ppm = scale.px_per_mm
        left = (sheet[0] - printed[0]) / 2 * ppm
        top = (sheet[1] - printed[1]) / 2 * ppm
        return QRectF(-left, -top, sheet[0] * ppm, sheet[1] * ppm)

    def _px_per_unit(self) -> float:
        """The page's pixels to one unit (``set_units``) on paper."""
        per_mm = self.px_per_mm() or 1.0
        return per_mm * (MM_PER_INCH if self._units == INCH else 1.0)

    def _show_guides(self) -> None:
        has = self._page_size is not None
        color = self.palette().color(QPalette.WindowText)
        self._grid_item.setVisible(has and self._grid)
        self._bounds_item.setVisible(has and self._bounds)
        self._sheet_item.setVisible(has and self._bounds)
        self._margin_item.setVisible(has and self._bounds)
        if not has:
            return
        width, height = self._page_size
        if self._grid:
            step = GRID_STEP[self._units] * self._px_per_unit()
            path = QPainterPath()
            x = step
            while x < width - 1e-6:
                path.moveTo(x, 0)
                path.lineTo(x, height)
                x += step
            y = step
            while y < height - 1e-6:
                path.moveTo(0, y)
                path.lineTo(width, y)
                y += step
            self._grid_item.setPath(path)
            # The theme's accent, half seen: on the white page and on the canvas around it alike. Cosmetic: a pixel
            # of the screen wide at any zoom.
            accent = self.palette().color(QPalette.Highlight)
            self._grid_item.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 130), 0))
        if self._bounds:
            sheet = self.sheet_rect()
            path = QPainterPath()
            path.addRect(sheet)
            self._bounds_item.setPath(path)
            self._bounds_item.setPen(QPen(color, 0))
            self._sheet_item.setRect(sheet)
            margin = self._paper.margin_mm * (self.px_per_mm() or 1.0)
            inner = QPainterPath()
            inner.addRect(sheet.adjusted(margin, margin, -margin, -margin))
            self._margin_item.setPath(inner)
            dashed = QPen(QColor(color.red(), color.green(), color.blue(), 140), 0, Qt.DashLine)
            self._margin_item.setPen(dashed)

    def ruler_ticks(self, horizontal: bool = True) -> list[tuple[float, float, bool]]:
        """The ticks the ruler along the top (``horizontal``) or the left shows: each one's place in the viewport (its
        logical pixels), its value on paper in the unit, and whether it is numbered. Empty with no page."""
        if self._page_size is None:
            return []
        per_unit_scene = self._px_per_unit()
        per_unit_screen = per_unit_scene * self.transform().m11()
        label, tick = ruler_steps(per_unit_screen, self._units)
        step = tick or label
        transform = self.viewportTransform()
        length = self.viewport().width() if horizontal else self.viewport().height()
        start = self.page_point(QPointF(0, 0))
        end = self.page_point(QPointF(length, length))
        low = (start.x() if horizontal else start.y()) / per_unit_scene
        high = (end.x() if horizontal else end.y()) / per_unit_scene
        ticks = []
        k = math.floor(low / step)
        while k * step <= high + 1e-9:
            value = k * step
            scene = value * per_unit_scene
            place = transform.map(QPointF(scene, scene))
            numbered = abs(value / label - round(value / label)) < 1e-6
            ticks.append((place.x() if horizontal else place.y(), round(value, 6), numbered))
            k += 1
        return ticks

    def _place_rulers(self) -> None:
        frame = self.frameWidth()
        viewport = self.viewport().geometry()
        self.rulers[0].setGeometry(viewport.left(), frame, viewport.width(), RULER_PX)
        self.rulers[1].setGeometry(frame, viewport.top(), RULER_PX, viewport.height())
        self._ruler_corner.setGeometry(frame, frame, RULER_PX, RULER_PX)

    def _update_rulers(self) -> None:
        if self._rulers:
            for ruler in self.rulers:
                ruler.update()
            self._ruler_corner.update()

    def _moved(self, _value: int = 0) -> None:
        self._update_rulers()
        self.view_moved.emit()

    # -- Drawing -------------------------------------------------------------------------------------
    def _zoomed(self) -> None:
        self._sharpen_timer.start()
        self.zoom_changed.emit(self.printed_zoom())
        self.view_moved.emit()
        self._update_rulers()
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
        _show_at(self._legend_item, self._legend, self._legend_size, zoom, self._shown, "legend")


class _Ruler(QWidget):
    """A ruler along the canvas's top or left edge: millimeters or inches on paper, from the page's top left corner."""

    def __init__(self, view: PreviewWidget, horizontal: bool):
        super().__init__(view)
        self.view = view
        self.horizontal = horizontal
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        palette = self.view.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.Window))
        painter.setPen(QPen(palette.color(QPalette.WindowText), 1))
        painter.setFont(_small_font(self.font()))
        if self.horizontal:
            painter.drawLine(0, RULER_PX - 1, self.width(), RULER_PX - 1)
        else:
            painter.drawLine(RULER_PX - 1, 0, RULER_PX - 1, self.height())
        for place, value, numbered in self.view.ruler_ticks(self.horizontal):
            reach = 9 if numbered else 4
            if self.horizontal:
                painter.drawLine(QLineF(place, RULER_PX - reach, place, RULER_PX - 1))
                if numbered:
                    painter.drawText(QPointF(place + 2, RULER_PX - 9), f"{value:g}")
            else:
                painter.drawLine(QLineF(RULER_PX - reach, place, RULER_PX - 1, place))
                if numbered:
                    painter.save()
                    painter.translate(RULER_PX - 9, place - 2)
                    painter.rotate(-90)
                    painter.drawText(QPointF(0, 0), f"{value:g}")
                    painter.restore()
        painter.end()


class _RulerCorner(QWidget):
    """The square where the rulers meet: their unit."""

    def __init__(self, view: PreviewWidget):
        super().__init__(view)
        self.view = view
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        palette = self.view.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.Window))
        painter.setPen(palette.color(QPalette.WindowText))
        painter.setFont(_small_font(self.font()))
        painter.drawText(QRectF(self.rect()), Qt.AlignCenter, self.view.units())
        painter.end()


def _small_font(font: QFont) -> QFont:
    small = QFont(font)
    if small.pointSizeF() > 0:
        small.setPointSizeF(max(6.0, small.pointSizeF() * 0.75))
    return small


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


def zoom_text(printed_zoom: float) -> str:
    """A zoom as the field shows it: a whole percentage, to a tenth under 10%."""
    percent = printed_zoom * 100
    return f"{round(percent)}%" if percent >= 9.95 else f"{percent:.1f}%"


def parse_zoom(text: str) -> float | None:
    """A zoom typed as a percentage ("150", "150%", "150 %", "37.5"), as a fraction (1.5); None if it isn't one."""
    cleaned = text.strip().rstrip("%").strip().replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100 if math.isfinite(value) and value > 0 else None


class ZoomBar(QWidget):
    """Buttons that zoom the preview, and how far in it is, in a field to type a zoom into or pick one from (100%, a
    pixel of the page as it prints to a pixel of the screen; Print size, as large as on paper), and Fit."""

    def __init__(self, preview: PreviewWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.preview = preview
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.out_button = QPushButton("−")
        self.out_button.setToolTip("Zoom out (Ctrl+-, or the mouse wheel)")
        self.out_button.setAccessibleName("Zoom out")
        self.zoom_field = QComboBox()
        self.zoom_field.setEditable(True)
        self.zoom_field.setInsertPolicy(QComboBox.NoInsert)
        self.zoom_field.setObjectName("zoomField")
        self.zoom_field.setAccessibleName("Zoom")
        self.zoom_field.setCompleter(None)
        self.zoom_field.setToolTip(
            "How far in: type a zoom and press Enter, or pick one. 100% shows a pixel of the page as it prints on a "
            "pixel of the screen; Print size, the page as large as on paper (approximate, by the size the screen "
            "reports)"
        )
        for preset in ZOOM_PRESETS:
            self.zoom_field.addItem(f"{preset * 100:.0f}%", preset)
        self.zoom_field.addItem("Print size", PRINT_SIZE)
        self.zoom_field.setItemData(
            self.zoom_field.count() - 1, "Approximate: by the size the screen reports", Qt.ToolTipRole
        )
        self.zoom_field.lineEdit().setPlaceholderText("Zoom")
        self.zoom_field.setMinimumContentsLength(4)
        self.zoom_field.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.in_button = QPushButton("+")
        self.in_button.setToolTip("Zoom in (Ctrl++, or the mouse wheel); drag the page to move it")
        self.in_button.setAccessibleName("Zoom in")
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("The whole page in the window (Ctrl+0)")
        for widget in (self.out_button, self.zoom_field, self.in_button, self.fit_button):
            layout.addWidget(widget)
        self.out_button.clicked.connect(preview.zoom_out)
        self.in_button.clicked.connect(preview.zoom_in)
        self.fit_button.clicked.connect(preview.fit_to_window)
        self.zoom_field.lineEdit().returnPressed.connect(self._typed)
        self.zoom_field.activated.connect(self._picked)
        preview.zoom_changed.connect(self._show_zoom)
        self._zoom = 1.0
        self._text = ""  # what the field was last set to show
        self.setEnabled(False)  # until there is a page to zoom into

    def zoom_text(self) -> str:
        return self.zoom_field.currentText()

    def _typed(self) -> None:
        text = self.zoom_field.currentText()
        if text == self._text:
            return  # what it shows already (Return can come twice, through the combo box and its line)
        value = parse_zoom(text)
        if value is None:
            self._show_zoom(self._zoom)  # what it was: nothing to go by
            return
        self.preview.zoom_to_printed(value)
        self._show_zoom(self.preview.printed_zoom())

    def _picked(self, index: int) -> None:
        value = self.zoom_field.itemData(index)
        if value == PRINT_SIZE:
            self.preview.zoom_to_print_size()
        elif value is not None:
            self.preview.zoom_to_printed(float(value))
        self._show_zoom(self.preview.printed_zoom())

    def _show_zoom(self, printed_zoom: float) -> None:
        self.setEnabled(True)
        self._zoom = printed_zoom
        self._text = zoom_text(printed_zoom)
        blocked = self.zoom_field.blockSignals(True)
        self.zoom_field.setCurrentIndex(-1)
        self.zoom_field.setEditText(self._text)
        self.zoom_field.blockSignals(blocked)


class VersionBar(QWidget):
    """A row of buttons, one per view of the page (its versions, then the picture itself), of which one is down: the
    view the canvas shows."""

    version_changed = Signal(object)  # Version, or ORIGINAL

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.buttons: dict[Hashable, QPushButton] = {}
        self._group = QButtonGroup(self)  # exclusive: one button down at a time
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Show"))
        for view in VIEWS:
            button = QPushButton(view_name(view))
            button.setCheckable(True)
            button.setToolTip(view_description(view))
            self._group.addButton(button)
            self.buttons[view] = button
            layout.addWidget(button)
            button.toggled.connect(lambda down, view=view: down and self.version_changed.emit(view))
        layout.addStretch(1)
        self.buttons[Version.PAGE].setChecked(True)

    def version(self) -> Hashable:
        return next(view for view, button in self.buttons.items() if button.isChecked())


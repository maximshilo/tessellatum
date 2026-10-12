"""The preview's zoom (T5): its arithmetic, the point under the cursor staying put, and the page as it prints shown past
the preview's own pixels, drawn at the screen's resolution."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QNativeGestureEvent, QPointingDevice, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui.preview_widget import (  # noqa: E402
    MAX_ZOOM_PRINTED,
    ZOOM_STEP,
    PreviewWidget,
    ZoomBar,
    pil_to_qimage,
)

_app = QApplication.instance() or QApplication([])

PAGE = (1100, 734)
PRINTS_AT = 2  # the page as it prints is twice the preview's size


def _picture(size: tuple[int, int], seed: int) -> Image.Image:
    """Noise, so that every pixel tells where it is from."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def _versions(size: tuple[int, int], seed: int) -> dict[Version, Image.Image]:
    return {version: _picture(size, seed + index) for index, version in enumerate(Version)}


@pytest.fixture
def preview():
    widget = PreviewWidget()
    widget.resize(500, 400)
    widget.show()
    _app.processEvents()
    widget.show_page(_versions(PAGE, 0), _picture((PAGE[0], 60), 9), print_scale=PRINTS_AT)
    _app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    _app.processEvents()


def _scene_at(widget: PreviewWidget, point: QPointF) -> QPointF:
    inverse, _ = widget.viewportTransform().inverted()
    return inverse.map(point)


def _view_at(widget: PreviewWidget, point: QPointF) -> QPointF:
    return widget.viewportTransform().map(point)


def _same(a: QImage, b: QImage) -> bool:
    return a.convertToFormat(QImage.Format_RGB888) == b.convertToFormat(QImage.Format_RGB888)


def test_a_new_page_is_fitted_to_the_window_and_the_bar_shows_how_far_in(preview):
    bar = ZoomBar(preview)
    assert not bar.isEnabled()
    preview.show_page(_versions(PAGE, 0), _picture((PAGE[0], 60), 9), print_scale=PRINTS_AT)

    rect = preview.mapFromScene(preview.sceneRect()).boundingRect()
    viewport = preview.viewport().rect()
    assert viewport.contains(rect)  # the page and its legend, whole
    assert rect.width() >= viewport.width() - 6 or rect.height() >= viewport.height() - 6  # and as large as fits
    assert preview.zoom() < 1 and preview.printed_zoom() == pytest.approx(preview.zoom() / PRINTS_AT)
    assert bar.isEnabled() and bar.zoom_label.text() == f"{preview.printed_zoom() * 100:.0f}%"


@pytest.mark.parametrize("anchor", [(40.0, 30.0), (250.0, 200.0), (470.5, 380.25)])
@pytest.mark.parametrize("factor", [ZOOM_STEP, 1 / ZOOM_STEP, 3.0])
def test_the_point_under_the_cursor_stays_put(preview, anchor, factor):
    preview.zoom_to(1.5)  # far enough in that the page reaches past the window on every side
    preview.horizontalScrollBar().setValue(300)
    preview.verticalScrollBar().setValue(200)
    anchor = QPointF(*anchor)
    under = _scene_at(preview, anchor)

    preview.zoom_by(factor, anchor)

    assert preview.zoom() == pytest.approx(1.5 * factor)
    moved = _view_at(preview, under) - anchor
    assert abs(moved.x()) <= 0.5 and abs(moved.y()) <= 0.5  # to the pixel: the scroll bars move in whole pixels


def test_the_wheel_and_a_pinch_zoom_about_the_cursor(preview):
    preview.zoom_to(1.5)
    anchor = QPointF(120, 90)
    under = _scene_at(preview, anchor)
    for modifiers in (Qt.NoModifier, Qt.ControlModifier):
        event = QWheelEvent(
            anchor, preview.viewport().mapToGlobal(anchor), QPoint(), QPoint(0, 120), Qt.NoButton, modifiers,
            Qt.NoScrollPhase, False,
        )
        QApplication.sendEvent(preview.viewport(), event)
    assert preview.zoom() == pytest.approx(1.5 * ZOOM_STEP**2)
    moved = _view_at(preview, under) - anchor
    assert abs(moved.x()) <= 0.5 and abs(moved.y()) <= 0.5

    pinch = QNativeGestureEvent(
        Qt.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2, anchor, anchor,
        preview.viewport().mapToGlobal(anchor), -0.2, QPointF(),
    )
    QApplication.sendEvent(preview.viewport(), pinch)
    assert preview.zoom() == pytest.approx(1.5 * ZOOM_STEP**2 * 0.8)
    moved = _view_at(preview, under) - anchor
    assert abs(moved.x()) <= 0.5 and abs(moved.y()) <= 0.5


def test_after_fit_a_zoom_about_the_middle_holds_the_point_shown_there(preview):
    # Scroll bars that stay put leave the viewport's middle where it was, across the fit.
    preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    preview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    _app.processEvents()
    preview.zoom_to(2.0)
    preview.horizontalScrollBar().setValue(0)
    preview.verticalScrollBar().setValue(0)
    preview.zoom_in()  # about the middle, a corner of the page there
    preview.fit_to_window()  # the page in the middle, not that corner
    middle = QRectF(preview.viewport().rect()).center()
    under = _scene_at(preview, middle)

    preview.zoom_in()

    moved = _view_at(preview, under) - middle
    assert abs(moved.x()) <= 0.5 and abs(moved.y()) <= 0.5


def test_a_touchpad_s_scrolling_pans(preview):
    preview.zoom_to(1.5)
    top = preview.verticalScrollBar().value()
    anchor = QPointF(120, 90)
    event = QWheelEvent(
        anchor, preview.viewport().mapToGlobal(anchor), QPoint(0, -40), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
        Qt.ScrollUpdate, False,
    )
    QApplication.sendEvent(preview.viewport(), event)

    assert preview.zoom() == pytest.approx(1.5) and preview.verticalScrollBar().value() > top


def test_the_zoom_stops_at_the_window_and_at_four_pixels_to_one_that_prints(preview):
    preview.fit_to_window()
    fit = preview.zoom()
    preview.zoom_by(0.1)
    assert preview.zoom() == pytest.approx(fit)
    preview.zoom_by(1000)
    assert preview.zoom() == pytest.approx(MAX_ZOOM_PRINTED * PRINTS_AT)
    assert preview.printed_zoom() == pytest.approx(MAX_ZOOM_PRINTED)

    preview.zoom_to_printed_pixels()
    assert preview.zoom() == pytest.approx(PRINTS_AT) and preview.printed_zoom() == pytest.approx(1)


def test_print_size_is_the_page_as_large_as_on_paper_by_the_screen_s_dpi(preview):
    preview.zoom_to_print_size()

    # A4 landscape less 10 mm margins: the page's 1100 x 734 px fill 277 x 184.8 mm, its width the limit.
    width_px = PAGE[0] * preview.transform().m11()
    assert width_px == pytest.approx(277 / 25.4 * preview.screen().physicalDotsPerInch(), rel=1e-6)


def test_the_fit_follows_the_window_until_zoomed(preview):
    preview.resize(700, 500)
    _app.processEvents()
    fit = preview.zoom()
    rect = preview.mapFromScene(preview.sceneRect()).boundingRect()
    viewport = preview.viewport().rect()
    assert viewport.contains(rect) and rect.height() >= viewport.height() - 6

    preview.zoom_to(1.2)
    preview.resize(600, 450)
    _app.processEvents()
    assert preview.zoom() == pytest.approx(1.2)
    preview.fit_to_window()
    assert preview.zoom() < fit


def test_switching_versions_keeps_the_zoom_and_the_place(preview):
    preview.zoom_by(3.0, QPointF(300, 100))
    transform = preview.transform()
    scrolled = (preview.horizontalScrollBar().value(), preview.verticalScrollBar().value())
    for version in Version:
        preview.show_version(version)
        assert preview.transform() == transform
        assert (preview.horizontalScrollBar().value(), preview.verticalScrollBar().value()) == scrolled
        assert _same(preview.shown_image(), preview._previews[version])


def test_a_new_page_of_the_same_size_can_keep_the_view(preview):
    preview.zoom_by(3.0, QPointF(300, 100))
    transform = preview.transform()
    scrolled = (preview.horizontalScrollBar().value(), preview.verticalScrollBar().value())

    preview.show_page(_versions(PAGE, 20), _picture((PAGE[0], 90), 29), print_scale=PRINTS_AT, keep_view=True)
    assert preview.transform() == transform
    assert (preview.horizontalScrollBar().value(), preview.verticalScrollBar().value()) == scrolled
    assert _same(preview.shown_image(), pil_to_qimage(_picture(PAGE, 20)))

    preview.show_page(_versions((900, 734), 30), _picture((900, 60), 39), print_scale=PRINTS_AT, keep_view=True)
    assert preview.zoom() < 1  # another shape: fitted


def test_a_new_page_of_the_picture_made_at_another_size_keeps_the_view_on_paper(preview):
    # D-012 (T8): another level makes its page at another size; it is shown as far in on paper, the same point of the
    # picture in the middle.
    preview.zoom_by(3.0, QPointF(300, 100))
    middle = QRectF(preview.viewport().rect()).center()
    point = _scene_at(preview, middle)
    zoom, on_paper = preview.zoom(), preview.printed_zoom()

    larger = (1650, 1101)  # 1.5 times as large: the page prints at 4/3 of its size, as at 2 times the preview's
    preview.show_page(_versions(larger, 40), _picture((larger[0], 90), 49), print_scale=4 / 3, keep_view=True)
    assert preview.printed_zoom() == pytest.approx(on_paper)
    assert preview.zoom() == pytest.approx(zoom / 1.5)
    there = _scene_at(preview, middle)
    assert abs(there.x() - point.x() * 1.5) <= 0.5 and abs(there.y() - point.y() * 1.5) <= 0.5
    assert _same(preview.shown_image(), pil_to_qimage(_picture(larger, 40)))


def test_past_the_preview_s_pixels_the_page_as_it_prints_is_shown_at_the_screen_s_resolution(preview):
    wanted = []
    preview.detail_wanted.connect(lambda: wanted.append(preview.zoom()))
    preview.zoom_to(1.0)
    assert wanted == [] and _same(preview.shown_image(), preview._previews[Version.PAGE])
    preview._sharpen()  # as the view does a moment after the zoom settles
    assert _same(preview._page_item.pixmap().toImage(), preview._previews[Version.PAGE])  # its own pixels

    preview.zoom_to(1.5)
    assert wanted == [pytest.approx(1.5)]  # the page as it prints, asked for
    assert _same(preview.shown_image(), preview._previews[Version.PAGE])  # the preview, scaled up until it comes

    printed = _versions((PAGE[0] * PRINTS_AT, PAGE[1] * PRINTS_AT), 40)
    preview.show_detail(printed)
    shown = preview._page_item.pixmap()
    assert (shown.width(), shown.height()) == (1650, 1101)  # drawn for the screen: 1.5 of its pixels to the preview's
    assert _same(preview.shown_image(), pil_to_qimage(printed[Version.PAGE]))
    preview.zoom_to(2.0)
    assert len(wanted) == 1  # it has it

    # At 100% a tile of the view is the page as it prints, pixel for pixel; past it, its pixels scaled up.
    preview._sharpen()
    assert _same(preview._page_item.pixmap().toImage(), pil_to_qimage(printed[Version.PAGE]))
    tile = preview.viewport().grab().toImage().copy(100, 100, 200, 150)
    corner = _scene_at(preview, QPointF(100, 100))
    expected = pil_to_qimage(printed[Version.PAGE]).copy(round(corner.x() * 2), round(corner.y() * 2), 200, 150)
    assert _same(tile, expected)
    preview.zoom_to(4.0)
    preview._sharpen()
    assert preview._page_item.pixmap().width() == PAGE[0] * PRINTS_AT

    preview.show_version(Version.TINTED)
    assert _same(preview.shown_image(), pil_to_qimage(printed[Version.TINTED]))
    preview.zoom_to(0.9)
    assert _same(preview.shown_image(), preview._previews[Version.TINTED])

    preview.show_page(_versions(PAGE, 50), _picture((PAGE[0], 60), 59), print_scale=PRINTS_AT, keep_view=True)
    assert preview._details == {}  # a new page: its own, when it is asked for


def test_zoomed_out_the_page_is_scaled_down_smoothly_to_the_screen_s_pixels(preview):
    preview.fit_to_window()
    preview._sharpen()
    shown = preview._page_item.pixmap()
    assert shown.width() == round(PAGE[0] * preview.zoom()) < PAGE[0]
    # It still covers the page's place in the scene.
    assert preview._page_item.sceneBoundingRect().width() == pytest.approx(PAGE[0])


def test_a_page_as_large_as_it_prints_asks_for_nothing_more(preview):
    wanted = []
    preview.show_page(_versions(PAGE, 0), _picture((PAGE[0], 60), 9), print_scale=1.0)
    preview.detail_wanted.connect(lambda: wanted.append(True))
    preview.zoom_to(3.0)
    assert wanted == [] and preview.printed_zoom() == pytest.approx(3.0)


def test_the_bar_s_buttons_zoom(preview):
    bar = ZoomBar(preview)
    preview.zoom_to(1.0)
    bar.in_button.click()
    assert preview.zoom() == pytest.approx(ZOOM_STEP)
    bar.out_button.click()
    assert preview.zoom() == pytest.approx(1.0)
    bar.pixels_button.click()
    assert preview.printed_zoom() == pytest.approx(1.0) and bar.zoom_label.text() == "100%"
    bar.fit_button.click()
    assert preview.zoom() < 1
    bar.paper_button.click()
    assert preview.transform().m11() * PAGE[0] == pytest.approx(277 / 25.4 * preview.screen().physicalDotsPerInch())

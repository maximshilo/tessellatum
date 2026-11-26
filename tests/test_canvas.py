"""The canvas (I1): the Original view mapped onto the page, the exact zoom field, the ways to pan and to click, the
position kept on a resize, the rulers' millimeters and inches on paper, the grid, the sheet's bounds, and the layers a
highlight is drawn in without drawing the page again."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QWidget  # noqa: E402

from tessellatum.core import pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.core.print_size import A4, MM_PER_INCH, print_scale  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.preview_widget import (  # noqa: E402
    ANNOTATION,
    ARTWORK,
    DIAGNOSTICS,
    INCH,
    INTERACTION,
    LAYERS,
    MM,
    ORIGINAL,
    RULER_LABEL_ROOM_PX,
    PreviewWidget,
    ZoomBar,
    parse_zoom,
    zoom_text,
    ruler_steps,
)

_app = QApplication.instance() or QApplication([])

PAGE = (1100, 734)
PRINTS_AT = 2
SCALE = print_scale(PAGE, A4)  # landscape: 277 mm across


def _noise(size, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def _views(original_size=(2200, 1468)):
    views = {version: _noise(PAGE, index) for index, version in enumerate(Version)}
    views[ORIGINAL] = _noise(original_size, 7)
    return views


@pytest.fixture
def canvas():
    widget = PreviewWidget()
    widget.resize(500, 400)
    widget.show()
    _app.processEvents()
    widget.show_page(_views(), _noise((PAGE[0], 60), 9), print_scale=PRINTS_AT)
    _app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    _app.processEvents()


def _mouse(widget, kind, point, button, buttons=None):
    event = QMouseEvent(kind, QPointF(point), widget.viewport().mapToGlobal(QPointF(point)), button,
                        button if buttons is None else buttons, Qt.NoModifier)
    QApplication.sendEvent(widget.viewport(), event)


def _drag(widget, start, end, button=Qt.LeftButton):
    start, end = QPointF(*start), QPointF(*end)
    _mouse(widget, QEvent.MouseButtonPress, start, button)
    middle = (start + end) / 2
    _mouse(widget, QEvent.MouseMove, middle, Qt.NoButton, button)
    _mouse(widget, QEvent.MouseMove, end, Qt.NoButton, button)
    _mouse(widget, QEvent.MouseButtonRelease, end, button, Qt.NoButton)


def _scroll(widget):
    return widget.horizontalScrollBar().value(), widget.verticalScrollBar().value()


# -- The Original view -------------------------------------------------------------------------------------------------
def test_the_original_is_mapped_onto_the_page_and_a_switch_keeps_the_view(canvas):
    assert canvas.views() == [*Version, ORIGINAL]
    canvas.zoom_to(1.7)
    canvas.horizontalScrollBar().setValue(240)
    canvas.verticalScrollBar().setValue(130)
    before = (canvas.transform(), _scroll(canvas))
    canvas.show_version(ORIGINAL)
    canvas._sharpen()
    assert canvas.view() == ORIGINAL and (canvas.transform(), _scroll(canvas)) == before
    # Its own pixels (twice the page's) are drawn over exactly the page's place.
    rect = canvas._page_item.sceneBoundingRect()
    assert rect.topLeft() == QPointF(0, 0) and rect.width() == pytest.approx(PAGE[0])
    assert rect.height() == pytest.approx(PAGE[1])
    canvas.show_version(Version.TINTED)
    assert (canvas.transform(), _scroll(canvas)) == before


def test_a_view_the_page_has_not_is_not_shown(canvas):
    canvas.show_page({version: _noise(PAGE, 1) for version in Version}, _noise((PAGE[0], 60), 9))
    canvas.show_version(ORIGINAL)
    assert canvas.view() == Version.PAGE


# -- The zoom field ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, value", [("150", 1.5), ("150%", 1.5), (" 37.5 % ", 0.375), ("12,5", 0.125)])
def test_a_zoom_is_read_as_a_percentage(text, value):
    assert parse_zoom(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["", "abc", "-50", "0", "inf", "nan"])
def test_what_isn_t_a_zoom_is_none(text):
    assert parse_zoom(text) is None


def test_a_zoom_typed_into_the_field_is_shown(canvas):
    bar = ZoomBar(canvas)
    canvas.fit_to_window()
    field = bar.zoom_field.lineEdit()
    for typed, printed in (("150", 1.5), ("62.5%", 0.625), ("100", 1.0)):
        field.selectAll()
        QTest.keyClicks(field, typed)
        QTest.keyClick(field, Qt.Key_Return)
        assert canvas.printed_zoom() == pytest.approx(printed)
        assert bar.zoom_text() == zoom_text(printed)
    field.selectAll()
    QTest.keyClicks(field, "lots")
    QTest.keyClick(field, Qt.Key_Return)
    assert canvas.printed_zoom() == pytest.approx(1.0) and bar.zoom_text() == "100%"  # nothing to go by: as it was
    bar.zoom_field.activated.emit(bar.zoom_field.findData(2.0))
    assert canvas.printed_zoom() == pytest.approx(2.0)
    # Past the furthest in, as far in as it goes.
    field.selectAll()
    QTest.keyClicks(field, "100000")
    QTest.keyClick(field, Qt.Key_Return)
    assert canvas.printed_zoom() == pytest.approx(4.0)


# -- Panning and clicking ----------------------------------------------------------------------------------------------
def test_the_middle_button_pans(canvas):
    canvas.zoom_to(2.0)
    canvas.horizontalScrollBar().setValue(400)
    canvas.verticalScrollBar().setValue(300)
    clicks = []
    canvas.clicked.connect(clicks.append)
    _drag(canvas, (200, 200), (150, 170), Qt.MiddleButton)
    assert _scroll(canvas) == (450, 330) and clicks == []


def test_space_and_the_left_button_pan_from_the_first_pixel(canvas):
    canvas.zoom_to(2.0)
    canvas.horizontalScrollBar().setValue(400)
    canvas.setFocus()
    QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))
    assert canvas.space_held()
    assert canvas.viewport().cursor().shape() == Qt.OpenHandCursor
    _mouse(canvas, QEvent.MouseButtonPress, QPointF(200, 200), Qt.LeftButton)
    _mouse(canvas, QEvent.MouseMove, QPointF(199, 200), Qt.NoButton, Qt.LeftButton)  # under the drag distance
    _mouse(canvas, QEvent.MouseButtonRelease, QPointF(199, 200), Qt.LeftButton, Qt.NoButton)
    assert _scroll(canvas)[0] == 401
    QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier, " "))
    assert not canvas.space_held() and canvas.viewport().cursor().shape() != Qt.OpenHandCursor


def test_a_left_drag_pans_and_a_click_says_where(canvas):
    canvas.zoom_to(2.0)
    canvas.horizontalScrollBar().setValue(400)
    canvas.verticalScrollBar().setValue(300)
    clicks = []
    canvas.clicked.connect(clicks.append)
    _drag(canvas, (200, 200), (120, 160))
    assert _scroll(canvas) == (480, 340) and clicks == []
    point = QPointF(101, 57)
    expected = canvas.page_point(point)
    _mouse(canvas, QEvent.MouseButtonPress, point, Qt.LeftButton)
    _mouse(canvas, QEvent.MouseButtonRelease, point, Qt.LeftButton, Qt.NoButton)
    assert len(clicks) == 1 and clicks[0] == expected
    assert canvas.page_pixel(point) == (int(expected.x()), int(expected.y()))


def test_a_space_typed_in_a_text_field_stays_there():
    holder = QWidget()
    layout = QHBoxLayout(holder)
    field = QLineEdit()
    canvas = PreviewWidget()
    layout.addWidget(field)
    layout.addWidget(canvas)
    holder.show()
    holder.activateWindow()
    field.setFocus()
    _app.processEvents()
    QTest.keyClick(field, Qt.Key_Space)
    assert field.text() == " " and not canvas.space_held()
    holder.close()
    holder.deleteLater()


def test_the_focus_leaving_lets_go_of_space(canvas):
    canvas.setFocus()
    QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))
    QApplication.sendEvent(canvas, QEvent(QEvent.FocusOut))
    assert not canvas.space_held()


def test_a_resize_keeps_the_middle(canvas):
    canvas.zoom_to(2.0)
    canvas.horizontalScrollBar().setValue(500)
    canvas.verticalScrollBar().setValue(300)
    middle = canvas.page_point(QRectF(canvas.viewport().rect()).center())
    canvas.resize(QSize(640, 360))
    _app.processEvents()
    after = canvas.page_point(QRectF(canvas.viewport().rect()).center())
    assert abs(after.x() - middle.x()) <= 1 and abs(after.y() - middle.y()) <= 1


# -- Rulers, grid, bounds ----------------------------------------------------------------------------------------------
def test_the_guides_are_hidden_at_first(canvas):
    assert not canvas.rulers_shown() and not canvas.grid_shown() and not canvas.bounds_shown()
    assert all(ruler.isHidden() for ruler in canvas.rulers)
    assert not canvas._grid_item.isVisible() and not canvas._bounds_item.isVisible()


@pytest.mark.parametrize("units", [MM, INCH])
@pytest.mark.parametrize("zoom", [0.3, 0.75, 1.0, 2.5, 8.0])
def test_the_rulers_show_paper_s_measure_at_every_zoom(canvas, units, zoom):
    canvas.set_units(units)
    canvas.set_rulers(True)
    _app.processEvents()
    canvas.zoom_to(zoom)
    canvas.horizontalScrollBar().setValue(canvas.horizontalScrollBar().maximum() // 3)
    per_unit = SCALE.px_per_mm * (MM_PER_INCH if units == INCH else 1)
    for horizontal in (True, False):
        ticks = canvas.ruler_ticks(horizontal)
        assert ticks and any(numbered for _place, _value, numbered in ticks)
        for place, value, _numbered in ticks:
            point = canvas.page_point(QPointF(place, 0) if horizontal else QPointF(0, place))
            measured = (point.x() if horizontal else point.y()) / per_unit
            assert measured == pytest.approx(value, abs=0.5 / (per_unit * canvas.transform().m11()) + 1e-6)
        numbered = [place for place, _value, numbered in ticks if numbered]
        assert all(b - a >= RULER_LABEL_ROOM_PX - 1 for a, b in zip(numbered, numbered[1:]))


def test_the_page_s_far_edge_is_its_printed_width():
    assert SCALE.printed_size_mm[0] == pytest.approx(277.0)
    canvas = PreviewWidget()
    canvas.resize(600, 400)
    canvas.show()
    canvas.show_page({Version.PAGE: _noise(PAGE, 0)}, _noise((PAGE[0], 60), 9))
    canvas.set_rulers(True)
    canvas.zoom_to(1.0)
    canvas.horizontalScrollBar().setValue(canvas.horizontalScrollBar().maximum())
    ticks = {value: place for place, value, _numbered in canvas.ruler_ticks(True)}
    edge = canvas.mapFromScene(QPointF(PAGE[0], 0)).x()
    assert 270 in ticks and abs(ticks[270] + (277 - 270) * SCALE.px_per_mm - edge) <= 1
    canvas.close()
    canvas.deleteLater()


def test_ruler_steps_leave_room_for_their_numbers():
    for per_unit in (0.2, 1.0, 3.7, 12.0, 80.0, 400.0):
        label, tick = ruler_steps(per_unit, MM)
        assert label * per_unit >= RULER_LABEL_ROOM_PX or label == 1000
        assert tick is None or (tick < label and abs(label / tick - round(label / tick)) < 1e-9)


def test_the_grid_is_every_centimeter_on_paper(canvas):
    canvas.set_grid(True)
    assert canvas._grid_item.isVisible()
    path = canvas._grid_item.path()
    xs = sorted({round(path.elementAt(i).x, 6) for i in range(path.elementCount()) if path.elementAt(i).y == 0})
    assert xs[0] == pytest.approx(10 * SCALE.px_per_mm)
    assert np.diff(xs) == pytest.approx(10 * SCALE.px_per_mm)
    assert len(xs) == 27  # 277 mm across: a line every 10 mm inside it
    canvas.set_units(INCH)
    path = canvas._grid_item.path()
    xs = sorted({round(path.elementAt(i).x, 6) for i in range(path.elementCount()) if path.elementAt(i).y == 0})
    assert xs[0] == pytest.approx(MM_PER_INCH * SCALE.px_per_mm) and len(xs) == 10


def test_the_bounds_are_the_sheet_with_the_page_in_the_middle_of_its_margins(canvas):
    canvas.set_bounds(True)
    sheet = canvas.sheet_rect()
    ppm = SCALE.px_per_mm
    assert sheet.width() == pytest.approx(297 * ppm) and sheet.height() == pytest.approx(210 * ppm)
    assert -sheet.left() == pytest.approx(10 * ppm)  # the page fills the width inside the margins
    assert -sheet.top() == pytest.approx((210 - PAGE[1] / ppm) / 2 * ppm)
    assert canvas._bounds_item.isVisible() and canvas._margin_item.isVisible()
    assert canvas.sceneRect().contains(sheet)
    # The legend, which prints on a sheet of its own, goes under this one.
    assert canvas._legend_item.pos().y() >= sheet.bottom()
    canvas.set_bounds(False)
    assert not canvas.sceneRect().contains(sheet) and canvas._legend_item.pos().y() == PAGE[1] + 24


# -- Layers ------------------------------------------------------------------------------------------------------------
def test_the_layers_are_stacked_in_order(canvas):
    heights = [canvas.layers[name].zValue() for name in (ARTWORK, "guides", ANNOTATION, DIAGNOSTICS, INTERACTION)]
    assert heights == sorted(heights) and len(set(heights)) == len(LAYERS)
    assert canvas._page_item.parentItem() is canvas.layers[ARTWORK]


def test_an_overlay_is_drawn_over_the_page_without_drawing_it_again(canvas):
    canvas._sharpen()
    key = canvas.page_pixmap_key()
    mask = QImage(PAGE[0] // 2, PAGE[1] // 2, QImage.Format_ARGB32)
    mask.fill(Qt.transparent)
    canvas.set_overlay(ANNOTATION, "highlight", mask)
    item = canvas.overlay(ANNOTATION, "highlight")
    assert item.parentItem() is canvas.layers[ANNOTATION]
    assert item.sceneBoundingRect() == QRectF(0, 0, *PAGE)
    assert canvas.page_pixmap_key() == key
    canvas.set_overlay(ANNOTATION, "highlight", None)
    assert canvas.overlay(ANNOTATION, "highlight") is None
    canvas.set_overlay(DIAGNOSTICS, "small", mask)
    canvas.clear_page()
    assert canvas.overlay(DIAGNOSTICS, "small") is None and not canvas.has_page()


# -- In the window -----------------------------------------------------------------------------------------------------
def _picture_file(path, size=(120, 90)):
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    _app.processEvents()
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _settle(window):
    deadline = time.monotonic() + 60
    _app.processEvents()
    while not window.jobs.is_idle():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.003)
    _app.processEvents()


def test_the_window_shows_the_picture_as_its_original_view(window, tmp_path):
    window.open_path(_picture_file(tmp_path / "noise.png"))
    window.generate_preview()
    _settle(window)
    assert window.preview.has_page() and ORIGINAL in window.preview.views()
    window.commands["show_original"].trigger()
    assert window.preview.view() == ORIGINAL and window.version_bar.version() == ORIGINAL
    assert window.commands["show_original"].shortcut().toString() == "Ctrl+4"
    # The picture's own pixels, as it was opened.
    image = window.preview.view_image(ORIGINAL).convertToFormat(QImage.Format_RGB888)
    source = window.current_image_bgr[:, :, ::-1]
    assert (image.width(), image.height()) == (source.shape[1], source.shape[0])
    row = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.bytesPerLine())[: 3 * image.width()]
    assert np.array_equal(row.reshape(-1, 3), source[0])
    window.commands["show_page"].trigger()
    assert window.preview.view() == Version.PAGE
    # Closed: the picture goes with the page.
    window.close_document()
    assert window._original_kept is None and not window.preview.has_page()


def test_the_guides_are_view_commands_kept_with_the_layout(window):
    for command_id in ("rulers", "grid", "page_bounds"):
        assert window.commands[command_id].isCheckable() and not window.commands[command_id].isChecked()
    window.commands["rulers"].trigger()
    window.commands["grid"].trigger()
    assert window.preview.rulers_shown() and window.preview.grid_shown()
    assert window.commands["rulers"].shortcut().toString() == "Ctrl+R"
    window.preferences.set_value(main_window.preferences.UNITS_KEY, "Inches")
    assert window.preview.units() == INCH
    window.reset_layout()
    assert not window.preview.rulers_shown() and not window.preview.grid_shown()

"""Comparing views (I2): split by a divider up or across the page, side by side in two canvases that move together, with
any view or the page made before (from the results kept); views of other pixels mapped onto the page's; and nothing
made by comparing."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from tessellatum.app.jobs import PREVIEW  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.preview_widget import (  # noqa: E402
    COMPARE_OFF,
    ORIGINAL,
    PREVIOUS,
    SIDE_BY_SIDE,
    SPLIT_HORIZONTAL,
    SPLIT_VERTICAL,
    PreviewWidget,
    SideBySide,
)

_app = QApplication.instance() or QApplication([])

PAGE = (1100, 734)


def _noise(size, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


@pytest.fixture
def canvas():
    widget = PreviewWidget()
    widget.resize(500, 400)
    widget.show()
    views = {version: _noise(PAGE, index) for index, version in enumerate(Version)}
    views[ORIGINAL] = _noise((2200, 1468), 7)
    widget.show_page(views, _noise((PAGE[0], 60), 9), print_scale=2)
    _app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    _app.processEvents()


def _mouse(widget, kind, point, button, buttons=None):
    event = QMouseEvent(kind, QPointF(point), widget.viewport().mapToGlobal(QPointF(point)), button,
                        button if buttons is None else buttons, Qt.NoModifier)
    QApplication.sendEvent(widget.viewport(), event)


# -- Split -------------------------------------------------------------------------------------------------------------
def test_a_split_shows_the_other_view_past_the_divider(canvas):
    canvas.set_split(ORIGINAL)
    canvas._sharpen()
    assert canvas.split_view() == ORIGINAL and canvas._split_clip.isVisible() and canvas._divider.isVisible()
    assert canvas._split_clip.rect() == QRectF(PAGE[0] / 2, 0, PAGE[0] / 2, PAGE[1])
    # Its own pixels, twice the page's, over exactly the page's place.
    assert canvas._split_item.sceneBoundingRect() == QRectF(0, 0, *PAGE)
    canvas.set_split(ORIGINAL, horizontal=True)
    assert canvas._split_clip.rect() == QRectF(0, PAGE[1] / 2, PAGE[0], PAGE[1] / 2)
    canvas.set_split(None)
    assert not canvas._split_clip.isVisible() and not canvas._divider.isVisible()


def test_a_split_with_a_view_the_page_has_not_is_none(canvas):
    canvas.set_split(PREVIOUS)
    assert canvas.split_view() is None and not canvas._divider.isVisible()


def test_the_divider_is_dragged_and_a_drag_of_it_neither_pans_nor_clicks(canvas):
    canvas.zoom_to(1.0)
    canvas.set_split(Version.TINTED)
    clicks = []
    canvas.clicked.connect(clicks.append)
    scroll = (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value())
    at = canvas.mapFromScene(QPointF(PAGE[0] / 2, PAGE[1] / 2))
    start = QPointF(at.x() + 3, 200)
    end = QPointF(start.x() - 100, 210)
    _mouse(canvas, QEvent.MouseButtonPress, start, Qt.LeftButton)
    assert canvas.viewport().cursor().shape() == Qt.SplitHCursor
    _mouse(canvas, QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton)
    _mouse(canvas, QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)
    expected = canvas.page_point(end).x() / PAGE[0]
    assert canvas.split_fraction() == pytest.approx(expected)
    assert (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value()) == scroll and clicks == []
    assert canvas._split_clip.rect().left() == pytest.approx(expected * PAGE[0])
    canvas.set_split_fraction(3.0)
    assert canvas.split_fraction() == 1.0


def test_a_view_of_its_own_is_mapped_onto_the_page(canvas):
    canvas.set_view_image(PREVIOUS, _noise((550, 367), 3))  # made at another size
    canvas.set_split(PREVIOUS)
    canvas._sharpen()
    rect = canvas._split_item.sceneBoundingRect()
    assert rect.width() == pytest.approx(PAGE[0]) and rect.height() == pytest.approx(PAGE[1])
    canvas.set_view_image(PREVIOUS, None)
    assert canvas.split_view() is None and PREVIOUS not in canvas.views()


def test_the_captions_name_each_side(canvas):
    canvas.set_split(ORIGINAL)
    canvas.set_captions("Page", "Original")
    assert canvas.captions() == ("Page", "Original")
    first, second = canvas._captions
    assert first.x() < 20 and second.geometry().right() > canvas.viewport().width() - 20
    canvas.set_captions(None)
    assert canvas.captions() == ("", "")


# -- Side by side ------------------------------------------------------------------------------------------------------
def test_a_mirror_shares_the_images_and_the_view(canvas):
    other = PreviewWidget()
    other.resize(500, 400)
    other.show()
    canvas.zoom_to(1.3)
    canvas.horizontalScrollBar().setValue(200)
    other.mirror(canvas)
    assert other.views() == canvas.views()
    assert all(other.view_image(v).cacheKey() == canvas.view_image(v).cacheKey() for v in canvas.views())
    assert other.view_state() == canvas.view_state()
    other.close()
    other.deleteLater()


@pytest.mark.parametrize("width", [700, 701])
def test_side_by_side_the_two_are_exactly_as_wide(width):
    first, second = QWidget(), QWidget()
    pair = SideBySide(first, second)
    pair.resize(width, 300)
    pair.set_both(True)
    assert first.width() == second.width() == width // 2 and second.geometry().right() == width - 1
    pair.set_both(False)
    assert first.width() == width and second.isHidden()


# -- In the window -----------------------------------------------------------------------------------------------------
def _picture_file(path, size=(120, 90)):
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1400, 860)
    window.show()
    _app.processEvents()
    window.open_path(_picture_file(tmp_path / "noise.png"))
    window.generate_preview()
    _settle(window)
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _settle(window):
    deadline = time.monotonic() + 60
    _app.processEvents()
    while not window.jobs.is_idle() or window._redraw_timer.isActive():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.003)
    _app.processEvents()


def _second_page(window):
    window.controls.controls["num_colors"].step(-3)
    _app.processEvents()
    window._record_settings()
    window.generate_preview()
    _settle(window)


def _aligned(window):
    """The page's points fall at the same place in the two canvases, to half a page pixel."""
    zoom = window.preview.transform().m11()
    for x, y in ((0.0, 0.0), (37.5, 12.25), (80.0, 60.0), (119.0, 89.0)):
        a = window.preview.mapFromScene(QPointF(x, y))
        b = window.compare_view.mapFromScene(QPointF(x, y))
        assert abs(a.x() - b.x()) / zoom <= 0.5 and abs(a.y() - b.y()) / zoom <= 0.5


def test_side_by_side_the_two_move_together_and_make_nothing(window, monkeypatch):
    started = []
    monkeypatch.setattr(window, "_start_preview", lambda *a, **k: started.append(a))
    window.commands["compare_side"].trigger()
    _app.processEvents()
    assert window.compare_mode() == SIDE_BY_SIDE and window.compare_view.isVisible()
    assert window.compare_view.view() == ORIGINAL and window.compare_button.isChecked()
    assert window.preview.width() == window.compare_view.width()
    assert window.compare_view.captions()[0] == "Original" and window.preview.captions()[0] == "Page"
    for zoom in (0.6, 1.0, 2.5, 4.0, 7.0):
        window.preview.zoom_to(zoom)
        window.preview.horizontalScrollBar().setValue(window.preview.horizontalScrollBar().maximum() // 3)
        _app.processEvents()
        assert window.compare_view.view_state() == window.preview.view_state()
        _aligned(window)
    # The second leads as well: its wheel zooms both.
    anchor = QPointF(100, 100)
    wheel = QWheelEvent(anchor, window.compare_view.viewport().mapToGlobal(anchor), QPoint(), QPoint(0, -120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(window.compare_view.viewport(), wheel)
    assert window.preview.view_state() == window.compare_view.view_state()
    _aligned(window)
    # A view chosen on the bar shows on the first; the second keeps its own.
    window.commands["show_tinted"].trigger()
    assert window.preview.view() == Version.TINTED and window.compare_view.view() == ORIGINAL
    window.commands["compare_with_completed"].trigger()
    assert window.compare_view.view() == Version.COMPLETED
    window.commands["compare_off"].trigger()
    assert window.compare_view.isHidden() and not window.compare_button.isChecked()
    assert window.preview.captions() == ("", "")
    assert started == [] and window.jobs.current(PREVIEW) is None  # the zoom may lay the page out to print, as ever


def test_a_split_in_the_window(window, monkeypatch):
    started = []
    monkeypatch.setattr(window, "_start_preview", lambda *a, **k: started.append(a))
    window.commands["compare_split"].trigger()
    assert window.preview.split_view() == ORIGINAL and window.compare_view.isHidden()
    assert window.preview.captions() == ("Page", "Original")
    window.commands["compare_split_h"].trigger()
    assert window.preview._split_horizontal and window.preview.split_view() == ORIGINAL
    window.commands["compare_with_tinted"].trigger()
    assert window.preview.split_view() == Version.TINTED
    window.set_compare(COMPARE_OFF)
    assert window.preview.split_view() is None
    assert started == []


def test_the_previous_version_comes_from_the_results_kept(window, monkeypatch):
    assert not window.commands["compare_with_previous"].isEnabled()  # the first page: none before it
    first = window.current_page
    _second_page(window)
    assert window.current_page is not first
    assert window.commands["compare_with_previous"].isEnabled()
    assert window.previous_label() == "Previous version (before: Colors)"
    hits = window.results.hits
    started = []
    monkeypatch.setattr(window, "_start_preview", lambda *a, **k: started.append(a))
    window.commands["compare_with_previous"].trigger()  # with no comparison: side by side
    _app.processEvents()
    assert window.compare_mode() == SIDE_BY_SIDE and window.compare_view.view() == PREVIOUS
    assert window.compare_view.captions()[0] == "Previous version (before: Colors)"
    image = window.compare_view.view_image(PREVIOUS)
    assert (image.width(), image.height()) == first.page.size
    assert window.results.hits == hits and started == []  # looked at, not counted as a hit, and nothing made
    # In the version shown.
    window.commands["show_completed"].trigger()
    assert window.preview.view_image(PREVIOUS).cacheKey() != image.cacheKey()
    window.commands["compare_split"].trigger()
    assert window.preview.split_view() == PREVIOUS
    # Gone from the results: none to compare with.
    window.results.clear()
    window._show_previous_state()
    assert not window.commands["compare_with_previous"].isEnabled() and window.compare_with() == ORIGINAL


def test_another_picture_has_no_previous_version(window, tmp_path):
    _second_page(window)
    assert window.commands["compare_with_previous"].isEnabled()
    window.open_path(_picture_file(tmp_path / "other.png", (100, 80)))
    window.generate_preview()
    _settle(window)
    assert not window.commands["compare_with_previous"].isEnabled()


def test_closing_ends_the_comparison_s_page(window):
    window.set_compare(SIDE_BY_SIDE)
    window.close_document()
    assert not window.compare_view.has_page() and not window.preview.has_page()
    window.set_compare(SPLIT_VERTICAL)
    assert window.preview.split_view() is None
    window.set_compare(SPLIT_HORIZONTAL)
    assert window.compare_view.isHidden()

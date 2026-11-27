"""The region inspector (I4): a region picked by a click in any view, at any zoom; measured exactly on paper; two-way
with the palette; the keyboard path through a color's regions; the pick kept across a page with the same regions and
dropped otherwise; and the lookup's speed on a dense page."""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import measures, pipeline  # noqa: E402
from tessellatum.core.labels import Label  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.preview_widget import ANNOTATION, VIEWS  # noqa: E402
from tessellatum.gui.region_inspector import MM_PER_POINT, region_info  # noqa: E402

_app = QApplication.instance() or QApplication([])


# -- Measures ----------------------------------------------------------------------------------------------------------
def test_region_stats_give_exact_areas_and_boxes():
    ids = np.full((40, 60), -1, dtype=np.int32)
    ids[5:15, 10:30] = 0  # a 20 x 10 rectangle
    ids[20:40, 0:5] = 2  # 5 x 20, and one pixel more far off: its box spans both
    ids[0, 59] = 2
    yy, xx = np.mgrid[:40, :60]
    disc = (xx - 45) ** 2 + (yy - 25) ** 2 <= 8 ** 2
    ids[disc] = 3
    stats = measures.region_stats(ids)
    assert stats.areas[0] == 200 and stats.box(0) == (10, 5, 30, 15)
    assert 1 not in stats and stats.areas[1] == 0
    assert stats.areas[2] == 101 and stats.box(2) == (0, 0, 60, 40)
    assert stats.areas[3] == disc.sum() and stats.box(3) == (37, 17, 54, 34)
    assert measures.widest_brush_px(ids, 0, stats.box(0)) == pytest.approx(10.0, abs=1.0)
    assert measures.widest_brush_px(ids, 3, stats.box(3)) == pytest.approx(16.0, abs=1.5)  # a 17 px disc, on the grid
    assert measures.widest_brush_px(ids, 2, stats.box(2)) == pytest.approx(5.0, abs=1.0)


class _Model:
    def __init__(self, ids, colors, labels):
        self.region_id_map, self.region_color, self.labels = ids, np.asarray(colors), labels


def test_region_info_on_paper():
    ids = np.full((40, 60), -1, dtype=np.int32)
    ids[5:15, 10:30] = 0
    ids[20:30, 40:50] = 1
    ids[30:32, 0:3] = 2
    labels = [
        Label(region_id=0, text="1", font_size=20.0, box=(15, 6, 25, 14)),
        Label(region_id=0, text="1", font_size=10.0, box=(0, 0, 1, 1), side=1),
        Label(region_id=1, text="2", font_size=8.0, box=(0, 0, 1, 1), leader=((0, 0), (45, 25))),
    ]
    model = _Model(ids, [0, 1, 0], labels)
    stats = measures.region_stats(ids)
    px_per_mm = 4.0
    info = region_info(model, 0, stats, px_per_mm, ["white", "rose"], 10.0)
    assert info.color_number == 1 and info.color_name == "white"
    assert info.area_mm2 == pytest.approx(200 / 16) and info.width_mm == 5.0 and info.height_mm == 2.5
    assert info.widest_brush_mm == 2.5 and info.placement == "inside" and info.sides == 1
    assert info.number_pt == pytest.approx(20 / 4 / MM_PER_POINT)
    assert "Also written on 1 other side" in info.lines()
    assert region_info(model, 1, stats, px_per_mm, ["white", "rose"], 9.0).placement == "on a leader"
    none = region_info(model, 2, stats, px_per_mm, [], 2.0)
    assert none.placement == "none" and none.number_pt is None and "Number: none" in none.lines()[-1]


# -- In the window -----------------------------------------------------------------------------------------------------
def _picture_file(path, size=(160, 120)):
    """Big flat blocks of color, so the regions are large enough to click into."""
    picture = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    colors = [(230, 60, 50), (40, 90, 200), (250, 220, 60), (30, 160, 70), (240, 240, 240), (90, 40, 120)]
    for index, color in enumerate(colors):
        x, y = (index % 3) * size[0] // 3, (index // 3) * size[1] // 2
        picture[y: y + size[1] // 2, x: x + size[0] // 3] = color
    Image.fromarray(picture).save(path)
    return path


def _settle(window):
    deadline = time.monotonic() + 60
    _app.processEvents()
    while not window.jobs.is_idle():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.003)
    _app.processEvents()


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1400, 860)
    window.show()
    _app.processEvents()
    window.open_path(_picture_file(tmp_path / "blocks.png"))
    window.generate_preview()
    _settle(window)
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _largest_region(window):
    model = window.current_page.model
    stats = measures.region_stats(model.region_id_map)
    return int(np.argmax(stats.areas)), stats


def _viewport_point_of(view, window, region):
    """A viewport point over a pixel of ``region``, in the middle of its widest part."""
    ids = window.current_page.model.region_id_map
    import cv2
    inside = np.pad(ids == region, 1).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    width, height = view.page_size()
    page_point = QPointF((x + 0.5) * width / ids.shape[1], (y + 0.5) * height / ids.shape[0])
    view.center_on_rect(view.mapToScene(view.viewport().rect()).boundingRect().translated(0, 0))
    view.centerOn(page_point)
    _app.processEvents()
    return view.mapFromScene(page_point)


@pytest.mark.parametrize("zoom", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_a_click_picks_the_region_under_it_at_every_zoom_and_view(window, zoom):
    region, _stats = _largest_region(window)
    view = window.preview
    for name in VIEWS:
        window.preview.show_version(name)
        view.zoom_to(zoom)
        _app.processEvents()
        point = _viewport_point_of(view, window, region)
        window.select_region(None)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        _app.processEvents()
        assert window.selected_region() == region, (name, zoom)
        assert window.region_at(view.page_point(QPointF(point))) == region


def test_picking_a_region_picks_its_color_and_measures_it(window):
    region, stats = _largest_region(window)
    model = window.current_page.model
    window.select_region(region)
    color = int(model.region_color[region])
    assert window.inspector.selected() == color
    info = window.region_inspector.info()
    px_per_mm = window._page_px_per_mm()
    assert info.region_id == region and info.color_number == color + 1
    assert info.area_mm2 == pytest.approx(stats.areas[region] / px_per_mm ** 2)
    assert window.region_inspector.isVisible()
    assert window.preview.overlay(ANNOTATION, "region") is not None
    # Another color picked in the palette lets the region go; its own color keeps it.
    window.inspector.select(color)
    assert window.selected_region() == region
    other = next(i for i in range(window.inspector.color_count()) if i != color)
    window.inspector.select(other)
    assert window.selected_region() is None and window.preview.overlay(ANNOTATION, "region") is None
    # A click off the regions lets go too.
    window.select_region(region)
    window.pick_at(QPointF(-5, -5))
    assert window.selected_region() is None


def test_the_keyboard_goes_through_a_colors_regions(window):
    region, stats = _largest_region(window)
    color = int(window.current_page.model.region_color[region])
    window.inspector.select(color)
    window.inspector.list.setFocus()
    QTest.keyClick(window.inspector.list, Qt.Key_Return)
    listed = window.region_inspector.listed()
    expected = measures.regions_of_color(window.current_page.model.region_id_map,
                                         window.current_page.model.region_color, color).tolist()
    assert listed == expected and window.selected_region() == listed[0]
    assert window.region_inspector.list.hasFocus()
    center = window.preview.mapToScene(window.preview.viewport().rect().center())
    assert window._page_rect(stats.box(listed[0])).adjusted(-2, -2, 2, 2).contains(center) or \
        window.preview.horizontalScrollBar().maximum() == 0
    if len(listed) > 1:
        QTest.keyClick(window.region_inspector.list, Qt.Key_Down)
        assert window.selected_region() == listed[1]
    QTest.keyClick(window.region_inspector.list, Qt.Key_Escape)
    assert window.inspector.list.hasFocus()


def test_the_pick_survives_a_page_with_the_same_regions_only(window):
    region, _stats = _largest_region(window)
    window.select_region(region)
    window._inspect_page()  # the same page again, as a redraw gives it
    _settle(window)
    assert window.selected_region() == region
    window.controls.controls["num_colors"].step(-3)  # another page, its regions drawn otherwise
    _app.processEvents()
    window._record_settings()
    window.generate_preview()
    _settle(window)
    old = window._region_page.model.region_id_map
    same = np.array_equal(old, window.current_page.model.region_id_map)
    assert (window.selected_region() == region) == same
    # A page whose regions are drawn otherwise: the same id is another region, so the pick goes.
    window.select_region(region)
    other = SimpleNamespace(model=SimpleNamespace(region_id_map=np.roll(old, 1, axis=1)))
    assert window._kept_region(other) is None
    assert window._kept_region(window.current_page) == region
    window.document.set_result(None)
    _settle(window)
    assert window.selected_region() is None and window.region_inspector.info() is None


def test_the_lookup_is_quick_on_a_dense_page():
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 12000, (2216, 1478), dtype=np.int32).T.copy()  # a dense page's region map, its size
    start = time.perf_counter()
    stats = measures.region_stats(ids)
    stats_s = time.perf_counter() - start
    times = []
    for region in rng.integers(0, 12000, 50):
        start = time.perf_counter()
        _ = int(ids[700, 1000])
        measures.widest_brush_px(ids, int(region), stats.box(int(region)))
        times.append(time.perf_counter() - start)
    # Off the GUI thread, once a page: well under a second. A pick: a frame's time.
    assert stats_s < 2.0
    assert np.median(times) < 0.05


def test_the_region_is_outlined_over_its_box_alone_in_both_canvases(window):
    region, stats = _largest_region(window)
    window.commands["compare_side"].trigger()
    window.select_region(region)
    for view in (window.preview, window.compare_view):
        item = view.overlay(ANNOTATION, "region")
        assert item is not None
        x0, y0, x1, y1 = stats.box(region)
        ids = window.current_page.model.region_id_map
        assert item.pixmap().width() == min(x1 + 1, ids.shape[1]) - max(x0 - 1, 0)
        shown = item.sceneBoundingRect()
        expected = window._page_rect((max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 1, ids.shape[1]),
                                      min(y1 + 1, ids.shape[0])))
        assert abs(shown.x() - expected.x()) < 1e-6 and abs(shown.width() - expected.width()) < 1e-6

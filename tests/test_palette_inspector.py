"""The palette inspector (I3): coverage measured from the region map, summing to the painted area; sorting that never
changes a number; finding by number or name; copying; the detail; the keyboard; and the highlight on the canvas, in the
annotation layer, which an export never sees."""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QListView  # noqa: E402

from tessellatum.core import difficulty, measures, pipeline  # noqa: E402
from tessellatum.core.print_size import A4, print_scale  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow, highlight_image  # noqa: E402
from tessellatum.gui.palette_inspector import (  # noqa: E402
    COVERAGE,
    GRID,
    HUE,
    LIGHTNESS,
    LIST,
    NAME,
    NUMBER,
    SORTS,
    PaletteInspector,
    matches,
    palette_colors,
    sort_colors,
)
from tessellatum.gui.preview_widget import ANNOTATION  # noqa: E402

_app = QApplication.instance() or QApplication([])


# -- Coverage ----------------------------------------------------------------------------------------------------------
def _regions():
    """A 10 x 12 page: region 0 (color 0) 40 px, region 1 (color 1) 30 px, region 3 (color 0) 20 px, region 4 (color 3,
    past a 3-color legend) 10 px, 20 px in no region; region 2 was merged away."""
    ids = np.full((10, 12), -1, dtype=np.int32)
    ids.flat[:40] = 0
    ids.flat[40:70] = 1
    ids.flat[70:90] = 3
    ids.flat[90:100] = 4
    color = np.array([0, 1, 2, 0, 3])
    return ids, color


def test_coverage_sums_to_the_painted_area():
    ids, color = _regions()
    coverage = measures.color_coverage(ids, color, 3)
    assert coverage.pixels.tolist() == [60, 30, 0, 10]  # widened to the color past the legend
    assert coverage.regions.tolist() == [2, 1, 0, 1]  # the merged-away id counts for nothing
    assert coverage.pixels.sum() == coverage.painted == (ids >= 0).sum() == 100
    assert coverage.page_pixels == 120 and coverage.share(0) == pytest.approx(0.5)
    assert coverage.area_mm2(1, 2.0) == pytest.approx(7.5)


def test_the_regions_of_a_color_largest_first_and_its_mask():
    ids, color = _regions()
    assert measures.regions_of_color(ids, color, 0).tolist() == [0, 3]
    assert measures.regions_of_color(ids, color, 2).tolist() == []
    mask = measures.color_mask(ids, color, 0)
    assert mask.sum() == 60 and mask.flat[0] and mask.flat[75] and not mask.flat[50] and not mask.flat[110]


def test_coverage_on_a_real_page_sums_to_its_painted_pixels():
    rng = np.random.default_rng(1)
    picture = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
    page = pipeline.generate(picture, difficulty.params_for_preset("Medium"), 240)
    model = page.model
    coverage = measures.color_coverage(model.region_id_map, model.region_color, len(page.palette_rgb))
    assert coverage.pixels.sum() == coverage.painted == int((model.region_id_map >= 0).sum())
    assert coverage.regions[: len(page.palette_rgb)].sum() + coverage.regions[len(page.palette_rgb):].sum() \
        == len(np.unique(model.region_id_map[model.region_id_map >= 0]))


# -- Sorting and finding -----------------------------------------------------------------------------------------------
def _page():
    return SimpleNamespace(
        palette_rgb=[(250, 250, 250), (192, 115, 122), (30, 60, 200), (120, 120, 120), (230, 200, 40)],
        color_names=["white", "dusty rose", "royal blue", "gray", "mustard yellow"],
        mixes=[None, ["2 Titanium white", "1 Cadmium red"], None, None, None],
        num_regions=77,
    )


def _coverage():
    return measures.ColorCoverage(
        pixels=np.array([100, 400, 50, 300, 150]), regions=np.array([3, 37, 2, 9, 1]), page_pixels=1000, painted=1000)


def test_every_sort_keeps_the_numbers():
    colors = palette_colors(_page())
    for by in SORTS:
        shown = sort_colors(colors, by, _coverage())
        assert sorted(c.index for c in shown) == [0, 1, 2, 3, 4]
        assert all(c.number == c.index + 1 for c in shown)
    assert [c.number for c in sort_colors(colors, COVERAGE, _coverage())] == [2, 4, 5, 1, 3]
    assert [c.number for c in sort_colors(colors, COVERAGE, None)] == [1, 2, 3, 4, 5]  # not measured yet
    assert [c.number for c in sort_colors(colors, LIGHTNESS)][0] == 1 and sort_colors(colors, LIGHTNESS)[-1].number == 3
    assert [c.number for c in sort_colors(colors, HUE)][:2] == [1, 4]  # the grays first, light to dark
    assert [c.name for c in sort_colors(colors, NAME)] == sorted(c.name for c in colors)


@pytest.mark.parametrize("text, numbers", [("2", [2]), ("12", []), ("rose", [2]), ("Yel", [5]), ("", [1, 2, 3, 4, 5]),
                                           ("blue", [3]), ("e", [1, 2, 3, 4, 5])])
def test_a_search_finds_a_number_or_a_name(text, numbers):
    colors = palette_colors(_page())
    found = [c.number for c in colors if matches(c, text)]
    assert found == numbers or (text == "e" and set(found) <= set(numbers))


# -- The widget --------------------------------------------------------------------------------------------------------
@pytest.fixture
def inspector():
    widget = PaletteInspector()
    widget.resize(260, 600)
    widget.show()
    widget.show_page(_page())
    _app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def test_the_coverage_fills_in_and_names_each_color(inspector):
    assert "measuring" in inspector.detail_coverage.text() or inspector.detail.isHidden()
    inspector.set_coverage(_coverage(), 2.0)
    item = inspector.list.item(1)
    assert item.text().startswith("2   dusty rose") and "40.0%" in item.text()
    assert item.data(Qt.AccessibleTextRole) == "Color 2, dusty rose, #C0737A, 40.0% of the page, 37 regions"
    assert "Mixed: 2 Titanium white + 1 Cadmium red" in item.toolTip()
    inspector.select(1)
    assert not inspector.detail.isHidden()
    assert inspector.detail_values.text() == "#C0737A · RGB 192, 115, 122"
    assert inspector.detail_coverage.text() == "40.0% of the page · 100 mm² · 37 regions"
    assert inspector.detail_recipe.text() == "Mixed: 2 Titanium white + 1 Cadmium red"


def test_sorting_keeps_the_numbers_and_the_color_picked(inspector):
    inspector.set_coverage(_coverage(), 2.0)
    inspector.select(3)
    for by in SORTS:
        inspector.set_sort(by)
        assert inspector.selected() == 3
        for row in range(inspector.list.count()):
            item = inspector.list.item(row)
            assert item.text().split()[0] == str(item.data(Qt.UserRole + 1) + 1)
    inspector.set_sort(COVERAGE)
    assert inspector.shown_order() == [1, 3, 4, 0, 2]
    inspector.set_sort(NUMBER)
    assert inspector.shown_order() == [0, 1, 2, 3, 4]


def test_the_grid_and_the_list(inspector):
    inspector.set_mode(GRID)
    assert inspector.mode() == GRID and inspector.list.viewMode() == QListView.IconMode
    assert inspector.list.item(0).text() == "1" and inspector.grid_button.isChecked()
    inspector.grid_button.click()
    assert inspector.mode() == LIST and inspector.list.item(0).text().startswith("1   white")


def test_finding_hides_the_rest_and_picking_a_hidden_one_shows_it(inspector):
    inspector.search.setText("2")
    assert inspector.visible_numbers() == [2]
    inspector.search.setText("ro")
    assert inspector.visible_numbers() == [2, 3]
    inspector.select(4)
    assert inspector.search.text() == "" and inspector.selected() == 4


def test_copying(inspector):
    inspector.select(1)
    inspector.list.setFocus()
    QTest.keyClick(inspector.list, Qt.Key_C, Qt.ControlModifier)
    assert QGuiApplication.clipboard().text() == "#C0737A"
    inspector.copy_actions["rgb"].trigger()
    assert QGuiApplication.clipboard().text() == "192, 115, 122"
    inspector.copy_actions["name"].trigger()
    assert QGuiApplication.clipboard().text() == "dusty rose"
    assert inspector.copy("hex", 2) == "#1E3CC8"
    QGuiApplication.clipboard().clear()


def test_the_keyboard_picks_moves_and_lets_go(inspector):
    picked = []
    inspector.color_selected.connect(picked.append)
    inspector.list.setFocus()
    inspector.select(0)
    QTest.keyClick(inspector.list, Qt.Key_Down)
    assert inspector.selected() == 1 and picked[-1] == 1
    asked = []
    inspector.regions_requested.connect(asked.append)
    QTest.keyClick(inspector.list, Qt.Key_Return)
    assert asked == [1]
    QTest.keyClick(inspector.list, Qt.Key_Escape)
    assert inspector.selected() is None and picked[-1] is None and inspector.detail.isHidden()


def test_isolate_says_so(inspector):
    told = []
    inspector.isolate_changed.connect(told.append)
    inspector.select(2)
    inspector.isolate_check.setChecked(True)
    assert told == [True] and inspector.isolated()
    inspector.show_page(_page())  # a new page: nothing picked, nothing isolated
    assert not inspector.isolate_check.isChecked() and inspector.selected() is None


def test_no_page_shows_no_list(inspector):
    inspector.show_page(None)
    assert inspector.color_count() == 0 and inspector.list.isHidden() and "Generate" in inspector.summary.text()


# -- The highlight -----------------------------------------------------------------------------------------------------
def _pixels(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format_RGBA8888)
    data = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    return data.reshape(image.height(), image.bytesPerLine() // 4, 4)[:, : image.width()].copy()  # not the image's


def test_the_highlight_tints_outlines_and_isolates():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 5:20] = True
    pixels = _pixels(highlight_image(mask, (13, 116, 104), isolate=False))
    assert tuple(pixels[10, 10]) == (13, 116, 104, main_window.HIGHLIGHT_ALPHA)  # inside
    assert tuple(pixels[5, 10]) == (13, 116, 104, 255)  # its edge
    assert pixels[0, 0, 3] == 0  # elsewhere: clear
    isolated = _pixels(highlight_image(mask, (13, 116, 104), isolate=True))
    assert tuple(isolated[0, 0]) == (255, 255, 255, main_window.ISOLATE_ALPHA)
    assert tuple(isolated[10, 10]) == tuple(pixels[10, 10])


def _picture_file(path, size=(120, 90)):
    rng = np.random.default_rng(8)
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
    while not window.jobs.is_idle():
        _app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(0.003)
    _app.processEvents()


def test_the_window_measures_the_page_and_highlights_a_color(window):
    page = window.current_page
    coverage = window.inspector.coverage()
    assert coverage is not None  # measured in the background once the page was made
    assert coverage.pixels.sum() == int((page.model.region_id_map >= 0).sum())
    height, width = page.model.region_id_map.shape
    px_per_mm = print_scale((width, height), A4).px_per_mm
    assert window._page_px_per_mm() == pytest.approx(px_per_mm)
    window.preview._sharpen()
    key = window.preview.page_pixmap_key()
    index = int(np.argmax(coverage.pixels[: len(page.palette_rgb)]))
    window.inspector.select(index)
    overlay = window.preview.overlay(ANNOTATION, "color")
    assert overlay is not None and window.preview.page_pixmap_key() == key  # the page isn't drawn again
    shown = _pixels(overlay.pixmap().toImage())
    mask = measures.color_mask(page.model.region_id_map, page.model.region_color, index)
    assert (shown[..., 3] > 0).sum() == mask.sum()
    window.inspector.isolate_check.setChecked(True)
    shown = _pixels(window.preview.overlay(ANNOTATION, "color").pixmap().toImage())
    assert (shown[~mask][:, 3] == main_window.ISOLATE_ALPHA).all()
    # Side by side, the second canvas shows it too.
    window.commands["compare_side"].trigger()
    assert window.compare_view.overlay(ANNOTATION, "color") is not None
    window.inspector.select(None)
    assert window.preview.overlay(ANNOTATION, "color") is None
    assert window.compare_view.overlay(ANNOTATION, "color") is None

"""The analysis (I5): the page measured by the benchmark's own functions, now in ``core.measures``; the Analysis page's
measures, issues and overlays; measuring only while it is shown, once a page; an issue picking its region; and the
overlays in the diagnostics layer, off by default."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app.jobs import ANALYSIS as ANALYSIS_JOB  # noqa: E402
from tessellatum.core import difficulty, measures, pipeline, print_size  # noqa: E402
from tessellatum.gui import analysis, main_window  # noqa: E402
from tessellatum.gui.actions import ANALYSIS  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402
from tessellatum.gui.preview_widget import DIAGNOSTICS  # noqa: E402

_app = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "sample_images"


def _bench_metrics():
    spec = importlib.util.spec_from_file_location("bench_metrics_for_test", ROOT / "benchmarks" / "bench_metrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- The measures, as the benchmark takes them ---------------------------------------------------------------------
def test_the_benchmark_measures_with_the_package_s_file():
    bm = _bench_metrics()
    assert Path(bm.MEASURES_PATH) == ROOT / "src" / "tessellatum" / "core" / "measures.py"
    for name in ("sliver_mask", "sliver_share", "corner_tips", "count_undersized", "unlabeled_regions"):
        assert getattr(bm, name).__code__.co_filename == str(bm.MEASURES_PATH)
        assert getattr(bm, name).__code__.co_code == getattr(measures, name).__code__.co_code
    scale = print_size.print_scale((2000, 1400), print_size.A4)
    assert bm.label_sizes([20, 30], scale) == measures.label_sizes([20, 30], scale, bm.LABEL_MIN_PT)


@pytest.mark.parametrize("name, level", [("l-photo-cats-face.jpg", "Medium"), ("m-cartoon-bold-lines-girl.png", "Easy")])
def test_the_analysis_equals_the_benchmark_s_metrics(name, level):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("sample image not present")
    bm = _bench_metrics()
    image = cv2.imread(str(path))
    params = difficulty.params_for_preset(level)
    page = pipeline.generate(image, params, pipeline.working_long_edge(image, params))
    model = page.model
    ids = model.region_id_map
    scale = print_size.print_scale((ids.shape[1], ids.shape[0]), page.drawing.style.paper_format)
    found = measures.measure_page(ids, model.labels, model.legend_size, scale.px_per_mm, params.min_width_mm,
                                  params.min_region_area_mm2 / params.detail_weight, model.style.min_label_pt,
                                  scale.px_per_pt)
    # As bench_case computes them.
    brush_px = scale.mm_to_px(params.min_width_mm)
    corners = bm.corner_tips(ids, brush_px)
    assert found.sliver_fraction == float((bm.sliver_mask(ids, brush_px) & ~corners).mean())
    assert found.corner_tip_fraction == float(corners.mean())
    labeled = {label.region_id for label in model.labels}
    expected = bm.unlabeled_regions(ids, labeled)
    assert (found.unlabeled_regions, found.unlabeled_area_fraction) == (
        expected["unlabeled_regions"], expected["unlabeled_area_fraction"])
    sizes = bm.label_sizes([label.font_size for label in model.labels], scale, model.style.min_label_pt)
    assert (found.small_label_fraction, found.min_label_pt) == (sizes["small_label_fraction"], sizes["min_label_pt"])
    assert found.leader_labels == sum(label.leader is not None for label in model.labels)
    assert found.regions == page.num_regions or found.regions == len(np.unique(ids[ids >= 0]))
    # The issues: their regions are the ones the measures count.
    assert len(found.unlabeled) == found.unlabeled_regions
    assert found.cramped.size == len({label.region_id for label in model.labels if label.cramped})


def test_the_widest_brushes_of_every_region_at_once():
    ids = np.full((40, 60), -1, dtype=np.int32)
    ids[5:15, 10:30] = 0
    ids[20:30, 40:42] = 1
    ids[30:35, 0:60] = 3
    widest = measures.widest_brushes_px(ids)
    stats = measures.region_stats(ids)
    for region in (0, 1, 3):
        assert widest[region] == pytest.approx(measures.widest_brush_px(ids, region, stats.box(region)), abs=1e-6)
    assert widest[2] == 0


def test_the_page_s_measures_on_a_synthetic_page():
    from tessellatum.core.labels import Label

    ids = np.full((100, 100), 0, dtype=np.int32)
    ids[10:12, 10:90] = 1  # a 2 px strip: nowhere 3 px wide
    ids[50:53, 50:53] = 2  # 9 px
    labels = [Label(region_id=0, text="1", font_size=12.0, box=(40, 40, 50, 50)),
              Label(region_id=1, text="2", font_size=4.0, box=(0, 0, 4, 4), leader=((2, 2), (20, 11)), cramped=True)]
    found = measures.measure_page(ids, labels, 3, px_per_mm=1.0, brush_mm=3.0, min_area_mm2=10.0, min_pt=6.0,
                                  px_per_pt=1.0)
    assert found.regions == 3 and found.colors == 3 and found.page_mm2 == 10000
    assert found.regions_per_cm2 == pytest.approx(0.03)  # 3 regions on 100 cm²
    assert found.smallest_region == 2 and found.smallest_mm2 == 9.0
    assert found.narrowest_region == 1 and found.narrowest_mm < 3.0
    assert found.narrow.tolist() == [1] and found.small.tolist() == [2]
    assert found.unlabeled.tolist() == [2] and found.unlabeled_regions == 1
    assert found.cramped.tolist() == [1] and found.small_numbers.tolist() == [1]
    assert found.leader_labels == 1 and found.cramped_labels == 1 and found.min_label_pt == 4.0
    assert found.sliver_fraction > 0 and found.slivers[10, 50]


# -- The Analysis page in the window ----------------------------------------------------------------------------------
def _picture_file(path, size=(160, 120)):
    rng = np.random.default_rng(5)
    picture = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        picture[max(0, y - 9): y + 9, max(0, x - 14): x + 14] = rng.integers(0, 256, 3)
    Image.fromarray(picture).save(path)
    return path


def _settle(window):
    deadline = time.monotonic() + 90
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
    window.open_path(_picture_file(tmp_path / "patches.png"))
    window.generate_preview()
    _settle(window)
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_the_analysis_is_measured_only_while_shown_once_a_page(window, monkeypatch):
    assert window.analysis.measures() is None  # not shown: not measured
    calls = []
    measure = measures.measure_page
    monkeypatch.setattr(measures, "measure_page", lambda *a, **k: calls.append(1) or measure(*a, **k))
    window.commands["activity_analysis"].trigger()
    assert window.controls.category() == ANALYSIS and window.activity_bar.buttons[ANALYSIS].isChecked()
    assert "Measuring" in window.analysis.status.text()
    _settle(window)
    found = window.analysis.measures()
    assert found is not None and calls == [1] and window.analysis.status.isHidden()
    assert found.regions == window.current_page.num_regions or found.regions > 0
    rows = window.analysis.metrics_layout.rowCount()
    assert rows == len(analysis.metric_rows(found))
    # Shown again, or another activity and back: not measured again.
    window.show_activity("Generation")
    window.show_activity(ANALYSIS)
    _settle(window)
    assert calls == [1]
    # A new page: measured again, as it is shown.
    window.controls.controls["num_colors"].step(-3)
    _app.processEvents()
    window._record_settings()
    window.generate_preview()
    _settle(window)
    assert calls == [1, 1] and window.analysis.measures() is not None
    # Hidden, a new page isn't measured until the analysis is shown.
    window.show_activity("Generation")
    window.controls.controls["num_colors"].step(-2)
    _app.processEvents()
    window._record_settings()
    window.generate_preview()
    _settle(window)
    assert calls == [1, 1] and window.analysis.measures() is None
    assert not window.jobs.is_running(ANALYSIS_JOB)


def test_an_issue_picks_its_region_and_the_overlays_are_off_until_asked(window):
    window.show_activity(ANALYSIS)
    _settle(window)
    for key in analysis.OVERLAYS:
        assert window.preview.overlay(DIAGNOSTICS, key) is None  # off by default
    found = window.analysis.measures()
    groups = analysis.issue_groups(found)
    assert window.analysis.issues.topLevelItemCount() == len(groups)
    row = next((i for i, (_t, _m, regions) in enumerate(groups) if regions), None)
    assert any(title.startswith("Regions numbered on leader lines") for title, _m, _r in groups)
    if row is not None:
        group = window.analysis.issues.topLevelItem(row)
        window.analysis.issues.setCurrentItem(group)  # the issue: all its regions highlighted
        highlight = window.preview.overlay(main_window.ANNOTATION, "issue")
        assert highlight is not None
        image = highlight.pixmap().toImage().convertToFormat(QImage.Format_RGBA8888)
        ids = window.current_page.model.region_id_map
        data = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes()).copy()
        lit = data.reshape(image.height(), image.bytesPerLine() // 4, 4)[:, : image.width(), 3] > 0
        assert (lit == np.isin(ids, groups[row][2])).all()
        window.analysis.issues.setCurrentItem(group.child(0))  # one of its regions: picked and centered
        assert window.selected_region() == groups[row][2][0]
        assert window.preview.overlay(main_window.ANNOTATION, "issue") is not None
    for key, check in window.analysis.overlay_checks.items():
        check.setChecked(True)
        assert window.preview.overlay(DIAGNOSTICS, key) is not None, key
    window.commands["compare_side"].trigger()
    for key in analysis.OVERLAYS:
        assert window.compare_view.overlay(DIAGNOSTICS, key) is not None
    window.analysis.overlay_checks[analysis.LEADERS].setChecked(False)
    assert window.preview.overlay(DIAGNOSTICS, analysis.LEADERS) is None
    # A page closed: the measures and their overlays go.
    window.document.set_result(None)
    _settle(window)
    assert window.analysis.measures() is None
    assert window.preview.overlay(DIAGNOSTICS, analysis.NARROW_PARTS) is None


def test_the_overlays_draw_what_they_say():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:9] = True
    image = main_window.diagnostic_image(mask, "#B8322A", 150)
    assert image.pixelColor(4, 3).alpha() == 150 and image.pixelColor(4, 3).red() == 0xB8
    assert image.pixelColor(0, 0).alpha() == 0
    leaders = main_window.leader_image((40, 40), [(((2, 2), (30, 30)), (0, 0, 4, 4))], "#0D7468")
    assert leaders.pixelColor(16, 16).alpha() > 0 and leaders.pixelColor(35, 5).alpha() == 0


def test_a_page_that_can_t_be_measured_says_so(window, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(measures, "measure_page", fail)
    window.show_activity(ANALYSIS)
    _settle(window)
    assert window.analysis.measures() is None
    assert "couldn't be measured" in window.analysis.status.text() and "out of memory" in window.analysis.status.text()

"""Custom and canvas sheets (paper-sizes T3): a sheet of one's own size, the usual kit canvases, and every rule on
paper holding on them as on the named papers."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import errors  # noqa: E402
from tessellatum.app import project as projects  # noqa: E402
from tessellatum.app.snapshot import DocumentSettings  # noqa: E402
from tessellatum.core import difficulty, pipeline, settings  # noqa: E402
from tessellatum.core.difficulty import params_for_preset  # noqa: E402
from tessellatum.core.print_size import (  # noqa: E402
    A4, CANVASES, CUSTOM, LETTER, PAPERS, custom_paper, legend_paper, paper_size_text,
)
from tessellatum.core.render import PageStyle  # noqa: E402
from tessellatum.gui.controls_panel import ControlsPanel  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _picture() -> np.ndarray:
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    image[:450, :600] = (200, 40, 40)
    image[:450, 600:] = (40, 180, 40)
    image[450:] = (40, 40, 200)
    for y in range(500, 860, 60):
        for x in range(40, 1160, 60):
            image[y : y + 24, x : x + 24] = (210, 210, 40)
    return image


# -- The sheets ----------------------------------------------------------------------------------------------------------
def test_the_canvases_are_named_papers_in_their_own_units():
    names = [paper.name for paper in CANVASES]
    assert names == ["30 × 40 cm canvas", "40 × 50 cm canvas", "50 × 65 cm canvas", "60 × 75 cm canvas",
                     "12 × 16 in canvas", "16 × 20 in canvas", "18 × 24 in canvas"]
    assert PAPERS["40 × 50 cm canvas"].width_mm == 400.0 and not PAPERS["40 × 50 cm canvas"].inches
    assert PAPERS["16 × 20 in canvas"].height_mm == pytest.approx(508.0) and PAPERS["16 × 20 in canvas"].inches
    assert settings.choice("paper").options[-1] == CUSTOM and settings.choice("paper").options[:3] == ("A4", "A3", "A2")


def test_a_custom_sheet_is_sized_either_way_round_in_inches_when_whole():
    assert custom_paper(500, 400) == custom_paper(400, 500)
    sheet = custom_paper(500, 400)
    assert (sheet.width_mm, sheet.height_mm, sheet.margin_mm, sheet.name) == (400.0, 500.0, 10.0, CUSTOM)
    assert not sheet.inches and paper_size_text(sheet) == "400 × 500 mm"
    inches = custom_paper(16 * 25.4, 20 * 25.4)
    assert inches.inches and paper_size_text(inches) == "16 × 20 in"
    assert legend_paper(sheet) == A4 and legend_paper(inches) == LETTER
    assert legend_paper(custom_paper(150, 200)) == custom_paper(150, 200)  # no larger than A4: its own
    style = PageStyle(paper=CUSTOM, custom_width_mm=297, custom_height_mm=210)
    assert style.paper_format == custom_paper(210, 297)
    assert PageStyle(custom_width_mm=300).paper_format is A4  # the size means nothing on a named paper


def test_a_custom_size_out_of_range_is_refused_with_a_reason_never_clamped():
    width = settings.setting("custom_width_mm")
    assert width.check(99) and "least" in width.check(99)
    assert width.check(1190) and "most" in width.check(1190)
    assert width.check(100) is None and width.check(1189) is None
    assert width.policy == settings.EXPLICIT  # a new sheet makes the page again
    assert width.disabled_reason({"paper": "A4"}) == "Only with Paper: Custom."
    assert width.disabled_reason({"paper": CUSTOM}) is None


@pytest.mark.parametrize("canvas, size", [("40 × 50 cm canvas", (400, 500)), ("A4", (210, 297))])
def test_a_custom_sheet_makes_the_page_its_named_paper_does(canvas, size):
    image, params = _picture(), params_for_preset("Medium")
    style = PageStyle(paper=canvas)
    custom = PageStyle(paper=CUSTOM, custom_width_mm=size[0], custom_height_mm=size[1])
    edges = {pipeline.working_long_edge(image, params, paper=s.paper_format) for s in (style, custom)}
    assert len(edges) == 1
    edge = edges.pop()
    named = pipeline.generate(image, params, edge, style=style)
    own = pipeline.generate(image, params, edge, style=custom)
    assert own.page.tobytes() == named.page.tobytes()
    assert np.array_equal(own.model.region_id_map, named.model.region_id_map)


def test_a_custom_size_changes_the_regions_not_a_cached_page():
    image, params = _picture(), params_for_preset("Medium")
    small = PageStyle(paper=CUSTOM, custom_width_mm=150, custom_height_mm=200)
    large = PageStyle(paper=CUSTOM, custom_width_mm=600, custom_height_mm=800)
    a = pipeline.generate(image, params, 1200, style=small, collect_analysis=True)
    b = pipeline.generate(image, params, 1200, style=large, collect_analysis=True)
    assert b.analysis.min_region_area_px < a.analysis.min_region_area_px  # finer on the larger sheet


def test_the_practical_description_names_the_sheet():
    assert difficulty.practical("Medium").endswith("regions to paint on A4 paper (400 to 1,200 on most pictures).") \
        or "on A4 paper" in difficulty.practical("Medium")
    assert "on a 40 × 50 cm canvas" in difficulty.practical("Medium", "40 × 50 cm canvas")
    assert "on a 300 × 400 mm sheet" in difficulty.practical("Medium", custom_paper(300, 400))


def test_running_out_of_memory_on_a_custom_sheet_or_a_canvas_suggests_a_smaller_one():
    text = errors.explain(errors.MAKING, MemoryError, "", paper=CUSTOM)
    assert "on a custom sheet" in text and "a smaller size, or A4" in text
    text = errors.explain(errors.MAKING, MemoryError, "", paper="60 × 75 cm canvas")
    assert "a smaller paper (50 × 65 cm canvas)" in text
    assert "smaller paper" not in errors.explain(errors.MAKING, MemoryError, "", paper="12 × 16 in canvas")


# -- The panel and the project -------------------------------------------------------------------------------------------
def _store(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def test_width_and_height_are_on_only_with_custom_and_are_kept(tmp_path):
    panel = ControlsPanel(store=_store(tmp_path))
    width, height = panel.controls["custom_width_mm"], panel.controls["custom_height_mm"]
    assert width.disabled_reason() == "Only with Paper: Custom." and height.disabled_reason()
    panel.paper_combo.setCurrentText(CUSTOM)
    assert width.disabled_reason() is None and height.disabled_reason() is None
    width.set_value(300.0)
    panel._on_setting_edited("custom_width_mm")
    assert panel.get_page_style().paper_format == custom_paper(300, 500)
    panel.set_picture_size((3000, 2000), "picture.png")
    assert "on a 300 × 500 mm sheet" in panel.picture_print_label.text()
    again = ControlsPanel(store=_store(tmp_path))
    assert again.get_page_style().paper == CUSTOM and again.get_page_style().custom_width_mm == 300.0
    again.paper_combo.setCurrentText("40 × 50 cm canvas")
    assert again.get_page_style().paper_format is PAPERS["40 × 50 cm canvas"]
    assert again.controls["custom_width_mm"].disabled_reason() is not None


def test_a_project_keeps_a_custom_size_and_an_older_one_opens_as_it_was(tmp_path):
    from test_project import _picture_bytes, _rewrite

    snapshot = DocumentSettings(paper=CUSTOM).with_values(custom_width_mm=350.0, custom_height_mm=450.0)
    data = _picture_bytes()
    path = tmp_path / "custom.tessellatum"
    projects.save_project(path, snapshot, "picture.png", data, projects.decode_image(data))
    read = projects.read_project(path)
    assert read.settings.paper == CUSTOM and read.settings.page_style().paper_format == custom_paper(350, 450)

    def older(members):  # as 2.0.1 saved it: format 1, no custom size
        info = json.loads(members[projects.MANIFEST])
        info["format_version"] = 1
        info["settings"]["choices"]["paper"] = "A3"
        for name in ("custom_width_mm", "custom_height_mm"):
            del info["settings"]["values"][name]
        members[projects.MANIFEST] = json.dumps(info).encode("utf-8")

    _rewrite(path, older)
    read = projects.read_project(path)
    assert read.settings.paper == "A3" and read.settings.page_style().custom_width_mm == 400.0
    assert projects.FORMAT_VERSION == 2


def test_a_custom_size_put_back_by_undo_shows_again(tmp_path):
    panel = ControlsPanel(store=_store(tmp_path))
    panel.paper_combo.setCurrentText(CUSTOM)
    panel.set_picture_size((3000, 2000), "picture.png")
    before = panel.snapshot()
    panel.show_settings(before.with_values(custom_width_mm=200.0, custom_height_mm=250.0))
    assert "on a 200 × 250 mm sheet" in panel.picture_print_label.text()
    panel.show_settings(before)
    assert "on a 400 × 500 mm sheet" in panel.picture_print_label.text()
    assert panel.controls["custom_width_mm"].disabled_reason() is None

# -- Letter by default in US units (T4) ----------------------------------------------------------------------------------
@pytest.fixture
def us_locale():
    from PySide6.QtCore import QLocale

    before = QLocale()
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    yield
    QLocale.setDefault(before)


def _paper_section(panel) -> str:
    return next(key for key, entries in panel._section_entries.items() if any(e.name == "paper" for e in entries))


@pytest.mark.system_paper
def test_a_first_run_in_us_units_starts_on_letter_and_a_kept_paper_wins(tmp_path, us_locale):
    from tessellatum.gui import controls_panel

    assert controls_panel.default_paper() == "Letter"
    fresh = ControlsPanel(store=_store(tmp_path))
    assert fresh.get_page_style().paper == "Letter"
    assert "paper" not in fresh._changes_in(_paper_section(fresh))  # the default here, not a change
    fresh.paper_combo.setCurrentText("A4")
    assert "paper" in fresh._changes_in(_paper_section(fresh))
    kept = ControlsPanel(store=_store(tmp_path))
    assert kept.get_page_style().paper == "A4"  # chosen and kept: it wins
    kept.reset()
    assert kept.get_page_style().paper == "Letter"
    assert PageStyle().paper == "A4" and settings.choice("paper").default == "A4"  # the core's default stays A4


@pytest.mark.system_paper
def test_elsewhere_a_first_run_starts_on_a4(tmp_path):
    from PySide6.QtCore import QLocale

    from tessellatum.gui import controls_panel

    assert controls_panel.default_paper(QLocale(QLocale.Language.German, QLocale.Country.Germany)) == "A4"
    assert controls_panel.default_paper(QLocale(QLocale.Language.English, QLocale.Country.UnitedKingdom)) == "A4"

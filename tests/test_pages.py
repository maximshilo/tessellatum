"""W4: the category pages at three levels of detail. The advanced settings wait in one folded group per page, a
section can be put back on its own, the Source page says how the picture prints and how it was read, the difficulty
says what it makes, no text names a paper but the one chosen, recipes from the paints list say where they come from,
and a newcomer's way through (open, pick a difficulty, generate, export) never needs an advanced setting."""

import dataclasses
import os
import re
import time

import numpy as np
import pytest
from PIL import Image

import pdf_reading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget  # noqa: E402

from tessellatum.core import difficulty, export, legend, pipeline, settings  # noqa: E402
from tessellatum.core.legend import NAME_HEX_RGB, NAME_RECIPE, RECIPES_HEADING, legend_sections, render_legend  # noqa: E402
from tessellatum.core.print_size import PAPERS  # noqa: E402
from tessellatum.gui import controls_panel, main_window  # noqa: E402
from tessellatum.gui.controls_panel import advanced_subsection_key, ADVANCED, ControlsPanel, advanced_key  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


# -- Levels of detail --------------------------------------------------------------------------
def test_each_page_has_its_advanced_settings_in_one_folded_group_at_its_foot():
    panel = ControlsPanel()
    for category, page in panel.pages.items():
        groups = [s for key, s in panel.sections.items() if s.title == ADVANCED and page.isAncestorOf(s)]
        advanced = [s for s in settings.SETTINGS if s.category == category and s.level == settings.ADVANCED]
        if not advanced:
            assert groups == [], category
            continue
        assert len(groups) == 1 and not groups[0].is_expanded(), category
        group = groups[0]
        assert group is panel.sections[advanced_key(category)]
        for s in advanced:
            assert group.body.isAncestorOf(panel.controls[s.name]), s.name
        # The group comes last among the page's sections.
        tops = [section.y() for section in panel.sections.values() if page.isAncestorOf(section)]
        assert group.y() == max(tops)
        # The intent and precise settings are on top, outside it.
        for s in settings.SETTINGS:
            if s.category == category and s.level != settings.ADVANCED:
                assert not group.isAncestorOf(panel.controls[s.name]), s.name
    # The advanced group's subsections name the registry's sections, without "Advanced ·": each folds on its own,
    # folded at first, inside the group.
    group = panel.sections[advanced_key(settings.GENERATION)]
    subs = {key: s for key, s in panel.sections.items() if key.startswith(advanced_key(settings.GENERATION) + " · ")}
    assert [s.title for s in subs.values()] == ["Faces and subject", "Line art", "Text", "Edges", "Corners",
                                               "Resolution"]
    for key, sub in subs.items():
        assert sub.nested and group.body.isAncestorOf(sub) and not sub.is_expanded(), key
    edges = subs[advanced_subsection_key(settings.GENERATION, "Advanced · Edges")]
    for s in settings.SETTINGS:
        if s.category == settings.GENERATION and s.section == "Advanced · Edges":
            assert edges.body.isAncestorOf(panel.controls[s.name]), s.name
            assert panel.section_of(panel.controls[s.name]) is edges  # the innermost


def test_the_difficulty_says_what_it_makes_and_how_far_it_was_moved():
    panel = ControlsPanel()
    medium = difficulty.describe(difficulty.params_for_preset("Medium"))
    assert panel.preset_description.text() == f"{difficulty.practical('Medium')}\n{medium}"
    assert "About 450 regions to paint on A4 paper" in panel.preset_description.text()
    panel.preset_combo.setCurrentText("Beginner")
    assert panel.preset_description.text().endswith(difficulty.describe(difficulty.params_for_preset("Beginner")))
    panel.controls["min_width_mm"].step(2)
    panel.controls["num_colors"].step(-1)
    colors, width = panel.controls["num_colors"].text(), panel.controls["min_width_mm"].text()
    # The modified indicator, with what was changed: in the settings' order.
    assert panel.preset_description.text() == (
        f"Beginner (modified, 2 changes): Colors {colors}, Brush width {width}.")


# -- Section resets ----------------------------------------------------------------------------
def test_a_section_is_put_back_on_its_own():
    panel = ControlsPanel()
    detail = panel.sections["Detail"]
    assert detail.reset_button.isHidden()
    panel.controls["min_width_mm"].step(3)
    panel.controls["blur_sigma"].step(2)
    panel.controls["line_width_mm"].step(2)  # another section's
    assert not detail.reset_button.isHidden() and detail.reset_button.accessibleName() == "Reset Detail"
    assert panel.preset_combo.currentText() == "Custom"
    detail.reset_button.click()
    assert panel.changed_settings() == ["line_width_mm"] and detail.reset_button.isHidden()
    assert panel.preset_combo.currentText() == "Medium"  # every difficulty setting the preset's again

    # Choices go back to their defaults.
    page = panel.sections["Page"]
    panel.paper_combo.setCurrentText("A2")
    assert not page.reset_button.isHidden()
    page.reset_button.click()
    assert panel.paper_combo.currentText() == "A4" and page.reset_button.isHidden()

    # The Advanced group puts back its own, and only them.
    group = panel.sections[advanced_key(settings.DOCUMENT)]
    panel.controls["leader_reach_mm"].step(4)
    panel.controls["line_gray"].step(3)
    assert panel.tone_combo.currentText() == "Custom" and not group.reset_button.isHidden()
    group.reset_button.click()
    assert panel.changed_settings() == ["line_width_mm"] and panel.tone_combo.currentText() == "Medium"
    # The Difficulty section has the app's Reset; the paints listed are never dropped by a section's.
    assert panel.sections["Difficulty"].reset_button.isHidden()


# -- The Source page -------------------------------------------------------------------------
def test_the_source_page_says_how_the_picture_prints_and_how_it_was_read():
    panel = ControlsPanel()
    assert panel.picture_print_label.text() == "" and panel.picture_reading_label.text() == ""
    panel.set_picture_size((3000, 2000), "lion.jpg")
    assert panel.picture_label.text() == "lion.jpg: 3,000 × 2,000 px"
    assert re.fullmatch(r"Prints \d+ × \d+ mm on A4, at \d+ of its pixels an inch\.", panel.picture_print_label.text())
    a4 = panel.picture_print_label.text()
    panel.paper_combo.setCurrentText("A2")
    assert "on A2" in panel.picture_print_label.text() and panel.picture_print_label.text() != a4
    assert panel.picture_reading_label.text() == "Generate a preview to see how the picture is read."
    panel.set_picture_reading(True)
    assert panel.picture_reading_label.text().startswith("Read as line art")
    panel.set_picture_reading(False)
    assert panel.picture_reading_label.text().startswith("Read as a picture of colors")
    panel.set_picture_size((100, 100), "other.png")  # a new picture: not read yet
    assert panel.picture_reading_label.text().startswith("Generate a preview")


# -- No paper named but the one chosen ---------------------------------------------------------
def _texts(widget: QWidget) -> list[str]:
    texts = []
    for child in [widget, *widget.findChildren(QWidget)]:
        texts += [child.toolTip(), child.accessibleName(), child.accessibleDescription(), child.statusTip()]
        if isinstance(child, (QLabel, QAbstractButton)):
            texts.append(child.text())
    return [t for t in texts if t]


@pytest.mark.parametrize("paper", ["A3", "A2"])
def test_no_text_names_a_paper_other_than_the_one_chosen(paper):
    panel = ControlsPanel()
    panel.set_picture_size((3000, 2000), "picture.png")
    panel.paper_combo.setCurrentText(paper)
    others = [name for name in PAPERS if name != paper]
    pattern = re.compile(r"\b(" + "|".join(others) + r")\b")
    # The paper choice itself names every paper; nothing else does.
    paper_texts = set(_texts(panel.paper_combo)) | {settings.PAPER_TOOLTIP}
    offending = [t for t in _texts(panel) if pattern.search(t) and t not in paper_texts]
    assert offending == []
    for name in difficulty.PRESETS:
        assert not pattern.search(difficulty.describe(difficulty.params_for_preset(name)))
    for entry in settings.entries():
        if entry.name != "paper":
            assert not re.search(r"\bA[0-9]\b", entry.tooltip + entry.label), entry.name


# -- Recipes from the paints list (D-005) ---------------------------------------------------------
PALETTE = [(200, 40, 40), (40, 160, 60), (230, 220, 200)]
RECIPES = [["1 Cadmium red"], ["2 Sap green", "1 Titanium white"], ["1 Titanium white"]]


def test_recipes_from_the_paints_list_say_where_they_come_from():
    sections = legend_sections(PALETTE, NAME_RECIPE, recipes=RECIPES)
    assert [s[0] for s in sections] == [RECIPES_HEADING]
    # Without recipes, or with another detail, the legend has no heading, as before.
    assert [s[0] for s in legend_sections(PALETTE, NAME_RECIPE)] == [""]
    assert [s[0] for s in legend_sections(PALETTE, NAME_HEX_RGB, recipes=RECIPES)] == [""]
    bgr = np.array([c[::-1] for c in PALETTE], np.uint8)
    with_heading = render_legend(bgr, 1200, 6.0, NAME_RECIPE, recipes=RECIPES)
    plain = render_legend(bgr, 1200, 6.0, NAME_HEX_RGB)
    assert with_heading.height == plain.height + round(legend.HEADING_MM * 6.0)


def test_the_pdf_legend_heads_recipes_from_the_paints_list(tmp_path):
    picture = np.zeros((200, 300, 3), np.uint8)
    picture[:, 150:] = (40, 170, 40)
    page = pipeline.generate(picture, difficulty.params_for_preset("Beginner"), 600, vector=True)
    recipes = [["1 Titanium white"]] * len(page.palette_rgb)
    style = dataclasses.replace(page.drawing.style, legend_detail=NAME_RECIPE)  # as the window writes the legend
    path = tmp_path / "page.pdf"
    export.save_pdf(dataclasses.replace(page.drawing, style=style), page.palette_rgb, path, recipes=recipes)
    sheet = pdf_reading.sheets(path.read_bytes())[1]["content"]
    assert f"({RECIPES_HEADING})" in sheet and "(1 Titanium white)" in sheet


# -- A newcomer's way through ------------------------------------------------------------------
def test_open_pick_a_difficulty_generate_and_export_without_an_advanced_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    picture = np.zeros((400, 600, 3), np.uint8)
    picture[:, 300:] = (40, 170, 40)
    picture[200:, :300] = (40, 40, 200)
    Image.fromarray(picture[:, :, ::-1]).save(tmp_path / "blocks.png")
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "blocks.png"), "")))
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "page.png"), "")))
    window = MainWindow()
    try:
        window.show()
        window.controls.set_export_dpi(150)
        advanced = [s for key, s in window.controls.sections.items() if s.title == ADVANCED]
        touched = []
        for s in settings.SETTINGS:
            if s.level == settings.ADVANCED:
                window.controls.controls[s.name].edited.connect(lambda name=s.name: touched.append(name))

        window.commands["open"].trigger()
        window.controls.preset_combo.setCurrentText("Easy")
        window.commands["generate"].trigger()
        deadline = time.monotonic() + 120
        while not window.jobs.is_idle() or window.current_page is None:
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.005)
        window.commands["export"].trigger()  # the export dialog: Export, with its options as they start
        assert window.export_dialog.isVisible() and not window.export_dialog.stale_shown()
        window.export_dialog.export_button.click()
        while not window.jobs.is_idle():
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.005)
        _app.processEvents()
        assert (tmp_path / "page.png").exists() and window.notification.text.text().startswith("Exported page.png")
        assert touched == [] and not any(group.is_expanded() for group in advanced)
        # Everything it took is outside the Advanced groups.
        for widget in (window.controls.preset_combo, window.controls.open_button, window.controls.generate_button,
                       window.controls.export_button):
            assert not any(group.isAncestorOf(widget) for group in advanced)
    finally:
        window.close()
        window.deleteLater()
        _app.processEvents()
        pipeline.clear_cache()


def test_the_panel_module_names_its_levels():
    assert controls_panel.ADVANCED == "Advanced"
    assert set(controls_panel.ADVANCED_SECTIONS) <= set(controls_panel.PAGE_SECTIONS)


def test_an_advanced_subsection_resets_on_its_own_and_the_group_resets_all():
    panel = ControlsPanel()
    edges = advanced_subsection_key(settings.GENERATION, "Advanced · Edges")
    corners = advanced_subsection_key(settings.GENERATION, "Advanced · Corners")
    edge = next(s.name for s in settings.SETTINGS if s.section == "Advanced · Edges" and s.category == settings.GENERATION)
    corner = next(s.name for s in settings.SETTINGS
                  if s.section == "Advanced · Corners" and s.category == settings.GENERATION)
    panel.controls[edge].step(-1)  # both start at their most
    panel.controls[corner].step(-1)
    assert not panel.sections[edges].reset_button.isHidden() and not panel.sections[corners].reset_button.isHidden()
    group = panel.sections[advanced_key(settings.GENERATION)]
    assert not group.reset_button.isHidden()
    panel.sections[edges].reset_button.click()
    assert panel.changed_settings() == [corner] and not group.reset_button.isHidden()
    group.reset_button.click()
    assert panel.changed_settings() == [] and group.reset_button.isHidden()

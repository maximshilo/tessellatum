"""Feature parity with 1.0.21: every capability the app had before the workspace was rebuilt is still reachable.

The inventory it was rebuilt against listed each one (FP-01 to FP-80) with where it was going. This test holds the
window to it: each row has a check that its equivalent exists and is reachable (a command in a menu, a control on its
page, a choice with its items, a function the core keeps). What each one does is tested where it lives; this is the
index that says nothing was dropped. The settings themselves (every registry entry has a control, every key kept
between runs still loads) are ``tests/test_registry.py``'s.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from tessellatum import selftest  # noqa: E402
from tessellatum.core import export, paints, pipeline, swatches  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui import main_window, paints_dialog, preview_widget  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window():
    w = MainWindow()
    yield w
    w.close()


def _menu_actions(window) -> list:
    found = []

    def walk(menu: QMenu) -> None:
        for action in menu.actions():
            if action.menu() is not None:
                walk(action.menu())
            else:
                found.append(action)

    for top in window.menuBar().actions():
        if top.menu() is not None:
            walk(top.menu())
    return found


def _in_menu(window, command_id: str) -> bool:
    return window.commands[command_id] in _menu_actions(window)


def _items(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _keys(window, command_id: str) -> set[str]:
    return {k.toString() for k in window.commands[command_id].shortcuts()}


CHECKS = {
    # 1. Opening a picture and the workflow
    "FP-01": lambda w: _in_menu(w, "open") and "Ctrl+O" in _keys(w, "open") and w.welcome.open_picture_button
    and all(p in main_window.IMAGE_PATTERNS for p in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff",
                                                      "*.gif")),
    "FP-02": lambda w: callable(pipeline.load_image_bgr),
    "FP-03": lambda w: w.notification is not None,
    "FP-04": lambda w: _in_menu(w, "show_original") and _in_menu(w, "activity_source"),
    "FP-05": lambda w: w.status.state_label is not None and w.status.page_label is not None,
    "FP-06": lambda w: _in_menu(w, "generate") and "F5" in _keys(w, "generate") and w.run_button is not None,
    "FP-07": lambda w: w.status.progress is not None and _in_menu(w, "cancel"),
    "FP-08": lambda w: _in_menu(w, "cancel") and "Esc" in _keys(w, "cancel"),
    "FP-09": lambda w: w.notification is not None and w.diagnostics is not None,
    "FP-10": lambda w: not w.commands["export"].isEnabled(),  # nothing to export yet: enablement, not a busy flag
    "FP-11": lambda w: callable(pipeline.warm_up),
    "FP-12": lambda w: callable(w.preview.show_page) and callable(w.preview.view_state),
    "FP-13": lambda w: callable(selftest.run),
    "FP-14": lambda w: _in_menu(w, "reset_layout"),
    # 2. Settings
    "FP-20": lambda w: _items(w.controls.preset_combo)[:5] == ["Beginner", "Easy", "Medium", "Hard", "Realistic"]
    and "Custom" in _items(w.controls.preset_combo),
    "FP-21": lambda w: w.controls.revert_preset_action is not None,
    "FP-22": lambda w: _in_menu(w, "reset") and w.controls.reset_button is not None,
    "FP-23": lambda w: len(w.controls.controls) >= 30,
    "FP-24": lambda w: len(w.controls.sections) >= 5,
    "FP-25": lambda w: _items(w.controls.paper_combo) == ["A4", "A3", "A2"],
    "FP-26": lambda w: len(_items(w.controls.medium_combo)) == 2 and w.controls.medium_hint is not None,
    "FP-27": lambda w: w.controls.colors_from_combo.count() == 2,
    "FP-28": lambda w: w.controls.paints_button is not None,
    "FP-29": lambda w: w.controls.mix_check is not None,
    "FP-30": lambda w: callable(w.controls.set_paints),
    "FP-31": lambda w: _items(w.controls.tone_combo)[:3] == ["Light", "Medium", "Dark"],
    "FP-32": lambda w: w.controls.legend_combo.count() == 4,
    "FP-33": lambda w: hasattr(w, "_recipes_for"),
    "FP-34": lambda w: all(c is not None for c in (w.controls.line_art_check, w.controls.detail_check,
                                                   w.controls.text_check)),
    "FP-35": lambda w: callable(getattr(w.controls, "_save", None) or getattr(w.controls, "save", None)),
    "FP-36": lambda w: w.controls.pages,  # the pages lay their fields out on one grid (test_fields)
    # 3. Paints dialog
    "FP-40": lambda w: _paints(w).table is not None and callable(_paints(w).delete),
    "FP-41": lambda w: callable(_paints(w).add_color) and _paints(w).add_button is not None,
    "FP-42": lambda w: callable(_paints(w).repick),
    "FP-43": lambda w: callable(_paints(w).start_from_kit) and _paints(w).kit_button is not None,
    "FP-44": lambda w: _paints(w).load_button is not None and callable(paints.read_palette_file),
    "FP-45": lambda w: callable(_paints(w).clear) and _paints(w).clear_button is not None,
    "FP-46": lambda w: _paints(w).text_button is not None and _paints(w).editor is not None,
    "FP-47": lambda w: _paints(w).status is not None,
    "FP-48": lambda w: _paints(w).findChild(paints_dialog.QDialogButtonBox) is not None,
    # 4. Preview and navigation
    "FP-50": lambda w: all(_in_menu(w, c) for c in ("show_page", "show_completed", "show_tinted", "show_original"))
    and {"Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4"} <= set().union(*(_keys(w, c) for c in (
        "show_page", "show_completed", "show_tinted", "show_original"))),
    "FP-51": lambda w: callable(w.preview.set_legend),
    "FP-52": lambda w: _in_menu(w, "zoom_in") and _in_menu(w, "zoom_out")
    and {"Ctrl++", "Ctrl+="} <= _keys(w, "zoom_in")
    and "Ctrl+-" in _keys(w, "zoom_out"),
    "FP-53": lambda w: callable(w.preview.wheelEvent),
    "FP-54": lambda w: callable(w.preview.mouseMoveEvent),  # its own drag, so a click can pick a region (below)
    "FP-55": lambda w: _in_menu(w, "fit") and "Ctrl+0" in _keys(w, "fit") and _in_menu(w, "actual_pixels")
    and _in_menu(w, "print_size"),
    "FP-56": lambda w: w.zoom_bar.zoom_field is not None and w.status.zoom_label is not None,
    "FP-57": lambda w: callable(w.preview.show_detail) and callable(w.preview.wants_detail),
    "FP-58": lambda w: preview_widget.SHARPEN_DELAY_MS == 120,
    "FP-59": lambda w: callable(w.preview.zoom_range),
    "FP-60": lambda w: w.welcome is not None,
    "FP-61": lambda w: _in_menu(w, "show_original"),
    # 5. Export
    "FP-70": lambda w: w.controls.version_combo.count() == 3,
    "FP-71": lambda w: all(r is not None for r in (w.controls.png_radio, w.controls.pdf_radio,
                                                   w.controls.palette_radio)),
    "FP-72": lambda w: w.controls.palette_combo.count() == 3 and callable(swatches.save_palette),
    "FP-73": lambda w: w.controls.screen_hint is not None,
    "FP-74": lambda w: w.controls.resolution_spin is not None and len(w.controls.dpi_buttons) == 3
    and w.controls.resolution_unit.count() == 2,
    "FP-75": lambda w: callable(export.save_pdf),
    "FP-76": lambda w: callable(export.export_png),
    "FP-77": lambda w: w.export_dialog.export_now is not None and w.export_dialog.export_previewed is not None,
    "FP-78": lambda w: w.controls.legend_combo is not None,
    "FP-79": lambda w: _in_menu(w, "export") and "Ctrl+E" in _keys(w, "export"),
    "FP-80": lambda w: w.notification is not None,
}


# Where a row's reachable part says little of what it does, the test that does: each must still exist.
BEHAVIOR = {
    "FP-03": "test_welcome.py::test_a_file_that_can_t_be_opened_says_why_and_leaves_the_work",
    "FP-10": "test_jobs.py::test_busy_follows_the_jobs",
    "FP-12": "test_preview_widget.py::test_switching_versions_keeps_the_zoom_and_the_place",
    "FP-35": "test_registry.py::test_every_key_kept_by_earlier_versions_loads",
    "FP-43": "test_paints_dialog.py::test_a_kit_replaces_the_list_or_is_added_to_it_and_asks_which",
    "FP-46": "test_paints_dialog.py::test_edit_as_text_round_trips_and_reports_lines_with_no_color",
    "FP-53": "test_preview_widget.py::test_a_touchpad_s_scrolling_pans",
    "FP-54": "test_canvas.py::test_a_left_drag_pans_and_a_click_says_where",
    "FP-77": "test_export_dialog.py::test_an_out_of_date_preview_is_made_again_by_default_or_written_as_previewed",
}


def _paints(window):
    if not hasattr(window, "_parity_paints"):
        window._parity_paints = paints_dialog.PaintsDialog([], window)
    return window._parity_paints


def test_every_row_of_the_inventory_has_a_check():
    rows = [f"FP-{n:02d}" for n in (*range(1, 15), *range(20, 37), *range(40, 49), *range(50, 62), *range(70, 81))]
    assert sorted(CHECKS) == rows


@pytest.mark.parametrize("row", sorted(CHECKS))
def test_the_feature_is_still_reachable(window, row):
    assert CHECKS[row](window), row


def test_the_tests_named_for_what_a_row_does_exist():
    here = os.path.dirname(__file__)
    for row, ref in BEHAVIOR.items():
        name, test = ref.split("::")
        with open(os.path.join(here, name), encoding="utf-8") as f:
            assert f"def {test}(" in f.read(), (row, ref)


def test_the_three_versions_and_the_original_are_views():
    assert {v.name for v in Version} >= {"PAGE", "COMPLETED", "TINTED"}

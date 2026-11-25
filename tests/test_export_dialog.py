"""W5: the export dialog. It shows the options beside what they will write (a thumbnail, the pixels and the size on
paper), warns without stopping (a soft PNG, a split Procreate palette, numbers too small for paper), refuses only an
export that can't be made, and says when the preview is out of date, writing the page made again by default or the
page as previewed when asked. The files are the old path's, byte for byte; what the canvas shows around the page never
reaches them; a failed export can be retried; it is all reachable from the keyboard."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from tessellatum.core import difficulty, export, pipeline, swatches  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.export_dialog import LOW_DPI, ExportDialog, ExportState, warnings_for  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    window.resize(1280, 840)
    window.show()
    window.controls.set_export_dpi(150)
    _app.processEvents()
    yield window
    window.export_dialog.reject()
    window.close()
    window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def _blocks(size=(600, 400)) -> np.ndarray:
    width, height = size
    picture = np.zeros((height, width, 3), np.uint8)
    picture[:, width // 2 :] = (40, 170, 40)
    picture[height // 2 :, : width // 2] = (40, 40, 200)
    picture[: height // 4, : width // 4] = (220, 220, 230)
    return picture


def _wait(window: MainWindow) -> None:
    deadline = time.monotonic() + 120
    while True:
        _app.processEvents()
        if window.jobs.is_idle():
            _app.processEvents()
            if window.jobs.is_idle():
                return
        assert time.monotonic() < deadline
        time.sleep(0.005)


def _previewed(window: MainWindow) -> None:
    window.current_image_bgr = _blocks()
    window.generate_preview()
    _wait(window)


def _save_to(monkeypatch, path, asked=None):
    def answer(*args, **_kwargs):
        if asked is not None:
            asked.append(args)
        return str(path), ""

    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", staticmethod(answer))


def _export_from_dialog(window: MainWindow, previewed: bool = False) -> None:
    window.commands["export"].trigger()
    assert window.export_dialog.isVisible()
    if previewed:
        window.export_dialog.export_previewed.setChecked(True)
    window.export_dialog.export_button.click()
    _wait(window)


# -- What it shows --------------------------------------------------------------------------
def test_without_a_preview_it_has_nothing_to_export():
    dialog = ExportDialog(main_window.QWidget())
    assert not dialog.export_button.isEnabled() and dialog.thumbnail.text() == "Nothing to export yet"
    dialog.show_state(ExportState("A PNG.", thumbnail=None, blocked="Generate a preview first."))
    assert not dialog.export_button.isEnabled() and dialog.export_button.toolTip() == "Generate a preview first."
    dialog.show_state(ExportState("A PNG."))
    assert dialog.export_button.isEnabled()


def test_it_shows_the_options_and_what_they_write(window):
    _previewed(window)
    window.commands["export"].trigger()
    dialog = window.export_dialog
    assert dialog.isVisible() and dialog.isAncestorOf(window.controls.png_radio)
    assert dialog.export_button.isDefault() and not dialog.stale_shown()
    assert not dialog.thumbnail.pixmap().isNull()
    assert dialog.summary.text().startswith("A PNG of the page: ") and "150 dpi" in dialog.summary.text()
    assert "mm on A4" in dialog.summary.text()
    # It follows the options as they change.
    window.controls.pdf_radio.setChecked(True)
    assert dialog.summary.text().startswith("A PDF of the page: vector art")
    window.controls.palette_radio.setChecked(True)
    colors = len(window.current_page.palette_rgb)
    assert dialog.summary.text().startswith(f"The legend's {colors} colors as a ")
    window.controls.png_radio.setChecked(True)
    window.controls.paper_combo.setCurrentText("A3")
    assert "mm on A3" in dialog.summary.text()


def test_it_warns_without_stopping(window):
    _previewed(window)
    window.controls.set_export_dpi(100)
    window.commands["export"].trigger()
    dialog = window.export_dialog
    assert dialog.warning_texts() == [f"Under {LOW_DPI} dpi, the lines and numbers print soft: 100 dpi is for a screen."]
    assert dialog.export_button.isEnabled()
    window.controls.set_export_dpi(300)
    assert dialog.warning_texts() == [] and dialog.warnings.isHidden()


def test_the_warnings():
    assert warnings_for("PNG", 149, ".swatches", 10, 3, False) != ()
    assert warnings_for("PNG", 150, ".swatches", 10, 3, False) == ()
    assert warnings_for("PDF", None, ".swatches", 10, 3, False) == ()
    split = warnings_for("Palette", None, ".swatches", 64, 3, False)
    assert len(split) == 1 and "as 3 files" in split[0]
    assert warnings_for("Palette", None, ".ase", 64, 3, False) == ()  # one file, however many colors
    assert warnings_for("Palette", None, ".swatches", 30, 3, False) == ()
    small = warnings_for("PDF", None, ".swatches", 10, 2, False)
    assert len(small) == 1 and "2 pt" in small[0]
    assert warnings_for("PDF", None, ".swatches", 10, 2, True) == ()  # on a screen, zoomed in, they read


def test_a_palette_is_written_to_the_files_it_names():
    assert swatches.palette_files(Path("p.swatches"), 30) == [Path("p.swatches")]
    assert [p.name for p in swatches.palette_files(Path("p.swatches"), 64)] == [
        "p 1-30.swatches", "p 31-60.swatches", "p 61-64.swatches",
    ]
    assert swatches.palette_files(Path("p.gpl"), 64) == [Path("p.gpl")]


# -- The files ----------------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["PNG", "PDF", "Palette"])
def test_the_dialog_writes_what_the_old_path_writes(window, tmp_path, monkeypatch, fmt):
    _previewed(window)
    {"PNG": window.controls.png_radio, "PDF": window.controls.pdf_radio, "Palette": window.controls.palette_radio}[
        fmt
    ].setChecked(True)
    name = "page" + {"PNG": ".png", "PDF": ".pdf", "Palette": ".swatches"}[fmt]  # a palette is titled by its name
    (tmp_path / "direct").mkdir()
    (tmp_path / "dialog").mkdir()
    _save_to(monkeypatch, tmp_path / "direct" / name)
    window.export_page()
    _wait(window)
    _save_to(monkeypatch, tmp_path / "dialog" / name)
    _export_from_dialog(window)
    assert (tmp_path / "dialog" / name).read_bytes() == (tmp_path / "direct" / name).read_bytes()


def test_what_the_canvas_shows_around_the_page_never_reaches_a_file(window, tmp_path, monkeypatch):
    _previewed(window)
    _save_to(monkeypatch, tmp_path / "plain.png")
    window.export_page()
    _wait(window)
    # Zoomed in, another version shown, a color picked in the inspector, the diagnostics open: the same file.
    window.preview.zoom_in()
    window.preview.zoom_in()
    window.version_bar.buttons[main_window.Version.COMPLETED].click()
    window.inspector.list.setCurrentRow(1)
    assert window.inspector.list.currentRow() == 1
    window.set_diagnostics_visible(True)
    _app.processEvents()
    _save_to(monkeypatch, tmp_path / "around.png")
    window.export_page()
    _wait(window)
    assert (tmp_path / "around.png").read_bytes() == (tmp_path / "plain.png").read_bytes()


def test_it_is_named_for_the_picture_and_put_beside_it(window, tmp_path, monkeypatch):
    picture = tmp_path / "harbor at dusk.png"
    from PIL import Image

    Image.fromarray(_blocks()[:, :, ::-1]).save(picture)
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(picture), "")))
    window.commands["open"].trigger()
    window.generate_preview()
    _wait(window)
    asked = []
    _save_to(monkeypatch, tmp_path / "out.png", asked)
    _export_from_dialog(window)
    assert asked[-1][2] == str(tmp_path / "harbor at dusk_page.png")
    window.controls.palette_radio.setChecked(True)
    _export_from_dialog(window)
    assert asked[-1][2].endswith("harbor at dusk_palette.swatches")


def test_a_palette_s_other_files_are_replaced_only_when_confirmed(window, tmp_path, monkeypatch):
    window.controls.controls["palette_margin_de00"].type_text("1")
    window.controls.controls["num_colors"].type_text("40")
    picture = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    window.current_image_bgr = np.repeat(np.repeat(picture[::10, ::10], 10, 0), 10, 1).copy()
    window.generate_preview()
    _wait(window)
    colors = len(window.current_page.palette_rgb)
    assert colors > swatches.PROCREATE_MAX_SWATCHES
    window.controls.palette_radio.setChecked(True)
    target = tmp_path / "set.swatches"
    files = swatches.palette_files(target, colors)
    files[1].write_bytes(b"mine")
    asked = []
    monkeypatch.setattr(main_window.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.Cancel))
    _save_to(monkeypatch, target)
    _export_from_dialog(window)
    assert files[1].name in asked[0] and files[1].read_bytes() == b"mine" and not files[0].exists()
    monkeypatch.setattr(main_window.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    _export_from_dialog(window)
    assert all(path.exists() for path in files) and files[1].read_bytes() != b"mine"


# -- Out of date ------------------------------------------------------------------------------
def test_an_out_of_date_preview_is_made_again_by_default_or_written_as_previewed(window, tmp_path, monkeypatch):
    _previewed(window)
    previewed = window.current_page
    _save_to(monkeypatch, tmp_path / "before.png")
    window.export_page()
    _wait(window)
    window.controls.preset_combo.setCurrentText("Realistic")  # another page altogether
    window.commands["export"].trigger()
    dialog = window.export_dialog
    assert dialog.stale_shown() and "Difficulty" in dialog.stale_text.text() and dialog.export_now.isChecked()
    dialog.reject()

    # As previewed: the page the preview shows, as the export before the change wrote it.
    _save_to(monkeypatch, tmp_path / "previewed.png")
    _export_from_dialog(window, previewed=True)
    assert (tmp_path / "previewed.png").read_bytes() == (tmp_path / "before.png").read_bytes()
    assert window.current_page is previewed

    # By default: made again with the settings as they are, as the old path did.
    _save_to(monkeypatch, tmp_path / "now.png")
    _export_from_dialog(window)
    _save_to(monkeypatch, tmp_path / "direct.png")
    window.export_page()
    _wait(window)
    assert (tmp_path / "now.png").read_bytes() == (tmp_path / "direct.png").read_bytes()
    assert (tmp_path / "now.png").read_bytes() != (tmp_path / "before.png").read_bytes()
    # The choice starts at its default each time.
    window.commands["export"].trigger()
    assert dialog.export_now.isChecked()


def test_the_legend_s_detail_alone_doesn_t_make_it_out_of_date(window):
    _previewed(window)
    window.controls.legend_combo.setCurrentIndex((window.controls.legend_combo.currentIndex() + 1) % 2)
    window.commands["export"].trigger()
    assert not window.export_dialog.stale_shown()


# -- Failure ----------------------------------------------------------------------------------
def test_a_failed_export_can_be_retried(window, tmp_path, monkeypatch):
    _previewed(window)
    real = export.export_png
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("the disk is full")
        return real(*args, **kwargs)

    monkeypatch.setattr(export, "export_png", flaky)
    _save_to(monkeypatch, tmp_path / "page.png")
    _export_from_dialog(window)
    bar = window.notification
    assert bar.property("level") == "error" and "the disk is full" in bar.text.text()
    assert bar.action_labels() == ["Retry", "Details"] and not (tmp_path / "page.png").exists()
    bar.trigger("Retry")
    _wait(window)
    assert (tmp_path / "page.png").exists() and bar.text.text().startswith("Exported page.png")


# -- The keyboard -----------------------------------------------------------------------------
def test_it_is_worked_from_the_keyboard(window, tmp_path, monkeypatch):
    _previewed(window)
    window.commands["export"].trigger()
    dialog = window.export_dialog
    _app.processEvents()
    # Tab reaches the options and the buttons.
    seen = set()
    widget = dialog.export_button
    for _ in range(40):
        widget = widget.nextInFocusChain()
        seen.add(widget)
    for wanted in (window.controls.version_combo, window.controls.png_radio, window.controls.resolution_spin,
                   dialog.export_button, dialog.buttons.button(dialog.buttons.StandardButton.Cancel)):
        assert wanted in seen
    # Esc closes it; Enter exports.
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert not dialog.isVisible()
    _save_to(monkeypatch, tmp_path / "keys.png")
    window.commands["export"].trigger()
    QTest.keyClick(dialog.export_button, Qt.Key_Return)
    _wait(window)
    assert (tmp_path / "keys.png").exists() and not dialog.isVisible()


def test_coloring_on_a_screen_points_to_the_palette_on_the_colors_page_and_in_the_dialog(window):
    window.controls.medium_combo.setCurrentText(difficulty.SCREEN_MEDIUM)
    assert not window.controls.medium_hint.isHidden() and not window.controls.screen_hint.isHidden()
    window.controls.medium_combo.setCurrentText(difficulty.DEFAULT_MEDIUM)
    assert window.controls.medium_hint.isHidden()

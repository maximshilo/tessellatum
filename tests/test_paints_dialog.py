"""The paints dialog lists the painter's paints as rows, adds them from a color picker or a kit, and round-trips them."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.core import paint_kits  # noqa: E402
from tessellatum.core.color import bgr_to_lab, ciede2000  # noqa: E402
from tessellatum.core.color_names import name_colors  # noqa: E402
from tessellatum.core.paints import MOST_PAINTS, paints_text, parse_paints  # noqa: E402
from tessellatum.gui import paints_dialog  # noqa: E402
from tessellatum.gui.paints_dialog import DELETE, NAME, SWATCH, PaintsDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])

PAINTS = (("Ultramarine", (32, 0, 177)), ("Sap green", (90, 107, 47)), ("", (250, 250, 250)))


@pytest.fixture
def picks(monkeypatch):
    """The colors the stubbed color dialog will pick, in turn (None: cancelled), and what it was opened with."""
    queue: list = []
    opened: list = []

    def pick(_parent, initial, _title):
        opened.append(initial)
        return queue.pop(0)

    monkeypatch.setattr(paints_dialog, "pick_color", pick)
    return queue, opened


def test_the_list_shows_each_paint_and_gives_them_back_unchanged():
    dialog = PaintsDialog(PAINTS)
    assert dialog.table.rowCount() == 3
    assert dialog.table.item(0, SWATCH).text() == "#2000B1" and dialog.table.item(0, NAME).text() == "Ultramarine"
    assert dialog.paints() == PAINTS
    assert parse_paints(paints_text(dialog.paints())).paints == PAINTS  # as the panel stores them
    assert dialog.status.text() == "3 paints."


def test_add_color_names_it_as_the_legend_would_and_never_adds_a_color_twice(picks):
    queue, _opened = picks
    dialog = PaintsDialog(PAINTS)
    queue.append((200, 30, 40))
    dialog.add_color()
    assert dialog.paints()[-1] == (name_colors([(200, 30, 40)])[0], (200, 30, 40))
    queue.append((90, 107, 47))
    dialog.add_color()
    assert len(dialog.paints()) == 4 and "listed already, as Sap green" in dialog.status.text()
    queue.append(None)  # cancelled
    dialog.add_color()
    assert len(dialog.paints()) == 4


def test_renaming_repicking_and_deleting_a_row(picks):
    queue, opened = picks
    dialog = PaintsDialog(PAINTS)
    dialog.table.item(1, NAME).setText("Olive  ")
    assert dialog.paints()[1] == ("Olive", (90, 107, 47))
    queue.append((60, 90, 40))
    dialog.table.cellClicked.emit(1, SWATCH)
    assert opened[-1] == (90, 107, 47)
    assert dialog.paints()[1] == ("Olive", (60, 90, 40))  # a name typed in is kept
    queue.append((32, 0, 177))  # another row's color: refused
    dialog.repick(1)
    assert dialog.paints()[1][1] == (60, 90, 40) and "listed already" in dialog.status.text()
    queue.append((0, 0, 0))
    dialog.repick(2)  # no name: it gets the new color's
    assert dialog.paints()[2] == (name_colors([(0, 0, 0)])[0], (0, 0, 0))
    dialog.table.cellClicked.emit(0, DELETE)
    assert [name for name, _ in dialog.paints()] == ["Olive", name_colors([(0, 0, 0)])[0]]


def test_a_name_the_dialog_gave_follows_a_repicked_color(picks):
    queue, _opened = picks
    dialog = PaintsDialog(())
    queue.append((200, 30, 40))
    dialog.add_color()
    queue.append((30, 60, 200))
    dialog.repick(0)
    assert dialog.paints() == ((name_colors([(30, 60, 200)])[0], (30, 60, 200)),)


def test_edit_as_text_round_trips_and_reports_lines_with_no_color():
    dialog = PaintsDialog(PAINTS)
    dialog.text_button.setChecked(True)
    assert parse_paints(dialog.editor.toPlainText()).paints == PAINTS
    assert dialog.paints() == PAINTS
    dialog.editor.setPlainText(dialog.editor.toPlainText() + "\nno color here\n#123456 Ink\n#123456 Again")
    assert "No color found on line 4" in dialog.status.text() and "counts once" in dialog.status.text()
    dialog.text_button.setChecked(False)
    assert dialog.paints() == PAINTS + (("Ink", (0x12, 0x34, 0x56)),)
    assert "No color found on line 4" in dialog.status.text()


def test_the_status_says_paints_were_left_off_only_when_some_were():
    exactly = tuple((f"P{i}", (i, 0, 0)) for i in range(MOST_PAINTS))
    dialog = PaintsDialog(exactly)
    dialog.text_button.setChecked(True)
    assert "left off" not in dialog.status.text() and not dialog.status.text().startswith("257")
    dialog.editor.setPlainText(paints_text(exactly + (("More", (0, 1, 0)),)))
    assert "1 left off" in dialog.status.text()
    dialog.text_button.setChecked(False)
    assert len(dialog.paints()) == MOST_PAINTS and not dialog.add_button.isEnabled()


def test_a_kit_replaces_the_list_or_is_added_to_it_and_asks_which(monkeypatch):
    acrylics = paint_kits.kit("acrylics")
    dialog = PaintsDialog(())
    dialog.start_from_kit(acrylics)  # nothing listed: nothing to ask
    assert dialog.paints() == acrylics.paints and "not any maker's" in dialog.status.text()
    answers = []
    monkeypatch.setattr(paints_dialog, "ask_replace", lambda _parent, _kit, listed: answers.pop(0))
    answers.append(None)  # cancelled
    dialog.start_from_kit(paint_kits.kit("grays"))
    assert dialog.paints() == acrylics.paints
    answers.append(False)  # add
    dialog.start_from_kit(paint_kits.kit("earth"))
    earth = paint_kits.kit("earth").paints
    added = [paint for paint in earth if paint[1] not in {rgb for _name, rgb in acrylics.paints}]
    assert dialog.paints() == acrylics.paints + tuple(added)
    assert f"{len(earth) - len(added)} listed already" in dialog.status.text()
    answers.append(True)  # replace
    dialog.start_from_kit(paint_kits.kit("primaries"))
    assert dialog.paints() == paint_kits.kit("primaries").paints


def test_the_kit_menu_offers_every_kit():
    dialog = PaintsDialog(())
    assert [action.text() for action in dialog.kit_button.menu().actions()] == [k.name for k in paint_kits.KITS]
    dialog.kit_button.menu().actions()[1].trigger()
    assert dialog.paints() == paint_kits.KITS[1].paints


@pytest.mark.parametrize("kit", paint_kits.KITS, ids=lambda kit: kit.key)
def test_each_kit_s_colors_are_distinct_named_and_keep_the_presets_margin(kit):
    import numpy as np

    colors = [rgb for _name, rgb in kit.paints]
    assert len(set(colors)) == len(colors) and all(name for name, _ in kit.paints)
    assert " " not in kit.key and paint_kits.kit(kit.key.upper()) is kit
    assert kit.name.endswith(f"({len(colors)})")
    lab = bgr_to_lab(np.array(colors, dtype=np.uint8)[:, ::-1].copy())
    apart = ciede2000(lab[:, None], lab[None])
    np.fill_diagonal(apart, np.inf)
    assert apart.min() >= 4.0  # the presets' color margin


def test_the_grays_kit_steps_evenly_in_lightness():
    import numpy as np

    grays = np.array([rgb for _name, rgb in paint_kits.kit("grays").paints], dtype=np.uint8)
    lightness = bgr_to_lab(grays[:, ::-1].copy())[:, 0]
    assert np.allclose(lightness, [0, 20, 40, 60, 80, 100], atol=0.5)

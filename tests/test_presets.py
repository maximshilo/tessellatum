"""W7: user presets. Saved, renamed and deleted by name in a versioned JSON file; read back the same; a file that can't
be read is left out with a warning and set aside, never lost; built-in names can't be taken."""

import json
import logging
import os

import pytest

from tessellatum.app.presets import FILE_NAME, VERSION, PresetStore, owned
from tessellatum.core import difficulty, settings


def _values(preset="Hard", **changes):
    values = settings.preset_values(preset)
    values.update(changes)
    return values


def test_a_preset_round_trips_through_the_file(tmp_path):
    path = tmp_path / FILE_NAME
    store = PresetStore(path)
    assert store.names() == []
    name = store.save("  Fine fur ", _values(num_colors=30))
    assert name == "Fine fur" and store.names() == ["Fine fur"]
    again = PresetStore(path)
    assert again.names() == ["Fine fur"]
    assert again.values("Fine fur") == {s.name: float(v) for s, v in
                                        ((s, _values(num_colors=30)[s.name]) for s in owned())}
    assert again.params("Fine fur").num_colors == 30
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == VERSION and set(data["presets"]["Fine fur"]) == {s.name for s in owned()}


def test_only_the_settings_a_preset_owns_are_kept(tmp_path):
    store = PresetStore(tmp_path / FILE_NAME)
    values = {**settings.factory_values(), "num_colors": 18}
    store.save("Mine", values)
    kept = store.values("Mine")
    assert set(kept) == {s.name for s in owned()} and kept["num_colors"] == 18
    assert "line_width_mm" not in kept  # a page setting, never part of a preset


def test_rename_and_delete(tmp_path):
    path = tmp_path / FILE_NAME
    store = PresetStore(path)
    store.save("A", _values())
    store.save("B", _values("Easy"))
    assert store.rename("A", "Zebra") == "Zebra"
    assert store.names() == ["B", "Zebra"]
    with pytest.raises(ValueError):
        store.rename("B", "Zebra")  # taken
    store.delete("B")
    assert PresetStore(path).names() == ["Zebra"]


def test_two_presets_never_differ_only_in_capitals(tmp_path):
    store = PresetStore(tmp_path / FILE_NAME)
    store.save("Fur", _values(num_colors=30))
    assert store.find(" fur ") == "Fur" and store.find("Feathers") is None
    assert store.save("fur", _values(num_colors=20)) == "fur"  # replaces it, in the new spelling
    assert store.names() == ["fur"] and store.params("fur").num_colors == 20
    store.save("Feathers", _values())
    with pytest.raises(ValueError):
        store.rename("Feathers", "FUR")  # taken, but for capitals
    assert store.rename("fur", "FUR") == "FUR"  # its own name in other capitals
    assert store.names() == ["Feathers", "FUR"]


def test_built_in_names_are_reserved(tmp_path):
    store = PresetStore(tmp_path / FILE_NAME)
    for name in (*difficulty.preset_names(), "medium", "  HARD ", ""):
        with pytest.raises(ValueError):
            store.save(name, _values())
    assert store.names() == []


def test_values_out_of_range_are_refused(tmp_path):
    store = PresetStore(tmp_path / FILE_NAME)
    with pytest.raises(ValueError):
        store.save("Too many", _values(num_colors=10_000))
    assert store.names() == []


@pytest.mark.parametrize("text", ["{not json", '{"version": 99, "presets": {}}', '{"version": 1}',
                                  '{"version": 1, "presets": {"X": {"num_colors": "many"}}}',
                                  '{"version": 1, "presets": {"Medium": {}}}'])
def test_a_file_that_cant_be_read_is_ignored_with_a_warning_and_kept_aside(tmp_path, caplog, text):
    path = tmp_path / FILE_NAME
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tessellatum.presets"):
        store = PresetStore(path)
    assert store.names() == [] and "can't be read" in caplog.text
    assert path.read_text(encoding="utf-8") == text  # untouched until a preset is saved
    store.save("New", _values())
    assert (tmp_path / (FILE_NAME + ".bak")).read_text(encoding="utf-8") == text
    assert PresetStore(path).names() == ["New"]


def test_a_missing_setting_takes_the_default_presets_value(tmp_path):
    path = tmp_path / FILE_NAME
    path.write_text(json.dumps({"version": 1, "presets": {"Short": {"num_colors": 12}}}), encoding="utf-8")
    store = PresetStore(path)
    defaults = settings.preset_values(difficulty.DEFAULT_PRESET)
    values = store.values("Short")
    assert values["num_colors"] == 12
    assert all(values[k] == float(v) for k, v in defaults.items() if k != "num_colors")


def test_a_store_without_a_path_keeps_nothing(tmp_path):
    store = PresetStore()
    store.save("Mine", _values())
    assert store.names() == ["Mine"] and list(tmp_path.iterdir()) == []


# -- In the panel and the window -----------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.controls_panel import CUSTOM, ControlsPanel  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
PAINTS = (("Titanium white", (245, 245, 240)), ("Mars black", (30, 30, 30)))


def _answers(panel, *names, yes=True):
    names = list(names)
    panel.ask_name = lambda *a, **k: names.pop(0) if names else None
    panel.ask_yes = lambda *a, **k: yes


def _settle():
    for _ in range(3):
        _app.processEvents()


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()
    _app.processEvents()


def test_saving_a_preset_keeps_the_difficulty_and_chooses_it(tmp_path):
    panel = ControlsPanel(presets=PresetStore(tmp_path / FILE_NAME))
    panel.paper_combo.setCurrentText("A3")
    panel.medium_combo.setCurrentText(difficulty.SCREEN_MEDIUM)
    panel.set_paints(PAINTS)
    panel.controls["num_colors"].step(3)
    assert panel.preset_combo.currentText() == CUSTOM
    _answers(panel, "Fine fur")
    assert panel.save_preset() == "Fine fur"
    assert panel.preset_combo.currentText() == "Fine fur"
    items = [panel.preset_combo.itemText(i) for i in range(panel.preset_combo.count())]
    assert items == [*difficulty.PRESETS, "Fine fur", CUSTOM]
    saved = panel.controls["num_colors"].value()
    # Another preset, then this one again: its values, and nothing else changed.
    panel.preset_combo.setCurrentText("Easy")
    panel.preset_combo.setCurrentText("Fine fur")
    assert panel.controls["num_colors"].value() == saved
    assert panel.paper_combo.currentText() == "A3" and panel.medium_combo.currentText() == difficulty.SCREEN_MEDIUM
    assert panel.get_paints() == PAINTS
    # The document made from it: the values, as Custom would make them, for the medium.
    snapshot = panel.snapshot()
    assert snapshot.preset == "Fine fur"
    assert snapshot.difficulty_params() == snapshot.replace(preset=CUSTOM).difficulty_params()
    # Kept for the next run.
    assert "Fine fur" in ControlsPanel(presets=PresetStore(tmp_path / FILE_NAME)).presets


def test_saving_or_renaming_onto_a_name_in_other_capitals_asks_first(tmp_path):
    panel = ControlsPanel(presets=PresetStore(tmp_path / FILE_NAME))
    panel.save_preset("Fur")
    panel.controls["num_colors"].step(2)
    asked = []
    panel.ask_name = lambda *a, **k: "fur"
    panel.ask_yes = lambda title, text: asked.append(text) or False
    assert panel.save_preset() is None and "Fur" in asked[-1] and panel.presets.names() == ["Fur"]
    panel.ask_yes = lambda title, text: True
    assert panel.save_preset() == "fur" and panel.presets.names() == ["fur"]
    panel.save_preset("Feathers")
    panel.ask_name = lambda *a, **k: "FUR"
    panel.ask_yes = lambda title, text: asked.append(text) or False
    assert panel.rename_preset() is None and "fur" in asked[-1]
    assert panel.presets.names() == ["Feathers", "fur"]


def test_a_built_in_name_is_asked_again_and_cancel_saves_nothing():
    panel = ControlsPanel()
    asked = []
    names = ["Medium", "  ", "Mine"]

    def ask(title, label, text=""):
        asked.append(label)
        return names.pop(0)

    panel.ask_name = ask
    assert panel.save_preset() == "Mine"
    assert asked == ["Name:", "Medium is a built-in preset's name. Another name:", "A preset needs a name:"]
    _answers(panel)  # cancelled
    panel.preset_combo.setCurrentText("Hard")
    assert panel.save_preset() is None and panel.presets.names() == ["Mine"]


def test_the_modified_indicator_and_back_to_the_preset():
    panel = ControlsPanel()
    panel.preset_combo.setCurrentText("Hard")
    assert not panel.revert_preset_action.isEnabled()
    hard = panel.snapshot().values
    panel.controls["num_colors"].step(-2)
    panel.controls["blur_sigma"].step(1)
    text = panel.preset_description.text()
    assert text.startswith("Hard (modified, 2 changes): Colors ") and "Smoothing" in text
    assert panel.revert_preset_action.isEnabled() and panel.revert_preset_action.text() == "Back to Hard"
    panel.revert_preset_action.trigger()
    assert panel.preset_combo.currentText() == "Hard" and panel.snapshot().values == hard
    assert not panel.revert_preset_action.isEnabled()


def test_a_snapshot_of_ones_own_preset_keeps_its_values_when_the_preset_has_changed_since(tmp_path):
    panel = ControlsPanel(presets=PresetStore(tmp_path / FILE_NAME))
    panel.controls["num_colors"].step(6)  # 30 colors
    _answers(panel, "Fine fur")
    panel.save_preset()
    saved = panel.snapshot()
    assert saved.preset == "Fine fur" and saved.values["num_colors"] == 30
    # The preset saved again with other values (or the project opened where it has others).
    panel.presets.save("Fine fur", {**saved.values, "num_colors": 20})
    panel.preset_combo.setCurrentText("Easy")
    panel.show_settings(saved)
    assert panel.controls["num_colors"].value() == 30  # the snapshot's, not the preset's 20
    assert panel.preset_combo.currentText() == CUSTOM and panel.snapshot().values == saved.values
    assert panel.snapshot().difficulty_params() == saved.difficulty_params()
    # Unchanged, it shows as the preset.
    panel.presets.save("Fine fur", saved.values)
    panel.show_settings(saved)
    assert panel.preset_combo.currentText() == "Fine fur" and panel.snapshot().values == saved.values


def test_rename_and_delete_in_the_panel(tmp_path):
    panel = ControlsPanel(presets=PresetStore(tmp_path / FILE_NAME))
    assert not panel.rename_preset_action.isEnabled() and not panel.delete_preset_action.isEnabled()
    panel.controls["num_colors"].step(2)
    _answers(panel, "Mine")
    panel.save_preset()
    values = panel.snapshot().values
    assert panel.rename_preset_action.isEnabled() and panel.delete_preset_action.isEnabled()
    _answers(panel, "Ours")
    assert panel.rename_preset() == "Ours" and panel.preset_combo.currentText() == "Ours"
    assert panel.presets.names() == ["Ours"]
    _answers(panel, yes=False)
    assert not panel.delete_preset() and panel.presets.names() == ["Ours"]
    _answers(panel)
    assert panel.delete_preset()
    assert panel.presets.names() == [] and panel.preset_combo.currentText() == CUSTOM
    assert panel.snapshot().values == values  # the settings stay as they were
    assert panel.preset_combo.findText("Ours") == -1


def test_a_preset_is_one_undo_step_and_undo_to_a_deleted_one_keeps_its_values(window):
    controls = window.controls
    controls.controls["num_colors"].step(3)
    _settle()
    _answers(controls, "Mine")
    controls.save_preset()
    _settle()
    mine = controls.snapshot()
    count = window.history.count()
    controls.preset_combo.setCurrentText("Beginner")
    _settle()
    assert window.commands["undo"].text().replace("&", "") == "Undo Difficulty: Beginner"
    assert window.history.count() == count + 1
    window.undo()
    assert controls.snapshot() == mine
    _answers(controls)
    controls.delete_preset()
    _settle()
    controls.preset_combo.setCurrentText("Hard")
    _settle()
    window.undo()  # Hard undone: the settings after the deletion, Custom
    window.undo()  # the deletion undone: the preset is gone, so Custom, with its values
    assert controls.preset_combo.currentText() == CUSTOM and controls.snapshot().values == mine.values


def test_only_the_global_reset_asks_first(window, monkeypatch):
    controls = window.controls
    asked = []
    answer = [main_window.QMessageBox.Cancel]
    monkeypatch.setattr(main_window.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[1]) or answer[0]))
    controls.controls["line_width_mm"].step(2)
    controls.controls["num_colors"].step(2)
    controls.sections["Detail"].reset_button.click()  # a section's reset: no question
    assert asked == [] and controls.preset_combo.currentText() == difficulty.DEFAULT_PRESET
    window.commands["reset"].trigger()
    assert asked == ["Reset All Settings"] and controls.changed_settings() == ["line_width_mm"]  # cancelled
    answer[0] = main_window.QMessageBox.Reset
    controls.reset_button.click()
    assert len(asked) == 2 and controls.changed_settings() == []


def test_the_preset_chosen_is_kept_between_runs(tmp_path):
    def store():
        return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)

    first_store = store()
    panel = ControlsPanel(store=first_store, presets=PresetStore(tmp_path / FILE_NAME))
    panel.controls["num_colors"].step(2)
    _answers(panel, "Mine")
    panel.save_preset()
    values = panel.snapshot().values
    first_store.sync()
    again = ControlsPanel(store=store(), presets=PresetStore(tmp_path / FILE_NAME))
    assert again.preset_combo.currentText() == "Mine" and again.snapshot().values == values
    # Its file gone: the values kept, Custom.
    (tmp_path / FILE_NAME).unlink()
    third = ControlsPanel(store=store(), presets=PresetStore(tmp_path / FILE_NAME))
    assert third.preset_combo.currentText() == CUSTOM and third.snapshot().values == values


def test_the_practical_description_is_for_the_paper_chosen():
    assert difficulty.practical("Medium") == ("About 450 regions to paint on A4 paper "
                                              "(200 to 1,250 on most pictures).")
    assert "on A3 paper" in difficulty.practical("Medium", "A3")
    assert difficulty.practical("Medium", "A2").startswith("About 2,000 regions")
    assert difficulty.practical("Custom") is None

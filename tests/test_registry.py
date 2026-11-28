"""The settings registry (``core.settings``): every pipeline field covered once, the presets' values, validation
without clamping, enablement with reasons, and parity with the panel: every entry has a control, and every key the
app has kept between runs still loads."""

import dataclasses
import os

import pytest

from tessellatum.core import difficulty, settings
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.pipeline import Handling
from tessellatum.core.render import TONE_FIELDS, TONES, PageStyle

OWNERS = {"DifficultyParams": DifficultyParams, "PageStyle": PageStyle, "Handling": Handling}
OWNER_NAMES = {settings.DIFFICULTY: "DifficultyParams", settings.STYLE: "PageStyle", settings.HANDLING: "Handling"}

# Every key the app kept between runs up to 1.0.21 (plans' FEATURE_PARITY, table 6): each must still load.
KEPT_KEYS = (
    "difficulty/preset", "style/tone", "style/paper", "difficulty/medium", "style/legend", "export/palette",
    "difficulty/colors_from", "difficulty/paints", "difficulty/mix_paints", "export/unit", "export/dpi",
    "export/pixels", *(f"settings/{s.name}" for s in settings.SETTINGS),
    *(f"handling/{name}" for name in ("line_art", "detail", "text")),
)


def test_every_pipeline_field_is_covered_once():
    covered = [f"{OWNER_NAMES[s.owner]}.{s.name}" for s in settings.SETTINGS]
    covered += [c.field for c in settings.CHOICES if c.field]
    covered += list(settings.INTERNAL_FIELDS)
    fields = [f"{name}.{f.name}" for name, owner in OWNERS.items() for f in dataclasses.fields(owner)]
    assert sorted(covered) == sorted(fields)


def test_ids_are_unique_and_every_entry_is_described():
    names = [e.name for e in settings.entries()]
    assert len(names) == len(set(names))
    for e in settings.entries():
        assert e.category in settings.CATEGORIES and e.level in settings.LEVELS and e.policy in settings.POLICIES
        assert e.section and e.label and e.tooltip and e.aliases, e.name
    for s in settings.SETTINGS:
        assert s.unit_kind in settings.UNIT_KINDS, s.name
    keys = [e.key for e in settings.entries()]
    assert len(keys) == len(set(keys))


def test_the_presets_and_defaults_match_difficulty():
    for preset in difficulty.PRESETS:
        params = difficulty.params_for_preset(preset)
        values = settings.preset_values(preset)
        assert set(values) == {f.name for f in dataclasses.fields(DifficultyParams)}
        assert all(values[name] == getattr(params, name) for name in values)
    assert {s.name for s in settings.SETTINGS if s.preset_owned} == {f.name for f in dataclasses.fields(DifficultyParams)}
    factory = settings.factory_values()
    assert {k: factory[k] for k in settings.preset_values(difficulty.DEFAULT_PRESET)} == settings.preset_values(
        difficulty.DEFAULT_PRESET)
    assert [factory[f] for f in TONE_FIELDS] == list(TONES[settings.choice("tone").default])
    for s in settings.SETTINGS:
        assert s.check(s.default) is None, s.name
    for c in settings.CHOICES:
        assert c.check(c.default) is None, c.name


def test_validation_gives_a_reason_and_never_clamps():
    colors = settings.setting("num_colors")
    assert colors.check(colors.maximum + 1) and "most" in colors.check(colors.maximum + 1)
    assert colors.check(colors.minimum - 1) and "least" in colors.check(colors.minimum - 1)
    assert colors.check(12.5) and "whole" in colors.check(12.5)
    assert colors.check("many")
    assert settings.setting("blur_sigma").check(0.95) is None
    assert settings.choice("paper").check("A0") and settings.choice("paper").check("Letter") is None
    assert settings.choice("text").check("yes") and settings.choice("text").check(False) is None
    assert settings.choice("paints").check((("White", (255, 255, 255)),)) is None
    assert settings.choice("paints").check(("White",))


def test_enablement_says_why():
    chosen = {"medium": difficulty.SCREEN_MEDIUM, "colors_from": settings.PICTURE_COLORS, "mix_paints": True,
              "detail": False, "line_art": True, "text": False}
    assert "screen" in settings.setting("paper_black_lstar").disabled_reason(chosen).lower()
    assert "My paints" in settings.setting("mix_gain_de00").disabled_reason(chosen)
    assert "My paints" in settings.choice("mix_paints").disabled_reason(chosen)
    assert settings.setting("mark_contrast").disabled_reason(chosen) == settings.ONLY_WITH_DETAIL
    assert settings.setting("text_max_height_mm").disabled_reason(chosen) == settings.ONLY_WITH_TEXT
    assert settings.setting("thin_ink_mm").disabled_reason(chosen) is None
    mine = {**chosen, "colors_from": settings.MY_PAINTS, "mix_paints": False}
    assert "Mix colors" in settings.setting("mix_min_share_pct").disabled_reason(mine)
    assert settings.setting("mix_min_share_pct").disabled_reason({**mine, "mix_paints": True}) is None
    assert settings.setting("num_colors").disabled_reason(chosen) is None


def test_search_finds_by_label_and_alias():
    assert settings.setting("blur_sigma") in settings.find("blur")
    assert settings.setting("min_width_mm") in settings.find("Brush")
    assert settings.choice("medium") in settings.find("procreate")
    assert settings.find("  ") == []


def test_only_the_legend_applies_at_once():
    assert [e.name for e in settings.entries() if e.policy == settings.IMMEDIATE] == ["legend_detail"]


# -- Parity with the panel ---------------------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_every_entry_has_a_control(qt_app):
    from tessellatum.gui.controls_panel import ControlsPanel

    panel = ControlsPanel()
    assert set(panel.controls) == {s.name for s in settings.SETTINGS}
    assert set(panel.choice_widgets) == {c.name for c in settings.CHOICES}
    for name, widget in panel.choice_widgets.items():
        assert widget.toolTip(), name
    panel.deleteLater()


def test_the_panel_disables_what_means_nothing_and_says_why(qt_app):
    from tessellatum.gui.controls_panel import ControlsPanel

    panel = ControlsPanel()
    panel.detail_check.setChecked(False)
    mark = panel.controls["mark_contrast"]
    assert not mark.isEnabled() and mark.toolTip().startswith(settings.ONLY_WITH_DETAIL)
    panel.detail_check.setChecked(True)
    assert mark.isEnabled() and mark.toolTip() == settings.setting("mark_contrast").tooltip
    panel.medium_combo.setCurrentText(difficulty.SCREEN_MEDIUM)
    assert not panel.controls["paper_black_lstar"].isEnabled()
    assert not panel.mix_check.isEnabled() and not panel.controls["mix_gain_de00"].isEnabled()
    panel.deleteLater()


def test_every_key_kept_by_earlier_versions_loads(qt_app, tmp_path):
    """A settings file as 1.0.21 wrote it, every key in it, read by this version's panel."""
    from PySide6.QtCore import QSettings

    from tessellatum.core.paints import paints_text
    from tessellatum.gui.controls_panel import ControlsPanel

    registry_keys = {e.key for e in settings.entries()}
    assert {k for k in KEPT_KEYS if not k.startswith("export/")} <= registry_keys

    old = QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat)
    custom = {s.name: s.default for s in settings.SETTINGS}
    custom.update(num_colors=30, blur_sigma=1.5, line_gray=60, brush_px=5.0, text_max_height_mm=20.0)
    paints = (("Ultramarine", (32, 0, 177)), ("Cadmium yellow", (255, 200, 0)))
    written = {
        "difficulty/preset": "Custom", "style/tone": "Custom", "style/paper": "A3", "difficulty/medium": "Screen",
        "style/legend": "Name", "export/palette": ".gpl", "difficulty/colors_from": "My paints",
        "difficulty/paints": paints_text(paints), "difficulty/mix_paints": True, "export/unit": "px",
        "export/dpi": 300, "export/pixels": 5000,
        **{f"settings/{name}": value for name, value in custom.items()},
        "handling/line_art": True, "handling/detail": False, "handling/text": True,
    }
    assert set(written) == set(KEPT_KEYS)
    for key, value in written.items():
        old.setValue(key, value)
    old.sync()

    panel = ControlsPanel(store=QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat))
    s = panel.snapshot()
    assert (s.preset, s.tone, s.paper, s.medium, s.legend, s.colors_from) == (
        "Custom", "Custom", "A3", "Screen", "Name", "My paints")
    assert s.paints == paints and s.mix and not s.detail and s.line_art and s.text
    assert (s.export.palette_format, s.export.unit, s.export.dpi, s.export.pixels) == (".gpl", "px", 300, 5000)
    for name in ("num_colors", "blur_sigma", "line_gray", "brush_px", "text_max_height_mm"):
        assert s.values[name] == pytest.approx(custom[name], abs=settings.setting(name).step / 2), name
    panel.deleteLater()

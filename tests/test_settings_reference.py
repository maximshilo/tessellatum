"""The wiki's settings reference is the registry's: generated from it, and current."""

import importlib.util
from pathlib import Path

from tessellatum.core import settings

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("make_settings_reference", ROOT / "docs" / "make_settings_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_settings_reference_is_current():
    generator = _generator()
    assert generator.PAGE.read_text(encoding="utf-8") == generator.page(), (
        "wiki/Settings-Reference.md is out of date: run docs/make_settings_reference.py")


def test_every_setting_and_choice_is_in_it():
    text = _generator().page()
    for entry in settings.entries():
        assert f"(`{entry.name}`)" in text, entry.name

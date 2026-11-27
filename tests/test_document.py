"""The document model and its settings snapshot, without any widget."""

import os

import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import snapshot  # noqa: E402
from tessellatum.app.document import Document, content_hash  # noqa: E402
from tessellatum.app.snapshot import CUSTOM, MY_PAINTS, DocumentSettings, default_values  # noqa: E402
from tessellatum.core import difficulty, settings  # noqa: E402
from tessellatum.core.pipeline import Handling  # noqa: E402
from tessellatum.core.render import TONE_FIELDS, TONES, PageStyle  # noqa: E402

# A QApplication though the model needs none: the first application made is the suite's, and the widget tests run
# after this file need one. What the model itself imports is checked below.
_app = QApplication.instance() or QApplication([])

PAINTS = (("Ultramarine", (32, 0, 177)), ("Titanium white", (250, 250, 250)))


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        _app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError
        time.sleep(0.005)


def test_the_model_needs_no_widgets():
    import re
    from pathlib import Path

    import tessellatum.app

    for module in Path(tessellatum.app.__file__).parent.glob("*.py"):
        imports = re.findall(r"^\s*(?:from|import)\s+(\S+)", module.read_text(encoding="utf-8"), re.M)
        assert not [i for i in imports if "QtWidgets" in i or i.startswith("tessellatum.gui")], module.name


# -- The snapshot -------------------------------------------------------------------
def test_defaults_are_the_panels():
    s = DocumentSettings()
    assert s.preset == difficulty.DEFAULT_PRESET and s.paper == "A4" and s.colors_from == snapshot.PICTURE_COLORS
    assert set(s.values) == {x.name for x in settings.SETTINGS}
    medium = difficulty.params_for_preset(difficulty.DEFAULT_PRESET)
    assert s.values["num_colors"] == medium.num_colors
    assert [s.values[f] for f in TONE_FIELDS] == list(TONES[s.tone])
    assert s.difficulty_params() == difficulty.for_medium(medium, s.medium)
    assert s.page_style() == PageStyle(paper="A4", legend_detail=s.legend)
    assert s.handling() == Handling()


def test_a_preset_is_the_presets_whatever_the_values_say():
    s = DocumentSettings(preset="Hard").with_values(num_colors=3)
    assert s.difficulty_params() == difficulty.for_medium(difficulty.params_for_preset("Hard"), s.medium)
    custom = s.replace(preset=CUSTOM)
    assert custom.difficulty_params().num_colors == 3


def test_paints_count_only_when_the_colors_come_from_them():
    s = DocumentSettings(paints=PAINTS, mix=True)
    assert s.handling().paints == () and not s.handling().mix_paints
    mine = s.replace(colors_from=MY_PAINTS)
    assert mine.handling().paints == PAINTS and mine.handling().mix_paints
    assert not mine.replace(paints=()).handling().mix_paints


def test_snapshots_are_values():
    a, b = DocumentSettings(), DocumentSettings()
    assert a == b and hash(a) == hash(b)
    c = a.with_values(line_width_mm=0.4)
    assert c != a and a.values["line_width_mm"] != 0.4  # the first is unchanged
    with pytest.raises(TypeError):
        c.values["line_width_mm"] = 1  # read-only
    with pytest.raises(KeyError):
        a.with_values(no_such_setting=1)
    with pytest.raises(ValueError):
        a.replace(paper="Letter")
    with pytest.raises(ValueError):
        DocumentSettings(values={"num_colors": 3})


def test_the_legend_and_export_options_dont_make_another_page():
    s = DocumentSettings()
    assert s.replace(legend="Number only").generation_key() == s.generation_key()
    assert s.replace(export=snapshot.ExportOptions(format=snapshot.PDF)).generation_key() == s.generation_key()
    assert s.with_values(line_width_mm=0.5).generation_key() != s.generation_key()
    assert s.replace(paper="A3").generation_key() != s.generation_key()


def test_default_values_follow_the_preset_and_tone():
    values = default_values("Beginner", "Dark")
    assert values["num_colors"] == difficulty.params_for_preset("Beginner").num_colors
    assert [values[f] for f in TONE_FIELDS] == list(TONES["Dark"])


# -- The document ---------------------------------------------------------------------
def test_one_signal_per_edit_and_none_for_no_change():
    doc = Document()
    seen = []
    doc.settings_changed.connect(seen.append)
    assert doc.set_settings(doc.settings().with_values(num_colors=12))
    assert not doc.set_settings(doc.settings())
    assert not doc.set_settings(DocumentSettings().with_values(num_colors=12))  # equal, though another object
    assert doc.set_settings(doc.settings().replace(preset="Hard"))
    assert seen == [1, 2] and doc.revision() == 2


def test_dirty_until_saved():
    doc = Document()
    changes = []
    doc.dirty_changed.connect(changes.append)
    assert not doc.is_dirty()
    doc.set_settings(doc.settings().replace(paper="A3"))
    doc.set_settings(doc.settings().replace(paper="A2"))
    assert doc.is_dirty() and changes == [True]
    doc.mark_saved()
    assert not doc.is_dirty() and changes == [True, False]


def test_a_new_picture_drops_the_old_result_and_is_hashed_off_the_gui_thread():
    doc = Document()
    image = np.zeros((20, 30, 3), np.uint8)
    doc.set_source(image)
    doc.set_result(object())
    found = []
    doc.source_identified.connect(found.append)
    other = np.full((20, 30, 3), 7, np.uint8)
    source = doc.set_source(other)
    assert doc.result() is None and doc.source() is source
    wait_for(lambda: found)
    assert found == [content_hash(other)] and source.content_hash == found[0]


def test_a_stale_hash_is_ignored():
    doc = Document()
    first = doc.set_source(np.zeros((10, 10, 3), np.uint8))
    second = doc.set_source(np.ones((10, 10, 3), np.uint8))
    found = []
    doc.source_identified.connect(found.append)
    doc._on_hashed(first.token, "stale")
    assert found == [] and second.content_hash != "stale"


def test_equal_pictures_hash_equal():
    a = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    assert content_hash(a) == content_hash(a.copy())
    assert content_hash(a) != content_hash(a.reshape(5, 4, 3))
    assert content_hash(a[:, ::-1]) == content_hash(np.ascontiguousarray(a[:, ::-1]))


def test_the_result_is_current_until_a_setting_that_makes_the_page_changes():
    doc = Document()
    doc.set_source(np.zeros((10, 10, 3), np.uint8))
    assert not doc.result_is_current()
    doc.set_result(object())
    assert doc.result_is_current()
    doc.set_settings(doc.settings().replace(legend="Name"))
    assert doc.result_is_current()
    doc.set_settings(doc.settings().with_values(num_colors=10).replace(preset=CUSTOM))
    assert not doc.result_is_current()


def test_a_document_deleted_while_hashing_leaves_the_thread_quiet():
    """The hashing thread reports through an object that outlives the document: deleting the document first raises
    nothing in the thread, and a document made after still gets its own hash."""
    import threading

    from PySide6.QtCore import QEvent

    errors = []
    previous = threading.excepthook
    threading.excepthook = errors.append
    try:
        for _ in range(5):
            doc = Document()
            doc.set_source(np.random.default_rng(0).integers(0, 255, (1500, 2000, 3), np.uint8))
            doc.deleteLater()
            _app.sendPostedEvents(None, QEvent.DeferredDelete)
            del doc
        survivor = Document()
        found = []
        survivor.source_identified.connect(found.append)
        image = np.full((50, 60, 3), 9, np.uint8)
        survivor.set_source(image)
        wait_for(lambda: found)
        assert found == [content_hash(image)]
        deadline = time.monotonic() + 5
        while threading.active_count() > 1 and any(t.name == "source-hash" for t in threading.enumerate()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        threading.excepthook = previous
    assert errors == []


def test_the_settings_changed_since_a_page_was_made_are_named_if_they_change_the_page():
    made = DocumentSettings()
    assert made.changes_from(made) == ()
    # The legend and the export's options are applied to the page as it is.
    assert made.replace(legend="Number only").changes_from(made) == ()
    assert made.replace(export=snapshot.ExportOptions(dpi=150)).changes_from(made) == ()
    assert made.with_values(line_width_mm=0.4).changes_from(made) == ("Line width",)
    assert made.replace(paper="A3").changes_from(made) == ("Paper",)
    # Another difficulty is named as such, not as each setting it moved.
    hard = made.replace(preset="Hard", values=default_values("Hard"))
    assert hard.changes_from(made) == ("Difficulty",)


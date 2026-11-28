"""H3: project files. A project keeps the picture's own bytes, the settings by their registry names, the paints and the
export options; it reads back exactly, its page made again identical; it is saved atomically; and a file that is
damaged, too large, of a newer version or reaching outside the archive is refused with a reason, the work kept."""

import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import project as projects  # noqa: E402
from tessellatum.app.project import ProjectError  # noqa: E402
from tessellatum.app.snapshot import MY_PAINTS, DocumentSettings, ExportOptions  # noqa: E402
from tessellatum.core import pipeline, settings  # noqa: E402
from tessellatum.core.painting import Version  # noqa: E402
from tessellatum.gui import main_window  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120


def _picture_bytes(size=(120, 160), fmt="PNG") -> bytes:
    height, width = size
    rng = np.random.default_rng(5)
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[:, : width // 2] = (200, 40, 40)
    rgb[: height // 2, width // 2:] = (40, 170, 40)
    rgb[height // 2:, width // 2:] = (40, 40, 200)
    rgb = np.clip(rgb + rng.integers(0, 40, rgb.shape), 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, fmt)
    return out.getvalue()


def _settings() -> DocumentSettings:
    """Settings far from the defaults in every kind of field."""
    paints = (("Cadmium red", (200, 30, 30)), ("Ultramarine", (30, 40, 180)), ("Titanium white", (250, 250, 248)))
    return DocumentSettings(preset="Hard", tone="Dark", paper="A3", legend="Name, hex and RGB",
                            colors_from=MY_PAINTS, paints=paints, mix=True, detail=False,
                            export=ExportOptions(Version.COMPLETED, "PDF", ".gpl", "px", 300, 4000)
                            ).with_values(line_width_mm=0.5, min_label_pt=5.0)


def _save(path: Path, snapshot=None, data=None, name="picture.png") -> bytes:
    data = data if data is not None else _picture_bytes()
    projects.save_project(path, snapshot or _settings(), name, data, projects.decode_image(data))
    return data


def _rewrite(path: Path, change) -> None:
    """Rewrite the project at ``path`` with ``change(members)`` applied to its {name: bytes}."""
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    change(members)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _manifest_change(path: Path, change) -> None:
    def apply(members):
        info = json.loads(members[projects.MANIFEST])
        change(info)
        members[projects.MANIFEST] = json.dumps(info).encode("utf-8")

    _rewrite(path, apply)


# -- The round trip ------------------------------------------------------------------------------------------------------
def test_a_project_reads_back_its_settings_paints_options_and_picture_bytes(tmp_path):
    path = tmp_path / "work.tessellatum"
    data = _save(path)
    loaded = projects.read_project(path)
    assert loaded.settings == _settings()
    assert loaded.settings.paints == _settings().paints and loaded.settings.export == _settings().export
    assert loaded.source_bytes == data and loaded.source_name == "picture.png"
    assert (loaded.image == pipeline.load_image_bgr(io.BytesIO(data))).all()


def test_the_page_made_again_from_a_project_is_the_page_made_before(tmp_path):
    data = _picture_bytes((150, 200), "JPEG")
    snapshot = DocumentSettings().with_values(num_colors=12)
    image = projects.decode_image(data)

    def page_of(picture, s):
        params, style, handling = s.difficulty_params(), s.page_style(), s.handling()
        long_edge = pipeline.working_long_edge(picture, params, handling, style.paper_format)
        return np.asarray(pipeline.generate(picture, params, long_edge, style=style, handling=handling).page)

    before = page_of(image, snapshot)
    path = tmp_path / "jpeg.tessellatum"
    projects.save_project(path, snapshot, "photo.jpg", data, image)
    loaded = projects.read_project(path)
    pipeline.clear_cache()
    assert np.array_equal(page_of(loaded.image, loaded.settings), before)


def test_the_manifest_names_every_setting_by_its_registry_name(tmp_path):
    path = tmp_path / "work.tessellatum"
    _save(path)
    with zipfile.ZipFile(path) as archive:
        info = json.loads(archive.read(projects.MANIFEST))
        names = set(archive.namelist())
    assert info["format"] == projects.FORMAT and info["format_version"] == projects.FORMAT_VERSION
    assert set(info["settings"]["values"]) == {s.name for s in settings.SETTINGS}
    assert info["source"]["member"] == "source.png" and names == {"manifest.json", "source.png", "thumbnail.png"}
    assert Image.open(io.BytesIO(projects.read_thumbnail(path))).size[0] <= projects.THUMBNAIL_EDGE


def test_a_setting_it_lacks_takes_its_default_and_one_it_doesn_t_know_is_ignored(tmp_path):
    path = tmp_path / "older.tessellatum"
    _save(path, DocumentSettings())

    def change(info):
        del info["settings"]["values"]["line_width_mm"]
        info["settings"]["values"]["a_setting_of_tomorrow"] = 3
        info["settings"]["a_section_of_tomorrow"] = {}

    _manifest_change(path, change)
    assert projects.read_project(path).settings == DocumentSettings()


def test_unicode_paths_and_names(tmp_path):
    folder = tmp_path / "Löwen 猫 ü"
    folder.mkdir()
    path = folder / "Bild é 画.tessellatum"
    data = _save(path, name="Löwe 猫.png")
    loaded = projects.read_project(path)
    assert loaded.source_bytes == data and loaded.source_name == "Löwe 猫.png"


# -- Saving ------------------------------------------------------------------------------------------------------------
def test_save_as_leaves_the_project_it_came_from_as_it_was(tmp_path):
    first, second = tmp_path / "first.tessellatum", tmp_path / "second.tessellatum"
    _save(first, DocumentSettings())
    before = first.read_bytes()
    _save(second, _settings())
    assert first.read_bytes() == before
    assert projects.read_project(second).settings == _settings()


def test_a_save_that_fails_leaves_the_file_saved_before_whole_and_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "work.tessellatum"
    _save(path, DocumentSettings())
    before = path.read_bytes()

    def broken(_path):
        raise ProjectError("it doesn't read back")

    monkeypatch.setattr(projects, "read_project", broken)
    with pytest.raises(ProjectError):
        _save(path, _settings())
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["work.tessellatum"]


# -- What is refused ---------------------------------------------------------------------------------------------------
def _refused(path: Path, words: str) -> None:
    with pytest.raises(ProjectError) as caught:
        projects.read_project(path)
    assert words in str(caught.value)


def test_a_file_that_isn_t_a_project_is_refused(tmp_path):
    path = tmp_path / "noise.tessellatum"
    path.write_bytes(os.urandom(5000))
    _refused(path, "isn't a Tessellatum project")
    path.write_bytes(_picture_bytes())  # a picture renamed
    _refused(path, "isn't a Tessellatum project")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "hello")
    _refused(path, "no manifest")


def test_a_truncated_project_is_refused(tmp_path):
    path = tmp_path / "cut.tessellatum"
    _save(path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with pytest.raises(ProjectError):
        projects.read_project(path)


@pytest.mark.parametrize("change, words", [
    (lambda i: i.update(format_version=projects.FORMAT_VERSION + 1), "newer Tessellatum"),
    (lambda i: i.pop("format_version"), "no format version"),
    (lambda i: i.update(format="something-else"), "isn't a Tessellatum project"),
    (lambda i: i["settings"]["values"].update(num_colors=10_000), "Colors"),
    (lambda i: i["settings"]["values"].update(num_colors="many"), "isn't a number"),
    (lambda i: i["settings"]["choices"].update(paper="A0"), "paper"),
    (lambda i: i["settings"]["switches"].update(text="yes"), "isn't on or off"),
    (lambda i: i["settings"].update(paints=[{"name": "Red", "rgb": [300, 0, 0]}]), "has no color"),
    (lambda i: i["settings"]["export"].update(format="TIFF"), "export's format"),
    (lambda i: i["source"].update(sha256="0" * 64), "doesn't match its checksum"),
    (lambda i: i["source"].update(member="elsewhere.png"), "picture is missing"),
    (lambda i: i["source"].update(member="../source.png"), "picture is missing"),
])
def test_a_manifest_that_is_wrong_is_refused_saying_what_is_wrong(tmp_path, change, words):
    path = tmp_path / "wrong.tessellatum"
    _save(path)
    _manifest_change(path, change)
    _refused(path, words)


def test_a_manifest_that_isn_t_json_is_refused(tmp_path):
    path = tmp_path / "bad.tessellatum"
    _save(path)
    _rewrite(path, lambda m: m.update({projects.MANIFEST: b"{not json"}))
    _refused(path, "manifest is damaged")


@pytest.mark.parametrize("name", ["../evil.png", "/abs.png", "C:/drive.png", "..\\up.png", "a/../../b"])
def test_a_member_reaching_outside_the_archive_is_refused(tmp_path, name):
    path = tmp_path / "traversal.tessellatum"
    _save(path)
    _rewrite(path, lambda m: m.update({name: b"x"}))
    _refused(path, "outside the project")


def test_members_past_their_bounds_are_refused(tmp_path, monkeypatch):
    path = tmp_path / "big.tessellatum"
    _save(path)
    monkeypatch.setattr(projects, "MAX_SOURCE", 1000)
    _refused(path, "more than a project holds")
    monkeypatch.undo()
    _rewrite(path, lambda m: m.update({projects.MANIFEST: b" " * (projects.MAX_MANIFEST + 1)}))
    _refused(path, "more than a project holds")


def test_a_file_too_large_to_be_a_project_isn_t_read(tmp_path, monkeypatch):
    path = tmp_path / "huge.tessellatum"
    _save(path)
    monkeypatch.setattr(projects, "MAX_FILE", path.stat().st_size - 1)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("read into memory"))
    _refused(path, "larger than a project can be")


def test_a_member_claiming_less_than_it_holds_is_read_only_up_to_the_bound(tmp_path, monkeypatch):
    # A zip whose header understates its size: reading stops at the bound, whatever the header says.
    archive = zipfile.ZipFile(io.BytesIO(), "w")
    archive.writestr("x", b"a" * 5000)
    info = archive.getinfo("x")
    info.file_size = 10
    with pytest.raises(ProjectError):
        projects._read_member(archive, info, 100, "test")


def test_a_picture_too_large_or_not_a_picture_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "MAX_PIXELS", 100)
    with pytest.raises(ProjectError, match="larger than Tessellatum opens"):
        projects.decode_image(_picture_bytes())
    monkeypatch.undo()
    with pytest.raises(ProjectError, match="can't be read"):
        projects.decode_image(b"%PDF-1.7 not a picture")
    out = io.BytesIO()
    Image.new("RGB", (10, 10)).save(out, "PPM")  # a picture, but not one Open Image takes
    with pytest.raises(ProjectError, match="not a picture Open Image takes"):
        projects.decode_image(out.getvalue())


# -- The window --------------------------------------------------------------------------------------------------------
class _Window:
    def __init__(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
        self.monkeypatch, self.tmp = monkeypatch, tmp_path
        self.made = 0
        real = pipeline.generate

        def counting(*args, **kwargs):
            self.made += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(pipeline, "generate", counting)
        self.asked: list[str] = []
        self.answer = main_window.DISCARD_CHANGES
        monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda window, name: self.asked.append(name) or self.answer)
        self.window = MainWindow()

    def open(self, path: Path) -> None:
        self.monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName",
                                 staticmethod(lambda *a, **k: (str(path), "")))
        self.window.commands["open"].trigger()
        self.settle()

    def save_as(self, path: Path) -> None:
        self.monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                                 staticmethod(lambda *a, **k: (str(path), "")))
        self.window.commands["save_as"].trigger()

    def settle(self) -> None:
        deadline = time.monotonic() + TIMEOUT_S
        window = self.window
        while not window.jobs.is_idle() or window._redraw_timer.isActive():
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.003)
        _app.processEvents()

    def edit(self) -> None:
        self.window.controls.controls["num_colors"].step(-3)
        _app.processEvents()
        self.window._record_settings()


@pytest.fixture
def win(monkeypatch, tmp_path):
    w = _Window(monkeypatch, tmp_path)
    picture = tmp_path / "lion cub.png"
    picture.write_bytes(_picture_bytes())
    w.picture = picture
    yield w
    w.window.close()
    w.window.deleteLater()
    _app.processEvents()
    pipeline.clear_cache()


def test_save_then_open_gives_the_settings_the_picture_and_its_page_back(win):
    win.open(win.picture)
    win.edit()
    win.window.generate_preview()
    win.settle()
    saved = win.window.controls.snapshot()
    path = win.tmp / "cub.tessellatum"
    win.save_as(path)
    assert path.exists() and win.window.history.is_clean() and not win.window.isWindowModified()
    assert win.window.windowTitle() == "cub.tessellatum[*] — Tessellatum"
    assert win.window.document.project_path() == path
    win.window.close_document()
    assert win.window.current_image_bgr is None and win.window.current_page is None
    win.window.controls.controls["num_colors"].step(5)  # the settings move on meanwhile
    made = win.made
    win.open(path)
    assert win.window.controls.snapshot() == saved
    assert win.window.document.source().data == win.picture.read_bytes()
    assert win.window.document.source().name == "lion cub.png"
    assert win.made == made + 1 and win.window.document.result_is_current()  # its page, made again
    assert win.window.history.count() == 0 and win.window.history.is_clean()


def test_save_writes_to_the_project_it_is_in(win):
    win.open(win.picture)
    path = win.tmp / "cub.tessellatum"
    win.save_as(path)
    win.edit()
    assert win.window.isWindowModified()
    win.monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: pytest.fail("Save asked where")))
    assert win.window.save_project()
    assert projects.read_project(path).settings == win.window.controls.snapshot()
    assert not win.window.isWindowModified()


def test_a_picture_from_no_file_is_saved_as_png(win):
    win.window.current_image_bgr = projects.decode_image(_picture_bytes())
    path = win.tmp / "pasted.tessellatum"
    win.save_as(path)
    loaded = projects.read_project(path)
    assert loaded.source_name.endswith(".png") and (loaded.image == win.window.current_image_bgr).all()


@pytest.mark.parametrize("answer, closed", [
    (main_window.KEEP_WORKING, False), (main_window.DISCARD_CHANGES, True), (main_window.SAVE_CHANGES, True),
])
def test_closing_with_unsaved_changes_asks_first(win, answer, closed):
    win.open(win.picture)
    win.window.close_document()
    assert win.asked == []  # nothing changed since it was opened: nothing asked
    win.open(win.picture)
    win.edit()
    win.answer = answer
    path = win.tmp / "asked.tessellatum"
    win.monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    assert win.window.close_document() is closed
    assert win.asked == ["lion cub.png"]
    assert (win.window.current_image_bgr is None) is closed
    assert path.exists() is (answer == main_window.SAVE_CHANGES)


def test_quitting_with_unsaved_changes_can_be_cancelled(win):
    win.open(win.picture)
    win.edit()
    win.answer = main_window.KEEP_WORKING
    win.window.show()
    win.window.close()
    assert win.window.isVisible() and win.asked == ["lion cub.png"]
    win.answer = main_window.DISCARD_CHANGES


def test_opening_a_project_asks_about_unsaved_changes_and_cancelling_keeps_the_work(win):
    path = win.tmp / "other.tessellatum"
    _save(path)
    win.open(win.picture)
    win.edit()
    before = win.window.controls.snapshot()
    win.answer = main_window.KEEP_WORKING
    win.open(path)
    assert win.asked and win.window.controls.snapshot() == before
    assert win.window.document.source().path == win.picture and win.window.document.project_path() is None


def test_a_damaged_project_says_why_and_leaves_the_work_as_it_was(win):
    win.open(win.picture)
    win.window.generate_preview()
    win.settle()
    page, before = win.window.current_page, win.window.controls.snapshot()
    path = win.tmp / "damaged.tessellatum"
    path.write_bytes(b"PK\x03\x04 not really")
    win.open(path)
    assert win.window.current_page is page and win.window.controls.snapshot() == before
    assert "Could not open damaged.tessellatum" in win.window.notification.text.text()


def test_opening_a_picture_into_a_project_is_one_step_to_undo(win):
    path = win.tmp / "cub.tessellatum"
    win.open(win.picture)
    win.save_as(path)
    other = win.tmp / "other.png"
    other.write_bytes(_picture_bytes((90, 140)))
    win.open(other)
    assert win.window.history.count() == 1 and win.window.document.project_path() == path
    win.window.undo()
    assert win.window.document.source().name == "lion cub.png"


def test_recent_projects(win):
    first, second = win.tmp / "first.tessellatum", win.tmp / "second.tessellatum"
    win.open(win.picture)
    win.save_as(first)
    win.save_as(second)
    assert win.window.recent_projects()[:2] == [second.absolute(), first.absolute()]
    first.unlink()
    win.window._show_recent()
    shown = [a.text() for a in win.window.recent_menu.actions() if not a.isSeparator()]
    assert shown == ["second.tessellatum", "lion cub.png", "&Clear Recent"]  # the project gone isn't shown
    assert first.absolute() in win.window.recent_projects()  # but kept: the welcome page greys it, with Remove
    win.window.recent_menu.actions()[0].trigger()
    win.settle()
    assert win.window.document.project_path() == second
    win.window.recent_menu.actions()[-1].trigger()
    assert win.window.recent_projects() == [] and win.window.recent_pictures() == []


def test_save_and_close_wait_for_a_picture(win):
    for command in ("save", "save_as", "close"):
        assert not win.window.commands[command].isEnabled()
    win.open(win.picture)
    for command in ("save", "save_as", "close"):
        assert win.window.commands[command].isEnabled()

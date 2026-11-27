"""H5: the welcome page and the ways in. The welcome page shows while no picture is open; the examples bundled open and
make their page; a file or an image dropped on the window or pasted opens, into the work as one step to undo, asking
only when a project would replace unsaved work; a file that can't be opened says why and leaves the work as it was;
the recent lists are capped, and a file no longer there is greyed with Remove."""

import io
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tessellatum.app import project as projects  # noqa: E402
from tessellatum.app.snapshot import DocumentSettings  # noqa: E402
from tessellatum.core import pipeline  # noqa: E402
from tessellatum.gui import main_window, welcome  # noqa: E402
from tessellatum.gui.main_window import MainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
TIMEOUT_S = 120


def _picture_bytes(size=(90, 120), color=(200, 60, 60)) -> bytes:
    rgb = np.zeros((*size, 3), np.uint8)
    rgb[:] = color
    rgb[: size[0] // 2, : size[1] // 2] = (40, 160, 60)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, "PNG")
    return out.getvalue()


class _Window:
    def __init__(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_window, "_warm_up_pipeline", lambda: None)
        self.monkeypatch, self.tmp = monkeypatch, tmp_path
        self.asked: list[str] = []
        self.answer = main_window.DISCARD_CHANGES
        monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda window, name: self.asked.append(name) or self.answer)
        self.window = MainWindow()

    def settle(self) -> None:
        deadline = time.monotonic() + TIMEOUT_S
        _app.processEvents()
        while not self.window.jobs.is_idle() or self.window._redraw_timer.isActive():
            _app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.003)
        _app.processEvents()

    def drop(self, mime: QMimeData) -> bool:
        enter = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(self.window, enter)
        if not enter.isAccepted():
            return False
        drop = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(self.window, drop)
        self.settle()
        return True

    def edit(self) -> None:
        self.window.controls.controls["num_colors"].step(-3)
        _app.processEvents()
        self.window._record_settings()


@pytest.fixture
def win(monkeypatch, tmp_path):
    w = _Window(monkeypatch, tmp_path)
    w.picture = tmp_path / "first.png"
    w.picture.write_bytes(_picture_bytes())
    w.other = tmp_path / "second.png"
    w.other.write_bytes(_picture_bytes((80, 100), (30, 60, 190)))
    yield w
    w.window.close()
    w.window.deleteLater()
    _app.processEvents()
    QApplication.clipboard().clear()
    pipeline.clear_cache()


def _urls(*paths: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


# -- The welcome page --------------------------------------------------------------------------------------------------
def test_the_welcome_page_shows_while_no_picture_is_open(win):
    window = win.window
    assert window.canvas_stack.currentWidget() is window.welcome
    window.open_path(win.picture)
    assert window.canvas_stack.currentWidget() is window.canvas_views
    window.close_document()
    assert window.canvas_stack.currentWidget() is window.welcome


def test_its_buttons_open_a_picture_and_a_project(win):
    path = win.tmp / "saved.tessellatum"
    data = _picture_bytes()
    projects.save_project(path, DocumentSettings(), "saved.png", data, projects.decode_image(data))
    answers = iter([(str(win.picture), ""), (str(path), "")])
    win.monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: next(answers)))
    win.window.welcome.open_picture_button.click()
    assert win.window.document.source().path == win.picture
    win.window.close_document()
    win.window.welcome.open_project_button.click()
    win.settle()
    assert win.window.document.project_path() == path


def test_the_examples_are_bundled_small_clean_and_credited():
    found = welcome.examples()
    assert 3 <= len(found) <= 4 and len(found) == len(welcome.EXAMPLES)
    notes = (welcome.examples_folder() / "README.md").read_text(encoding="utf-8")
    total = 0
    for example in found:
        total += example.path.stat().st_size
        image = Image.open(example.path)
        assert max(image.size) <= 1200
        assert not image.getexif() and not {"exif", "xmp", "icc_profile", "comment"} & set(image.info)
        assert f"`{example.file}`" in notes and "CC0" in example.credit
    assert total < 1_500_000


def test_the_examples_icons_are_one_square_so_their_names_line_up(win):
    buttons = win.window.welcome.example_buttons
    assert len(buttons) >= 2
    sizes = {b.icon().availableSizes()[0].toTuple() for b in buttons}
    assert sizes == {(welcome.THUMBNAIL, welcome.THUMBNAIL)}  # a tall picture and a wide one alike
    image = buttons[-1].icon().pixmap(welcome.THUMBNAIL).toImage()
    corners = [image.pixelColor(x, y).alpha() for x in (0, image.width() - 1) for y in (0, image.height() - 1)]
    assert 0 in corners  # the picture on transparency, not stretched to the square


def test_an_example_opens_and_makes_its_page(win):
    button = win.window.welcome.example_buttons[0]
    assert not button.icon().isNull()
    button.click()
    win.settle()
    source = win.window.document.source()
    assert source.name == welcome.EXAMPLES[0].file and source.path is None  # not the app's own folder
    assert source.data == welcome.EXAMPLES[0].path.read_bytes()
    assert win.window.document.result_is_current()
    assert win.window.recent_pictures() == []  # an example isn't a file of one's own


# -- Drag and drop -----------------------------------------------------------------------------------------------------
def test_dropping_a_picture_file_opens_it_and_into_work_is_one_step_to_undo(win):
    assert win.drop(_urls(win.picture))
    assert win.window.document.source().path == win.picture
    win.edit()
    assert win.drop(_urls(win.other))
    assert win.asked == []  # a picture into the work asks nothing: it is undoable
    assert win.window.document.source().path == win.other
    win.window.undo()
    assert win.window.document.source().path == win.picture


def test_dropping_image_data_opens_it_on_white(win):
    image = QImage(40, 30, QImage.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    for x in range(20):
        for y in range(30):
            image.setPixelColor(x, y, QColor(10, 20, 200))
    mime = QMimeData()
    mime.setImageData(image)
    assert win.drop(mime)
    source = win.window.document.source()
    assert source.name == main_window.PASTED_NAME and source.image.shape == (30, 40, 3)
    assert tuple(source.image[5, 5]) == (200, 20, 10) and tuple(source.image[5, 35]) == (255, 255, 255)  # BGR
    assert source.data is not None and (projects.decode_image(source.data) == source.image).all()


def test_dropping_a_project_over_unsaved_work_asks_first(win):
    path = win.tmp / "dropped.tessellatum"
    data = _picture_bytes()
    projects.save_project(path, DocumentSettings(), "dropped.png", data, projects.decode_image(data))
    win.drop(_urls(win.picture))
    win.edit()
    win.answer = main_window.KEEP_WORKING
    assert win.drop(_urls(path))
    assert win.asked == ["first.png"] and win.window.document.source().path == win.picture
    win.answer = main_window.DISCARD_CHANGES
    assert win.drop(_urls(path))
    assert win.window.document.project_path() == path


@pytest.mark.parametrize("name, content, words", [
    ("notes.txt", b"hello", "Tessellatum opens pictures"),
    ("broken.png", b"\x89PNG not really", "Could not open broken.png"),
    ("broken.tessellatum", b"PK\x03\x04 not really", "Could not open broken.tessellatum"),
])
def test_a_file_that_can_t_be_opened_says_why_and_leaves_the_work(win, name, content, words):
    win.window.open_path(win.picture)
    win.edit()
    before = win.window.controls.snapshot()
    bad = win.tmp / name
    bad.write_bytes(content)
    assert win.drop(_urls(bad))
    assert words in win.window.notification.text.text()
    assert win.window.document.source().path == win.picture and win.window.controls.snapshot() == before


def test_what_isn_t_a_file_or_a_picture_isn_t_taken(win):
    mime = QMimeData()
    mime.setText("just words")
    assert not win.drop(mime)
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.com/picture.png")])
    assert not win.drop(mime)


def test_the_canvas_lets_the_window_take_a_drop(win):
    assert win.window.acceptDrops()
    assert not win.window.preview.acceptDrops() and not win.window.preview.viewport().acceptDrops()


# -- Paste -------------------------------------------------------------------------------------------------------------
def test_pasting_an_image(win):
    image = QImage(30, 20, QImage.Format_RGB32)
    image.fill(QColor(250, 200, 0))
    QApplication.clipboard().setImage(image)
    win.window.commands["paste"].trigger()
    source = win.window.document.source()
    assert source is not None and source.name == main_window.PASTED_NAME and tuple(source.image[0, 0]) == (0, 200, 250)


def test_pasting_a_file_or_its_path(win):
    QApplication.clipboard().setMimeData(_urls(win.picture))
    win.window.paste()
    assert win.window.document.source().path == win.picture
    QApplication.clipboard().setText(f'"{win.other}"')
    win.window.paste()
    assert win.window.document.source().path == win.other


def test_pasting_nothing_says_so(win):
    QApplication.clipboard().setText("not a path")
    win.window.paste()
    assert win.window.document.source() is None
    assert "no picture to paste" in win.window.notification.text.text()


def test_ctrl_v_in_a_text_field_pastes_text_there(win):
    win.window.show()
    field = win.window.controls.search.field
    field.setFocus()
    QApplication.clipboard().setText(str(win.picture))
    QTest.keyClick(field, Qt.Key_V, Qt.ControlModifier)
    _app.processEvents()
    assert win.window.document.source() is None and str(win.picture) in field.text()


# -- Recent files ------------------------------------------------------------------------------------------------------
def test_the_recent_pictures_are_capped_latest_first(win):
    paths = []
    for index in range(main_window.RECENT_MAX + 3):
        path = win.tmp / f"p{index}.png"
        path.write_bytes(_picture_bytes())
        win.window.open_path(path)
        paths.append(path.absolute())
    win.window.open_path(paths[5])  # opened again: to the top, once
    recent = win.window.recent_pictures()
    assert len(recent) == main_window.RECENT_MAX and recent[0] == paths[5] and recent.count(paths[5]) == 1


def test_a_recent_file_no_longer_there_is_greyed_with_remove(win):
    win.window.open_path(win.picture)
    win.window.open_path(win.other)
    win.window.close_document()
    win.other.unlink()
    win.window._show_canvas()
    rows = win.window.welcome.recent_rows
    name, remove = rows[win.other.absolute()]
    assert not name.isEnabled() and remove is not None
    there, none = rows[win.picture.absolute()]
    assert there.isEnabled() and none is None
    remove.click()
    _app.processEvents()
    assert win.window.recent_pictures() == [win.picture.absolute()]
    assert win.other.absolute() not in win.window.welcome.recent_rows
    win.window.welcome.recent_rows[win.picture.absolute()][0].click()
    assert win.window.document.source().path == win.picture.absolute()

"""The dialog where the painter lists the paints or pencils they have (see ``core.paints``).

A list, a row a paint: its swatch (click it to pick the color again), its name (double-click to rename) and a button
that deletes it. Colors are added with a color picker, whose Pick Screen Color is an eyedropper: from a photo of one's
swatches, or from the picture in the preview. A sample kit (``core.paint_kits``) or a palette file can start the list
or add to it, and Edit as text shows it as lines of text, to paste a list or type one.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core.color_names import hex_code, name_colors
from tessellatum.core.paint_kits import KITS, Kit
from tessellatum.core.paints import MOST_PAINTS, Paint, distinct, paints_text, parse_paints, read_palette_file

INTRO = (
    "The paints or pencils you have. Add each with Add color…: pick it on the color wheel, type its hex code, or use "
    "Pick Screen Color to take it from the screen (a photo of your swatches, or the picture in the preview). Click a "
    "swatch to pick its color again, double-click a name to change it. Or start from a sample kit, or load a palette "
    "file: GIMP (.gpl), Adobe Swatch Exchange (.ase) or Procreate (.swatches). The page then takes its colors from "
    "these, at most as many as Colors asks for, the ones that paint the picture best."
)
TEXT_INTRO = (
    "One paint a line: its color as a hex code or as three numbers from 0 to 255, and its name. For example:\n"
    "    #2000B1  Ultramarine\n"
    "    90 107 47  Sap green"
)
KIT_NOTE = "A kit's colors are typical of its paints, not any maker's: real tubes and pencils differ a little."
PALETTE_FILTER = "Palettes (*.gpl *.ase *.swatches *.txt);;All files (*)"
SWATCH, NAME, DELETE = range(3)
_SWATCH_PX = 22
_RGB = Qt.UserRole


def pick_color(parent: QWidget, initial: tuple[int, int, int] | None, title: str) -> tuple[int, int, int] | None:
    """A color picked in Qt's color dialog (its own on every system, for its Pick Screen Color), or None if cancelled."""
    start = QColor(*initial) if initial is not None else QColor(128, 128, 128)
    picked = QColorDialog.getColor(start, parent, title, QColorDialog.DontUseNativeDialog)
    if not picked.isValid():
        return None
    return (picked.red(), picked.green(), picked.blue())


def ask_replace(parent: QWidget, kit: Kit, listed: int) -> bool | None:
    """Whether ``kit`` replaces the ``listed`` paints (True), is added to them (False), or neither (None)."""
    box = QMessageBox(parent)
    box.setWindowTitle(kit.name)
    box.setText(f"Replace the {listed} paint{'s' if listed != 1 else ''} listed with {kit.name}, or add it to them?")
    box.setInformativeText(KIT_NOTE)
    replace = box.addButton("Replace", QMessageBox.AcceptRole)
    add = box.addButton("Add", QMessageBox.AcceptRole)
    box.addButton(QMessageBox.Cancel)
    box.setDefaultButton(replace)
    box.exec()
    if box.clickedButton() is replace:
        return True
    if box.clickedButton() is add:
        return False
    return None


def swatch_icon(rgb: tuple[int, int, int], size: int = _SWATCH_PX) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class PaintsDialog(QDialog):
    """The paints as a list of rows (or, with Edit as text, as lines of text); ``paints()`` gives them back."""

    def __init__(self, paints: tuple[Paint, ...], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("My paints")
        self.resize(560, 600)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Color", "Name", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(SWATCH, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(NAME, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(DELETE, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("#2000B1  Ultramarine")
        text_page = QWidget()
        text_layout = QVBoxLayout(text_page)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_intro = QLabel(TEXT_INTRO)
        text_intro.setWordWrap(True)
        text_layout.addWidget(text_intro)
        text_layout.addWidget(self.editor, 1)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.stack.addWidget(text_page)

        self.add_button = QPushButton("Add color…")
        self.add_button.setToolTip("Pick a color to add: on the wheel, by its hex code, or from the screen.")
        self.kit_button = QToolButton()
        self.kit_button.setText("Start from a kit…")
        self.kit_button.setPopupMode(QToolButton.InstantPopup)
        self.kit_button.setToolTip("A sample set of paints or pencils, to start from. " + KIT_NOTE)
        kit_menu = QMenu(self.kit_button)
        for kit in KITS:
            action = QAction(kit.name, kit_menu)
            action.triggered.connect(lambda _checked=False, kit=kit: self.start_from_kit(kit))
            kit_menu.addAction(action)
        self.kit_button.setMenu(kit_menu)
        self.load_button = QPushButton("Load palette file…")
        self.load_button.setToolTip("Add the colors of a palette file to the list.")
        self.clear_button = QPushButton("Clear")
        self.text_button = QPushButton("Edit as text")
        self.text_button.setCheckable(True)
        self.text_button.setToolTip("The list as lines of text: to paste a list, or type one.")
        self.status = QLabel()
        self.status.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        row = QHBoxLayout()
        for widget in (self.add_button, self.kit_button, self.load_button, self.clear_button):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.text_button)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(row)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

        self._note = ""  # what the last action has to say, after the count
        self.add_button.clicked.connect(self.add_color)
        self.load_button.clicked.connect(self._load)
        self.clear_button.clicked.connect(self.clear)
        self.text_button.toggled.connect(self._show_as_text)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.itemChanged.connect(lambda _item: self._show_status())
        self.editor.textChanged.connect(self._show_status)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._set_rows(distinct(paints))

    # -- What the dialog holds ---------------------------------------------
    def paints(self) -> tuple[Paint, ...]:
        """The paints listed, each color once, ``MOST_PAINTS`` at most."""
        if self.text_button.isChecked():
            return distinct(parse_paints(self.editor.toPlainText()).paints)
        return tuple(self._row_paint(row) for row in range(self.table.rowCount()))

    def _row_paint(self, row: int) -> Paint:
        return (self.table.item(row, NAME).text().strip(), self.table.item(row, SWATCH).data(_RGB))

    def _colors(self) -> list[tuple[int, int, int]]:
        return [self.table.item(row, SWATCH).data(_RGB) for row in range(self.table.rowCount())]

    def _set_rows(self, paints: tuple[Paint, ...] | list[Paint]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for name, rgb in paints:
            self._append_row(name, rgb)
        self.table.blockSignals(False)
        self._show_status()

    def _append_row(self, name: str, rgb: tuple[int, int, int]) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        swatch = QTableWidgetItem()
        swatch.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(row, SWATCH, swatch)
        self._paint_swatch(row, tuple(int(v) for v in rgb))
        self.table.setItem(row, NAME, QTableWidgetItem(name))
        delete = QTableWidgetItem("✕")
        delete.setFlags(Qt.ItemIsEnabled)
        delete.setTextAlignment(Qt.AlignCenter)
        delete.setToolTip("Delete this paint.")
        self.table.setItem(row, DELETE, delete)
        return row

    def _paint_swatch(self, row: int, rgb: tuple[int, int, int]) -> None:
        swatch = self.table.item(row, SWATCH)
        swatch.setData(_RGB, rgb)
        swatch.setIcon(swatch_icon(rgb))
        swatch.setText(hex_code(rgb))
        swatch.setToolTip(f"{hex_code(rgb)}  RGB {rgb[0]} {rgb[1]} {rgb[2]}: click to pick it again.")

    # -- Actions -----------------------------------------------------------
    def add_color(self) -> None:
        """Pick a color and add it, named as the legend would name it, its name ready to be typed over."""
        self._to_list()
        if self.table.rowCount() >= MOST_PAINTS:
            self._say(f"The list is full: at most {MOST_PAINTS} paints.")
            return
        rgb = pick_color(self, None, "Add a color")
        if rgb is None:
            return
        if rgb in self._colors():
            row = self._colors().index(rgb)
            self.table.selectRow(row)
            self._say(f"{hex_code(rgb)} is listed already, as {self._row_paint(row)[0] or 'row ' + str(row + 1)}.")
            return
        self.table.blockSignals(True)
        row = self._append_row(name_colors([rgb])[0], rgb)
        self.table.blockSignals(False)
        self._say("")
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, NAME))
        self.table.setCurrentCell(row, NAME)
        self.table.editItem(self.table.item(row, NAME))  # its name selected, to type over

    def repick(self, row: int) -> None:
        """Pick row ``row``'s color again. A name the dialog gave it follows the new color."""
        old = self.table.item(row, SWATCH).data(_RGB)
        rgb = pick_color(self, old, "Pick the color again")
        if rgb is None or rgb == old:
            return
        if rgb in self._colors():
            other = self._colors().index(rgb)
            self._say(f"{hex_code(rgb)} is listed already, as {self._row_paint(other)[0] or 'row ' + str(other + 1)}.")
            return
        self.table.blockSignals(True)
        self._paint_swatch(row, rgb)
        name = self.table.item(row, NAME)
        if not name.text().strip() or name.text().strip() == name_colors([old])[0]:
            name.setText(name_colors([rgb])[0])
        self.table.blockSignals(False)
        self._say("")

    def delete(self, row: int) -> None:
        self.table.removeRow(row)
        self._say("")

    def clear(self) -> None:
        if self.text_button.isChecked():
            self.editor.clear()
        else:
            self._set_rows(())
        self._say("")

    def start_from_kit(self, kit: Kit, replace: bool | None = None) -> None:
        """Put ``kit``'s paints in the list: in place of those listed (``replace``) or after them. Asked if None and
        paints are listed."""
        self._to_list()
        if replace is None:
            replace = True if self.table.rowCount() == 0 else ask_replace(self, kit, self.table.rowCount())
            if replace is None:
                return
        if replace:
            self._set_rows(kit.paints)
            self._say(KIT_NOTE)
        else:
            self._add(kit.paints, KIT_NOTE)

    def _add(self, paints: list[Paint] | tuple[Paint, ...], note: str = "") -> None:
        """``paints`` after those listed: each color once, ``MOST_PAINTS`` at most, and said so if not all fit."""
        listed = set(self._colors())
        new = [paint for paint in distinct(list(paints)) if paint[1] not in listed]
        room = MOST_PAINTS - self.table.rowCount()
        self.table.blockSignals(True)
        for name, rgb in new[: max(0, room)]:
            self._append_row(name, rgb)
        self.table.blockSignals(False)
        said = []
        twice = len(paints) - len(new)
        if twice:
            said.append(f"{twice} listed already.")
        if len(new) > room:
            said.append(f"{len(new) - max(0, room)} left off: at most {MOST_PAINTS} paints.")
        if note:
            said.append(note)
        self._say(" ".join(said))

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if column == SWATCH:
            self.repick(row)
        elif column == DELETE:
            self.delete(row)

    def _load(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load palette file", "", PALETTE_FILTER)
        if not path_str:
            return
        try:
            loaded = read_palette_file(Path(path_str))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not read the palette", str(exc))
            return
        self._to_list()
        self._add(loaded)

    # -- Edit as text --------------------------------------------------------
    def _to_list(self) -> None:
        if self.text_button.isChecked():
            self.text_button.setChecked(False)  # reads the text back (see _show_as_text)

    def _show_as_text(self, as_text: bool) -> None:
        if as_text:
            self.editor.setPlainText(paints_text(self.paints_from_rows()))
            self.stack.setCurrentIndex(1)
            self._say("")
            return
        parsed = parse_paints(self.editor.toPlainText())
        self._set_rows(distinct(parsed.paints))
        self.stack.setCurrentIndex(0)
        self._say(_text_notes(parsed))

    def paints_from_rows(self) -> tuple[Paint, ...]:
        return tuple(self._row_paint(row) for row in range(self.table.rowCount()))

    # -- Status --------------------------------------------------------------
    def _say(self, note: str) -> None:
        self._note = note
        self._show_status()

    def _show_status(self) -> None:
        if self.text_button.isChecked():
            parsed = parse_paints(self.editor.toPlainText())
            count = len(distinct(parsed.paints))
            notes = _text_notes(parsed)
        else:
            count = self.table.rowCount()
            notes = self._note
        self.add_button.setEnabled(count < MOST_PAINTS or self.text_button.isChecked())
        text = f"{count} paint{'s' if count != 1 else ''}."
        self.status.setText(f"{text} {notes}".strip())


def _text_notes(parsed) -> str:
    """What reading a text's paints has to say: a color listed twice, paints left off, lines with no color."""
    notes = []
    colors = len({rgb for _name, rgb in parsed.paints})
    if len(parsed.paints) > colors:
        notes.append("A color listed twice counts once.")
    if colors > MOST_PAINTS:
        notes.append(f"{colors - MOST_PAINTS} left off: the first {MOST_PAINTS} are used.")
    if parsed.rejected:
        lines = ", ".join(str(number) for number, _ in parsed.rejected[:8])
        more = "…" if len(parsed.rejected) > 8 else ""
        notes.append(f"No color found on line{'s' if len(parsed.rejected) > 1 else ''} {lines}{more}: skipped.")
    return " ".join(notes)

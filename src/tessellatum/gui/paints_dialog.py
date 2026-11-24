"""The dialog where the painter lists the paints or pencils they have (see ``core.paints``)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core.paints import MOST_PAINTS, Paint, distinct, paints_text, parse_paints, read_palette_file

INTRO = (
    "List the paints or pencils you have, one a line: its color as a hex code or as three numbers from 0 to 255, and "
    "its name, for the legend. For example:\n"
    "    #2000B1  Ultramarine\n"
    "    90 107 47  Sap green\n"
    "Or load a palette file: GIMP (.gpl), Adobe Swatch Exchange (.ase) or Procreate (.swatches). The page then takes "
    "its colors from these, at most as many as Colors asks for, the ones that paint the picture best."
)
PALETTE_FILTER = "Palettes (*.gpl *.ase *.swatches *.txt);;All files (*)"


class PaintsDialog(QDialog):
    """A text box of paints, a line each, read as it is typed; Load adds a palette file's to it."""

    def __init__(self, paints: tuple[Paint, ...], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("My paints")
        self.resize(520, 520)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        self.editor = QPlainTextEdit(paints_text(paints))
        self.editor.setPlaceholderText("#2000B1  Ultramarine")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.load_button = QPushButton("Load palette file…")
        self.clear_button = QPushButton("Clear")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        row = QHBoxLayout()
        row.addWidget(self.load_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.status)
        layout.addLayout(row)
        layout.addWidget(buttons)
        self.editor.textChanged.connect(self._show_status)
        self.load_button.clicked.connect(self._load)
        self.clear_button.clicked.connect(self.editor.clear)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._show_status()

    def paints(self) -> tuple[Paint, ...]:
        """The paints listed, each color once."""
        return distinct(parse_paints(self.editor.toPlainText()).paints)

    def _show_status(self) -> None:
        parsed = parse_paints(self.editor.toPlainText())
        count = len(distinct(parsed.paints))
        text = f"{count} paint{'s' if count != 1 else ''}."
        if len(parsed.paints) > count:
            text += " A color listed twice counts once."
        if count >= MOST_PAINTS:
            text += f" The first {MOST_PAINTS} are used."
        if parsed.rejected:
            lines = ", ".join(str(number) for number, _ in parsed.rejected[:8])
            more = "…" if len(parsed.rejected) > 8 else ""
            text += f" No color found on line{'s' if len(parsed.rejected) > 1 else ''} {lines}{more}: skipped."
        self.status.setText(text)

    def _load(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load palette file", "", PALETTE_FILTER)
        if not path_str:
            return
        try:
            loaded = read_palette_file(Path(path_str))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not read the palette", str(exc))
            return
        current = self.editor.toPlainText().rstrip()
        added = paints_text(loaded)
        self.editor.setPlainText(f"{current}\n{added}" if current else added)

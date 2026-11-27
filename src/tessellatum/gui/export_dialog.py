"""The export dialog: what will be written, before it is.

The export's options (the version, the format, the PNG's resolution, the palette's format) live on the settings panel,
which keeps them between runs; the dialog shows them beside a thumbnail of the file, a line on what will be written
(its pixels, its size on paper), and warnings that never stop it (a PNG under 150 dpi, a Procreate palette split into
several files, numbers too small to read on paper). Only an export that can't be made at all is refused.

When the settings changed since the preview was made, it says so, and asks which page to write: the page made again
with the settings as they are now (the default), or the page as previewed, which the window can write without making
it again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import swatches
from tessellatum.gui.i18n import translate

THUMBNAIL = 220  # px, the thumbnail's box
LOW_DPI = 150  # under this, a PNG's lines and numbers print soft
SMALL_NUMBER_PT = 3.0  # under this, numbers are for a screen, zoomed in, rather than for paper


@dataclass(frozen=True)
class ExportState:
    """What the dialog shows: ``summary``, a line on what will be written; ``warnings``, each a sentence;
    ``changed``, the settings changed since the preview was made (by label; empty if it is up to date);
    ``thumbnail``, the file's picture, or None; ``blocked``, why it can't be exported at all, or None."""

    summary: str
    warnings: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    thumbnail: QPixmap | None = None
    blocked: str | None = None


def warnings_for(fmt: str, dpi: float | None, palette_format: str, colors: int, smallest_pt: float,
                 on_screen: bool) -> tuple[str, ...]:
    """The warnings an export of ``fmt`` gets: at ``dpi`` (a PNG's), as ``palette_format`` (a palette's), of a page of
    ``colors`` colors whose numbers print down to ``smallest_pt``, colored on a screen or on paper."""
    found = []
    if fmt == "PNG" and dpi is not None and dpi < LOW_DPI:
        found.append(translate("export_dialog", "Under {LOW_DPI} dpi, the lines and numbers print soft: {dpi:.0f} "
                                                "dpi is for a screen.").format(LOW_DPI=LOW_DPI, dpi=dpi))
    if fmt == "Palette" and palette_format == swatches.PROCREATE and colors > swatches.PROCREATE_MAX_SWATCHES:
        files = -(-colors // swatches.PROCREATE_MAX_SWATCHES)
        found.append(
            translate("export_dialog", "Procreate holds {0} colors a palette: the {colors} colors are written as "
                                       "{files} files, each named for the colors it holds.").format(swatches.PROCREATE_MAX_SWATCHES, colors=colors, files=files)
        )
    if fmt != "Palette" and not on_screen and smallest_pt < SMALL_NUMBER_PT:
        found.append(
            translate("export_dialog", "Numbers as small as {smallest_pt:g} pt are hard to read on paper; "
                                       "{SMALL_NUMBER_PT:g} pt or more reads well (Smallest number, on the "
                                       "Document page).").format(smallest_pt=smallest_pt, SMALL_NUMBER_PT=SMALL_NUMBER_PT)
        )
    return tuple(found)


def pixmap_of(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    data = rgb.tobytes()
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # its own pixels: ``data`` goes away


def palette_pixmap(palette_rgb: Sequence[tuple[int, int, int]], size: int = THUMBNAIL) -> QPixmap:
    """The palette's colors as swatches in rows, as a painting app shows them."""
    columns = 6
    cell = size // columns
    rows = max(1, -(-len(palette_rgb) // columns))
    pixmap = QPixmap(cell * columns, cell * rows)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    for index, rgb in enumerate(palette_rgb):
        row, column = divmod(index, columns)
        painter.fillRect(column * cell + 2, row * cell + 2, cell - 4, cell - 4, QColor(*rgb))
    painter.end()
    return pixmap


class ExportDialog(QDialog):
    """The export's options, beside what they will write. ``export_requested(previewed)`` asks for the export: of the
    page as previewed (True) or made with the settings as they are now (False). The window fills it with ``show_state``
    whenever an option changes."""

    export_requested = Signal(bool)

    def __init__(self, options: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle(translate("export_dialog", "Export"))
        self.thumbnail = QLabel()
        self.thumbnail.setObjectName("exportThumbnail")
        self.thumbnail.setFixedSize(THUMBNAIL, THUMBNAIL)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setAccessibleName(translate("export_dialog", "What will be exported"))
        self.options = options
        self.summary = QLabel()
        self.summary.setObjectName("exportSummary")
        self.summary.setWordWrap(True)
        self.warnings = QLabel()
        self.warnings.setObjectName("exportWarnings")
        self.warnings.setWordWrap(True)
        self.warnings.setProperty("role", "warning")

        # Out of date: which page to write.
        self.stale = QFrame()
        self.stale.setObjectName("exportStale")
        self.stale_text = QLabel()
        self.stale_text.setWordWrap(True)
        self.export_now = QRadioButton(translate("export_dialog", "Generate and export: make the page again with the "
                                                                  "settings as they are now"))
        self.export_previewed = QRadioButton(translate("export_dialog", "Export the previewed page, made with the "
                                                                        "settings it was previewed with"))
        self.export_now.setChecked(True)
        stale_layout = QVBoxLayout(self.stale)
        stale_layout.addWidget(self.stale_text)
        stale_layout.addWidget(self.export_now)
        stale_layout.addWidget(self.export_previewed)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.export_button = self.buttons.addButton(translate("export_dialog", "Export…"), QDialogButtonBox.AcceptRole)
        self.export_button.setDefault(True)
        self.export_button.setProperty("primary", True)
        self.export_button.setAutoDefault(True)
        self.buttons.accepted.connect(self._on_export)
        self.buttons.rejected.connect(self.reject)

        side = QVBoxLayout()
        side.addWidget(self.options)
        side.addWidget(self.summary)
        side.addWidget(self.warnings)
        side.addStretch(1)
        top = QHBoxLayout()
        top.addWidget(self.thumbnail, 0, Qt.AlignTop)
        top.addLayout(side, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.stale)
        layout.addWidget(self.buttons)
        self.export_button.setDefault(True)  # once laid out: the button box picks its own default when added
        self._blocked: str | None = None
        self.show_state(ExportState("", blocked=translate("export_dialog", "Generate a preview first.")))

    def show_state(self, state: ExportState) -> None:
        self.summary.setText(state.summary)
        self.warnings.setText("\n".join(f"⚠ {text}" for text in state.warnings))
        self.warnings.setVisible(bool(state.warnings))
        if state.changed:
            named = ", ".join(state.changed[:4]) + (" " + translate("export_dialog", "and {0} more")
                                                          .format(len(state.changed) - 4) if len(state.changed) > 4 else "")
            self.stale_text.setText(translate("export_dialog", "The preview is out of date. Changed since it was "
                                                               "made: {named}.").format(named=named))
        self.stale.setVisible(bool(state.changed))
        if state.thumbnail is None:
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText(translate("export_dialog", "Nothing to export yet"))
        else:
            self.thumbnail.setText("")
            self.thumbnail.setPixmap(
                state.thumbnail.scaled(THUMBNAIL, THUMBNAIL, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self._blocked = state.blocked
        self.export_button.setEnabled(state.blocked is None)
        self.export_button.setToolTip(state.blocked or "")

    def stale_shown(self) -> bool:
        return not self.stale.isHidden()

    def previewed_chosen(self) -> bool:
        """Whether the page as previewed is to be written: only when it is out of date and that was chosen."""
        return self.stale_shown() and self.export_previewed.isChecked()

    def warning_texts(self) -> list[str]:
        return [line[2:] for line in self.warnings.text().splitlines() if line]

    def _on_export(self) -> None:
        if self._blocked is not None:
            return
        previewed = self.previewed_chosen()
        self.accept()
        self.export_requested.emit(previewed)

    def open_for_export(self) -> None:
        """Shown, the out-of-date choice back at its default, the focus on Export."""
        self.export_now.setChecked(True)
        self.open()
        self.export_button.setDefault(True)
        self.export_button.setFocus()

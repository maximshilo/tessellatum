"""Left-hand controls: open image, difficulty, output format, generate/export."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from tessellatum.core import difficulty

THUMBNAIL_SIZE = 220


class ControlsPanel(QWidget):
    open_image_requested = Signal()
    generate_requested = Signal()
    export_requested = Signal()
    abort_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.open_button = QPushButton("Open Image…")
        self.thumbnail_label = QLabel("No image loaded")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.thumbnail_label.setStyleSheet("border: 1px solid #999; color: #777;")

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(difficulty.preset_names())
        self.preset_combo.setCurrentText(difficulty.DEFAULT_PRESET)

        self.custom_group = QGroupBox("Custom settings")
        self.colors_slider, colors_row = _slider_row(*difficulty.CUSTOM_COLORS_RANGE, default=12)
        self.min_region_slider, min_region_row = _slider_row(0, 1000, default=250)
        self.blur_slider, blur_row = _slider_row(
            int(difficulty.CUSTOM_BLUR_RANGE[0] * 10), int(difficulty.CUSTOM_BLUR_RANGE[1] * 10), default=50
        )
        custom_form = QFormLayout()
        custom_form.addRow("Colors", colors_row)
        custom_form.addRow("Region size", min_region_row)
        custom_form.addRow("Smoothing", blur_row)
        self.custom_group.setLayout(custom_form)
        self.custom_group.setVisible(False)

        format_group = QGroupBox("Output format")
        self.png_radio = QRadioButton("PNG (image)")
        self.pdf_radio = QRadioButton("PDF (printable)")
        self.png_radio.setChecked(True)
        format_layout = QHBoxLayout()
        format_layout.addWidget(self.png_radio)
        format_layout.addWidget(self.pdf_radio)
        format_group.setLayout(format_layout)

        self.generate_button = QPushButton("Generate Preview")
        self.export_button = QPushButton("Export…")
        self.export_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)

        self.abort_button = QPushButton("✕")
        self.abort_button.setToolTip("Cancel generation")
        self.abort_button.setFixedWidth(28)

        self.progress_row = QWidget()
        progress_row_layout = QHBoxLayout(self.progress_row)
        progress_row_layout.setContentsMargins(0, 0, 0, 0)
        progress_row_layout.addWidget(self.progress_bar)
        progress_row_layout.addWidget(self.abort_button)
        self.progress_row.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.open_button)
        layout.addWidget(self.thumbnail_label, alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("Difficulty"))
        layout.addWidget(self.preset_combo)
        layout.addWidget(self.custom_group)
        layout.addWidget(format_group)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.progress_row)
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        self.open_button.clicked.connect(self.open_image_requested)
        self.generate_button.clicked.connect(self.generate_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.abort_button.clicked.connect(self._on_abort_clicked)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

    def _on_abort_clicked(self) -> None:
        # Cancellation takes effect at the pipeline's next stage boundary, so
        # disable the button immediately to avoid double-clicks while we wait.
        self.abort_button.setEnabled(False)
        self.abort_requested.emit()

    def _on_preset_changed(self, name: str) -> None:
        self.custom_group.setVisible(name == "Custom")

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumbnail_label.setPixmap(scaled)

    def get_difficulty_params(self) -> difficulty.DifficultyParams:
        preset = self.preset_combo.currentText()
        if preset != "Custom":
            return difficulty.params_for_preset(preset)

        min_lo, min_hi = difficulty.CUSTOM_MIN_REGION_RANGE
        min_fraction = min_lo + (self.min_region_slider.value() / 1000.0) * (min_hi - min_lo)
        return difficulty.custom_params(
            num_colors=self.colors_slider.value(),
            min_region_fraction=min_fraction,
            blur_sigma=self.blur_slider.value() / 10.0,
        )

    def get_output_format(self) -> str:
        return "PDF" if self.pdf_radio.isChecked() else "PNG"

    def set_busy(self, busy: bool) -> None:
        self.progress_row.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
            self.abort_button.setEnabled(True)
        self.generate_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.export_button.property("hasResult") is True)
        self.open_button.setEnabled(not busy)

    def set_progress(self, percent: int) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setProperty("hasResult", enabled)
        self.export_button.setEnabled(enabled)


def _slider_row(minimum: int, maximum: int, default: int) -> tuple[QSlider, QWidget]:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)
    value_label = QLabel(str(default))
    value_label.setFixedWidth(40)
    slider.valueChanged.connect(lambda v: value_label.setText(str(v)))

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(slider)
    row_layout.addWidget(value_label)
    return slider, row

"""Left-hand controls: open image, difficulty and every setting, page style, picture handling, export, generate.

Each setting the app offers (see ``core.settings``) gets a slider, built from
its entry there: its range, its step, its unit and its tooltip. The difficulty
presets fill in the difficulty's settings, and moving any of those picks
Custom. Reset puts every setting back to its default.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import difficulty, settings
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import Handling
from tessellatum.core.render import DEFAULT_TONE, TONES, PageStyle
from tessellatum.core.settings import Setting

THUMBNAIL_SIZE = 220

# A slider over a setting that moves in equal ratios (``Setting.log``) has this many steps: the region size's, from 2
# to 500 mm², about 5% apart, so that 2 -> 4 mm² is as big a move as 250 -> 500 mm².
LOG_SLIDER_STEPS = 120

TONE_TOOLTIPS = {
    "Light": "Fainter lines and numbers, which vanish under the palest paints.",
    "Medium": "Gray lines, and lighter gray numbers that don't read as writing in the picture.",
    "Dark": "Darker lines and numbers, easier to follow on paper.",
}

CUSTOM_TOOLTIP = "Set every difficulty setting yourself. Moving any of them picks Custom, starting from the preset."

# Where the panel keeps its settings between runs (see ``ControlsPanel``): the preset, the tone, each setting by name,
# and each switch.
_PRESET_KEY = "difficulty/preset"
_TONE_KEY = "style/tone"
_SETTING_KEY = "settings/{}"
_SWITCH_KEY = "handling/{}"


def slider_steps(setting: Setting) -> int:
    """How many steps a slider over ``setting`` has: its range in steps of its own, or ``LOG_SLIDER_STEPS``."""
    if setting.log:
        return LOG_SLIDER_STEPS
    return round((setting.maximum - setting.minimum) / setting.step)


def slider_value(setting: Setting, position: int) -> float:
    """The value at a position of a slider over ``setting``."""
    if setting.log:
        value = setting.minimum * (setting.maximum / setting.minimum) ** (position / LOG_SLIDER_STEPS)
    else:
        value = round(setting.minimum + position * setting.step, 6)
    return setting.clamp(value)


def slider_position(setting: Setting, value: float) -> int:
    """The position of a slider over ``setting`` nearest to ``value``."""
    value = setting.clamp(value)
    if setting.log:
        ratio = math.log(value / setting.minimum) / math.log(setting.maximum / setting.minimum)
        return round(LOG_SLIDER_STEPS * ratio)
    return round((value - setting.minimum) / setting.step)


class SettingSlider(QWidget):
    """A slider over one setting, its value and unit beside it.

    It holds the value it was set to exactly, though that may fall between two
    of its steps (a preset's 125 mm², say), until the slider is moved.
    ``edited`` is emitted only when the user moves it.
    """

    edited = Signal()

    def __init__(self, setting: Setting, parent: QWidget | None = None):
        super().__init__(parent)
        self.setting = setting
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, slider_steps(setting))
        self.value_label = QLabel()
        self.value_label.setFixedWidth(72)
        self.setToolTip(setting.tooltip)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        self._value = setting.default
        self.set_value(setting.default)
        self.slider.valueChanged.connect(self._on_moved)

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Show ``value``, held within the setting's range, without counting it as the user's edit."""
        self._value = self.setting.clamp(value)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_position(self.setting, self._value))
        self.slider.blockSignals(False)
        self.value_label.setText(self.setting.text(self._value))

    def _on_moved(self, position: int) -> None:
        self._value = slider_value(self.setting, position)
        self.value_label.setText(self.setting.text(self._value))
        self.edited.emit()


class Section(QWidget):
    """A titled part of the panel that folds away: a button with its title, and the controls under it."""

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(12, 0, 0, 4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)
        self.toggle.setChecked(expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()


class ControlsPanel(QWidget):
    open_image_requested = Signal()
    generate_requested = Signal()
    export_requested = Signal()
    abort_requested = Signal()

    def __init__(self, parent: QWidget | None = None, store: QSettings | None = None):
        """``store``, if given, is where the panel keeps its settings between runs: read now, written on every change."""
        super().__init__(parent)
        self._store = store
        self._loading = False

        self.open_button = QPushButton("Open Image…")
        self.thumbnail_label = QLabel("No image loaded")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.thumbnail_label.setStyleSheet("border: 1px solid #999; color: #777;")

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(difficulty.preset_names())
        for index, name in enumerate(difficulty.PRESETS):
            self.preset_combo.setItemData(index, difficulty.describe(difficulty.params_for_preset(name)), Qt.ToolTipRole)
        self.preset_combo.setItemData(self.preset_combo.findText("Custom"), CUSTOM_TOOLTIP, Qt.ToolTipRole)
        self.preset_combo.setCurrentText(difficulty.DEFAULT_PRESET)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Put every setting back to its default, and the difficulty to Medium.")
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(self.preset_combo, 1)
        preset_layout.addWidget(self.reset_button)

        # One slider per setting, by the field's name, in the section its group names.
        self.controls: dict[str, SettingSlider] = {s.name: SettingSlider(s) for s in settings.SETTINGS}
        self.sections: dict[str, Section] = {
            group: Section(group, expanded=group in (settings.REGIONS, settings.LINES)) for group in settings.GROUPS
        }
        for s in settings.SETTINGS:
            self.sections[s.group].form.addRow(s.label, self.controls[s.name])

        self.tone_combo = QComboBox()
        self.tone_combo.addItems(list(TONES))
        for index, name in enumerate(TONES):
            self.tone_combo.setItemData(index, TONE_TOOLTIPS[name], Qt.ToolTipRole)
        self.tone_combo.setCurrentText(DEFAULT_TONE)
        self.sections[settings.LINES].form.insertRow(1, "Tone", self.tone_combo)

        self.line_art_check = QCheckBox("Print line art's own ink")
        self.line_art_check.setToolTip(
            "On a cartoon or a comic, print its ink lines and paint the areas they enclose. "
            "Off, it is drawn from its colors like a photograph."
        )
        self.detail_check = QCheckBox("More detail on faces and subject")
        self.detail_check.setToolTip(
            "On a photograph or a painting, regions in the faces and the subject found may be smaller (see Face and "
            "subject detail), a face is painted in a few tones, and its thin dark marks (pupils, lip lines) are printed."
        )
        self.text_check = QCheckBox("Print text")
        self.text_check.setToolTip(
            "Print the letters of the signs, titles and captions found in the picture, their ground left bare, "
            "and keep the numbers off them."
        )
        self._switches = {"line_art": self.line_art_check, "detail": self.detail_check, "text": self.text_check}
        handling_group = QGroupBox("Picture handling")
        handling_layout = QVBoxLayout()
        for check in self._switches.values():
            check.setChecked(True)
            handling_layout.addWidget(check)
        handling_group.setLayout(handling_layout)

        export_group = QGroupBox("Export")
        self.version_combo = QComboBox()
        for index, version in enumerate(Version):
            self.version_combo.addItem(version.value, version)
            self.version_combo.setItemData(index, version.description, Qt.ToolTipRole)
        self.version_combo.setToolTip("Which version of the page to export: the preview can show each of them.")
        self.png_radio = QRadioButton("PNG (image)")
        self.pdf_radio = QRadioButton("PDF (A4)")
        self.png_radio.setChecked(True)
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.png_radio)
        format_layout.addWidget(self.pdf_radio)
        export_form = QFormLayout()
        export_form.addRow("Version", self.version_combo)
        export_form.addRow("Format", format_row)
        export_group.setLayout(export_form)

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
        layout.addWidget(preset_row)
        layout.addWidget(self.sections[settings.REGIONS])
        layout.addWidget(self.sections[settings.LINES])
        layout.addWidget(handling_group)
        for group in (settings.FACES, settings.LINE_ART, settings.TEXT):
            layout.addWidget(self.sections[group])
        layout.addWidget(export_group)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.progress_row)
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        self.open_button.clicked.connect(self.open_image_requested)
        self.generate_button.clicked.connect(self.generate_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.abort_button.clicked.connect(self._on_abort_clicked)
        self.reset_button.clicked.connect(self.reset)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        for name, control in self.controls.items():
            control.edited.connect(lambda name=name: self._on_setting_edited(name))
        self.tone_combo.currentTextChanged.connect(lambda _text: self._save())
        for check in self._switches.values():
            check.toggled.connect(lambda _checked: self._save())

        self._show_preset(difficulty.DEFAULT_PRESET)
        self._load()

    # -- Settings --------------------------------------------------------
    def _difficulty_controls(self) -> dict[str, SettingSlider]:
        return {name: c for name, c in self.controls.items() if c.setting.owner == settings.DIFFICULTY}

    def _show_preset(self, name: str) -> None:
        """Set the difficulty's sliders to the preset ``name``'s values."""
        params = difficulty.params_for_preset(name)
        for field, control in self._difficulty_controls().items():
            control.set_value(getattr(params, field))

    def _on_preset_changed(self, name: str) -> None:
        if name != "Custom":
            self._show_preset(name)
        self._save()

    def _on_setting_edited(self, name: str) -> None:
        # A difficulty setting moved off its preset is a Custom difficulty, starting from the preset's other values.
        if self.controls[name].setting.owner == settings.DIFFICULTY and self.preset_combo.currentText() != "Custom":
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
        self._save()

    def reset(self) -> None:
        """Every setting back to its default: the default preset, the default style, every switch on."""
        self._loading = True
        try:
            self.preset_combo.setCurrentText(difficulty.DEFAULT_PRESET)
            self._show_preset(difficulty.DEFAULT_PRESET)
            for control in self.controls.values():
                if control.setting.owner != settings.DIFFICULTY:
                    control.set_value(control.setting.default)
            self.tone_combo.setCurrentText(DEFAULT_TONE)
            for check in self._switches.values():
                check.setChecked(True)
        finally:
            self._loading = False
        self._save()

    def _save(self) -> None:
        if self._store is None or self._loading:
            return
        self._store.setValue(_PRESET_KEY, self.preset_combo.currentText())
        self._store.setValue(_TONE_KEY, self.tone_combo.currentText())
        for name, control in self.controls.items():
            self._store.setValue(_SETTING_KEY.format(name), control.value())
        for name, check in self._switches.items():
            self._store.setValue(_SWITCH_KEY.format(name), check.isChecked())

    def _load(self) -> None:
        """The settings kept in the store, where it has them; anything it lacks or can't read keeps its default."""
        if self._store is None:
            return
        self._loading = True
        try:
            preset = self._store.value(_PRESET_KEY)
            if preset in difficulty.preset_names():
                self.preset_combo.setCurrentText(preset)
            tone = self._store.value(_TONE_KEY)
            if tone in TONES:
                self.tone_combo.setCurrentText(tone)
            for name, control in self.controls.items():
                kept = self._store.value(_SETTING_KEY.format(name))
                try:
                    value = float(kept)
                except (TypeError, ValueError):
                    continue
                # A preset's difficulty is the preset's, whatever was kept beside it.
                if control.setting.owner != settings.DIFFICULTY or preset == "Custom":
                    control.set_value(value)
            for name, check in self._switches.items():
                kept = self._store.value(_SWITCH_KEY.format(name))
                if kept is not None:
                    check.setChecked(kept in (True, "true", "True", 1, "1"))
        finally:
            self._loading = False

    # -- Buttons -----------------------------------------------------------
    def _on_abort_clicked(self) -> None:
        # Cancellation takes effect at the pipeline's next stage boundary, so
        # disable the button immediately to avoid double-clicks while we wait.
        self.abort_button.setEnabled(False)
        self.abort_requested.emit()

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumbnail_label.setPixmap(scaled)

    # -- What the pipeline is asked for -------------------------------------
    def _chosen(self, owner: str) -> dict[str, float]:
        return {name: c.value() for name, c in self.controls.items() if c.setting.owner == owner}

    def get_difficulty_params(self) -> difficulty.DifficultyParams:
        preset = self.preset_combo.currentText()
        if preset != "Custom":
            return difficulty.params_for_preset(preset)
        return difficulty.custom_params(**self._chosen(settings.DIFFICULTY))

    def get_page_style(self) -> PageStyle:
        return PageStyle.from_settings(tone=self.tone_combo.currentText(), **self._chosen(settings.STYLE))

    def get_handling(self) -> Handling:
        switches = {name: check.isChecked() for name, check in self._switches.items()}
        return Handling(**switches, **self._chosen(settings.HANDLING))

    def get_output_format(self) -> str:
        return "PDF" if self.pdf_radio.isChecked() else "PNG"

    def get_export_version(self) -> Version:
        return self.version_combo.currentData()

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

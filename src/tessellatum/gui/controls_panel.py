"""Left-hand controls: open image, difficulty and every setting, page style, picture handling, export, generate.

Each setting the app offers (see ``core.settings``) gets a slider, built from
its entry there: its range, its step, its unit and its tooltip. The difficulty
presets fill in the difficulty's settings, and moving any of those picks
Custom; the tones fill in the ink's three grays alike. Reset puts every
setting back to its default.
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
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import difficulty, export, settings
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import Handling
from tessellatum.core.print_size import MM_PER_INCH, PAPERS, print_scale
from tessellatum.core.render import DEFAULT_TONE, TONE_FIELDS, TONES, PageStyle
from tessellatum.core.settings import Setting

THUMBNAIL_SIZE = 220

# A slider over a setting that moves in equal ratios (``Setting.log``) has this many steps: the region size's, from 5
# to 500 mm², about 4% apart, so that 5 -> 10 mm² is as big a move as 250 -> 500 mm².
LOG_SLIDER_STEPS = 120

# A slider is never narrower than this, however narrow the panel is dragged: below it a handle has no room to move.
MIN_SLIDER_WIDTH = 80

TONE_TOOLTIPS = {
    "Light": "Fainter lines, numbers and leaders, which vanish under the palest paints.",
    "Medium": "Gray lines, lighter gray numbers that don't read as writing in the picture, and the numbers written "
    "outside their regions in a gray of their own, with their leaders.",
    "Dark": "Darker lines, numbers and leaders, easier to follow on paper.",
}
CUSTOM_TONE = "Custom"
CUSTOM_TONE_TOOLTIP = "Set the three grays yourself. Moving any of them picks Custom, starting from the tone."

PAPER_TOOLTIP = (
    "The sheet the page is made for: every size on paper (the brush, the smallest region, the lines, the numbers) "
    "applies on it. A larger sheet prints the picture larger, so it makes a finer page, with more and smaller regions, "
    "and takes longer; the PDF's sheets are this size."
)
DEFAULT_PAPER = "A4"

# The PNG's resolution (T3c): in dots per inch on its paper, or in pixels along its long edge.
DPI, PIXELS = "dpi", "px"
DPI_PRESETS = (150, 300, 600)
# 600 dpi: an A4 export of the benchmark's heaviest page (the lion at Realistic) lays out in 6 s and draws at 600 dpi in
# 1.6 s more, at 1.1 GB, within the 15 s and 2 GB a default may take (D-011).
DEFAULT_DPI = 600
MIN_DPI = 72
MIN_PIXELS = 500
RESOLUTION_TOOLTIP = (
    "How finely the PNG is drawn: in dots per inch on its paper, or in pixels along its long edge, the other worked "
    "out from the paper and the picture's shape. 300 dpi prints as sharp as a photo print; 600 is sharper still, for "
    "small numbers and for coloring zoomed in on a screen. A PNG is at most 16,384 px along a side: as much as "
    "Procreate's canvases take, which hold 33 to 134 megapixels by iPad. The PDF is vector art, sharp at any size: "
    "an app that turns it into pixels picks its own resolution."
)

CUSTOM_TOOLTIP = "Set every difficulty setting yourself. Moving any of them picks Custom, starting from the preset."

# Where the panel keeps its settings between runs (see ``ControlsPanel``): the preset, the tone, each setting by name,
# and each switch.
_PRESET_KEY = "difficulty/preset"
_TONE_KEY = "style/tone"
_PAPER_KEY = "style/paper"
_UNIT_KEY = "export/unit"
_DPI_KEY = "export/dpi"
_PIXELS_KEY = "export/pixels"
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
        self.slider.setMinimumWidth(MIN_SLIDER_WIDTH)
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
        """``store``, if given, is where the panel keeps its settings between runs: read now, written on each change."""
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
            label = QLabel(s.label)
            label.setToolTip(s.tooltip)
            self.sections[s.group].form.addRow(label, self.controls[s.name])

        self.tone_combo = QComboBox()
        self.tone_combo.addItems([*TONES, CUSTOM_TONE])
        for index, name in enumerate(TONES):
            self.tone_combo.setItemData(index, TONE_TOOLTIPS[name], Qt.ToolTipRole)
        self.tone_combo.setItemData(len(TONES), CUSTOM_TONE_TOOLTIP, Qt.ToolTipRole)
        self.tone_combo.setCurrentText(DEFAULT_TONE)
        self.sections[settings.LINES].form.insertRow(1, "Tone", self.tone_combo)
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(list(PAPERS))
        self.paper_combo.setCurrentText(DEFAULT_PAPER)
        self.paper_combo.setToolTip(PAPER_TOOLTIP)
        self.sections[settings.REGIONS].form.insertRow(0, "Paper", self.paper_combo)
        self.sections[settings.REGIONS].form.labelForField(self.paper_combo).setToolTip(PAPER_TOOLTIP)
        # Every section's labels as wide as the widest, so that the sliders line up from one section to the next.
        labels = [
            section.form.labelForField(section.form.itemAt(row, QFormLayout.FieldRole).widget())
            for section in self.sections.values()
            for row in range(section.form.rowCount())
        ]
        label_width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setMinimumWidth(label_width)

        self.line_art_check = QCheckBox("Print line art's own ink")
        self.line_art_check.setToolTip(
            "On a cartoon or a comic, print its ink lines and paint the areas they enclose. "
            "Off, it is drawn from its colors like a photograph."
        )
        self.detail_check = QCheckBox("More detail on faces and subject")
        self.detail_check.setToolTip(
            "On a photograph or a painting, regions in the faces and the subject found may be smaller (see Face and "
            "subject detail), a face is painted in a few tones, and its thin dark marks (pupils, lip lines) are "
            "printed."
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
        self.pdf_radio = QRadioButton("PDF")
        self.png_radio.setChecked(True)
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.png_radio)
        format_layout.addWidget(self.pdf_radio)
        self._picture_size: tuple[int, int] | None = None
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setGroupSeparatorShown(True)
        self.resolution_spin.setToolTip(RESOLUTION_TOOLTIP)
        self.resolution_unit = QComboBox()
        self.resolution_unit.addItem("dpi", DPI)
        self.resolution_unit.addItem("px", PIXELS)
        self.resolution_unit.setItemData(0, "Dots per inch on the paper", Qt.ToolTipRole)
        self.resolution_unit.setItemData(1, "Pixels along the page's long edge", Qt.ToolTipRole)
        self.resolution_unit.setToolTip(RESOLUTION_TOOLTIP)
        resolution_row = QWidget()
        resolution_layout = QHBoxLayout(resolution_row)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.addWidget(self.resolution_spin, 1)
        resolution_layout.addWidget(self.resolution_unit)
        self.dpi_buttons: dict[int, QPushButton] = {}
        presets_row = QWidget()
        presets_layout = QHBoxLayout(presets_row)
        presets_layout.setContentsMargins(0, 0, 0, 0)
        for dpi in DPI_PRESETS:
            button = QPushButton(f"{dpi} dpi")
            button.setToolTip(f"Draw the PNG at {dpi} dots per inch on its paper.")
            button.clicked.connect(lambda _checked=False, dpi=dpi: self.set_export_dpi(dpi))
            presets_layout.addWidget(button)
            self.dpi_buttons[dpi] = button
        self.resolution_readout = QLabel()
        self.resolution_readout.setWordWrap(True)
        self.resolution_readout.setStyleSheet("color: #666;")
        # Each mode's value, the one not shown kept for it: dots per inch, and pixels along the long edge (None until
        # first shown, then what the dpi came to).
        self._dpi, self._pixels = DEFAULT_DPI, None
        export_form = QFormLayout()
        export_form.addRow("Version", self.version_combo)
        export_form.addRow("Format", format_row)
        export_form.addRow("Resolution", resolution_row)
        export_form.addRow("", presets_row)
        export_form.addRow(self.resolution_readout)
        export_form.labelForField(resolution_row).setToolTip(RESOLUTION_TOOLTIP)
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
        self.tone_combo.currentTextChanged.connect(self._on_tone_changed)
        self.paper_combo.currentTextChanged.connect(lambda _name: self._on_print_changed())
        self.resolution_unit.currentIndexChanged.connect(lambda _index: self._on_unit_changed())
        self.resolution_spin.valueChanged.connect(lambda _value: self._on_resolution_edited())
        self.png_radio.toggled.connect(lambda _checked: self._show_resolution())
        for check in self._switches.values():
            check.toggled.connect(lambda _checked: self._save())

        self._show_preset(difficulty.DEFAULT_PRESET)
        self._show_resolution()
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

    def _show_tone(self, name: str) -> None:
        """Set the gray sliders to the tone ``name``'s grays."""
        for field, gray in zip(TONE_FIELDS, TONES[name]):
            self.controls[field].set_value(gray)

    def _on_tone_changed(self, name: str) -> None:
        if name in TONES:
            self._show_tone(name)
        self._save()

    def _on_setting_edited(self, name: str) -> None:
        # A difficulty setting moved off its preset is a Custom difficulty, starting from the preset's other values;
        # a gray moved off its tone, a Custom tone, likewise.
        if self.controls[name].setting.owner == settings.DIFFICULTY and self.preset_combo.currentText() != "Custom":
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
        if name in TONE_FIELDS and self.tone_combo.currentText() != CUSTOM_TONE:
            self.tone_combo.blockSignals(True)
            self.tone_combo.setCurrentText(CUSTOM_TONE)
            self.tone_combo.blockSignals(False)
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
            self._show_tone(DEFAULT_TONE)
            self.paper_combo.setCurrentText(DEFAULT_PAPER)
            for check in self._switches.values():
                check.setChecked(True)
            self._dpi, self._pixels = DEFAULT_DPI, None
            self.resolution_unit.setCurrentIndex(self.resolution_unit.findData(DPI))
            self._show_resolution()
        finally:
            self._loading = False
        self._save()

    def _save(self) -> None:
        if self._store is None or self._loading:
            return
        self._store.setValue(_PRESET_KEY, self.preset_combo.currentText())
        self._store.setValue(_TONE_KEY, self.tone_combo.currentText())
        self._store.setValue(_PAPER_KEY, self.paper_combo.currentText())
        for name, control in self.controls.items():
            self._store.setValue(_SETTING_KEY.format(name), control.value())
        for name, check in self._switches.items():
            self._store.setValue(_SWITCH_KEY.format(name), check.isChecked())
        self._store.setValue(_UNIT_KEY, self._unit())
        self._store.setValue(_DPI_KEY, self._dpi)
        if self._pixels is not None:
            self._store.setValue(_PIXELS_KEY, self._pixels)

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
            if tone in TONES or tone == CUSTOM_TONE:
                self.tone_combo.setCurrentText(tone)
            tone = self.tone_combo.currentText()
            paper = self._store.value(_PAPER_KEY)
            if paper in PAPERS:
                self.paper_combo.setCurrentText(paper)
            for name, control in self.controls.items():
                kept = self._store.value(_SETTING_KEY.format(name))
                try:
                    value = float(kept)
                except (TypeError, ValueError):
                    continue
                # A preset's difficulty is the preset's, and a tone's grays the tone's, whatever was kept beside them.
                if control.setting.owner == settings.DIFFICULTY and preset != "Custom":
                    continue
                if name in TONE_FIELDS and tone != CUSTOM_TONE:
                    continue
                control.set_value(value)
            for name, check in self._switches.items():
                kept = self._store.value(_SWITCH_KEY.format(name))
                if kept is not None:
                    check.setChecked(kept in (True, "true", "True", 1, "1"))
            for key, attr in ((_DPI_KEY, "_dpi"), (_PIXELS_KEY, "_pixels")):
                try:
                    setattr(self, attr, int(self._store.value(key)))
                except (TypeError, ValueError):
                    pass
            unit = self._store.value(_UNIT_KEY)
            if unit in (DPI, PIXELS):
                self.resolution_unit.setCurrentIndex(self.resolution_unit.findData(unit))
            self._show_resolution()
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
        # The grays on their sliders, the tone's unless Custom.
        return PageStyle(**self._chosen(settings.STYLE), paper=self.paper_combo.currentText())

    def get_handling(self) -> Handling:
        switches = {name: check.isChecked() for name, check in self._switches.items()}
        return Handling(**switches, **self._chosen(settings.HANDLING))

    def get_output_format(self) -> str:
        return "PDF" if self.pdf_radio.isChecked() else "PNG"

    def get_export_version(self) -> Version:
        return self.version_combo.currentData()

    # -- The PNG's resolution ------------------------------------------------
    def set_picture_size(self, size: tuple[int, int] | None) -> None:
        """The picture's pixels (width, height), whose shape sets how large the page prints, or None."""
        self._picture_size = size
        self._show_resolution()

    def set_export_dpi(self, dpi: float) -> None:
        """Draw the PNG at ``dpi``, shown in dots per inch."""
        self._dpi = round(dpi)
        self.resolution_unit.blockSignals(True)
        self.resolution_unit.setCurrentIndex(self.resolution_unit.findData(DPI))
        self.resolution_unit.blockSignals(False)
        self._show_resolution()
        self._save()

    def get_export_resolution(self) -> tuple[str, int]:
        """The PNG's resolution as asked for: (``DPI``, dots per inch) or (``PIXELS``, pixels along the long edge)."""
        return self._unit(), self.resolution_spin.value()

    def get_export_dpi(self, printed_long_mm: float | None = None) -> float:
        """The dots per inch the PNG is drawn at, on a page ``printed_long_mm`` long on paper (by default, the
        picture's): as asked, or for a size in pixels, what puts that many along the long edge."""
        if self._unit() == DPI:
            return float(self.resolution_spin.value())
        long_mm = printed_long_mm if printed_long_mm is not None else self._printed_long_mm()
        return self.resolution_spin.value() * MM_PER_INCH / long_mm

    def _unit(self) -> str:
        return self.resolution_unit.currentData()

    def _printed_long_mm(self) -> float:
        """How long the page prints along its long edge: the picture's shape on the paper, or with no picture the
        paper's printable area."""
        paper = PAPERS[self.paper_combo.currentText()]
        if self._picture_size is None:
            return max(paper.printable_mm())
        return max(print_scale(self._picture_size, paper).printed_size_mm)

    def _on_unit_changed(self) -> None:
        # The other mode starts at what the shown one asked for; loaded, at what it was kept at.
        long_mm = self._printed_long_mm()
        if self._loading:
            pass
        elif self._unit() == PIXELS:
            self._pixels = round(long_mm * self._dpi / MM_PER_INCH)
        else:
            self._dpi = round(self._pixels * MM_PER_INCH / long_mm) if self._pixels else self._dpi
        self._show_resolution()
        self._save()

    def _on_resolution_edited(self) -> None:
        if self._unit() == DPI:
            self._dpi = self.resolution_spin.value()
        else:
            self._pixels = self.resolution_spin.value()
        self._show_resolution()
        self._save()

    def _on_print_changed(self) -> None:
        self._show_resolution()
        self._save()

    def _show_resolution(self) -> None:
        """The resolution's box in its mode, within what a PNG takes on this page (see ``export.MAX_RASTER_LONG_EDGE``),
        and what it comes to: the file's pixels and the dpi."""
        long_mm = self._printed_long_mm()
        most_dpi = int((export.MAX_RASTER_LONG_EDGE - 0.5) * MM_PER_INCH / long_mm)
        spin = self.resolution_spin
        spin.blockSignals(True)
        if self._unit() == DPI:
            spin.setRange(MIN_DPI, most_dpi)
            spin.setSingleStep(25)
            spin.setValue(self._dpi)  # within the range: the most this paper takes, if more was asked
            dpi = float(spin.value())
        else:
            spin.setRange(MIN_PIXELS, export.MAX_RASTER_LONG_EDGE)
            spin.setSingleStep(100)
            if self._pixels is None:
                self._pixels = round(long_mm * self._dpi / MM_PER_INCH)
            spin.setValue(self._pixels)
            dpi = spin.value() * MM_PER_INCH / long_mm
        spin.blockSignals(False)
        png = self.png_radio.isChecked()
        for widget in (spin, self.resolution_unit, *self.dpi_buttons.values()):
            widget.setEnabled(png)
        if not png:
            self.resolution_readout.setText("The PDF is vector art: it prints sharp at any size.")
            return
        paper = self.paper_combo.currentText()
        if self._picture_size is None:
            text = f"{round(long_mm * dpi / MM_PER_INCH):,} px along the long edge at {dpi:.0f} dpi on {paper}."
        else:
            printed = print_scale(self._picture_size, PAPERS[paper]).printed_size_mm
            width, height = (max(1, round(mm * dpi / MM_PER_INCH)) for mm in printed)
            text = f"{width:,} × {height:,} px ({width * height / 1e6:.0f} Mpx) at {dpi:.0f} dpi on {paper}."
        capped = self._dpi > most_dpi if self._unit() == DPI else spin.value() == export.MAX_RASTER_LONG_EDGE
        if capped:
            text += f" The most a PNG takes: {export.MAX_RASTER_LONG_EDGE:,} px along a side."
        self.resolution_readout.setText(text)

    def set_busy(self, busy: bool, cancellable: bool = True) -> None:
        """Busy, or not: while busy the progress row shows, and the buttons that start something are off. A step that
        can't be ``cancellable`` shows a busy bar with no percentage and no abort."""
        self.progress_row.setVisible(busy)
        self.progress_bar.setRange(0, 100 if cancellable else 0)
        if busy:
            self.progress_bar.setValue(0)
            self.abort_button.setEnabled(cancellable)
        self.generate_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.export_button.property("hasResult") is True)
        self.open_button.setEnabled(not busy)

    def set_progress(self, percent: int) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setProperty("hasResult", enabled)
        self.export_button.setEnabled(enabled)

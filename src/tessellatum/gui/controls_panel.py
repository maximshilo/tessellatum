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

from tessellatum.app.snapshot import (  # noqa: F401 - the panel's names for them, kept for its callers
    COLORS_FROM,
    CUSTOM,
    CUSTOM_TONE,
    DEFAULT_DPI,
    DEFAULT_PAPER,
    DPI,
    MY_PAINTS,
    PALETTE,
    PDF,
    PICTURE_COLORS,
    PIXELS,
    PNG,
    DocumentSettings,
    ExportOptions,
)
from tessellatum.core import difficulty, export, settings, swatches
from tessellatum.core.legend import DEFAULT_LEGEND_DETAIL, LEGEND_DETAILS, NAME, NAME_HEX_RGB, NAME_RECIPE, NUMBER_ONLY
from tessellatum.core.paints import distinct, paints_text, parse_paints
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import Handling
from tessellatum.core.print_size import MM_PER_INCH, PAPERS, print_scale
from tessellatum.core.render import DEFAULT_TONE, TONE_FIELDS, TONES, PageStyle
from tessellatum.core.settings import (  # noqa: F401 - the registry's strings, by the panel's names for them
    COLORS_FROM_TOOLTIP,
    CUSTOM_TONE_TOOLTIP,
    CUSTOM_TOOLTIP,
    LEGEND_TOOLTIP,
    LEGEND_TOOLTIPS,
    MEDIUM_TOOLTIP,
    MEDIUM_TOOLTIPS,
    MIX_TOOLTIP,
    PAPER_TOOLTIP,
    TONE_TOOLTIPS,
    Setting,
)

THUMBNAIL_SIZE = 220

# A slider over a setting that moves in equal ratios (``Setting.log``) has this many steps: the region size's, from 5
# to 500 mm², about 4% apart, so that 5 -> 10 mm² is as big a move as 250 -> 500 mm².
LOG_SLIDER_STEPS = 120

# A slider is never narrower than this, however narrow the panel is dragged: below it a handle has no room to move.
MIN_SLIDER_WIDTH = 80


PALETTE_TOOLTIP = (
    "The legend's colors as a palette file, for coloring the page in a painting app with exactly its colors, each "
    "named by its number and its name on the legend (\"7 dusty rose\"). Procreate: a .swatches file of 30 colors at "
    "most, so a larger legend is saved as several (\"1-30\", \"31-60\"). Adobe Swatch Exchange (.ase): Photoshop, "
    "Illustrator, Affinity, Procreate and others. GIMP palette (.gpl): GIMP, Krita, Inkscape and others."
)
SCREEN_HINT = "Coloring on a screen: export the Palette to load the legend's colors into your painting app."


# The PNG's resolution (T3c): in dots per inch on its paper, or in pixels along its long edge.
DPI_PRESETS = (150, 300, 600)
# DEFAULT_DPI, 600 dpi: the benchmark's slowest A4 export, the complex cartoon at Realistic, lays its preview out in
# 7.2 s and draws it at 600 dpi in 3.1 s more, at 1.1 GB, within the 15 s and 2 GB a default may take (D-012).
MIN_DPI = 72
MIN_PIXELS = 500
RESOLUTION_TOOLTIP = (
    "How finely the PNG is drawn: in dots per inch on its paper, or in pixels along its long edge, the other worked "
    "out from the paper and the picture's shape. 300 dpi prints as sharp as a photo print; 600 is sharper still, for "
    "small numbers and for coloring zoomed in on a screen. A PNG is at most 16,384 px along a side: as much as "
    "Procreate's canvases take, which hold 33 to 134 megapixels by iPad. The PDF is vector art, sharp at any size: "
    "an app that turns it into pixels picks its own resolution."
)


# Where the panel keeps its settings between runs (see ``ControlsPanel``): the preset, the tone, each setting by name,
# and each switch.
# The document's settings are kept under their registry keys (``Setting.key``, ``Choice.key``); the export's options,
# which are no document setting (D-008), under keys of their own.
_PRESET_KEY = settings.choice("preset").key
_TONE_KEY = settings.choice("tone").key
_PAPER_KEY = settings.choice("paper").key
_MEDIUM_KEY = settings.choice("medium").key
_LEGEND_KEY = settings.choice("legend_detail").key
_COLORS_FROM_KEY = settings.choice("colors_from").key
_PAINTS_KEY = settings.choice("paints").key
_MIX_KEY = settings.choice("mix_paints").key
_PALETTE_FORMAT_KEY = "export/palette"
_UNIT_KEY = "export/unit"
_DPI_KEY = "export/dpi"
_PIXELS_KEY = "export/pixels"


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
        self.toggle.setObjectName("sectionHeader")  # styled by the theme (see ``theme.style``)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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
    # What the legend writes beside each swatch was changed: it applies to the page shown, which isn't made again.
    legend_detail_changed = Signal(str)
    paints_changed = Signal()  # the paints listed, which the legend's recipes are mixed from
    # Any setting changed: the settings as they are now, a ``DocumentSettings`` (see ``snapshot``). Not while the
    # panel is loading or resetting, which ends with one.
    settings_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None, store: QSettings | None = None):
        """``store``, if given, is where the panel keeps its settings between runs: read now, written on each change."""
        super().__init__(parent)
        self.setObjectName("sidebar")  # the theme's sidebar surface
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._store = store
        self._loading = False
        self._announced: DocumentSettings | None = None  # the settings last sent with settings_changed

        self.open_button = QPushButton("Open Image…")
        self.thumbnail_label = QLabel("No image loaded")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.thumbnail_label.setObjectName("thumbnail")

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(difficulty.preset_names())
        for index, name in enumerate(difficulty.PRESETS):
            self.preset_combo.setItemData(index, difficulty.describe(difficulty.params_for_preset(name)), Qt.ToolTipRole)
        self.preset_combo.setItemData(self.preset_combo.findText(CUSTOM), CUSTOM_TOOLTIP, Qt.ToolTipRole)
        self.preset_combo.setCurrentText(difficulty.DEFAULT_PRESET)
        self.preset_combo.setToolTip(settings.choice("preset").tooltip)
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
        self.tone_combo.setToolTip(settings.choice("tone").tooltip)
        self.sections[settings.LINES].form.insertRow(1, "Tone", self.tone_combo)
        self.legend_combo = QComboBox()
        self.legend_combo.addItems(list(LEGEND_DETAILS))
        for index, name in enumerate(LEGEND_DETAILS):
            self.legend_combo.setItemData(index, LEGEND_TOOLTIPS[name], Qt.ToolTipRole)
        self.legend_combo.setCurrentText(DEFAULT_LEGEND_DETAIL)
        self.legend_combo.setToolTip(LEGEND_TOOLTIP)
        self.sections[settings.LINES].form.insertRow(2, "Legend", self.legend_combo)
        self.sections[settings.LINES].form.labelForField(self.legend_combo).setToolTip(LEGEND_TOOLTIP)
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(list(PAPERS))
        self.paper_combo.setCurrentText(DEFAULT_PAPER)
        self.paper_combo.setToolTip(PAPER_TOOLTIP)
        self.sections[settings.REGIONS].form.insertRow(0, "Paper", self.paper_combo)
        self.sections[settings.REGIONS].form.labelForField(self.paper_combo).setToolTip(PAPER_TOOLTIP)
        self.medium_combo = QComboBox()
        self.medium_combo.addItems(list(difficulty.MEDIA))
        for index, name in enumerate(difficulty.MEDIA):
            self.medium_combo.setItemData(index, MEDIUM_TOOLTIPS[name], Qt.ToolTipRole)
        self.medium_combo.setCurrentText(difficulty.DEFAULT_MEDIUM)
        self.medium_combo.setToolTip(MEDIUM_TOOLTIP)
        self.sections[settings.REGIONS].form.insertRow(1, "Coloring on", self.medium_combo)
        self.sections[settings.REGIONS].form.labelForField(self.medium_combo).setToolTip(MEDIUM_TOOLTIP)
        self._paints: tuple = ()
        self.colors_from_combo = QComboBox()
        self.colors_from_combo.addItems(list(COLORS_FROM))
        self.colors_from_combo.setToolTip(COLORS_FROM_TOOLTIP)
        self.paints_button = QPushButton("Paints…")
        self.paints_button.setToolTip(settings.choice("paints").tooltip)
        colors_from_row = QWidget()
        colors_from_layout = QHBoxLayout(colors_from_row)
        colors_from_layout.setContentsMargins(0, 0, 0, 0)
        colors_from_layout.addWidget(self.colors_from_combo, 1)
        colors_from_layout.addWidget(self.paints_button)
        self.sections[settings.REGIONS].form.insertRow(2, "Colors from", colors_from_row)
        self.sections[settings.REGIONS].form.labelForField(colors_from_row).setToolTip(COLORS_FROM_TOOLTIP)
        self.mix_check = QCheckBox(settings.choice("mix_paints").label)
        self.mix_check.setToolTip(MIX_TOOLTIP)
        self.sections[settings.REGIONS].form.insertRow(3, QLabel(""), self.mix_check)
        # Every section's labels as wide as the widest, so that the sliders line up from one section to the next.
        labels = [
            section.form.labelForField(section.form.itemAt(row, QFormLayout.FieldRole).widget())
            for section in self.sections.values()
            for row in range(section.form.rowCount())
        ]
        label_width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setMinimumWidth(label_width)

        self._switches: dict[str, QCheckBox] = {}
        for name in settings.SWITCHES:
            check = QCheckBox(settings.choice(name).label)
            check.setToolTip(settings.choice(name).tooltip)
            self._switches[name] = check
        self.line_art_check, self.detail_check, self.text_check = (self._switches[n] for n in settings.SWITCHES)
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
        self.palette_radio = QRadioButton("Palette")
        self.palette_radio.setToolTip(PALETTE_TOOLTIP)
        self.png_radio.setChecked(True)
        self.palette_combo = QComboBox()
        for suffix, title in swatches.FORMATS.items():
            self.palette_combo.addItem(title.split(" (")[0], suffix)
        self.palette_combo.setToolTip(PALETTE_TOOLTIP)
        self.screen_hint = QLabel(SCREEN_HINT)
        self.screen_hint.setWordWrap(True)
        self.screen_hint.setProperty("role", "hint")
        self.screen_hint.setVisible(False)  # shown coloring on a screen
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.png_radio)
        format_layout.addWidget(self.pdf_radio)
        format_layout.addWidget(self.palette_radio)
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
        self.resolution_readout.setProperty("role", "hint")
        # Each mode's value, the one not shown kept for it: dots per inch, and pixels along the long edge (None until
        # first shown, then what the dpi came to).
        self._dpi, self._pixels = DEFAULT_DPI, None
        export_form = QFormLayout()
        export_form.addRow("Version", self.version_combo)
        export_form.addRow("Format", format_row)
        export_form.addRow("Palette", self.palette_combo)
        export_form.labelForField(self.palette_combo).setToolTip(PALETTE_TOOLTIP)
        export_form.addRow(self.screen_hint)
        export_form.addRow("Resolution", resolution_row)
        export_form.addRow("", presets_row)
        export_form.addRow(self.resolution_readout)
        export_form.labelForField(resolution_row).setToolTip(RESOLUTION_TOOLTIP)
        export_group.setLayout(export_form)

        self.generate_button = QPushButton("Generate Preview")
        self.generate_button.setProperty("primary", True)  # the one primary action
        self.export_button = QPushButton("Export…")
        self.export_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # The percentage beside the bar, not on it: text on a bar half filled with the accent can't keep its contrast.
        self.progress_bar.setTextVisible(False)
        self.progress_label = QLabel("0%")
        self.progress_label.setProperty("role", "secondary")
        self.progress_label.setMinimumWidth(self.progress_label.fontMetrics().horizontalAdvance("100%") + 4)
        self.progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.abort_button = QPushButton("✕")
        self.abort_button.setObjectName("abortButton")  # a square button: the theme gives it no side padding
        self.abort_button.setToolTip("Cancel generation")
        self.abort_button.setFixedWidth(28)

        self.progress_row = QWidget()
        progress_row_layout = QHBoxLayout(self.progress_row)
        progress_row_layout.setContentsMargins(0, 0, 0, 0)
        progress_row_layout.addWidget(self.progress_bar, 1)
        progress_row_layout.addWidget(self.progress_label)
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
        self.medium_combo.currentTextChanged.connect(lambda _name: self._on_medium_changed())
        self.legend_combo.currentTextChanged.connect(self._on_legend_changed)
        self.colors_from_combo.currentTextChanged.connect(lambda _name: self._on_colors_from_changed())
        self.mix_check.toggled.connect(lambda _checked: self._on_mix_changed())
        self.paints_button.clicked.connect(self.edit_paints)
        self.resolution_unit.currentIndexChanged.connect(lambda _index: self._on_unit_changed())
        self.resolution_spin.valueChanged.connect(lambda _value: self._on_resolution_edited())
        for radio in (self.png_radio, self.pdf_radio, self.palette_radio):
            radio.toggled.connect(lambda checked: checked and self._on_format_changed())
        self.version_combo.currentIndexChanged.connect(lambda _index: self._save())
        self.palette_combo.currentIndexChanged.connect(lambda _index: self._save())
        for check in self._switches.values():
            check.toggled.connect(lambda _checked: self._on_switch_changed())

        # Each of the registry's choices, by name, and the control that edits it.
        self.choice_widgets: dict[str, QWidget] = {
            "preset": self.preset_combo, "paper": self.paper_combo, "medium": self.medium_combo,
            "colors_from": self.colors_from_combo, "paints": self.paints_button, "mix_paints": self.mix_check,
            "tone": self.tone_combo, "legend_detail": self.legend_combo, **self._switches,
        }

        self._show_preset(difficulty.DEFAULT_PRESET)
        self._show_resolution()
        self._show_paints()  # the mixing settings stand aside until the colors come from the paints
        self._load()
        self._announced = self.snapshot()

    # -- Settings --------------------------------------------------------
    def _difficulty_controls(self) -> dict[str, SettingSlider]:
        return {name: c for name, c in self.controls.items() if c.setting.owner == settings.DIFFICULTY}

    def _show_preset(self, name: str) -> None:
        """Set the difficulty's sliders to the preset ``name``'s values."""
        params = difficulty.params_for_preset(name)
        for field, control in self._difficulty_controls().items():
            control.set_value(getattr(params, field))

    def _on_preset_changed(self, name: str) -> None:
        if name != CUSTOM:
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
        if self.controls[name].setting.owner == settings.DIFFICULTY and self.preset_combo.currentText() != CUSTOM:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(CUSTOM)
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
            self.medium_combo.setCurrentText(difficulty.DEFAULT_MEDIUM)
            self.legend_combo.setCurrentText(DEFAULT_LEGEND_DETAIL)
            self.palette_combo.setCurrentIndex(0)
            self.colors_from_combo.setCurrentText(PICTURE_COLORS)  # the paints listed are kept, for next time
            self.mix_check.setChecked(False)
            self._show_paints()
            for check in self._switches.values():
                check.setChecked(True)
            self._dpi, self._pixels = DEFAULT_DPI, None
            self.resolution_unit.setCurrentIndex(self.resolution_unit.findData(DPI))
            self._show_resolution()
        finally:
            self._loading = False
        self._save()

    def _on_format_changed(self) -> None:
        self._show_resolution()
        self._save()

    def _save(self) -> None:
        """Announce the settings if they changed, and keep them in the store."""
        if self._loading:
            return
        snapshot = self.snapshot()
        if snapshot != self._announced:
            self._announced = snapshot
            self.settings_changed.emit(snapshot)
        if self._store is None:
            return
        self._store.setValue(_PRESET_KEY, self.preset_combo.currentText())
        self._store.setValue(_TONE_KEY, self.tone_combo.currentText())
        self._store.setValue(_PAPER_KEY, self.paper_combo.currentText())
        self._store.setValue(_MEDIUM_KEY, self.medium_combo.currentText())
        self._store.setValue(_LEGEND_KEY, self.legend_combo.currentText())
        self._store.setValue(_PALETTE_FORMAT_KEY, self.palette_combo.currentData())
        self._store.setValue(_COLORS_FROM_KEY, self.colors_from_combo.currentText())
        self._store.setValue(_PAINTS_KEY, paints_text(self._paints))
        self._store.setValue(_MIX_KEY, self.mix_check.isChecked())
        for name, control in self.controls.items():
            self._store.setValue(control.setting.key, control.value())
        for name, check in self._switches.items():
            self._store.setValue(settings.choice(name).key, check.isChecked())
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
            medium = self._store.value(_MEDIUM_KEY)
            if medium in difficulty.MEDIA:
                self.medium_combo.setCurrentText(medium)
            legend = self._store.value(_LEGEND_KEY)
            if legend in LEGEND_DETAILS:
                self.legend_combo.setCurrentText(legend)
            kept_paints = self._store.value(_PAINTS_KEY)
            if isinstance(kept_paints, str):
                self._paints = distinct(parse_paints(kept_paints).paints)
            colors_from = self._store.value(_COLORS_FROM_KEY)
            if colors_from in COLORS_FROM:
                self.colors_from_combo.setCurrentText(colors_from)
            self.mix_check.setChecked(self._store.value(_MIX_KEY) in (True, "true", "True", 1, "1"))
            self._show_paints()
            palette_format = self._store.value(_PALETTE_FORMAT_KEY)
            if palette_format in swatches.FORMATS:
                self.palette_combo.setCurrentIndex(self.palette_combo.findData(palette_format))
            for name, control in self.controls.items():
                kept = self._store.value(control.setting.key)
                try:
                    value = float(kept)
                except (TypeError, ValueError):
                    continue
                # A preset's difficulty is the preset's, and a tone's grays the tone's, whatever was kept beside them.
                if control.setting.owner == settings.DIFFICULTY and preset != CUSTOM:
                    continue
                if name in TONE_FIELDS and tone != CUSTOM_TONE:
                    continue
                control.set_value(value)
            for name, check in self._switches.items():
                kept = self._store.value(settings.choice(name).key)
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

    # -- The settings as one value -----------------------------------------
    def snapshot(self) -> DocumentSettings:
        """The settings as the panel shows them: what the document keeps, and what every ``get_`` below reads."""
        return DocumentSettings(
            preset=self.preset_combo.currentText(),
            values={name: control.value() for name, control in self.controls.items()},
            tone=self.tone_combo.currentText(),
            paper=self.paper_combo.currentText(),
            medium=self.medium_combo.currentText(),
            legend=self.legend_combo.currentText(),
            colors_from=self.colors_from_combo.currentText(),
            paints=self._paints,
            mix=self.mix_check.isChecked(),
            **{name: check.isChecked() for name, check in self._switches.items()},
            export=ExportOptions(
                version=self.version_combo.currentData(),
                format=PALETTE if self.palette_radio.isChecked() else PDF if self.pdf_radio.isChecked() else PNG,
                palette_format=self.palette_combo.currentData(),
                unit=self._unit(),
                dpi=self._dpi,
                pixels=self._pixels,
            ),
        )

    def show_settings(self, snapshot: DocumentSettings) -> None:
        """Show ``snapshot``: every control set to it, then one ``settings_changed`` if that changed anything. The
        sliders show a value to their own step, so the panel's snapshot after may differ from it by that much."""
        self._loading = True
        try:
            self.preset_combo.setCurrentText(snapshot.preset)
            self.tone_combo.setCurrentText(snapshot.tone)
            for name, control in self.controls.items():
                control.set_value(snapshot.values[name])
            if snapshot.preset != CUSTOM:
                self._show_preset(snapshot.preset)
            if snapshot.tone != CUSTOM_TONE:
                self._show_tone(snapshot.tone)
            self.paper_combo.setCurrentText(snapshot.paper)
            self.medium_combo.setCurrentText(snapshot.medium)
            self.legend_combo.setCurrentText(snapshot.legend)
            self._paints = distinct(snapshot.paints)
            self.colors_from_combo.setCurrentText(snapshot.colors_from)
            self.mix_check.setChecked(snapshot.mix)
            self._show_paints()
            for name, check in self._switches.items():
                check.setChecked(getattr(snapshot, name))
            options = snapshot.export
            self.version_combo.setCurrentIndex(self.version_combo.findData(options.version))
            radio = {PNG: self.png_radio, PDF: self.pdf_radio, PALETTE: self.palette_radio}[options.format]
            radio.setChecked(True)
            self.palette_combo.setCurrentIndex(self.palette_combo.findData(options.palette_format))
            self._dpi, self._pixels = options.dpi, options.pixels
            self.resolution_unit.setCurrentIndex(self.resolution_unit.findData(options.unit))
            self._show_resolution()
        finally:
            self._loading = False
        self._save()
        self.paints_changed.emit()
        self.legend_detail_changed.emit(self.legend_combo.currentText())

    # -- What the pipeline is asked for -------------------------------------
    def get_difficulty_params(self) -> difficulty.DifficultyParams:
        return self.snapshot().difficulty_params()

    def get_page_style(self) -> PageStyle:
        # The grays on their sliders, the tone's unless Custom.
        return self.snapshot().page_style()

    def get_legend_detail(self) -> str:
        """What the legend writes beside each swatch (one of ``legend.LEGEND_DETAILS``)."""
        return self.legend_combo.currentText()

    def get_handling(self) -> Handling:
        return self.snapshot().handling()

    def get_paints(self) -> tuple:
        """The painter's paints the page is to take its colors from: none unless Colors from is My paints."""
        return self.snapshot().paints_used()

    def set_paints(self, paints: tuple) -> None:
        """The paints listed (name, RGB each), kept whether or not the page takes its colors from them."""
        self._paints = distinct(paints)
        if self._paints and self.colors_from_combo.currentText() != MY_PAINTS and not self._wants_recipes():
            self.colors_from_combo.setCurrentText(MY_PAINTS)  # listing paints is asking for them
        self._show_paints()
        self._save()
        self.paints_changed.emit()

    def listed_paints(self) -> tuple:
        """The paints listed, (name, RGB) each, whether or not the page takes its colors from them: what the legend's
        recipes are mixed from."""
        return self._paints

    def _wants_recipes(self) -> bool:
        return self.legend_combo.currentText() == NAME_RECIPE

    def edit_paints(self) -> None:
        """Ask for the paints in their dialog."""
        from tessellatum.gui.paints_dialog import PaintsDialog

        dialog = PaintsDialog(self._paints, self)
        if dialog.exec():
            self.set_paints(dialog.paints())

    def _on_colors_from_changed(self) -> None:
        if self.colors_from_combo.currentText() == MY_PAINTS and not self._paints and not self._loading:
            self.edit_paints()
        self._show_paints()
        self._save()

    def _on_mix_changed(self) -> None:
        self._show_paints()
        self._save()

    def _on_switch_changed(self) -> None:
        self._show_enablement()
        self._save()

    def _chosen(self) -> dict:
        """The choices the registry's enablement reads, as shown (see ``Setting.disabled_reason``)."""
        return {
            "medium": self.medium_combo.currentText(), "colors_from": self.colors_from_combo.currentText(),
            "mix_paints": self.mix_check.isChecked(),
            **{name: check.isChecked() for name, check in self._switches.items()},
        }

    def _show_enablement(self) -> None:
        """Every setting that means nothing with the choices as they are turned off, its reason first in its
        tooltip: the darkest black on a screen, the mixing settings without My paints and Mix, the faces', line
        art's and text's settings without their switches."""
        chosen = self._chosen()
        screen = chosen["medium"] == difficulty.SCREEN_MEDIUM
        for name, control in self.controls.items():
            tip = control.setting.tooltip
            if name == "palette_margin_de00" and screen:
                tip += (
                    f" Coloring on a screen, the page keeps its colors at most {difficulty.SCREEN_MARGIN_DE00:g} ΔE00 "
                    "apart, whatever this says."
                )
            reason = control.setting.disabled_reason(chosen)
            control.setEnabled(reason is None)
            control.setToolTip(tip if reason is None else f"{reason} {tip}")
        mix = settings.choice("mix_paints")
        reason = mix.disabled_reason(chosen)
        self.mix_check.setEnabled(reason is None)
        self.mix_check.setToolTip(mix.tooltip if reason is None else f"{reason} {mix.tooltip}")

    def _show_paints(self) -> None:
        # Mixing, and its two thresholds, only mean something when the colors come from the paints.
        self._show_enablement()
        count = len(self._paints)
        self.paints_button.setText(f"Paints ({count})…" if count else "Paints…")
        if self.colors_from_combo.currentText() == MY_PAINTS and not self._paints:
            self.colors_from_combo.setToolTip("No paints listed: the colors come from the picture. " + COLORS_FROM_TOOLTIP)
        else:
            self.colors_from_combo.setToolTip(COLORS_FROM_TOOLTIP)

    def get_output_format(self) -> str:
        """"PNG", "PDF", or "Palette": the legend's colors as a palette file (see ``get_palette_format``)."""
        return self.snapshot().export.format

    def get_palette_format(self) -> str:
        """The palette file's suffix: one of ``swatches.FORMATS``."""
        return self.palette_combo.currentData()

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

    def _on_medium_changed(self) -> None:
        # On a screen there is no darkest black on paper: its slider stands aside. The color margin is capped there,
        # which the margin's and the levels' tooltips say.
        medium = self.medium_combo.currentText()
        screen = medium == difficulty.SCREEN_MEDIUM
        self._show_enablement()  # the darkest black off, the margin's tooltip saying its cap
        self.screen_hint.setVisible(screen)
        for index, name in enumerate(difficulty.PRESETS):
            params = difficulty.for_medium(difficulty.params_for_preset(name), medium)
            self.preset_combo.setItemData(index, difficulty.describe(params), Qt.ToolTipRole)
        self._save()

    def _on_legend_changed(self, detail: str) -> None:
        self._save()
        if not self._loading:
            self.legend_detail_changed.emit(detail)

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
        palette = self.palette_radio.isChecked()
        for widget in (spin, self.resolution_unit, *self.dpi_buttons.values()):
            widget.setEnabled(png)
        self.palette_combo.setEnabled(palette)
        self.version_combo.setEnabled(not palette)
        if palette:
            self.resolution_readout.setText("The palette holds the legend's colors, named as on the legend.")
            return
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
        self.progress_label.setVisible(cancellable)
        if busy:
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")
            self.abort_button.setEnabled(cancellable)
        self.generate_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.export_button.property("hasResult") is True)
        self.open_button.setEnabled(not busy)

    def set_progress(self, percent: int) -> None:
        percent = max(0, min(100, percent))
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{percent}%")

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setProperty("hasResult", enabled)
        self.export_button.setEnabled(enabled)

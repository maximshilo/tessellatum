"""The fields the settings are edited in, built from the registry (``core.settings``).

- ``SettingField``: a slider over a setting, and beside it the value with its unit in a box to type an exact one in.
  Typing takes the locale's decimals ("0,7" in German or French) or the plain ones ("0.7"), with or without the unit.
  A value out of range isn't held to the range: the box says what the range is, in an inline message, and the value
  stays as it was until a valid one is typed (Esc puts the box back). Up and Down step it, Page Up and Page Down ten
  steps, in the box as on the slider.
- A length or an area shows in millimetres or inches (``set_units``); the value is kept in millimetres, so switching
  back and forth never changes it.
- A value changed from its reference (the difficulty's preset, the tone, or the default) shows a dot after it: a
  shape, not a color only, with a tooltip saying what it was ("Changed from Medium: 0.7 mm"). Clicking the dot puts
  it back; the field's context menu offers the reference and the factory default.
- A setting that means nothing with the document's choices is turned off with its reason under it ("Only with Colors
  from: My paints.").
- A drag of the slider is announced as it starts and ends (``edit_started``, ``edit_finished``): one edit, for undo.
- ``SegmentedControl``: two to four exclusive choices side by side, for the paper, the medium, the colors' source and
  the tone; it answers to the same calls as a combo box.
"""

from __future__ import annotations

import math
import re

from PySide6.QtCore import QEvent, QLocale, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import settings
from tessellatum.core.print_size import MM_PER_INCH
from tessellatum.core.settings import Setting
from tessellatum.gui.i18n import term, translate

MM, INCH = "mm", "in"
UNIT_SYSTEMS = (MM, INCH)
# A slider over a setting that moves in equal ratios (``Setting.log``) has this many steps: the region size's, from 2
# to 500 mm², about 5% apart, so that 5 -> 10 mm² is as big a move as 250 -> 500 mm².
LOG_SLIDER_STEPS = 120
# A slider is never narrower than this, however narrow the panel is dragged: below it a handle has no room to move.
MIN_SLIDER_WIDTH = 80
PAGE_STEPS = 10  # how many steps Page Up and Page Down move
MARKER = "●"


# -- Numbers as typed and shown ------------------------------------------------------
def parse_number(text: str, locale: QLocale | None = None) -> float | None:
    """The number ``text`` holds, typed the way ``locale`` writes numbers (the default locale's if None) or the plain
    way ("1.5"), with any unit after it ("0,7 mm"); None if it holds none."""
    text = text.replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ")  # minus, no-break spaces
    # The number at its start, whatever unit follows (even one with digits in it, as \u0394E00 has).
    match = re.match(r"\s*([+-]?[\d.,' ]*\d)", text)
    if match is None:
        return None
    text = match.group(1)
    locale = locale or QLocale()
    for attempt in (text, text.replace(" ", "")):
        for where in (locale, QLocale.c()):
            value, ok = where.toDouble(attempt)
            if ok and math.isfinite(value):
                return value
    return None


def format_number(value: float, decimals: int, locale: QLocale | None = None) -> str:
    return (locale or QLocale()).toString(float(value), "f", decimals)


def slider_steps(setting: Setting) -> int:
    """How many steps a slider over ``setting`` has: its range in steps of its own, or ``LOG_SLIDER_STEPS``."""
    if setting.log:
        return LOG_SLIDER_STEPS
    return round((setting.maximum - setting.minimum) / setting.step)


def slider_value(setting: Setting, position: int) -> float:
    """The value at a position of a slider over ``setting``."""
    position = max(0, min(slider_steps(setting), position))
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


def same_value(setting: Setting, a: float, b: float) -> bool:
    """Whether ``a`` and ``b`` are one value of ``setting``: closer than a hundredth of its step."""
    return abs(float(a) - float(b)) <= setting.step / 100


class Display:
    """How a setting's value is shown and typed: in its own unit, as a lightness for a gray, or in inches for a length
    or an area (the value itself is kept as the setting holds it)."""

    def __init__(self, setting: Setting, units: str = MM):
        self.setting = setting
        self.units = units
        kind = setting.unit_kind
        if units == INCH and kind == settings.LENGTH:
            self.unit, self.factor, self.decimals = "in", 1 / MM_PER_INCH, setting.decimals + 2
        elif units == INCH and kind == settings.AREA:
            self.unit, self.factor, self.decimals = "in²", 1 / MM_PER_INCH**2, setting.decimals + 4
        else:
            self.unit, self.factor = setting.unit, 1.0
            self.decimals = 0 if setting.shown is not None else setting.decimals

    def shown(self, value: float) -> float:
        """``value`` in the unit shown."""
        if self.setting.shown is not None:
            return self.setting.shown(value)
        return value * self.factor

    def held(self, shown: float) -> float:
        """The value of the setting that shows as ``shown``: a gray found by its lightness, a length from inches."""
        setting = self.setting
        if setting.shown is None:
            return shown / self.factor
        # The grays: whole numbers, their lightness rising with them. The nearest by lightness.
        low, high = int(setting.minimum), int(setting.maximum)
        if shown <= setting.shown(low):
            return low
        if shown >= setting.shown(high):
            return high
        while high - low > 1:
            middle = (low + high) // 2
            if setting.shown(middle) < shown:
                low = middle
            else:
                high = middle
        return low if shown - setting.shown(low) <= setting.shown(high) - shown else high

    def text(self, value: float, locale: QLocale | None = None) -> str:
        """``value`` as shown, with its unit: "0.7 mm", "34 L*", "0.028 in"."""
        if self.setting.whole and self.setting.shown is None:
            number, decimals = round(value), 0
        else:
            number, decimals = self.shown(value), self.decimals
        return f"{format_number(number, decimals, locale)} {self.unit}".strip()

    def check(self, shown: float) -> bool:
        """Whether ``shown``, typed in the unit shown, is within the setting's range (as shown, to its decimals)."""
        slack = 0.5 * 10 ** -self.decimals
        low, high = self.shown(self.setting.minimum), self.shown(self.setting.maximum)
        return low - slack <= shown <= high + slack

    def range_text(self, locale: QLocale | None = None) -> str:
        low, high = self.text(self.setting.minimum, locale), self.text(self.setting.maximum, locale)
        if self.unit:  # "0.5 to 6.0 mm" rather than "0.5 mm to 6.0 mm"
            low = low[: -len(self.unit)].strip()
        return f"{low} to {high}"


class _Entry(QLineEdit):
    """The box the exact value is typed in: Up, Down, Page Up and Page Down step it."""

    stepped = Signal(int)
    cancelled = Signal()

    def _takes_escape(self) -> bool:
        return self.isModified() or bool(self.property("invalid"))

    def event(self, event) -> bool:
        # Esc puts back the value being typed, before the window's Cancel can take it as its shortcut.
        if event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape and self._takes_escape():
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        steps = {Qt.Key_Up: 1, Qt.Key_Down: -1, Qt.Key_PageUp: PAGE_STEPS, Qt.Key_PageDown: -PAGE_STEPS}
        if event.key() in steps:
            self.stepped.emit(steps[event.key()])
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._takes_escape():
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SettingField(QWidget):
    """A slider over one setting, the exact value beside it, its changed-value marker and reset, and a line under it
    for an error or the reason it is off.

    It holds the value it was set to exactly, though that may fall between two of its steps (a preset's 125 mm², say),
    until the slider is moved. ``edited`` is emitted only for the user's own edits: a move, a typed value, a step or a
    reset.
    """

    edited = Signal()
    edit_started = Signal()  # a drag of the slider began: the edits until edit_finished are one
    edit_finished = Signal()

    def __init__(self, setting: Setting, parent: QWidget | None = None):
        super().__init__(parent)
        self.setting = setting
        self.display = Display(setting)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, slider_steps(setting))
        self.slider.setPageStep(PAGE_STEPS)
        self.slider.setMinimumWidth(MIN_SLIDER_WIDTH)
        self.slider.setAccessibleName(term(setting.label))
        self.slider.setAccessibleDescription(term(setting.tooltip))
        self.entry = _Entry()
        self.entry.setObjectName("fieldEntry")
        self.entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.entry.setAccessibleName(translate("fields", "{0}, exact value").format(term(setting.label)))
        self.entry.setFixedWidth(_entry_width(self.entry))
        self.marker = QToolButton()
        self.marker.setObjectName("changedMarker")
        self.marker.setText(MARKER)
        self.marker.setAutoRaise(True)
        self.marker.setFocusPolicy(Qt.TabFocus)
        policy = self.marker.sizePolicy()
        policy.setRetainSizeWhenHidden(True)  # the row's width doesn't change as it comes and goes
        self.marker.setSizePolicy(policy)
        self.marker.hide()
        self.message = QLabel()
        self.message.setObjectName("fieldMessage")
        self.message.setWordWrap(True)
        self.message.hide()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.slider, 1)
        row.addWidget(self.entry)
        row.addWidget(self.marker)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.message)
        self.setToolTip(term(setting.tooltip))
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.slider.setContextMenuPolicy(Qt.CustomContextMenu)
        self.slider.customContextMenuRequested.connect(self._show_menu)

        self._value = setting.default
        self._reference: tuple[str, float] = (translate("fields", "the default"), setting.default)
        self._error: str | None = None
        self._reason: str | None = None
        self._say_reason = True
        self.set_value(setting.default)
        self.slider.valueChanged.connect(self._on_moved)
        self.slider.sliderPressed.connect(self.edit_started)
        self.slider.sliderReleased.connect(self.edit_finished)
        self.entry.editingFinished.connect(self._on_typed)
        self.entry.stepped.connect(self.step)
        self.entry.cancelled.connect(self._cancel_typing)
        self.marker.clicked.connect(self.reset_to_reference)

    # -- The value -------------------------------------------------------------------
    def value(self) -> float:
        return self._value

    def text(self) -> str:
        """The value as shown, with its unit."""
        return self.display.text(self._value)

    def set_value(self, value: float) -> None:
        """Show ``value``, held within the setting's range, without counting it as the user's edit."""
        self._value = self.setting.clamp(value)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_position(self.setting, self._value))
        self.slider.blockSignals(False)
        self._clear_error()
        self._show_value()

    def _edit(self, value: float) -> None:
        """The user's edit: ``value`` taken, and announced unless it is the value already."""
        value = self.setting.clamp(value)
        changed = not same_value(self.setting, value, self._value)
        self.set_value(value)
        if changed:
            self.edited.emit()

    def _show_value(self) -> None:
        text = self.text()
        self.entry.setText(text)
        self.entry.setModified(False)
        self.slider.setAccessibleDescription(translate("fields", "{text}. {0}")
                                             .format(term(self.setting.tooltip), text=text))
        self._show_marker()

    def _on_moved(self, position: int) -> None:
        self._value = slider_value(self.setting, position)
        self._clear_error()
        self._show_value()
        self.edited.emit()

    def step(self, steps: int) -> None:
        """``steps`` steps up (or down), as the slider moves."""
        if not self.isEnabled():
            return
        self._edit(slider_value(self.setting, slider_position(self.setting, self._value) + steps))

    # -- Typing -----------------------------------------------------------------------
    def type_text(self, text: str) -> bool:
        """Take ``text`` as typed in the box (as Enter does); whether it was a value the setting takes."""
        self.entry.setText(text)
        self.entry.setModified(True)
        return self._on_typed()

    def _on_typed(self) -> bool:
        if not self.entry.isModified() and self._error is None:
            return True
        number = parse_number(self.entry.text())
        if number is None:
            self._show_error(translate("fields", "Type a number from {0}.").format(self.display.range_text()))
            return False
        if not self.display.check(number):
            self._show_error(translate("fields", "Out of range: from {0}.").format(self.display.range_text()))
            return False
        self._edit(self.display.held(number))  # held within the range to the last bit, and whole if it takes that
        return True

    def _cancel_typing(self) -> None:
        self._clear_error()
        self._show_value()

    def error(self) -> str | None:
        """What is wrong with the value typed, while it is shown; else None."""
        return self._error

    def _show_error(self, text: str) -> None:
        self._error = text
        self.entry.setProperty("invalid", True)
        self.entry.setAccessibleDescription(text)
        self._restyle(self.entry)
        self._show_message()

    def _clear_error(self) -> None:
        if self._error is None:
            return
        self._error = None
        self.entry.setProperty("invalid", False)
        self.entry.setAccessibleDescription("")
        self._restyle(self.entry)
        self._show_message()

    # -- Its reference, the marker and reset ----------------------------------------------
    def set_reference(self, value: float, name: str) -> None:
        """What the value is compared with for its marker, and put back to by the marker: the preset's ``value`` (or
        the tone's, or the default), ``name`` saying which ("Medium", "the default")."""
        self._reference = (name, float(value))
        self._show_marker()

    def reference(self) -> tuple[str, float]:
        return self._reference

    def is_changed(self) -> bool:
        """Whether the value differs from its reference."""
        return not same_value(self.setting, self._value, self._reference[1])

    def _show_marker(self) -> None:
        name, value = self._reference
        changed = self.is_changed()
        self.marker.setVisible(changed)
        shown = self.display.text(value)
        self.marker.setToolTip(translate("fields", "Changed from {name}: {shown}. Click to put it back.")
                               .format(name=name, shown=shown))
        self.marker.setAccessibleName(translate("fields", "Reset {0} to {shown}")
                                      .format(term(self.setting.label), shown=shown))

    def reset_to_reference(self) -> None:
        self._edit(self._reference[1])

    def reset_to_default(self) -> None:
        self._edit(self.setting.default)

    def menu(self) -> QMenu:
        """The field's context menu: back to its reference, or to the factory default."""
        menu = QMenu(self)
        name, value = self._reference
        to_reference = menu.addAction(translate("fields", "Reset to {name} ({0})")
                                      .format(self.display.text(value), name=name), self.reset_to_reference)
        to_reference.setEnabled(self.isEnabled() and self.is_changed())
        if not same_value(self.setting, value, self.setting.default):
            factory = menu.addAction(
                translate("fields", "Reset to the factory default ({0})")
                .format(self.display.text(self.setting.default)), self.reset_to_default,
            )
            factory.setEnabled(self.isEnabled() and not same_value(self.setting, self._value, self.setting.default))
        return menu

    def _show_menu(self, point) -> None:
        source = self.sender() if isinstance(self.sender(), QWidget) else self
        self.menu().exec(source.mapToGlobal(point))

    # -- Units, and the reason it is off ---------------------------------------------------
    def set_units(self, units: str) -> None:
        """Show a length or an area in ``units`` (``MM`` or ``INCH``); the value itself doesn't change."""
        if units not in UNIT_SYSTEMS:
            raise ValueError(f"no units {units!r}")
        self.display = Display(self.setting, units)
        self._clear_error()
        self._show_value()

    def set_disabled_reason(self, reason: str | None, say: bool = True) -> None:
        """Turn the field off with ``reason``, said under it unless ``say`` is False (a field above says it for both),
        or on again if None."""
        self._reason = reason
        self._say_reason = say
        self.setEnabled(reason is None)
        self._show_message()

    def disabled_reason(self) -> str | None:
        return self._reason

    def _show_message(self) -> None:
        if self._error:
            text, role = self._error, "error"
        elif self._reason and self._say_reason:
            text, role = term(self._reason), "reason"
        else:
            text, role = "", ""
        self.message.setText(text or "")
        self.message.setProperty("role", role)
        self._restyle(self.message)
        self.message.setVisible(bool(text))

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.EnabledChange:
            self._show_marker()


_ENTRY_WIDTHS: dict[str, int] = {}


def _entry_width(entry: QLineEdit) -> int:
    """As wide as the widest value any setting shows, in either units: the boxes line up down a page."""
    key = entry.font().key()
    if key not in _ENTRY_WIDTHS:
        metrics = entry.fontMetrics()
        texts = [
            Display(setting, units).text(value)
            for setting in settings.SETTINGS for units in UNIT_SYSTEMS
            for value in (setting.minimum, setting.maximum, setting.default)
        ]
        _ENTRY_WIDTHS[key] = max(metrics.horizontalAdvance(text) for text in texts) + 14
    return _ENTRY_WIDTHS[key]


VALUE_ROLE = Qt.UserRole + 64  # a ChoiceCombo item's value


class ChoiceCombo(QComboBox):
    """A combo box of fixed choices, each shown in the interface's language and known to the code by its value (its
    English name): ``currentText``, ``setCurrentText``, ``findText`` and ``itemText`` take and give the value, and
    ``valueChanged`` says it, where Qt's ``currentTextChanged`` says the words shown. So code that compares, keeps or
    restores a choice by name works whatever the language. ``context`` is the choices' translation context."""

    valueChanged = Signal(str)

    def __init__(self, context: str = "settings", parent: QWidget | None = None):
        super().__init__(parent)
        self._context = context
        self.currentIndexChanged.connect(lambda _index: self.valueChanged.emit(self.currentText()))

    def addItem(self, text: str, userData=None) -> None:  # noqa: N803 - Qt's name
        super().addItem(translate(self._context, text), userData)
        self.setItemData(self.count() - 1, text, VALUE_ROLE)

    def addItems(self, texts) -> None:
        for text in texts:
            self.addItem(text)

    def itemText(self, index: int) -> str:
        value = self.itemData(index, VALUE_ROLE)
        return value if value is not None else super().itemText(index)

    def currentText(self) -> str:
        index = self.currentIndex()
        return self.itemText(index) if index >= 0 else ""

    def findText(self, text: str, flags=None) -> int:
        return next((i for i in range(self.count()) if self.itemText(i) == text), -1)

    def setCurrentText(self, text: str) -> None:
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)


class Reflow(QWidget):
    """Widgets side by side while the width given holds them all at their size, one under another when it doesn't: in
    a longer language or at a larger scale, a row keeps its words whole rather than cutting them. Its minimum width is
    its widest widget's, so a narrow sidebar gives it less room than the row needs and it stacks; ``side_by_side``
    measures what a panel needs to hold every row in one line, the width it starts at (see ``preferred_width``)."""

    measuring_rows = False  # while True, every Reflow's minimum is its row's: see ``side_by_side``

    def __init__(self, parent: QWidget | None = None, spacing: int | None = None, stacked_spacing: int | None = None):
        super().__init__(parent)
        self._layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSizeConstraint(QBoxLayout.SetNoConstraint)  # its minimum is minimumSizeHint's, even alone
        if spacing is not None:
            self._layout.setSpacing(spacing)
        self._row_spacing = self._layout.spacing()
        self._stacked_spacing = self._row_spacing if stacked_spacing is None else stacked_spacing

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
        self._layout.addWidget(widget, stretch)

    def _widgets(self) -> list[QWidget]:
        items = (self._layout.itemAt(i).widget() for i in range(self._layout.count()))
        return [w for w in items if w is not None and not w.isHidden()]

    def row_width(self) -> int:
        """The width the widgets need side by side, none cut (a row within, side by side too)."""
        widgets = self._widgets()
        need = sum(w.row_width() if isinstance(w, Reflow) else w.minimumSizeHint().width() for w in widgets)
        return need + max(0, self._row_spacing) * max(0, len(widgets) - 1)

    def is_stacked(self) -> bool:
        return self._layout.direction() == QBoxLayout.TopToBottom

    def minimumSizeHint(self) -> QSize:
        if Reflow.measuring_rows:
            return QSize(self.row_width(), super().minimumSizeHint().height())
        widest = max((w.minimumSizeHint().width() for w in self._widgets()), default=0)
        return QSize(widest, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow(event.size().width())

    def _reflow(self, width: int) -> None:
        stacked = width < self.row_width()
        if stacked == self.is_stacked():
            return
        self._layout.setDirection(QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight)
        self._layout.setSpacing(self._stacked_spacing if stacked else self._row_spacing)
        self._stacked(stacked)
        self.updateGeometry()

    def _stacked(self, stacked: bool) -> None:
        """What else changes when the widgets stack, or go back side by side."""


def preferred_width(widget: QWidget) -> int:
    """The minimum width ``widget`` would have with every ``Reflow`` in it holding its widgets side by side."""
    rows = widget.findChildren(Reflow)

    def ask_again() -> None:
        # Every layout from each row up to ``widget`` keeps the sizes it was told: each is told again.
        for row in rows:
            w = row
            while w is not None and w is not widget:
                w.updateGeometry()
                if w.layout() is not None:
                    w.layout().invalidate()
                w = w.parentWidget()
        if widget.layout() is not None:
            widget.layout().invalidate()

    Reflow.measuring_rows = True
    try:
        ask_again()
        return widget.minimumSizeHint().width()
    finally:
        Reflow.measuring_rows = False
        ask_again()


class SegmentedControl(Reflow):
    """Two to four exclusive choices side by side, one of them down (one under another where the width doesn't hold
    them: see ``Reflow``). It answers to a combo box's calls (``addItems``, ``currentText``, ``setCurrentText``,
    ``currentTextChanged`` …), so it takes a combo's place as it is; Left and Right move the choice."""

    currentTextChanged = Signal(str)
    currentIndexChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None, context: str = "settings"):
        super().__init__(parent, spacing=0, stacked_spacing=2)
        self._context = context  # the choices' translation context (see ``ChoiceCombo``)
        self.setObjectName("segmented")
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._group.idToggled.connect(self._on_toggled)

    def addItems(self, texts) -> None:
        for text in texts:
            self.addItem(text)

    def addItem(self, text: str) -> None:
        """A choice: ``text`` is its value, shown in the interface's language."""
        shown = translate(self._context, text)
        button = QPushButton(shown)
        button.setObjectName("segment")
        button.setCheckable(True)
        button.setAccessibleName(shown)
        button.setProperty("value", text)
        button.installEventFilter(self)
        index = len(self._buttons)
        button.setProperty("position", "only")
        self._buttons.append(button)
        self._group.addButton(button, index)
        self._layout.addWidget(button, 1)
        self._place()
        if index == 0:
            button.setChecked(True)
        self._update_focus()

    def _place(self) -> None:
        # For the stylesheet's rounded ends: joined in a row, each whole when stacked.
        last = len(self._buttons) - 1
        for i, button in enumerate(self._buttons):
            position = ("only" if last == 0 or self.is_stacked() else "first" if i == 0 else "last" if i == last
                        else "middle")
            if button.property("position") != position:
                button.setProperty("position", position)
                button.style().unpolish(button)
                button.style().polish(button)

    def _stacked(self, stacked: bool) -> None:
        self._place()

    def count(self) -> int:
        return len(self._buttons)

    def itemText(self, index: int) -> str:
        """The choice's value (as ``ChoiceCombo``'s: the words shown may be a translation)."""
        return self._buttons[index].property("value")

    def findText(self, text: str) -> int:
        return next((i for i in range(len(self._buttons)) if self.itemText(i) == text), -1)

    def currentIndex(self) -> int:
        return self._group.checkedId()

    def currentText(self) -> str:
        index = self.currentIndex()
        return self.itemText(index) if index >= 0 else ""

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def setCurrentText(self, text: str) -> None:
        self.setCurrentIndex(self.findText(text))

    def setItemData(self, index: int, value, role=Qt.UserRole) -> None:
        if role == Qt.ToolTipRole:
            self._buttons[index].setToolTip(value)
        else:
            self._buttons[index].setProperty("itemData", value)

    def itemData(self, index: int, role=Qt.UserRole):
        if role == Qt.ToolTipRole:
            return self._buttons[index].toolTip()
        return self._buttons[index].property("itemData")

    def buttons(self) -> list[QPushButton]:
        return list(self._buttons)

    def _on_toggled(self, index: int, checked: bool) -> None:
        if not checked:
            return
        self._update_focus()
        self.currentIndexChanged.emit(index)
        self.currentTextChanged.emit(self.itemText(index))

    def _update_focus(self) -> None:
        # One stop in the Tab order, on the choice made, as a group of radio buttons has.
        for button in self._buttons:
            button.setFocusPolicy(Qt.StrongFocus if button.isChecked() else Qt.ClickFocus)
        index = self.currentIndex()
        if index >= 0:
            had_focus = any(b.hasFocus() for b in self._buttons)
            self.setFocusProxy(self._buttons[index])
            if had_focus:
                self._buttons[index].setFocus()

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Left, Qt.Key_Right):
            step = -1 if event.key() == Qt.Key_Left else 1
            if self.layoutDirection() == Qt.RightToLeft:
                step = -step
            self.setCurrentIndex(max(0, min(len(self._buttons) - 1, self.currentIndex() + step)))
            return True
        return False

    def setAccessibleName(self, name: str) -> None:
        super().setAccessibleName(name)
        for button in self._buttons:
            button.setAccessibleDescription(name)

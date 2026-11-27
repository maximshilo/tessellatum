"""The palette inspector, on the right: the page's colors, as its legend numbers them, to look at and pick.

- Each color: its number, swatch and name; its hex and RGB; its recipe, if it is mixed from the paints; how much of
  the page it paints (a share of the page and square millimeters on paper) and in how many regions. The coverage is
  measured off the GUI thread once the page is made (see ``core.measures``), and fills in when it is ready.
- A **list** or a **grid**, sorted by number, coverage, lightness, hue or name. Sorting changes only the order shown:
  never a number, a region's color or the export's numbering.
- **Find** by number ("12") or by a word of the name.
- **Copy** a color's hex, RGB or name: its context menu, the buttons under the list, or Ctrl+C (the hex).
- Picking a color highlights its regions on the canvas (in the annotation layer, never in an export); **Isolate** dims
  the rest. The selection is outlined, not told by color alone, and each color's accessible name says all of it:
  "Color 12, dusty rose, #C0737A, 4.2% of the page, 37 regions".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QT_TRANSLATE_NOOP, QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core.color import bgr_to_lab
from tessellatum.core.measures import ColorCoverage
from tessellatum.gui.fields import ChoiceCombo
from tessellatum.gui.i18n import number, translate

NUMBER, COVERAGE, LIGHTNESS, HUE, NAME = (  # the sorts, shown translated by the combo box (``ChoiceCombo``)
    QT_TRANSLATE_NOOP("palette_inspector", "Number"), QT_TRANSLATE_NOOP("palette_inspector", "Coverage"),
    QT_TRANSLATE_NOOP("palette_inspector", "Lightness"), QT_TRANSLATE_NOOP("palette_inspector", "Hue"),
    QT_TRANSLATE_NOOP("palette_inspector", "Name"),
)
SORTS = (NUMBER, COVERAGE, LIGHTNESS, HUE, NAME)
LIST, GRID = "list", "grid"
GRAY_CHROMA = 6.0  # a color less colorful than this (CIELAB chroma) has no hue to sort by: the grays go first
INDEX_ROLE = Qt.UserRole + 1  # an item's palette index: its number less one


@dataclass(frozen=True)
class PaletteColor:
    """A color of the page's legend, as the inspector shows it."""

    index: int  # into the palette; numbered index + 1
    rgb: tuple[int, int, int]
    name: str
    recipe: tuple[str, ...]  # how it is mixed from the paints, a line each; empty if it isn't
    lab: tuple[float, float, float]

    @property
    def number(self) -> int:
        return self.index + 1

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    @property
    def rgb_text(self) -> str:
        return "{}, {}, {}".format(*self.rgb)

    def hue_key(self) -> tuple:
        lightness, a, b = self.lab
        chroma = math.hypot(a, b)
        if chroma < GRAY_CHROMA:
            return (0, -lightness)  # the grays, light to dark
        return (1, math.degrees(math.atan2(b, a)) % 360.0)


def palette_colors(page) -> list[PaletteColor]:
    """``page``'s legend colors (a ``GeneratedPage``), in legend order."""
    rgbs = [tuple(int(v) for v in rgb) for rgb in page.palette_rgb]
    if not rgbs:
        return []
    labs = bgr_to_lab(np.asarray([rgb[::-1] for rgb in rgbs], dtype=np.uint8))
    names = list(page.color_names or [])
    mixes = list(page.mixes or [])
    return [
        PaletteColor(
            index=index, rgb=rgb, name=names[index] if index < len(names) else "",
            recipe=tuple(mixes[index]) if index < len(mixes) and mixes[index] else (),
            lab=tuple(float(v) for v in labs[index]),
        )
        for index, rgb in enumerate(rgbs)
    ]


def sort_colors(colors: list[PaletteColor], by: str, coverage: ColorCoverage | None = None) -> list[PaletteColor]:
    """``colors`` in the order ``by`` (one of ``SORTS``) shows them: by number; most coverage first (by number until
    it is measured); light to dark; by hue, the grays first; by name. Ties keep the numbers' order."""
    if by == COVERAGE and coverage is not None:
        return sorted(colors, key=lambda c: (-int(coverage.pixels[c.index]), c.index))
    if by == LIGHTNESS:
        return sorted(colors, key=lambda c: (-c.lab[0], c.index))
    if by == HUE:
        return sorted(colors, key=lambda c: (c.hue_key(), c.index))
    if by == NAME:
        return sorted(colors, key=lambda c: (c.name.casefold(), c.index))
    return sorted(colors, key=lambda c: c.index)


def matches(color: PaletteColor, text: str) -> bool:
    """Whether ``color`` is what a search for ``text`` asks for: its number, or a word of its name it starts with."""
    text = text.strip().casefold().lstrip("#")
    if not text:
        return True
    if text.isdigit():
        return color.number == int(text)
    return any(word.startswith(text) for word in color.name.casefold().split()) or text in color.name.casefold()


def describe(color: PaletteColor, coverage: ColorCoverage | None, px_per_mm: float | None) -> str:
    """The color in words, for its accessible name and tooltip."""
    parts = [translate("palette_inspector", "Color {0}").format(color.number)]
    if color.name:
        parts.append(color.name)
    parts.append(color.hex)
    if coverage is not None and color.index < len(coverage.pixels):
        parts.append(translate("palette_inspector", "{0:.1f}% of the page").format(coverage.share(color.index) * 100))
        regions = int(coverage.regions[color.index])
        parts.append(f"{regions} region" + ("" if regions == 1 else "s"))
    return ", ".join(parts)


def coverage_text(color: PaletteColor, coverage: ColorCoverage | None, px_per_mm: float | None) -> str:
    if coverage is None or color.index >= len(coverage.pixels):
        return translate("palette_inspector", "Coverage: measuring…")
    regions = int(coverage.regions[color.index])
    text = translate("palette_inspector", "{0:.1f}% of the page").format(coverage.share(color.index) * 100)
    if px_per_mm:
        text += translate("palette_inspector", " · {0:,.0f} mm²").format(coverage.area_mm2(color.index, px_per_mm))
    count = (translate("palette_inspector", "1 region") if regions == 1
             else translate("palette_inspector", "{regions:,} regions").format(regions=regions))
    return f"{text} · {count}"


def swatch(color: QColor, size: int, outline: QColor | None = None) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(color)
    if outline is not None:
        painter = QPainter(pixmap)
        painter.setPen(QPen(outline, 1))
        painter.drawRect(0, 0, size - 1, size - 1)
        painter.end()
    return QIcon(pixmap)


class PaletteInspector(QWidget):
    """The page's colors, to look at, sort, find, copy and pick (see the module's docstring)."""

    SWATCH = 18
    GRID_SWATCH = 34
    color_selected = Signal(object)  # the palette index picked, or None
    isolate_changed = Signal(bool)
    regions_requested = Signal(int)  # Enter on a color: list its regions (the region inspector's)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("inspector")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._colors: list[PaletteColor] = []
        self._coverage: ColorCoverage | None = None
        self._px_per_mm: float | None = None
        self._regions = 0
        self.title = QLabel(translate("palette_inspector", "Palette"))
        self.title.setObjectName("panelTitle")
        self.summary = QLabel()
        self.summary.setProperty("role", "secondary")
        self.summary.setWordWrap(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText(translate("palette_inspector", "Find a color: number or name"))
        self.search.setAccessibleName(translate("palette_inspector", "Find a color"))
        self.search.setClearButtonEnabled(True)
        self.sort_combo = ChoiceCombo("palette_inspector")
        self.sort_combo.addItems(SORTS)
        self.sort_combo.setAccessibleName(translate("palette_inspector", "Sort colors by"))
        self.sort_combo.setToolTip(translate("palette_inspector", "The order the colors are shown in. Their "
                                                                  "numbers never change."))
        self.grid_button = QPushButton(translate("palette_inspector", "Grid"))
        self.grid_button.setCheckable(True)
        self.grid_button.setAccessibleName(translate("palette_inspector", "Show the colors as a grid"))
        self.grid_button.setToolTip(translate("palette_inspector", "Show the colors as a grid of swatches, or as "
                                                                   "a list"))
        self.list = QListWidget()
        self.list.setObjectName("paletteList")
        self.list.setIconSize(QSize(self.SWATCH, self.SWATCH))
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setFocusPolicy(Qt.StrongFocus)
        self.list.setAccessibleName(translate("palette_inspector", "Palette"))
        self.list.setUniformItemSizes(True)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.installEventFilter(self)
        # The color picked, in full.
        self.detail = QWidget()
        self.detail.setObjectName("paletteDetail")
        self.detail_swatch = QLabel()
        self.detail_swatch.setFixedSize(40, 40)
        self.detail_title = QLabel()
        self.detail_title.setObjectName("detailTitle")
        self.detail_title.setWordWrap(True)
        self.detail_values = QLabel()
        self.detail_values.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_values.setWordWrap(True)
        self.detail_recipe = QLabel()
        self.detail_recipe.setWordWrap(True)
        self.detail_recipe.setProperty("role", "secondary")
        self.detail_coverage = QLabel()
        self.detail_coverage.setWordWrap(True)
        # Copy: the hex at a click, the RGB or the name from its menu.
        self.copy_button = QToolButton()
        self.copy_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.copy_button.setAccessibleName(translate("palette_inspector", "Copy the color"))
        copy_menu = QMenu(self.copy_button)
        self.copy_actions = {}
        for label, what in (
            (translate("palette_inspector", "Copy Hex"), "hex"),
            (translate("palette_inspector", "Copy RGB"), "rgb"),
            (translate("palette_inspector", "Copy Name"), "name"),
        ):
            action = copy_menu.addAction(label)
            action.triggered.connect(lambda _checked=False, w=what: self.copy(w))
            self.copy_actions[what] = action
        self.copy_button.setMenu(copy_menu)
        self.copy_button.setDefaultAction(self.copy_actions["hex"])
        self.copy_button.setToolTip(translate("palette_inspector", "Copy the hex (Ctrl+C); its menu copies the "
                                                                   "RGB or the name"))
        self.isolate_check = QCheckBox(translate("palette_inspector", "Isolate"))
        self.isolate_check.setToolTip(translate("palette_inspector", "On the canvas, dim every region but this "
                                                                     "color's"))
        self.isolate_check.setAccessibleDescription(translate("palette_inspector", "Dim the other colors on the "
                                                                                   "canvas"))
        top = QHBoxLayout()
        top.addWidget(self.detail_swatch)
        top.addWidget(self.detail_title, 1)
        copies = QHBoxLayout()
        copies.addWidget(self.copy_button)
        copies.addStretch(1)
        detail_layout = QVBoxLayout(self.detail)
        detail_layout.setContentsMargins(10, 8, 10, 8)
        detail_layout.addLayout(top)
        for widget in (self.detail_values, self.detail_recipe, self.detail_coverage):
            detail_layout.addWidget(widget)
        copies.insertWidget(0, self.isolate_check)
        copies.insertSpacing(1, 8)
        detail_layout.addLayout(copies)
        self.detail.hide()
        tools = QHBoxLayout()
        tools.setContentsMargins(10, 4, 10, 4)
        tools.addWidget(self.sort_combo, 1)
        tools.addWidget(self.grid_button)
        search_row = QHBoxLayout()
        search_row.setContentsMargins(10, 4, 10, 0)
        search_row.addWidget(self.search)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title)
        layout.addWidget(self.summary)
        self.tools = QWidget()
        tools_layout = QVBoxLayout(self.tools)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(0)
        tools_layout.addLayout(search_row)
        tools_layout.addLayout(tools)
        layout.addWidget(self.tools)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.detail)
        self._sections = layout.count()
        layout.addStretch(0)  # takes the room while the list is hidden, so the title stays at the top
        self.search.textChanged.connect(self._filter)
        self.sort_combo.currentTextChanged.connect(lambda _text: self._fill())
        self.grid_button.toggled.connect(self.set_mode)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.isolate_check.toggled.connect(self.isolate_changed)
        self.show_page(None)

    # -- What it shows --------------------------------------------------------------------------------
    def show_page(self, page) -> None:
        """List ``page``'s colors (a ``GeneratedPage``), or say there are none yet. The coverage waits for
        ``set_coverage``; nothing is picked."""
        self._colors = palette_colors(page) if page is not None else []
        self._coverage = None
        self._regions = page.num_regions if page is not None else 0
        blocked = self.isolate_check.blockSignals(True)
        self.isolate_check.setChecked(False)
        self.isolate_check.blockSignals(blocked)
        self._fill(keep=None)
        if page is None:
            self.summary.setText(translate("palette_inspector", "Generate a preview to see its colors here."))
            self.list.setVisible(False)
            self.tools.setVisible(False)
            return
        self.summary.setText(translate("palette_inspector", "{0} colors, {1:,} regions")
                             .format(len(self._colors), page.num_regions))
        self.list.setVisible(True)
        self.tools.setVisible(True)

    def add_section(self, widget: QWidget) -> None:
        """Show ``widget`` under the color picked (the region inspector)."""
        self.layout().insertWidget(self._sections, widget)
        self._sections += 1

    def set_coverage(self, coverage: ColorCoverage | None, px_per_mm: float | None) -> None:
        """How much of the page each color paints, measured: its texts fill in, and Coverage sorts by it."""
        self._coverage, self._px_per_mm = coverage, px_per_mm
        self._fill()

    def coverage(self) -> ColorCoverage | None:
        return self._coverage

    def color_count(self) -> int:
        return self.list.count()

    def colors(self) -> list[PaletteColor]:
        return list(self._colors)

    def shown_order(self) -> list[int]:
        """The palette indices in the order the list shows them, hidden ones too."""
        return [self.list.item(row).data(INDEX_ROLE) for row in range(self.list.count())]

    def visible_numbers(self) -> list[int]:
        return [self.list.item(row).data(INDEX_ROLE) + 1 for row in range(self.list.count())
                if not self.list.item(row).isHidden()]

    def selected(self) -> int | None:
        """The palette index picked, or None."""
        items = self.list.selectedItems()
        return items[0].data(INDEX_ROLE) if items else None

    def select(self, index: int | None) -> None:
        """Pick the color of palette index ``index`` (None: none), as a click would, but showing it however sorted
        and found: a search that hides it is cleared."""
        if index is None:
            self.list.clearSelection()
            self.list.setCurrentItem(None)
            return
        item = self._item(index)
        if item is None:
            return
        if item.isHidden():
            self.search.clear()
        self.list.setCurrentItem(item)
        item.setSelected(True)
        self.list.scrollToItem(item)

    def isolated(self) -> bool:
        return self.isolate_check.isChecked() and self.selected() is not None

    def mode(self) -> str:
        return GRID if self.list.viewMode() == QListView.IconMode else LIST

    def set_mode(self, mode) -> None:
        """Show the colors as a ``GRID`` of swatches or a ``LIST``; True and False stand for them too."""
        grid = mode is True or mode == GRID
        blocked = self.grid_button.blockSignals(True)
        self.grid_button.setChecked(grid)
        self.grid_button.blockSignals(blocked)
        if grid:
            self.list.setViewMode(QListView.IconMode)
            self.list.setIconSize(QSize(self.GRID_SWATCH, self.GRID_SWATCH))
            self.list.setGridSize(QSize(self.GRID_SWATCH + 18, self.GRID_SWATCH + 22))
            self.list.setResizeMode(QListView.Adjust)
            self.list.setMovement(QListView.Static)
            self.list.setWrapping(True)
        else:
            self.list.setViewMode(QListView.ListMode)
            self.list.setIconSize(QSize(self.SWATCH, self.SWATCH))
            self.list.setGridSize(QSize())
            self.list.setWrapping(False)
        self._fill()

    def set_sort(self, by: str) -> None:
        self.sort_combo.setCurrentText(by)

    # -- Copying ----------------------------------------------------------------------------------------
    def copy(self, what: str, index: int | None = None) -> str | None:
        """Copy the color's ``what`` ("hex", "rgb" or "name") to the clipboard: the color of ``index``, or the one
        picked. Returns what was copied."""
        index = self.selected() if index is None else index
        if index is None or not 0 <= index < len(self._colors):
            return None
        color = self._colors[index]
        text = {"hex": color.hex, "rgb": color.rgb_text, "name": color.name or color.hex}[what]
        QGuiApplication.clipboard().setText(text)
        return text

    def eventFilter(self, watched, event) -> bool:
        if watched is self.list and event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape \
                and self.selected() is not None:
            event.accept()  # Esc lets go of the color, before the window's Cancel can take it
            return True
        if watched is self.list and event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Copy):
                self.copy("hex")
                return True
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.selected() is not None:
                self.regions_requested.emit(self.selected())
                return True
            if event.key() == Qt.Key_Escape and self.selected() is not None:
                self.select(None)
                return True
        return super().eventFilter(watched, event)

    def _context_menu(self, position) -> None:
        item = self.list.itemAt(position)
        if item is None:
            return
        index = item.data(INDEX_ROLE)
        menu = QMenu(self.list)
        for label, what in (
            (translate("palette_inspector", "Copy Hex"), "hex"),
            (translate("palette_inspector", "Copy RGB"), "rgb"),
            (translate("palette_inspector", "Copy Name"), "name"),
        ):
            menu.addAction(label, lambda w=what: self.copy(w, index))
        menu.exec(self.list.viewport().mapToGlobal(position))

    # -- The list ---------------------------------------------------------------------------------------
    def _item(self, index: int) -> QListWidgetItem | None:
        for row in range(self.list.count()):
            if self.list.item(row).data(INDEX_ROLE) == index:
                return self.list.item(row)
        return None

    _KEEP = object()

    def _fill(self, keep=_KEEP) -> None:
        """Lay the list out again in the order asked for, keeping the color picked (or ``keep``)."""
        picked = self.selected() if keep is self._KEEP else keep
        grid = self.mode() == GRID
        blocked = self.list.blockSignals(True)
        self.list.clear()
        size = self.GRID_SWATCH if grid else self.SWATCH
        outline = QColor(0, 0, 0, 60)
        for color in sort_colors(self._colors, self.sort_combo.currentText(), self._coverage):
            share = ""
            if self._coverage is not None and color.index < len(self._coverage.pixels):
                share = f"   {number(self._coverage.share(color.index) * 100, 1)}%"
            text = str(color.number) if grid else f"{color.number}   {color.name}".rstrip() + share
            item = QListWidgetItem(swatch(QColor(*color.rgb), size, outline), text)
            item.setData(INDEX_ROLE, color.index)
            words = describe(color, self._coverage, self._px_per_mm)
            tip = f"{color.number} {color.name}: {color.hex}, RGB {color.rgb_text}".replace(" :", ":")
            if color.recipe:
                tip += translate("palette_inspector", "\nMixed: ") + " + ".join(color.recipe)
            if self._coverage is not None:
                tip += "\n" + coverage_text(color, self._coverage, self._px_per_mm)
            item.setToolTip(tip)
            item.setData(Qt.AccessibleTextRole, words)
            if grid:
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.list.addItem(item)
        self.list.blockSignals(blocked)
        self._filter(self.search.text())
        if picked is not None and self._item(picked) is not None:
            self.select(picked)
        else:
            self._show_detail(None)
            if keep is not self._KEEP or picked is not None:
                self.color_selected.emit(None)

    def _filter(self, text: str) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(not matches(self._colors[item.data(INDEX_ROLE)], text))

    def _on_current(self, current, _previous) -> None:
        if current is not None and not current.isSelected():
            current.setSelected(True)

    def _on_selection(self) -> None:
        index = self.selected()
        self._show_detail(index)
        self.color_selected.emit(index)

    def _show_detail(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self._colors):
            self.detail.hide()
            return
        color = self._colors[index]
        self.detail_swatch.setPixmap(swatch(QColor(*color.rgb), 40, QColor(0, 0, 0, 60)).pixmap(40, 40))
        self.detail_title.setText(f"{color.number}  {color.name}".rstrip())
        self.detail_values.setText(translate("palette_inspector", "{0} · RGB {1}").format(color.hex, color.rgb_text))
        self.detail_recipe.setText(
            translate("palette_inspector", "Mixed: ") + " + ".join(color.recipe) if color.recipe else ""
        )
        self.detail_recipe.setVisible(bool(color.recipe))
        self.detail_coverage.setText(coverage_text(color, self._coverage, self._px_per_mm))
        self.detail.setAccessibleName(describe(color, self._coverage, self._px_per_mm))
        self.detail.show()

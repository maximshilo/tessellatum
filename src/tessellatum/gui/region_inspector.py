"""The region inspector, under the palette: the region picked on the canvas, measured on paper, and its number.

- Click a region on the canvas, in any view, to pick it: its color is picked in the palette too, and the region is
  outlined on the canvas. Picking another color in the palette lets the region go.
- It shows the region's number on the page and its color; its area in mm² and its bounding size in mm; the widest
  brush that fits in it; and how its number is written: inside it, outside it on a leader line, or cramped where
  lines may cross it, at what size in points, and on how many other sides of the region.
- From the keyboard: Enter on a color in the palette lists its regions here, largest first, and picks the first; Up
  and Down go through them, each centered on the canvas; Esc goes back to the palette.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from tessellatum.gui.i18n import translate

MM_PER_POINT = 25.4 / 72.0


@dataclass(frozen=True)
class RegionInfo:
    """A region as the inspector shows it, measured on paper."""

    region_id: int
    color_number: int  # as the legend numbers it; past the legend for a color of its own
    color_name: str
    area_mm2: float
    width_mm: float
    height_mm: float
    widest_brush_mm: float
    placement: str  # "inside", "on a leader", "cramped" or "none"
    number_pt: float | None  # the number's size in points, or None if it has none
    sides: int  # the other sides of the region its number is also written on

    def lines(self) -> list[str]:
        color = translate("region_inspector", "Color {0}").format(self.color_number) + (
            f", {self.color_name}" if self.color_name else ""
        )
        lines = [
            color,
            translate("region_inspector", "Area: {0:,.1f} mm²").format(self.area_mm2),
            translate("region_inspector", "Size: {0:.1f} × {1:.1f} mm").format(self.width_mm, self.height_mm),
            translate("region_inspector", "Widest brush: {0:.1f} mm").format(self.widest_brush_mm),
        ]
        if self.placement == "none":
            lines.append(translate("region_inspector", "Number: none (too small to write it in)"))
        else:
            where = {"inside": translate("region_inspector", "inside"),
                     "on a leader": translate("region_inspector", "outside, on a leader line"),
                     "cramped": translate("region_inspector", "cramped: lines may run through it")}[self.placement]
            lines.append(
                translate("region_inspector", "Number: {where}, {0:.1f} pt").format(self.number_pt, where=where)
            )
        if self.sides:
            lines.append(translate("region_inspector", "Also written on 1 other side") if self.sides == 1
                         else translate("region_inspector", "Also written on {0} other sides").format(self.sides))
        return lines

    def describe(self) -> str:
        return translate("region_inspector", "Region {0}: ").format(self.region_id) + "; ".join(self.lines())


def region_info(model, region_id: int, stats, px_per_mm: float, color_names, widest_px: float) -> RegionInfo:
    """``region_id`` of ``model`` (a ``PageModel``), measured on paper of ``px_per_mm``, its number found among the
    page's labels. ``stats`` is the page's ``measures.RegionStats``; ``widest_px`` its widest brush in pixels."""
    color = int(model.region_color[region_id])
    x0, y0, x1, y1 = stats.box(region_id)
    own = [label for label in model.labels if label.region_id == region_id]
    first = next((label for label in own if label.side == 0), None)
    if first is None:
        placement, points = "none", None
    else:
        placement = "cramped" if first.cramped else "on a leader" if first.leader is not None else "inside"
        points = first.font_size / px_per_mm / MM_PER_POINT
    names = list(color_names or [])
    return RegionInfo(
        region_id=region_id, color_number=color + 1, color_name=names[color] if color < len(names) else "",
        area_mm2=float(stats.areas[region_id]) / px_per_mm ** 2,
        width_mm=(x1 - x0) / px_per_mm, height_mm=(y1 - y0) / px_per_mm,
        widest_brush_mm=widest_px / px_per_mm, placement=placement, number_pt=points,
        sides=sum(1 for label in own if label.side > 0),
    )


class RegionInspector(QWidget):
    """The region picked, and the regions of a color to go through (see the module's docstring)."""

    region_chosen = Signal(int)  # a region picked in the list of a color's regions
    back = Signal()  # Esc in the list: back to the palette

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("regionInspector")
        self.title = QLabel(translate("region_inspector", "Region"))
        self.title.setObjectName("detailTitle")
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.list_title = QLabel()
        self.list_title.setProperty("role", "secondary")
        self.list = QListWidget()
        self.list.setObjectName("paletteList")
        self.list.setAccessibleName(translate("region_inspector", "Regions of the color"))
        self.list.setMaximumHeight(140)
        self.list.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        for widget in (self.title, self.details, self.list_title, self.list):
            layout.addWidget(widget)
        self.list.currentItemChanged.connect(self._on_current)
        self._info: RegionInfo | None = None
        self.show_region(None)
        self.show_regions(None, [])

    def info(self) -> RegionInfo | None:
        return self._info

    def show_region(self, info: RegionInfo | None) -> None:
        """Show ``info``, or hide (with no list either) when there is none."""
        self._info = info
        if info is None:
            self.details.clear()
            self.setVisible(self.list.count() > 0)
            return
        self.title.setText(translate("region_inspector", "Region {0}").format(info.region_id))
        self.details.setText("\n".join(info.lines()))
        self.details.setAccessibleName(info.describe())
        self.setVisible(True)
        item = self._item(info.region_id)
        if item is not None and self.list.currentItem() is not item:
            blocked = self.list.blockSignals(True)
            self.list.setCurrentItem(item)
            self.list.blockSignals(blocked)

    def show_regions(self, color_number: int | None, regions: list[tuple[int, float]]) -> None:
        """List color ``color_number``'s regions, as (id, mm²), largest first; None: none."""
        blocked = self.list.blockSignals(True)
        self.list.clear()
        for region_id, area in regions:
            item = QListWidgetItem(
                translate("region_inspector", "Region {region_id}   {area:,.1f} mm²").format(
                    region_id=region_id, area=area
                )
            )
            item.setData(Qt.UserRole, region_id)
            self.list.addItem(item)
        self.list.blockSignals(blocked)
        listed = color_number is not None and bool(regions)
        self.list_title.setText(translate("region_inspector", "The {0:,} regions of color {color_number}, largest "
                                                              "first").format(len(regions), color_number=color_number) if listed else "")
        self.list_title.setVisible(listed)
        self.list.setVisible(listed)
        self.setVisible(listed or self._info is not None)

    def listed(self) -> list[int]:
        return [self.list.item(row).data(Qt.UserRole) for row in range(self.list.count())]

    def _item(self, region_id: int) -> QListWidgetItem | None:
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.UserRole) == region_id:
                return self.list.item(row)
        return None

    def _on_current(self, current, _previous) -> None:
        if current is not None:
            self.region_chosen.emit(current.data(Qt.UserRole))

    def eventFilter(self, watched, event) -> bool:
        if watched is self.list and event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape:
            event.accept()  # Esc goes back to the palette, before the window's Cancel can take it
            return True
        if watched is self.list and event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.back.emit()
            return True
        return super().eventFilter(watched, event)

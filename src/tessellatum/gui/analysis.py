"""The Analysis page of the sidebar: the page measured as the benchmark judges it, its issues, and overlays to see them.

- **Measures**, on paper: the colors and regions, regions per cm², the smallest region and the narrowest one, the share
  of the page thinner than the brush (its corners' tips left out, as a painter fills those with the brush's point),
  the numbers written on leaders, cramped or smaller than the page's smallest size, and the regions with no number.
  They are ``core.measures.measure_page``'s, the benchmark's own functions (see ``benchmarks/bench_metrics.py``),
  measured in the background while this page is shown, once a page.
- **Issues**: the regions under each measure, largest first. Activate one (double-click, or Enter) to pick it on the
  canvas, centered, with the region inspector telling its measures.
- **Overlays**, off until asked for: the parts thinner than the brush, the small regions and the numbers on leaders,
  drawn in the canvas's diagnostics layer. Like every layer but the artwork, an export never sees them.
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tessellatum.gui.i18n import number, translate

NARROW_PARTS, SMALL_REGIONS, LEADERS = "narrow", "small", "leaders"
OVERLAYS = (NARROW_PARTS, SMALL_REGIONS, LEADERS)
OVERLAY_LABELS = {
    NARROW_PARTS: QT_TRANSLATE_NOOP("analysis", "Parts thinner than the brush"),
    SMALL_REGIONS: QT_TRANSLATE_NOOP("analysis", "Small regions"),
    LEADERS: QT_TRANSLATE_NOOP("analysis", "Numbers on leader lines"),
}
ISSUES_LISTED = 500  # the regions listed under an issue, at most; its count says how many there are
REGION_ROLE = Qt.UserRole + 1


def _percent(fraction: float | None) -> str:
    return "–" if fraction is None else number(fraction * 100, 1) + "%"


def metric_rows(measures) -> list[tuple[str, str]]:
    """``measures`` (``core.measures.PageMeasures``) as the Analysis page lists them: (name, value) rows."""
    m = measures
    rows = [
        (translate("analysis", "Colors"), f"{m.colors}"),
        (translate("analysis", "Regions"), number(m.regions)),
        (translate("analysis", "Regions per cm²"), number(m.regions_per_cm2, 1)),
        (translate("analysis", "Smallest region"), translate("analysis", "{0:.1f} mm²").format(m.smallest_mm2)),
        (
            translate("analysis", "Narrowest region"),
            translate("analysis", "{0:.1f} mm across at most").format(m.narrowest_mm),
        ),
        (
            translate("analysis", "Thinner than the brush"),
            translate("analysis", "{0} of the page ({1:g} mm brush)").format(_percent(m.sliver_fraction), m.brush_mm),
        ),
        (translate("analysis", "Numbers on leaders"), number(m.leader_labels)),
        (translate("analysis", "Cramped numbers"), number(m.cramped_labels)),
        (
            translate("analysis", "Numbers under {0:g} pt").format(m.min_pt),
            _percent(m.small_label_fraction)
            + (
                ""
                if m.min_label_pt is None
                else " " + translate("analysis", "(smallest {0:.1f} pt)").format(m.min_label_pt)
            ),
        ),
        (
            translate("analysis", "Regions with no number"),
            translate("analysis", "{0:,} ({1} of the page)").format(
                m.unlabeled_regions, _percent(m.unlabeled_area_fraction)
            ),
        ),
    ]
    return rows


def issue_groups(measures) -> list[tuple[str, str, list[int]]]:
    """The issues: (title, what it means, the region ids, largest first), each a measure's regions."""
    m = measures
    return [
        (
            translate("analysis", "Regions nowhere {0:g} mm wide").format(m.brush_mm),
            translate("analysis", "The brush can't fit in them anywhere."),
            m.narrow.tolist(),
        ),
        (
            translate("analysis", "Regions under {0:g} mm²").format(m.min_area_mm2),
            translate("analysis", "Smaller than a page of these settings keeps anywhere."),
            m.small.tolist(),
        ),
        (
            translate("analysis", "Regions with no number"),
            translate("analysis", "No number was written for them."),
            m.unlabeled.tolist(),
        ),
        (
            translate("analysis", "Cramped numbers"),
            translate("analysis", "Written where lines may run through them: no room was found."),
            m.cramped.tolist(),
        ),
        (
            translate("analysis", "Numbers under {0:g} pt").format(m.min_pt),
            translate("analysis", "Printed smaller than the page's smallest number."),
            m.small_numbers.tolist(),
        ),
        (
            translate("analysis", "Regions numbered on leader lines"),
            translate("analysis", "Too small for their number inside: it is written beside them."),
            m.on_leaders.tolist(),
        ),
    ]


class AnalysisPanel(QWidget):
    """The Analysis page (see the module's docstring)."""

    region_activated = Signal(int)  # an issue's region, to pick and center
    issue_chosen = Signal(object)  # the region ids of the issue chosen (or of the region's issue), to highlight; None
    overlays_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("analysisPanel")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setProperty("role", "secondary")
        self.metrics = QWidget()
        self.metrics_layout = QFormLayout(self.metrics)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.issues = QTreeWidget()
        self.issues.setObjectName("issueList")
        self.issues.setHeaderHidden(True)
        self.issues.setAccessibleName(translate("analysis", "Issues"))
        self.issues.setMinimumHeight(180)
        self.overlay_checks: dict[str, QCheckBox] = {}
        overlays = QWidget()
        overlays_layout = QVBoxLayout(overlays)
        overlays_layout.setContentsMargins(0, 0, 0, 0)
        for key in OVERLAYS:
            check = QCheckBox(translate("analysis", OVERLAY_LABELS[key]))
            check.setToolTip(translate("analysis", "Shown on the canvas only: an export never includes it"))
            check.toggled.connect(lambda _on: self.overlays_changed.emit())
            self.overlay_checks[key] = check
            overlays_layout.addWidget(check)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self.status)
        layout.addWidget(self._heading(translate("analysis", "Measures")))
        layout.addWidget(self.metrics)
        layout.addWidget(self._heading(translate("analysis", "Show on the canvas")))
        layout.addWidget(overlays)
        layout.addWidget(self._heading(translate("analysis", "Issues")))
        layout.addWidget(self.issues, 1)
        self.issues.itemActivated.connect(self._activated)
        self.issues.currentItemChanged.connect(self._on_current)
        self._measures = None
        self.show_measures(None)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("detailTitle")
        return label

    def measures(self):
        return self._measures

    def overlays(self) -> set[str]:
        """The overlays asked for."""
        return {key for key, check in self.overlay_checks.items() if check.isChecked()}

    def set_measuring(self) -> None:
        self.status.setText(translate("analysis", "Measuring the page…"))
        self.status.show()

    def show_measures(self, measures, empty: str | None = None) -> None:
        """Show ``measures`` (``core.measures.PageMeasures``), or ``empty`` with none (by default, that a preview is
        to be made)."""
        if empty is None:
            empty = translate("analysis", "Generate a preview to analyze it here.")
        self._measures = measures
        while self.metrics_layout.rowCount():
            self.metrics_layout.removeRow(0)
        self.issues.clear()
        if measures is None:
            self.status.setText(empty)
            self.status.show()
            return
        self.status.hide()
        for name, value in metric_rows(measures):
            text = QLabel(value)
            text.setWordWrap(True)
            text.setAccessibleName(translate("analysis", "{name}: {value}").format(name=name, value=value))
            self.metrics_layout.addRow(name, text)
        for title, meaning, regions in issue_groups(measures):
            group = QTreeWidgetItem([f"{title}: {number(len(regions))}"])
            group.setToolTip(0, meaning)
            group.setFlags(group.flags() & ~Qt.ItemIsSelectable if not regions else group.flags())
            for region in regions[:ISSUES_LISTED]:
                child = QTreeWidgetItem([translate("analysis", "Region {region}").format(region=region)])
                child.setData(0, REGION_ROLE, int(region))
                group.addChild(child)
            if len(regions) > ISSUES_LISTED:
                hidden = len(regions) - ISSUES_LISTED
                more = QTreeWidgetItem([translate("analysis", "… and {0:,} more").format(hidden)])
                more.setFlags(Qt.ItemIsEnabled)
                group.addChild(more)
            self.issues.addTopLevelItem(group)

    def issue_regions(self, row: int) -> list[int]:
        group = self.issues.topLevelItem(row)
        return [group.child(i).data(0, REGION_ROLE) for i in range(group.childCount())
                if group.child(i).data(0, REGION_ROLE) is not None]

    def chosen_issue(self) -> list[int] | None:
        """The region ids of the issue chosen, or of the issue the region chosen is under; None."""
        item = self.issues.currentItem()
        if item is None or self._measures is None:
            return None
        group = item if item.parent() is None else item.parent()
        return issue_groups(self._measures)[self.issues.indexOfTopLevelItem(group)][2]

    def _on_current(self, current, _previous) -> None:
        self.issue_chosen.emit(self.chosen_issue())
        if current is not None:
            self._activated(current, 0)

    def _activated(self, item: QTreeWidgetItem, _column: int) -> None:
        region = item.data(0, REGION_ROLE)
        if region is not None:
            self.region_activated.emit(int(region))

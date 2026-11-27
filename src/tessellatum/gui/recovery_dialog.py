"""The recovery dialog: the work an unclean exit left, offered back as the app starts (see ``app.recovery``).

Each snapshot is listed with its thumbnail, the work's name and when it was written. **Restore** opens the one
selected (it is unsaved, and says where it came from), **Discard** removes it for good, and **Keep for Later** leaves
them all for the next start.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QDateTime, QLocale, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tessellatum.app.recovery import RecoveryStore, Snapshot
from tessellatum.gui.i18n import translate


def when(seconds: float) -> str:
    """When a snapshot was written, as the list says it, in the language and the date and time formats of the
    system's locale: "today at 14:05", or "12 Nov 2026 at 14:05"."""
    locale = QLocale()
    at = QDateTime.fromSecsSinceEpoch(int(seconds))
    clock = locale.toString(at.time(), QLocale.ShortFormat)
    if time.strftime("%Y-%m-%d", time.localtime(seconds)) == time.strftime("%Y-%m-%d"):
        return translate("recovery_dialog", "today at {time}").format(time=clock)
    day = locale.toString(at.date(), QLocale.ShortFormat)
    return translate("recovery_dialog", "{day} at {time}").format(day=day, time=clock)


class RecoveryDialog(QDialog):
    """The snapshots ``pending`` (from ``store``); ``chosen`` is the one to restore once it is accepted."""

    THUMBNAIL = 72

    def __init__(self, pending: list[Snapshot], store: RecoveryStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(translate("recovery_dialog", "Recover Unsaved Work"))
        self.setObjectName("recovery")
        self.store = store
        self.chosen: Snapshot | None = None
        intro = QLabel(translate("recovery_dialog", "Tessellatum didn't close properly last time. This work had "
                                                    "changes that weren't saved:"))
        intro.setWordWrap(True)
        self.list = QListWidget()
        self.list.setIconSize(QSize(self.THUMBNAIL, self.THUMBNAIL))
        self.list.setAccessibleName(translate("recovery_dialog", "Unsaved work"))
        for snapshot in pending:
            item = QListWidgetItem(translate("recovery_dialog", "{name}\nkept {when}").format(
                name=snapshot.name, when=when(snapshot.saved_at)))
            item.setData(Qt.UserRole, snapshot)
            thumbnail = snapshot.thumbnail()
            if thumbnail:
                pixmap = QPixmap()
                if pixmap.loadFromData(thumbnail):
                    item.setIcon(QIcon(pixmap))
            self.list.addItem(item)
        self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _item: self.restore())
        buttons = QDialogButtonBox()
        self.restore_button = buttons.addButton(translate("recovery_dialog", "Restore"), QDialogButtonBox.AcceptRole)
        self.discard_button = buttons.addButton(translate("recovery_dialog", "Discard"),
                                                QDialogButtonBox.DestructiveRole)
        self.later_button = QPushButton(translate("recovery_dialog", "Keep for Later"))
        buttons.addButton(self.later_button, QDialogButtonBox.RejectRole)
        self.restore_button.setDefault(True)
        self.restore_button.setToolTip(translate("recovery_dialog", "Open the work selected as it was, unsaved"))
        self.discard_button.setToolTip(translate("recovery_dialog", "Remove the work selected for good"))
        self.later_button.setToolTip(translate("recovery_dialog", "Leave it all for the next time Tessellatum "
                                                                  "starts"))
        self.restore_button.clicked.connect(self.restore)
        self.discard_button.clicked.connect(self.discard)
        self.later_button.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)
        self.resize(460, 320)

    def selected(self) -> Snapshot | None:
        item = self.list.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def restore(self) -> None:
        self.chosen = self.selected()
        if self.chosen is not None:
            self.accept()

    def discard(self) -> None:
        snapshot = self.selected()
        if snapshot is None:
            return
        self.store.discard(snapshot)
        self.list.takeItem(self.list.currentRow())
        if self.list.count() == 0:
            self.reject()

"""Tessellatum entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from tessellatum.gui.main_window import MainWindow

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":  # a check of a build (see ``selftest``), no window
        from tessellatum import selftest

        sys.exit(selftest.run(Path(sys.argv[2])))

    app = QApplication(sys.argv)
    app.setApplicationName("Tessellatum")
    app.setOrganizationName("Tessellatum")
    app.setWindowIcon(QIcon(str(RESOURCES_DIR / "icon.png")))

    # The panel's settings are kept between runs, under the organization and application names above.
    window = MainWindow(store=QSettings())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Tessellatum entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from tessellatum.gui import theme
from tessellatum.gui.main_window import MainWindow

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
APP_ICON_DIR = RESOURCES_DIR / "app-icon"  # the logo at each size, the small ones simplified (packaging/icons)
APP_ICON_SIZES = (16, 20, 24, 32, 48, 64, 128, 256, 512)


def window_icon() -> QIcon:
    """The app's icon at every size it was drawn at, so the title bar and taskbar take the drawing made for their
    size (a 16 px icon is the simplified logo, not the large one shrunk)."""
    icon = QIcon()
    for size in APP_ICON_SIZES:
        path = APP_ICON_DIR / f"{size}.png"
        if path.is_file():
            icon.addFile(str(path), QSize(size, size))
    return icon


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":  # a check of a build (see ``selftest``), no window
        from tessellatum import selftest

        sys.exit(selftest.run(Path(sys.argv[2])))

    app = QApplication(sys.argv)
    app.setApplicationName("Tessellatum")
    app.setOrganizationName("Tessellatum")
    app.setWindowIcon(window_icon())

    # The panel's settings and the theme picked are kept between runs, under the organization and application names
    # above.
    theme.install(app, QSettings())
    window = MainWindow(store=QSettings())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

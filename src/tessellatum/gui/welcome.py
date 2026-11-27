"""The welcome page: the canvas while no picture is open.

A few words on what to do, **Open Picture…** and **Open Project…**, a reminder that a file can be dropped anywhere on
the window or pasted, the example pictures bundled with the app (``resources/examples``, public domain or CC0; see
its ``README.md``), and the recent projects and pictures. A recent file no longer there is greyed, with **Remove**.
It is guidance only: there is no tour, and nothing here is needed to use the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

THUMBNAIL = 112


@dataclass(frozen=True)
class Example:
    file: str
    title: str
    credit: str  # its title and author, as its source gives them; all are public domain or CC0

    @property
    def path(self) -> Path:
        return examples_folder() / self.file


EXAMPLES = (
    Example("lion.jpg", "Lion", '"African Lion", Mathias Appel, CC0'),
    Example("cat.jpg", "Cat", '"Headshot of orange cat with white face", Derrick Coetzee, CC0'),
    Example("lakeside-castle.jpg", "Castle", '"Switzerland - Oberhofen castle", Andrea Einaudi, CC0'),
    Example("cartoon-girl.png", "Cartoon", '"Cute girl", dcatcherex (Openclipart), CC0'),
)


def examples_folder() -> Path:
    return Path(str(resources.files("tessellatum").joinpath("resources", "examples")))


def examples() -> list[Example]:
    """The examples bundled, the ones whose file is there."""
    return [example for example in EXAMPLES if example.path.is_file()]


class WelcomePage(QScrollArea):
    """The page; it says what was asked through its signals, and the window does it."""

    open_picture_requested = Signal()
    open_project_requested = Signal()
    open_requested = Signal(Path)  # a recent project or picture
    example_requested = Signal(Path)
    remove_requested = Signal(Path)  # a recent file no longer there

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("welcome")
        self.setFocusPolicy(Qt.NoFocus)  # its buttons are the stops on the Tab path, and it scrolls to them
        self.setAccessibleName("Welcome")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("welcomeBody")
        self.setWidget(body)
        outer = QHBoxLayout(body)
        column = QVBoxLayout()
        column.setSpacing(12)
        outer.addStretch(1)
        outer.addLayout(column, 4)
        outer.addStretch(1)

        title = QLabel("Make a paint-by-numbers page")
        font = QFont(title.font())
        font.setPointSizeF(font.pointSizeF() * 1.6)
        font.setBold(True)
        title.setFont(font)
        title.setAccessibleName("Welcome")
        guide = QLabel("Open a photo or a drawing. The sidebar's difficulty sets how many regions and colors it gets; "
                       "Generate Preview draws the page, and Export prints it.")
        guide.setWordWrap(True)
        guide.setProperty("role", "secondary")
        self.open_picture_button = QPushButton("Open Picture…")
        self.open_picture_button.setProperty("primary", True)
        self.open_picture_button.setToolTip("A photo or a drawing to make a page of")
        self.open_picture_button.clicked.connect(self.open_picture_requested)
        self.open_project_button = QPushButton("Open Project…")
        self.open_project_button.setToolTip("A project saved before: its picture and settings")
        self.open_project_button.clicked.connect(self.open_project_requested)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_picture_button)
        buttons.addWidget(self.open_project_button)
        buttons.addStretch(1)
        hint = QLabel("Or drop a picture or a project anywhere on the window, or paste a picture (Ctrl+V).")
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")

        column.addStretch(1)
        column.addWidget(title)
        column.addWidget(guide)
        column.addLayout(buttons)
        column.addWidget(hint)

        self.example_buttons: list[QToolButton] = []
        found = examples()
        if found:
            column.addSpacing(8)
            column.addWidget(self._heading("Try an example"))
            row = QHBoxLayout()
            for example in found:
                button = QToolButton()
                button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                button.setText(example.title)
                button.setToolTip(f"Open the example picture: {example.credit}")
                button.setAccessibleName(f"Example: {example.title}")
                pixmap = QPixmap(str(example.path))
                if not pixmap.isNull():
                    button.setIcon(QIcon(pixmap.scaled(THUMBNAIL, THUMBNAIL, Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation)))
                    button.setIconSize(QSize(THUMBNAIL, THUMBNAIL))
                button.clicked.connect(lambda _checked=False, p=example.path: self.example_requested.emit(p))
                self.example_buttons.append(button)
                row.addWidget(button)
            row.addStretch(1)
            column.addLayout(row)
            credit = QLabel("Example pictures: public domain or CC0.")
            credit.setProperty("role", "hint")
            column.addWidget(credit)

        self.recent_box = QWidget()
        self.recent_layout = QGridLayout(self.recent_box)
        self.recent_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_layout.setColumnStretch(1, 1)
        column.addSpacing(8)
        column.addWidget(self.recent_box)
        column.addStretch(2)
        self.recent_rows: dict[Path, tuple[QPushButton, QPushButton | None]] = {}

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        font = QFont(label.font())
        font.setBold(True)
        label.setFont(font)
        return label

    def set_recent(self, recent_projects: list[Path], recent_pictures: list[Path]) -> None:
        """Show the recent projects and pictures: the ones there to open, the others greyed with Remove."""
        while self.recent_layout.count():
            item = self.recent_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.recent_rows = {}
        row = 0
        for heading, paths in (("Recent projects", recent_projects), ("Recent pictures", recent_pictures)):
            if not paths:
                continue
            self.recent_layout.addWidget(self._heading(heading), row, 0, 1, 3)
            row += 1
            for path in paths:
                there = path.is_file()
                name = QPushButton(path.name)
                name.setFlat(True)
                name.setStyleSheet("text-align: left;")
                name.setToolTip(str(path) if there else f"{path}\nNo longer there")
                name.setEnabled(there)
                name.clicked.connect(lambda _checked=False, p=path: self.open_requested.emit(p))
                folder = QLabel(str(path.parent) if there else "No longer there")
                folder.setProperty("role", "hint")
                folder.setEnabled(there)
                self.recent_layout.addWidget(name, row, 0)
                self.recent_layout.addWidget(folder, row, 1)
                remove = None
                if not there:
                    remove = QPushButton("Remove")
                    remove.setToolTip("Take it off the list")
                    remove.clicked.connect(lambda _checked=False, p=path: self.remove_requested.emit(p))
                    self.recent_layout.addWidget(remove, row, 2)
                self.recent_rows[path] = (name, remove)
                row += 1
        self.recent_box.setVisible(row > 0)

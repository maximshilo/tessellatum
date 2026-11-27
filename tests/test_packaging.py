"""What a wheel or a plain ``pip install .`` carries: every data file the app reads, not only the code."""

import glob
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "tessellatum"


def test_the_package_data_patterns_take_every_data_file_of_the_package():
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"][
        "package-data"]["tessellatum"]
    taken = {Path(p).resolve() for pattern in patterns for p in glob.glob(str(PACKAGE / pattern)) if os.path.isfile(p)}
    data = {p.resolve() for p in PACKAGE.rglob("*")
            if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts}
    missing = sorted(str(p.relative_to(PACKAGE)) for p in data - taken)
    assert not missing, missing
    # The interface's own: its font, its icons and the window's icons among them.
    for folder in ("fonts", "icons", "app-icon", "examples"):
        assert any(p.parent.name == folder for p in taken), folder

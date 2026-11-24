"""The self-test the release workflow runs on every standalone build (``Tessellatum --self-test DIR``)."""

from __future__ import annotations

import subprocess
import sys

from PIL import Image

from tessellatum import selftest
from tessellatum.core.painting import Version


def test_the_self_test_makes_and_exports_a_page(tmp_path):
    assert selftest.run(tmp_path) == 0
    assert (tmp_path / "self-test.pdf").read_bytes().startswith(b"%PDF")
    for version in Version:
        with Image.open(tmp_path / f"self-test-{version.name.lower()}.png") as png:
            assert min(png.size) > 300 and png.getextrema() != ((255, 255),) * 3
    assert not (tmp_path / selftest.ERROR_FILE).exists()


def test_a_failing_self_test_exits_1_and_leaves_its_traceback(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("no picture")

    monkeypatch.setattr(selftest, "picture", broken)
    assert selftest.run(tmp_path) == 1
    assert "no picture" in (tmp_path / selftest.ERROR_FILE).read_text(encoding="utf-8")


def test_the_app_runs_the_self_test_from_its_command_line(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "tessellatum.main", "--self-test", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "self-test passed" in done.stdout


def test_the_self_test_fails_a_build_without_the_interface_assets(tmp_path, monkeypatch):
    from tessellatum.gui.theme import icons

    monkeypatch.setattr(icons, "ICONS_DIR", tmp_path / "nowhere")
    assert selftest.run(tmp_path) == 1
    assert "interface's assets are missing" in (tmp_path / selftest.ERROR_FILE).read_text(encoding="utf-8")

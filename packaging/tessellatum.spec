# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Tessellatum.

Produces a standalone "onedir" build (double-click to run, no installer)
under dist/Tessellatum/ on Windows and Linux, and dist/Tessellatum.app on
macOS. Build with the platform build script in this directory rather than
calling pyinstaller directly, so the venv/deps are set up consistently.
"""

import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH)
SRC_DIR = SPEC_DIR.parent / "src"
RESOURCES_DIR = SRC_DIR / "tessellatum" / "resources"

block_cipher = None

a = Analysis(
    [str(SRC_DIR / "tessellatum" / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[(str(RESOURCES_DIR), "tessellatum/resources")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

windows_icon = RESOURCES_DIR / "icon.ico"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Tessellatum",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(windows_icon) if windows_icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Tessellatum",
)

if sys.platform == "darwin":
    macos_icon = RESOURCES_DIR / "icon.icns"
    app = BUNDLE(
        coll,
        name="Tessellatum.app",
        icon=str(macos_icon) if macos_icon.exists() else None,
        bundle_identifier="app.tessellatum.desktop",
    )

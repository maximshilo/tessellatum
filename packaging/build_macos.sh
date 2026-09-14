#!/usr/bin/env bash
# Build a standalone Tessellatum.app for macOS (no installer required).
# Usage: ./packaging/build_macos.sh
#
# NOTE: written for cross-platform completeness but not yet run/tested on an
# actual Mac. To get a proper Dock icon, generate src/tessellatum/resources/icon.icns
# first (e.g. via `iconutil -c icns icon.iconset`); the spec falls back to no
# custom icon if it's missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pyinstaller --noconfirm --clean \
    --workpath build/_pyinstaller_scratch --distpath dist \
    packaging/tessellatum.spec

echo
echo "Build complete: dist/Tessellatum.app  <-- run THIS one"
echo "(build/_pyinstaller_scratch is just intermediate scratch space -- ignore it)"
echo "Copy it to /Applications (or anywhere) and double-click to launch it."

#!/usr/bin/env bash
# Build a standalone Tessellatum.app for macOS (no installer required).
# Usage: ./packaging/build_macos.sh
#
# The release workflow (.github/workflows/release.yml) makes the same build on every
# release and checks it with `--self-test`; run the result from a terminal to see
# any error.
# The Dock icon is src/tessellatum/resources/icon.icns, rendered with the other
# platforms' icons from the logo's SVG masters by packaging/icons/make_icons.py.
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

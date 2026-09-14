#!/usr/bin/env bash
# Build a standalone Tessellatum binary for Linux (no installer required).
# Usage: ./packaging/build_linux.sh
#
# NOTE: written for cross-platform completeness but not yet run/tested on an
# actual Linux machine -- if PyInstaller's OpenCV/Qt hooks miss a shared
# library on your distro, run the resulting dist/Tessellatum/Tessellatum
# from a terminal to see the missing-library error and report it.
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
echo "Build complete: dist/Tessellatum/Tessellatum  <-- run THIS one"
echo "(build/_pyinstaller_scratch is just intermediate scratch space -- ignore it)"
echo "Copy the whole dist/Tessellatum folder anywhere and run ./Tessellatum to launch it."

#!/usr/bin/env bash
# Build a standalone Tessellatum binary for Linux (no installer required).
# Usage: ./packaging/build_linux.sh
#
# The release workflow (.github/workflows/release.yml) makes the same build on every
# release and checks it with `--self-test`; run the result from a terminal to see
# any error, e.g. a system library missing on your distro.
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

# Build a standalone Tessellatum.exe for Windows (no installer required).
# Usage: powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Test-Path ".venv")) {
        py -3 -m venv .venv
    }

    & ".venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
    & ".venv\Scripts\pyinstaller.exe" --noconfirm --clean `
        --workpath build\_pyinstaller_scratch --distpath dist `
        packaging\tessellatum.spec

    Write-Host ""
    Write-Host "Build complete: dist\Tessellatum\Tessellatum.exe  <-- run THIS one"
    Write-Host "(build\_pyinstaller_scratch is just intermediate scratch space -- ignore it, don't run anything from there)"
    Write-Host "Copy the whole dist\Tessellatum folder anywhere and double-click Tessellatum.exe to run it."
} finally {
    Pop-Location
}

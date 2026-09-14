# Working in this repo

## Branching & versioning

- Do all work on a feature branch off `main`, never commit directly to `main`.
- Bump the patch version for every feature/fix, kept in sync in both:
  - `pyproject.toml` (`[project] version`)
  - `src/tessellatum/__init__.py` (`__version__`)
  Use a minor/major bump instead only when the change clearly warrants it.
- Push the branch and open a PR with `gh pr create` rather than merging
  locally and pushing `main` directly.

## Running

```
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -m tessellatum.main
```

## Tests

```
.venv\Scripts\python -m pytest tests/
```

## Building a standalone app

See `packaging/build_windows.ps1` / `build_linux.sh` / `build_macos.sh`.
The real output is `dist/Tessellatum/` — PyInstaller's `build/` folder is
just scratch space, never the thing to run.

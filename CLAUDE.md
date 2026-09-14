# Working in this repo

## Public repo — no sensitive info

This repo is public. Never commit secrets/API keys/tokens, real personal
data, or local machine details (absolute filesystem paths, usernames,
internal hostnames, etc.). Sample/test assets must be synthetic or
otherwise safe to publish, not pulled from anything private. If you
generate scratch/debug files while working, keep them out of the repo
(the scratchpad directory, not a tracked path) rather than relying on
.gitignore to catch it after the fact.

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

## Benchmarks

Benchmark any change to the generation pipeline (`src/tessellatum/core/`)
for speed *and* output quality against `main` before opening the PR:

```
.venv\Scripts\python benchmarks\bench.py run main WORKTREE
.venv\Scripts\python benchmarks\bench.py compare main-<commit> worktree-<commit>-dirty
```

See `benchmarks/README.md` for what is measured and how the quality verdict
works. The region and rendering stages are checked for pixel-identical
output against the original implementation kept in `tests/reference_impl.py`
(`tests/test_regions_equivalence.py`); if a change is meant to alter their
output, update the reference deliberately and say so in the PR.

## Building a standalone app

See `packaging/build_windows.ps1` / `build_linux.sh` / `build_macos.sh`.
The real output is `dist/Tessellatum/` — PyInstaller's `build/` folder is
just scratch space, never the thing to run.

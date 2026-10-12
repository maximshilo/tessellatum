# Working in this repo

## Start here

Before changing the generation pipeline, or planning work on it, get
familiar with how its output is judged:

- `benchmarks/QUALITY_BENCHMARKS.md`: what a good paint-by-numbers page is
  for each kind of image (photos, bold-line cartoons, faces, text) and how
  each property is measured.
- `benchmarks/README.md`: the speed + quality benchmark harness and its
  pass/fail verdict.

Longer-running work is planned and tracked in `plans/`. It is git-ignored,
so it only exists in a local checkout. If it exists, read `plans/README.md`
first and follow it: work from the active plan, and update task status and
the progress log before you finish.

## Public repo — no sensitive info

This repo is public. Never commit secrets/API keys/tokens, real personal
data, or local machine details (absolute filesystem paths, usernames,
internal hostnames, etc.). Sample/test assets must be synthetic or
otherwise safe to publish, not pulled from anything private. Photos can
carry metadata (EXIF/XMP: GPS position, author, camera serial numbers):
strip it before committing an image. `tests/sample_images/` may also hold
local-only images that must not be committed; only commit images whose
license has been confirmed, and list each one's source and license in
`tests/sample_images/SOURCES.md` (it also has the rules for adding images). If you
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
- Add the version's entry to `CHANGELOG.md` in the same PR.
- Releasing is a separate step, done when the user asks to ship: after the
  PR is merged, tag the merge commit `vX.Y.Z` (matching the version) and
  push the tag. `.github/workflows/release.yml` then builds the Windows app
  and publishes the GitHub release with that `CHANGELOG.md` section as its
  notes.

## Public presentation: README, wiki, gallery

The GitHub page showcases the project, so it has to stay true as the code
changes. Update it in the same PR as the change it describes:

- `README.md` is the landing page: brief, leading with the machine-learning
  and image-analysis techniques and tools the project uses; no
  implementation detail (that goes in the wiki). Keep its numbers (tests,
  settings, preview times) and its technique list current.
- The wiki's source is `wiki/` in this repo. `.github/workflows/wiki.yml`
  publishes it to the GitHub wiki on every push to `main` that touches it,
  overwriting edits made in GitHub's editor, so edit `wiki/` only. A change
  to behavior, settings, techniques, architecture or the workflow updates
  the matching page (`User-Guide`, `How-It-Works`, `Techniques`,
  `Architecture`, `Quality-and-Performance`, `Development`).
- `docs/make_assets.py` regenerates the gallery, the README banner and the
  social preview from the sample images with the current pipeline;
  `docs/make_screenshot.py` the app screenshot. Rerun them when pages change
  visibly, and before a release. The social preview is uploaded by hand
  (Settings > Social preview): tell the user when it changed.
- The repository's About (description, topics) and badges should match what
  the project is; mention it to the user when a change makes them stale.

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
works, and `benchmarks/QUALITY_BENCHMARKS.md` for the quality goals the
metrics serve. The region and rendering stages are checked for pixel-identical
output against the original implementation kept in `tests/reference_impl.py`
(`tests/test_regions_equivalence.py`); if a change is meant to alter their
output, update the reference deliberately and say so in the PR.

## Building a standalone app

See `packaging/build_windows.ps1` / `build_linux.sh` / `build_macos.sh`.
The real output is `dist/Tessellatum/` — PyInstaller's `build/` folder is
just scratch space, never the thing to run.

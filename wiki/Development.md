# Development

## Workflow

- Work on a feature branch off `main`; never commit to `main` directly.
- Bump the version in both `pyproject.toml` and `src/tessellatum/__init__.py`, and add an entry to `CHANGELOG.md`.
- Push the branch and open a pull request. The **tests** workflow runs the suite on Windows, Linux and macOS for every
  pull request and every push to `main`.
- Changes to the pipeline (`src/tessellatum/core/`) are benchmarked for speed and quality against `main` before review.

## Tests

```
.venv\Scripts\python -m pytest tests/
```

The suite (1,700+ tests) covers every pipeline stage, the guarantees a page makes (each boundary drawn once, every
region enclosed and numbered, no number on ink), the vector PDF against the raster page, the GUI (driven headless on
Qt's offscreen platform) and the benchmark harness itself. The region and rendering stages are checked for
pixel-identical output against the original implementations kept in `tests/reference_impl.py` and
`tests/reference_line_art.py`; a change meant to alter their output updates the reference deliberately.

## Translations

Every string the interface shows goes through `gui.i18n.translate(context, text)` (Qt's
`QCoreApplication.translate`), its context the module it is in; a string kept in a constant is marked with
`QT_TRANSLATE_NOOP` and translated where it is shown. The core has no Qt, so its words (the settings' registry, the
versions' and stages' names) are translated as they are shown, with `i18n.term`, under the context `settings`.
Placeholders are named and filled after the translation, and the string `translate` returns writes the numbers it is
filled with in the locale's format.

```
.venv\Scripts\python packaging\update_translations.py            # what is collected, by context
.venv\Scripts\python packaging\update_translations.py de fr      # write or update translations/tessellatum_de.ts, _fr.ts
.venv\Scripts\python packaging\update_translations.py --release  # compile them into the app's .qm files
```

A `.ts` file is edited in Qt Linguist. Updating keeps every translation still in use and marks the ones whose string
is gone as vanished rather than deleting them. Two checks help a translation:
- `TESSELLATUM_PSEUDO_LOCALE=1` runs the app in a pseudo-locale: every string accented and about 40% longer, in
  brackets. Text without brackets isn't translatable, and a label that clips has a fixed width.
- `-reverse` lays the window out right to left.

The tests check that every string the interface asks for is in the catalog and that the pseudo-locale leaves no English
shown (`tests/test_i18n.py`).

## Benchmarks

```
.venv\Scripts\python benchmarks\bench.py run main WORKTREE
.venv\Scripts\python benchmarks\bench.py compare main-<commit> worktree-<commit>-dirty
```

See [Quality and performance](Quality-and-Performance) for what is measured, and `benchmarks/README.md` for every
metric and tolerance.

## Releasing

1. Merge the version bump (with its `CHANGELOG.md` entry) into `main`.
2. Tag the merge commit and push the tag:
   ```
   git tag -a v1.2.3 -m "Tessellatum 1.2.3"
   git push origin v1.2.3
   ```
3. The **release** workflow builds the standalone app with PyInstaller on Windows, Linux (Ubuntu 22.04, for the
   oldest glibc) and macOS (Apple silicon), checks that each build makes and exports a page
   (`Tessellatum --self-test`, `src/tessellatum/selftest.py`), and publishes them as one GitHub release:
   `Tessellatum-1.2.3-windows-x64.zip`, `-macos-arm64.zip` and `-linux-x64.tar.gz`. Its notes
   (`packaging/release_notes.py`) are every version's `CHANGELOG.md` section since the previous release's tag, then how
   to download and run the app. A pull request that touches the packaging builds and checks all three without
   publishing.

## Keeping the docs current

The README, these wiki pages and the gallery describe the project as it is, so they change with it:

- **Wiki:** these pages live in the main repository under `wiki/`, and the **wiki** workflow publishes them to the
  GitHub wiki on every push to `main` that touches them. Edit them there, in the same pull request as the change they
  describe; edits made in GitHub's wiki editor are overwritten.
- **README:** the landing page; brief, with the details here.
- **Images:** `docs/make_assets.py` regenerates the gallery, the README banner and the social preview with the current
  pipeline, and `docs/make_screenshot.py` the app screenshot. Run them when the pages change visibly, and before a
  release. The social preview is uploaded by hand in the repository's settings.
- **Logo and icons:** edit the logo in `packaging/icons/make_logo.py`, then run it and `make_icons.py` (both in
  `packaging/icons/`) to write the SVG masters and every platform's icon into `src/tessellatum/resources/`, and
  `docs/make_assets.py` for the banner.

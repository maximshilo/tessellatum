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

The suite (2,000+ tests) covers every pipeline stage, the guarantees a page makes (each boundary drawn once, every
region enclosed and numbered, no number on ink), the vector PDF against the raster page, the GUI (driven headless on
Qt's offscreen platform) and the benchmark harness itself. The region and rendering stages are checked for
pixel-identical output against the original implementations kept in `tests/reference_impl.py` and
`tests/reference_line_art.py`; a change meant to alter their output updates the reference deliberately.

**Interface tests.** They build the real window on Qt's offscreen platform and drive it as a user would: clicks, keys,
menu commands, jobs run to the end or held mid-way by a fake pipeline, file dialogs answered by the test. A window
gets no settings store (`MainWindow()` keeps nothing), so a test never reads or writes the user's settings; one that
needs a store uses a throwaway `.ini` file in a temporary folder. Offscreen fonts are about 40% wider than native ones,
so a test checks layout by relation (a row stacks where it doesn't fit; the inspector folds away below a width), never
by pixels; what the window looks like is checked on a desktop session, below. `tests/test_parity.py` holds the
window to every capability the app had before its workspace was rebuilt (1.0.21): each is still reachable.

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

## Layout at every size

```
.venv\Scripts\python docs\check_layout.py OUT_DIR                        # 100-200%, 1280×720 and 1920×1080
.venv\Scripts\python docs\check_layout.py OUT_DIR --pseudo --scales 1 1.5 --sizes 1280x720 2560x1440
```

It draws the window, the export dialog and the Preferences at each interface scale (a process each, with
`QT_SCALE_FACTOR`) and window size, saves the screenshots and writes `layout.md`: whether the window fits and which
text is cut. It needs a desktop session (Qt's offscreen platform has no fonts); the **layout** workflow runs it on
the three platforms.

## Screenshots

```
.venv\Scripts\python docs\make_screenshot.py --set OUT_DIR                 # 9 views x 2 themes x 2 sizes
.venv\Scripts\python docs\make_screenshot.py --compare REF_DIR OUT_DIR     # a set against its references
```

The set is the welcome, the Generation and Colors pages, a region selected, the analysis, two comparisons, the export
dialog and the Preferences, in the light and dark themes, at 1280×720 and 1920×1080. `--compare` fails when more than
1% of a shot's pixels differ by more than a few levels. Fonts and anti-aliasing differ by platform, so references are
kept per platform and display, and the comparison runs locally, not in CI: it checks that a change meant to leave the
window alone did.

## Verification matrix

What has been checked where:

| | Windows 11 | macOS (CI, Apple silicon) | Linux (CI, X11 under Xvfb) |
|---|---|---|---|
| Test suite | every pull request | every pull request | every pull request |
| Standalone build and self-test | every release, and every pull request touching the packaging | same | same |
| Layout at 100-200%, both themes | `check_layout.py`, 1280×720 to 3840×2160 | the **layout** workflow, to 2560×1440 | the **layout** workflow, to 2560×1440 |
| Pseudo-locale (strings 40% longer) | locally and in CI | in CI | in CI |
| Keyboard-only use; names and roles for assistive technology | tested; the UI Automation tree walked once | tested headless | tested headless |
| Screen readers (Narrator, NVDA, VoiceOver, Orca) | not yet | not yet | not yet |
| Real display scaling, mixed-DPI monitors | not yet (scales set with `QT_SCALE_FACTOR`) | Retina not checked | not yet |
| Native file dialogs, the app menu | by hand | menu roles tested; not by hand | not by hand; Wayland not checked |

## Benchmarks

```
.venv\Scripts\python benchmarks\bench.py run main WORKTREE
.venv\Scripts\python benchmarks\bench.py compare main-<commit> worktree-<commit>-dirty
.venv\Scripts\python benchmarks\ui_bench.py OUT_DIR       # the app's responsiveness, through the real window
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
  pipeline, and `docs/make_screenshot.py` the app screenshot (and, with `--set`, the screenshots above). Run them
  when the pages change visibly, and before a release. The social preview is uploaded by hand in the repository's
  settings.
- **Logo and icons:** edit the logo in `packaging/icons/make_logo.py`, then run it and `make_icons.py` (both in
  `packaging/icons/`) to write the SVG masters and every platform's icon into `src/tessellatum/resources/`, and
  `docs/make_assets.py` for the banner.

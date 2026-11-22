# Changelog

All notable changes to Tessellatum are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).
Each release's notes on GitHub are its section below.

## [1.0.8] - 2026-11-22

### Changed

- **Smallest number** goes down to 1 pt (was 3 pt). Under about 3 pt a number is for coloring on screen, zoomed in,
  rather than on paper.
- A PDF's numbers are placed for the PDF: as small as **Smallest number** asks, in fractions of a pixel, however few
  pixels the picture has, rather than no smaller than 10 px of the page. On a picture exported at its own size, a
  3 pt number used to print at 4-6 pt; now it prints at 3 pt, so more small regions take their number inside them
  rather than on a leader. The regions are the preview's, as before; the preview, its zoom and a PNG are unchanged.


### Changed

- The page's ink is in three grays: the lines, the numbers inside their regions, and the numbers written outside their
  regions with the leaders pointing them in and the leaders' dots, so a number with a leader is told at a glance from
  one inside its region. Medium, the default, prints the lines as before (L* 38), the numbers inside their regions a
  little darker than before (L* 54, was 58), and the leaders and their numbers lighter (L* 70). Light and Dark move all
  three by 15 L*.
- Leaders print thinner than the lines: 0.1 mm by default, against the lines' 0.2 mm, their dots three times as wide.
  Numbers keep as clear of a leader as of a line however thin it is, so no number moves and no region changes.

### Added

- Settings **Line gray**, **Number gray** and **Leader gray** (shown as a lightness, L*), which the Tone sets all three
  of; moving one picks a Custom tone. Setting **Leader width**.

## [1.0.6] - 2026-11-22

### Added

- A region pinched so thin somewhere that it reads as two regions on paper gets its number on each side. A neck the
  lines close off, or one whose paper narrows to less than the new **Split at necks under** setting (0.35 mm by
  default), splits it; each side at least as large as the smallest region gets the number, inside it if it fits, else
  with a leader. Every region's own number stays exactly where it was, and the regions themselves don't change. A side
  with no room for a number anywhere keeps none, rather than one written over the lines.
- The benchmark reports **unnumbered sides**: sides of a region that read as separate regions and carry no number of
  their region, measured independently of the pipeline.

## [1.0.5] - 2026-11-22

### Fixed

- Two tests failed with OpenCV 5.1, which the test runs now install; the pages it makes are as good. One asked the
  boxes found round lines of lettering to cover the blank rows above the letters too, where 5.1's boxes start a
  fraction of a pixel lower: it now asks them to hold every pixel of the letters. The other, of an export rendered
  again where a number moves, relied on a number happening to move on one sample page, and none does with 5.1: it
  now moves one itself.

## [1.0.4] - 2026-10-22

### Changed

- The test suite runs on Linux and macOS as well as Windows, for every pull request and every push to `main`.

### Fixed

- A face test asked for at least five fewer regions where a face's tones are settled, a margin that only held on
  Windows; macOS (arm64) paints the lion slightly differently and settles three. It now asks for fewer.

## [1.0.3] - 2026-10-22

### Fixed

- The README banner and the social preview no longer stretch the lion sideways: the picture is cropped at the shape of
  the space it fills instead of being resized into it.

### Changed

- The README lists macOS and Linux as platforms, with how to run Tessellatum from source on them.

## [1.0.2] - 2026-10-12

### Fixed

- Two window tests no longer depend on timing: they hold the page being drawn at print size until they have checked
  it, rather than racing it. One of them failed on CI when the drawing stopped before the test looked.

## [1.0.1] - 2026-10-12

### Fixed

- Release notes keep their non-ASCII characters (such as "²" and "–"): the release workflow had them pass through a
  Windows console code page, which garbled them on the 1.0.0 release page (corrected there by hand).

## [1.0.0] - 2026-10-12

The first stable release: a desktop app that turns any picture into a printable paint-by-numbers page.

### Highlights

- **Every kind of picture.** Photographs, paintings, cartoons, comics and scans, each handled its own way: line art is
  drawn from its own ink lines and fill colors, photographs from their colors, with texture edges settled into shapes a
  brush can follow.
- **On-device perception.** Faces (YuNet, plus an LBP cascade for drawn faces), the salient subject (U²-Net) and lines of
  text (PP-OCR) are detected offline; faces and subjects get finer regions, and lettering is printed rather than
  painted.
- **Paintable by construction.** Physical rules on an A4 sheet: no part of a region narrower than the brush, no number
  under 3 pt or on a line, every boundary drawn exactly once, every two palette colors a set CIEDE2000 margin apart.
- **Five difficulty levels and 22 settings**, from Beginner (16 colors, a 1 mm brush) to Realistic (32 colors, a 0.5 mm
  brush), each setting in physical units with a tooltip.
- **Three versions of every page**: the page, the page painted in, and a tinted painting guide, in a zoomable preview
  that shows the page exactly as it prints.
- **Print output**: a 300 dpi PNG, or an A4 PDF of vector art with an embedded font, drawn from the very page the preview
  shows.
- **Fast enough to tune live**: a preview in about 1–3 s, from Numba-compiled kernels, multithreading and per-image
  caching.
- **Verified**: 1,220 automated tests and a benchmark harness that scores every version on speed and image quality.

### Distribution

- A standalone Windows build, attached to this release; no installer or Python needed.
- Continuous integration runs the test suite on every pull request.

Development before 1.0 (versions 0.1.0 to 0.1.56) is recorded in the
[pull requests](https://github.com/maximshilo/tessellatum/pulls?q=is%3Apr+is%3Amerged) and the git history.

[1.0.2]: https://github.com/maximshilo/tessellatum/releases/tag/v1.0.2
[1.0.1]: https://github.com/maximshilo/tessellatum/releases/tag/v1.0.1
[1.0.0]: https://github.com/maximshilo/tessellatum/releases/tag/v1.0.0

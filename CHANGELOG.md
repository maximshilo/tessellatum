# Changelog

All notable changes to Tessellatum are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).
Each release's notes on GitHub are its section below.

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

[1.0.1]: https://github.com/maximshilo/tessellatum/releases/tag/v1.0.1
[1.0.0]: https://github.com/maximshilo/tessellatum/releases/tag/v1.0.0

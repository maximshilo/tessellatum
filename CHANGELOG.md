# Changelog

All notable changes to Tessellatum are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).
Each release's notes on GitHub are its section below.

## [1.0.21] - 2026-11-24

### Added

- **Mixed colors.** With Colors from: My paints, a new box, **Mix colors from my paints**, lets the page also take
  colors mixed from two of your paints (1 + 1, 2 + 1 or 3 + 1 parts, predicted as paint mixes), where the picture has
  colors none of your paints comes close to. Your paints come first, chosen as before; a mix is added only if the
  parts of the picture it would paint come out at least **3 ΔE00** closer on average (**Mix if closer by**) and are
  at least **1%** of the picture (**and it paints at least**), both settings. There is no fixed number of mixes: on
  the benchmark pictures a 5-paint kit gets about 6 and a 24-pencil box about 4, each bringing the page clearly
  closer. The legend lists them in a section of their own, **Mixed colors**, after **Your paints**, each with its
  name and recipe, in the preview, the PNG and the PDF; a palette file names each with its recipe.
- **An easier paints list.** Paints… is now a list of swatches: **Add color…** opens a color picker whose **Pick
  Screen Color** takes a color from anywhere on the screen (a photo of your swatches, or the picture in the preview),
  named for you; click a swatch to pick it again, double-click a name to change it, ✕ to delete. **Edit as text**
  still takes a pasted list.
- **Sample kits** to start from: Basic acrylics (12), Primaries (5), Colored pencils (24), Earth palette (8) and
  Grays (6), generic pigments in typical colors (approximate: real tubes differ), replacing the list or added to it.
- The benchmark makes pages from the kits (`Medium-KitPrimaries`) and with mixing (`Medium-Paints24-Mix`), and
  counts a page's mixed colors.

### Fixed

- The paints list no longer says "The first 256 are used" when exactly 256 are listed.

## [1.0.20] - 2026-11-24

### Added

- **Mixing recipes.** A fourth **Legend** option, **Name and recipe**, writes under each color's name how to mix it
  from the paints you list (Paints…): "3 Titanium white", "1 Ultramarine", one to three of them in whole parts adding
  up to 10 at most. The mixes are worked out as paint mixes, not as light does: each paint's color becomes a
  reflectance curve and the paints are mixed by Kubelka-Munk theory, so blue and yellow make green. A color no mix
  comes within 5 ΔE00 of says how far off its closest is. The paints need not be the page's colors: with Colors from:
  The picture, the page keeps the picture's colors and the legend says how to mix each. The model is spectral.js's
  (MIT), whose data ships with the app with its license.

## [1.0.19] - 2026-11-24

### Added

- **Colors from your own paints.** A new setting, **Colors from** (under Regions): **The picture**, as before, or
  **My paints**: the page's colors are chosen from the paints or pencils you list, at most as many as Colors asks for,
  the ones that paint the picture best (the faces and the subject counting twice), and the legend calls each by its
  own name. Every color on the page is then one you own, exactly. Two paints closer than the color margin are never
  both used, on screen or as printed. **Paints…** lists them: one a line, a hex code (`#5A6B2F`) or three RGB numbers,
  the rest of the line its name; or **Load palette file…** reads a GIMP (`.gpl`), Adobe (`.ase`) or Procreate
  (`.swatches`) palette, or a text file. The list is kept for next time.
- The benchmark makes pages from synthetic paint sets (`Medium-Paints24`: 24 paints spread over the gamut).

## [1.0.18] - 2026-11-24

### Added

- **Palette export** for coloring on a screen: Export > Format > **Palette** saves the legend's colors as a
  Procreate palette (`.swatches`), an Adobe Swatch Exchange file (`.ase`: Photoshop, Illustrator, Affinity and others)
  or a GIMP palette (`.gpl`: GIMP, Krita, Inkscape and others), each color named by its number and its name on the
  legend. A Procreate palette holds 30 colors, so a larger legend is saved as several. The panel suggests it when
  Coloring on is Screen.

## [1.0.17] - 2026-11-24

### Added

- **Color names, hex codes and RGB on the legend.** Beside each swatch, its everyday name ("dusty rose", "olive
  green"), and under it its hex code and RGB values (`#C0737A  RGB 192 115 122`), exactly the color the page is
  painted with: to find it in a painting app, or match a paint or pencil to it. The name is the nearest by CIEDE2000
  of about 800 names from the xkcd color survey (CC0), every color on a legend a different one. A new setting,
  **Legend** (under Lines and numbers): Name, hex and RGB (the default), Name, or Number only, as before. Changing it
  redraws the preview's legend at once and applies to the next export without making the page again. In the PDF the
  names and codes are text, which a reader can search and copy; 64 colors with their details still fit an A4 sheet.

### Changed

- The preview draws its legend at 300 dpi, so its small print stays sharp zoomed in.

## [1.0.16] - 2026-11-23

### Added

- **Colors as they print.** Paper and paint can't go darker than their own black, so near-blacks a screen shows
  apart look alike on paper, and a dark picture (a night sky, a painting's dark ground) used to spend many legend
  colors on them. The palette now keeps its colors apart as printed too: black point compensation to the paper's
  darkest black, a new setting, **Darkest black on paper** (L* 25 by default, 0 to 40), merges the near-blacks it
  can't tell apart, and k-means is asked again for the colors that frees, so the legend keeps its count and spends it
  on the picture's other colors. Colors mostly in the faces and the subject keep their dark shades. A picture without
  such darks gets the same palette as before.
- **Coloring on**: Paper and paint (the default) or Screen, for coloring in an app with an eyedropper: no darkest
  black, and a color margin of at most 2.5 ΔE00, so more of the picture's own shades stay on the legend.
- Benchmark metrics: the legend as printed (its smallest difference and its pairs closer than the margin, with a paper
  black of L* 25), its colors near black and near white, the error on the picture's most colorful tenth, and
  S-CIELAB, the error as the eye sees the page from 40 cm (informational). The verdict allows the trade the printed
  palette makes, a little more error in a large dark area for better colors elsewhere, within set bounds. The
  benchmark also takes a preset colored on a screen, such as `Medium-Screen`.
- A dark portrait among the benchmark images: Velázquez's *Juan de Pareja* (The Met, CC0), a dark-skinned face
  in a dark painting; it is in the gallery too.

## [1.0.15] - 2026-11-23

### Added

- **macOS and Linux apps.** Each release now has the standalone app for macOS (Apple silicon) and Linux (x86-64) as
  well as Windows: unpack it and run it, no Python needed. The macOS app isn't notarized by Apple, so it has to be
  allowed once in System Settings > Privacy & Security.
- Every build is checked before it is released: `Tessellatum --self-test DIR` makes a page of a picture it draws
  itself and exports its PDF and a PNG of each version into `DIR`.

### Changed

- A release's notes hold every version since the previous release, rather than its own alone.

## [1.0.14] - 2026-11-23

### Changed

- **Medium**, the default tone, prints the lines and the numbers inside their regions in one gray, L* 34 (they were
  38 and 54), so a number reads as clearly as the line beside it; the numbers written outside their regions, with
  their leaders and dots, stay a lighter gray of their own (L* 70), told at a glance from a number inside its region.
  Light and Dark keep 15 L* either side: 49, 49 and 85, and 19, 19 and 55. The app picks up the new grays
  for a tone it kept from an earlier run; a Custom tone keeps its own.

## [1.0.13] - 2026-11-23

### Fixed

- **A large PNG lost its lines.** A PNG is the PDF's page drawn by QtPdf, whose renderer (PDFium, also Chrome's and
  Edge's) silently stops drawing a path past about 4.2 million pixel cells, and every line on the page was one path.
  On a dense page, large enough, the lines stopped partway down with a thick dark line where they did: the lion on A2
  at 724 dpi kept its lines over the top 42% only, and on A2 even 300 dpi lost some. The PDF is now written in paths
  that stay within a quarter of that at the largest PNG: the lines in groups by where they lie, a paint's area in
  groups of its shapes, and line art's ink and the lettering in bands overlapping by half a millimeter. A PDF drawn
  whole by a PDFium viewer or printer at a high resolution prints every line too. PDFs are written a little faster.
  Where two groups meet, a few anti-aliased pixels at the ends of lines print a shade darker: on an A4 page at
  300 dpi, at most one pixel in 6,000.

## [1.0.12] - 2026-11-23

### Changed

- **Crowded numbers no longer print across lines** where they can help it. In a cluster of tiny regions, a number
  that fits neither in its region at the smallest size nor outside it on a leader used to be written where it was,
  lines through it. On a page laid out to print (every export, and the zoom), it now goes smaller inside its own
  region, down to half **Smallest number** (1 pt at the least), and only a number that fits nowhere even then is
  written across its lines. On the benchmark's print pages, numbers on lines fall from 18.7 to 6.6 a page (lion at
  Realistic: 102 to 48 on A4, 1,403 to 784 on A2); 0.4% of the numbers print under the smallest size.

## [1.0.11] - 2026-11-23

### Added

- **Resolution** in the export box: the PNG in dots per inch (150, 300 or 600, or any other) or in pixels along its
  long edge, the other shown as it is set, with the file's pixel size. 600 dpi by default. A PNG is at most 16,384 px
  along a side, the most a Procreate canvas takes; the box says when a paper caps it. The legend is drawn at the same
  resolution, and the PNG is drawn in the background.

## [1.0.10] - 2026-11-23

### Changed

- Every export is laid out once to print, at 300 dpi on its paper, however few pixels the picture has (it used to
  stop at the picture's own size), and the PDF and the PNG are both drawn from that layout: the PNG is the PDF's page
  rendered in pixels, so the two number the page the same way, each number at the same place on paper.
- A picture with fewer pixels than its page needs is scaled up first: a page is made where its brush is 4 px across,
  so a small picture, or one printed on A3 or A2, gives regions a brush fits and room for their numbers rather than a
  crowded page with its numbers on the lines. Such pictures take longer to preview.

## [1.0.9] - 2026-11-23

### Added

- **Paper**: A4, A3 or A2. A page is made for its sheet: the brush, the smallest region, the lines and the numbers are
  sizes on that sheet, so a larger one makes a finer page of the same picture, at more pixels where the picture has
  them. The PDF's sheets are that paper, and a PNG records the resolution that prints it there. A4 pages are as
  before.

## [1.0.8] - 2026-11-22

### Changed

- **Smallest number** goes down to 1 pt (was 3 pt). Under about 3 pt a number is for coloring on screen, zoomed in,
  rather than on paper.
- A PDF's numbers are placed for the PDF: as small as **Smallest number** asks, in fractions of a pixel, however few
  pixels the picture has, rather than no smaller than 10 px of the page. On a picture exported at its own size, a
  3 pt number used to print at 4-6 pt; now it prints at 3 pt, so more small regions take their number inside them
  rather than on a leader. The regions are the preview's, as before; the preview, its zoom and a PNG are unchanged.

## [1.0.7] - 2026-11-22

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

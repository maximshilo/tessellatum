# Changelog

All notable changes to Tessellatum are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).
Each release's notes on GitHub are its section below.

## [1.1.30] - 2026-11-26

### Added

- **Analysis**, a fifth page of the sidebar (View > Analysis, Ctrl+Shift+5): the page measured on paper as the
  project's quality benchmark measures it, in the background while the page is open:
  - its measures: colors, regions, regions per cm², the smallest and the narrowest region, the share thinner than
    the brush, numbers on leaders, cramped or too small, and regions with no number;
  - its issues, the regions behind them: an issue chosen highlights them all on the canvas, a region chosen is
    picked and centered;
  - overlays on the canvas for the parts thinner than the brush, the small regions and the numbers on leader lines,
    off until asked for and never exported.

### Changed

- The benchmark measures the slivers, corners' tips, undersized and unnumbered regions and the numbers' sizes with the
  app's own functions, loaded from this checkout as the print model is: one implementation for both. Its numbers are
  unchanged.

## [1.1.29] - 2026-11-26

### Added

- Click a region on the canvas to pick it: its color is picked in the palette and the region outlined, and the
  inspector shows its area and size on paper, the widest brush that fits, and how its number is written (inside, on a
  leader line or cramped, at what size in points, and on how many other sides).
- Enter on a color lists its regions, largest first; Up and Down go through them, each centered on the canvas, and Esc
  goes back to the palette.

### Changed

- A color's highlight on the canvas is built about four times as fast.

## [1.1.28] - 2026-11-26

### Added

- The palette inspector, on the right:
  - each color's share of the page, its area on paper and its number of regions, measured in the background once the
    page is made;
  - sort by number, coverage, lightness, hue or name, without changing a number;
  - a grid of swatches or a list;
  - find a color by number or name;
  - copy its hex (Ctrl+C), RGB or name;
  - pick a color to see it in full and highlight its regions on the canvas; **Isolate** dims the rest. The highlight
    never reaches an export.

## [1.1.27] - 2026-11-26

### Added

- **Compare**, on the canvas bar and in the View menu: two views of the page side by side, following each other's zoom
  and scroll, or one over the other split left and right or top and bottom, with a divider to drag. Compare the view
  shown with Page, Completed, Tinted, Original or the **Previous version**: the page shown before the last change, if
  it is still kept, labeled with that change. Each view is named in its corner. Comparing never makes a page.

## [1.1.26] - 2026-11-26

### Added

- **Original** view (View > Original, Ctrl+4): the picture itself in the page's place, beside Page, Completed and
  Tinted, keeping the zoom and the place. It is never exported.
- A zoom field above the canvas: type a zoom (150, 37.5%) and press Enter, or pick one from its list, where 100% and
  Print size now are (they were buttons).
- Pan with the middle button, or by holding Space and dragging, as well as by dragging. Space only pans while the canvas
  has the keyboard.
- **View > Rulers** (Ctrl+R), **Grid** (Ctrl+') and **Page Bounds**: rulers in millimeters or inches on paper, a grid
  every centimeter or inch, and the sheet and its margins around the page. Hidden until asked for, then kept.

### Changed

- A window resized keeps the same point of the page in its middle, rather than its top left corner.
- **Print size** says it is approximate: it goes by the size the screen reports.

## [1.1.25] - 2026-11-26

### Added

- A welcome page while no picture is open. It has a few words on what to do, **Open Picture…** and **Open
  Project…**, the recent projects and pictures, and four example pictures to try, bundled with the app (public domain
  or CC0). A recent file no longer there is greyed, with **Remove**.
- Drag and drop: a picture or a project dropped anywhere on the window opens. A picture goes into the work as one
  step to undo; a project over unsaved changes asks first.
- **Edit > Paste Picture** (Ctrl+V) opens the picture copied, or the picture or project whose file or path was
  copied. Ctrl+V in a text field still pastes text there.
- **File > Open Recent** lists the recent pictures too.

### Changed

- A file Tessellatum can't open (not a picture, damaged) says why, and the work open stays as it was.

## [1.1.24] - 2026-11-26

### Added

- Recovery: while the work has changes that aren't saved, it is kept every 2 minutes in the app's own folder, never
  over a project.
  - If Tessellatum ends without closing properly (a crash, a power cut), the next start offers it back with its
    thumbnail and when it was kept: **Restore** (it opens unsaved, saying where it came from), **Discard**, or
    **Keep for Later**.
  - Saving, closing the picture and quitting remove it.
  - A window still open is never offered to another.
  - What is kept goes after 14 days, and past the latest 20.
- Preferences > Workspace > **Keep for recovery**: every 1, 2, 5 or 10 minutes, or Off.

## [1.1.23] - 2026-11-26

### Added

- Projects: **File > Save Project** (Ctrl+S) and **Save Project As…** (Ctrl+Shift+S) save the picture and every setting
  as a `.tessellatum` file, to open again later or on another computer. **File > Open…** opens a project or a picture,
  and **Open Recent** lists the projects opened or saved last. **Close** (Ctrl+W) closes the picture.
  - A project keeps the picture's own file, byte for byte, so it doesn't depend on the picture staying where it was.
    It also keeps the settings, the paints listed and the export options, and a small thumbnail.
  - Pages aren't stored: opening a project makes its page again, and the page is the one that was saved.
- Closing the picture, quitting, or opening a project over changes not saved asks whether to save them first.
- The window's title names the project (or the picture), and marks unsaved changes.

### Changed

- Saving is atomic: a project is written beside its file, read back and checked, and only then put in the file's
  place, so a save that fails leaves the file saved before untouched.
- A project that is damaged, too large, made by a newer version, or holding a file outside itself is refused with the
  reason, and the work open stays as it was.
- Export suggests its file beside the project, if there is one, else beside the picture, named after the picture.

## [1.1.22] - 2026-11-26

### Added

- Preferences > Performance > **Generate automatically**: whether a change that makes the page again (the colors, the
  difficulty, the paper, what is done with line art, faces and text) makes it by itself, a second after the last change.
  - **Off**, the default: such a change waits for Generate Preview, as before.
  - **When it's quick**: only while the picture's last page took under 2 s. After a slower one, changes wait for
    Generate Preview, and the preview's state says why and how long the last page took.
  - **Always**.

  A slider dragged makes one page, when it stops. The page replaces one still being made, waits for an export to
  finish, and is called off if the change is put back. A change back to a page kept is shown at once, whatever the
  setting.

## [1.1.21] - 2026-11-26

### Added

- Pages made are kept, so Undo, Redo and a setting put back show their page at once, without making it again (about
  0.1 s for the lion at Medium, rather than 2 s). This holds for undoing an Open as well. A page kept brings its
  print-size layout with it, so the zoom and the export don't lay it out again.
- When a page undo or redo goes back to is no longer kept, the settings come back at once and the page is made again in
  the background. Meanwhile the page shown is kept and labeled out of date. A state that never had a page still waits
  for Generate Preview.
- Preferences > Performance > **Memory for pages made**: how much memory the kept pages may take. Automatic uses a
  quarter of the computer's memory, up to 2 GB. The page shown and the ones Undo and Redo go to are always kept.
  **Clear Now** frees the kept pages too.

### Changed

- A page being made with settings that have since been put back is dropped, rather than replacing the page that is
  now current.

## [1.1.20] - 2026-11-25

### Fixed

- An export of the preview that was waiting for its page to be laid out to print, when another picture took its place
  (an undo of Open can do that while it runs), waited for ever: the busy bar stayed and Generate Preview did nothing.
  It now stops, and says so.

## [1.1.19] - 2026-11-25

### Fixed

- The pipeline's warm-up as the app starts could throw away the stages kept for a picture opened and previewed while
  it ran, so the next preview or redraw built them all again (a redraw of the lion at Medium took 1.7 s rather than
  1.0 s). The warm-up no longer touches what is kept.

## [1.1.18] - 2026-11-25

### Added

- The status bar says what the page being made is doing, step by step: "Sizing the picture", "Finding faces and the
  subject", "Choosing the colors", "Building the regions", "Placing the numbers and drawing the page", "Painting the
  versions", each with how far along it is.
- The diagnostics give each step's time in a page's details, say how many steps were kept from before (a change to
  the lines or numbers keeps most), and, for a page cancelled, how long it took to stop.

### Changed

- Cancelling says at once that the page stops after the current step, and which step that is.
- A failure says what to try: a page too large for memory suggests a smaller paper (A3 rather than A2), an easier
  difficulty or a lower resolution; a file that can't be written says whether it may be open elsewhere, the disk is
  full or its folder is gone. The exception and its traceback stay in the diagnostics, behind Details.

## [1.1.17] - 2026-11-25

### Added

- The bar above the page says whether the page shown is **Up to date**, **Out of date** (its tooltip names what
  changed), **Generating…**, or **Cancelled** / **Failed** with **· earlier preview** when the page asked for didn't
  come and the one shown is from before.
- A change to the lines, the numbers or the tone now draws the preview again by itself, 0.4 s after the last change:
  a slider dragged draws the page once, when it stops, from the regions already found. Every other setting still
  waits for Generate Preview, and the legend's detail still applies at once. **Preferences > Performance > Redraw
  lines and numbers** sets the delay (0.2, 0.4 or 1 s) or turns it off.

### Changed

- The out-of-date status names only the settings that change the page; the legend's detail is no longer among them
  when other settings changed too.

## [1.1.16] - 2026-11-25

### Changed

- A change to the lines or numbers (line width, grays, number size, leaders, smoothing, split necks, the gap beside
  text) no longer builds the page's regions again: they are kept with the rest of a picture's stages, and the page is
  drawn again from them, as it would be from scratch, pixel for pixel. The lion at Medium redraws in 0.9 s rather
  than 1.6 s; line art, whose regions with no room for a number merge as the numbers are placed, in 2.0 s rather than
  3.1 s on the comic.
- The pipeline can announce each stage by name as it starts and ends, with how long it took and whether its results
  were kept (`generate(..., stage_callback=...)`), for the app's progress and diagnostics.

## [1.1.15] - 2026-11-25

### Changed

- Each part of a page's **Advanced** group (Edges, Corners, Resolution, Lines, Numbers…) now folds on its own, under a
  header of its own with its own Reset, folded at first: opening the group lists its parts by name. Finding a setting
  unfolds its part too, and the search names it ("Generation · Advanced · Edges"). What is folded is kept between
  runs.

### Fixed

- Focus mode said nothing about how to leave it once its button was hidden: it now says so over the page (Esc, or
  the same shortcut), and **Exit Focus Mode** stays on the bar above the page.
- The ✕ beside the progress bar did nothing: it turned itself off before asking to cancel, which turned off the
  Cancel command it triggers. It cancels now, as the toolbar's Cancel does.
- "The preview is out of date" was said only after an undo: it is now said as soon as a setting that changes the page
  moves, naming what changed, and "Preview ready." comes back once the settings are as the page was made with.

## [1.1.14] - 2026-11-25

### Added

- **Preferences** (Edit > Preferences, Ctrl+, ; on macOS in the app menu): how the app looks and works on this
  computer, kept apart from the page's settings, never undone with them and kept between runs:
  - **Appearance**: the theme and motion (as in View > Appearance), the interface's scale (90% to 200%, applied when
    the app next starts; the page's zoom stays exact whatever the scale), and the canvas's background behind the page
    (the theme's, a neutral gray to judge colors against, white or black).
  - **Units**: lengths and areas shown and typed in millimeters or inches; the page's values are kept in millimeters
    either way.
  - **Workspace**: Reset Layout.
  - **Performance**: how many pages' intermediate results are kept to make the next preview faster, and Clear Now.
  - **Shortcuts**: every command's keys.

### Fixed

- A command whose standard key the platform doesn't define (Quit on Windows) no longer shows an empty "()" in its
  tooltip.
- The settings search's results hide when the focus leaves the search field, giving the sidebar back, and show again
  when it returns.

## [1.1.13] - 2026-11-25

### Added

- **Presets of your own**: the Presets menu beside the difficulty saves the difficulty's settings as a preset under a
  name of your own (Save as Preset), and renames or deletes it. They are listed after the built-in levels, kept in a
  file of their own in the app's preferences folder, and set only the difficulty's settings: the paper, what the page
  is colored on and the paints stay as they are. A presets file that can't be read is left out, with a warning in the
  log, and kept aside rather than overwritten.
- **What a level makes, in regions**: each built-in level now says how many regions it typically gives on the paper
  chosen ("About 450 regions to paint on an A4 page (200 to 1,250 on most pictures)" for Medium), measured on the
  benchmark's pictures.
- **Back to the level**: a level changed reads "Hard (modified, 2 changes): Colors 26, Smoothing 0.9." under it, and
  Presets > Back to Hard puts its settings back, as one step to undo.

### Changed

- **Reset** beside the difficulty (and Edit > Reset All Settings) asks before putting every setting back; a section's
  Reset and a setting's own don't ask, and Undo takes any of them back.

## [1.1.12] - 2026-11-25

### Added

- **Search the settings** (Edit > Find Setting, Ctrl+F): a field at the top of the sidebar finds a setting by its
  name, by the words it may be called by ("outline" finds Line width, "pupils" Dark mark contrast), by what it does,
  or by where it is, best match first. Each result says where the setting is and what it is set to. Choosing one
  (Enter, or a click) opens its page, unfolds its section, the Advanced group too, scrolls to it, puts the focus on it
  and outlines it for a moment. Up and Down move through the results; Esc clears the search.

## [1.1.11] - 2026-11-25

### Added

- **Undo and Redo** (Edit menu; Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z). Every change to the settings is a step back,
  named for what it did: "Undo Brush width", "Undo Difficulty: Hard", "Undo Reset Detail", "Undo Paints". A slider
  dragged is one step from press to release, and a preset or a reset is one step however many settings it moves.
  Opening another picture can be undone too. The zoom, the panels and the version shown are never in it. Undoing
  back to settings the preview wasn't made with says so in the status bar.

## [1.1.10] - 2026-11-25

### Changed

- **Export opens a dialog first.** It shows the export's options (the version, the format, the PNG's resolution, the
  palette's format) beside a thumbnail of what will be written and a line on it: "A PNG of the page: 6,543 × 4,363 px
  at 600 dpi, printing 277 × 185 mm on A4, with its legend under it." The options have moved there from the Document
  page, and are kept between runs as before.
- **Warnings that don't stop you:** a PNG under 150 dpi, a Procreate palette of more than 30 colors written as
  several files, numbers under 3 pt on a page for paper. Only an export with nothing to write is refused.
- **No more silent re-making.** When the settings changed since the preview, the dialog says which, and asks which
  page to write: the page made again with the settings as they are now (the default, as before), or the page as
  previewed.
- Exports are named for the picture and suggested beside it ("harbor_page.png", "harbor_palette.swatches"). A
  Procreate palette written as several files asks before replacing any of them that exist.
- A failed export offers **Retry**, which writes the same file with the same options again.
- With Coloring on: Screen, the Colors page now also points to exporting the palette.
- Radio buttons have a clear outline in the dark theme.

## [1.1.9] - 2026-11-25

### Changed

- **Each settings page shows what most pages need first.** The everyday settings are on top; the rest wait in one
  **Advanced** group, folded, at the foot of the page, under small headings (Faces and subject, Line art, Text,
  Edges, Corners, Resolution on Generation; Mixing on Colors; Tone, Lines, Numbers on Document). Opening a picture,
  picking a difficulty, generating and exporting never needs one of them.
- **Reset a section on its own.** A section with something changed in it shows **Reset** in its header, which puts
  its sliders back to their preset (or tone, or default) and its choices to their defaults, and leaves the rest of the
  page as it is.
- **The difficulty says what it makes**, under its choice ("Up to 24 colors, at least 4 ΔE00 apart. Regions of at
  least 6 mm² ..."), or, once a setting is moved, how far: "Custom, from Medium: 2 settings changed."
- **The Source page tells you more about the picture:** how large it prints on the paper chosen and at how many of its
  pixels an inch, and, after a preview, whether it was read as line art or as a picture of colors.
- **Recipes from your paints list say so.** With Legend: Name and recipe and the colors from the picture, the legend's
  recipes are now headed "Recipes from your paints list: mix each in the parts given", in the preview, the PNG and the
  PDF, so they don't read as the paints the page was made of.

### Fixed

- Tooltips no longer say sizes are on "the A4 page" when the paper is A3 or A2.
- A settings page scrolled down comes back where it was left in full on the next run, not partway, when its layout
  takes more than one pass to settle.

## [1.1.8] - 2026-11-25

### Changed

- **Type an exact value.** Every slider has its value beside it in a box to type in: "1.2", "1.2 mm", or, in a
  German or French setup, "1,2". A value out of range is not quietly held to it: the box turns red and says the range
  under it ("Out of range: from 0.5 to 6.0 mm."), and the setting keeps its value until a good one is typed; Esc puts
  the box back. Up and Down step the value, Page Up and Page Down ten steps, in the box as on the slider. A gray is
  typed as its lightness (L*).
- **What you changed is marked.** A setting moved off its preset (or the tone's grays off the tone, or any other
  setting off its default) shows a dot after it; its tooltip says what it was ("Changed from Medium: 0.7 mm"), and a
  click puts it back. Right-click a setting to reset it to its preset or to the factory default. When every
  difficulty setting is its preset's again, the difficulty shows the preset again instead of Custom.
- **Why a setting is off is said under it**, once per section ("Only with More detail on faces and subject."), not only
  in its tooltip.
- The paper, what the page is colored on, where its colors come from, and the tone are now buttons side by side
  rather than drop-down lists: every choice in sight, one click away. Left and Right move the choice.

## [1.1.7] - 2026-11-25

### Changed

- **No more message boxes in the way.** What just happened is told in a banner over the page: an export done, with
  **Show in Folder**; a failure, with **Details**. It never takes the keyboard focus; a success goes by itself after a
  few seconds, a warning or an error stays until closed.
- **The status bar** says what the app is doing, with an icon and a progress bar while it works, and keeps the page's
  colors and regions, its size in pixels and on paper, and the zoom on its right.
- **Diagnostics** (View > Diagnostics, Ctrl+Shift+D): a log under the page of each step the app ran, how long it took
  and how it ended, and every warning and error with its details, to copy into a bug report. It never opens by
  itself: the status bar counts its warnings and errors, and a click on the count opens it.

### Fixed

- With no page yet, the inspector's title sat halfway down the panel.

## [1.1.6] - 2026-11-25

### Changed

- **A new workspace.** A toolbar along the top (Open, Generate Preview, which turns into Cancel while a page is made,
  and Export); an activity bar down the left that picks the settings shown: **Source** (the picture and how it is
  read), **Generation** (the difficulty, the regions and their detail), **Colors** (what the page is colored with,
  and its palette) and **Document** (the paper, lines, numbers, legend and export). Each category has its own page,
  in sections; sections of advanced settings start folded. Generate Preview and Export stay at the foot of every page.
- **The palette beside the page**: an inspector on the right lists the page's colors, numbered as on its legend, with
  their names; hover one for its hex and RGB.
- The canvas has its own toolbar: the version shown and the zoom. On a narrow window, what doesn't fit goes into its
  overflow menu.
- The sidebar and the inspector can be resized and folded away (View > Sidebar, Ctrl+B; View > Inspector,
  Ctrl+Alt+B; or click the page already shown in the activity bar). **Focus Mode** (Ctrl+Shift+F) hides every panel
  around the page, and Esc brings them back as they were. The categories have shortcuts too (Ctrl+Shift+1 to 4).
- The layout is kept between runs: the window's size, the panels' sizes and whether they show, the page shown, and
  each page's scroll position and folded sections. **View > Reset Layout** puts it back as it starts.
- On a narrow window the inspector folds away by itself, then the sidebar, and they come back as the window widens.
- The large thumbnail of the picture is gone from the panel: the Source page shows it small, with its file name and
  size.

## [1.1.5] - 2026-11-25

### Changed

- Every slow step runs as a background job with an identity: making the preview, laying it out to print, making a
  page to export, writing the file, and mixing the legend's recipes. A job's result is used only if it is still
  wanted: a page of a picture replaced while it was made is never shown, and a page made while a setting changed is
  kept with the settings it was made with, never taken for one made with the new ones. One job of each kind runs at a
  time; a newer one stops the last first.
- Writing a PDF or a palette file, and mixing a legend's recipes (up to a second with many paints), no longer hold up
  the window: they run in the background too.
- Closing the window while a page is being made closes it at once, rather than waiting for the page; the app finishes
  stopping it as it quits.

## [1.1.4] - 2026-11-25

### Fixed

- Closing the window within moments of opening a large picture raised an error in the background thread that
  identifies the picture (its content hash), and raced the window's teardown. The thread now reports through an
  object that outlives every window.

## [1.1.3] - 2026-11-25

### Added

- **A logo**, "Tesserae": a rounded tile cut into regions, as a page is, in the app's Viridian with a warm ochre. It
  is the app's icon on Windows (Explorer, the taskbar), macOS (the Dock) and Linux, and its window icon, drawn
  simpler at 16 to 24 px so it stays legible there.
- The README banner and the social preview carry the logo, in the app's typeface and accent.

## [1.1.2] - 2026-11-24

### Changed

- A setting that means nothing with the choices made is turned off, and its tooltip says why first: the faces' and
  subject's settings without **More detail on faces and subject**, the line-art settings without **Print line art's
  own ink**, Print lettering up to and Gap beside text without **Print text**, joining the darkest black on a screen
  and the mixing settings without My paints and Mix, which were already off.
- Under the hood, every setting, sliders and choices alike, is described once in a settings registry: where it
  belongs, how advanced it is, what kind of quantity it is, when a change applies, when it means nothing and why, the
  words to search it by, and the key it is kept under. Settings kept by earlier versions load as before.

## [1.1.1] - 2026-11-24

### Added

- **A menu bar** (File, Edit, View, Generate, Help) with every command the app has, and **keyboard shortcuts** for
  them: Open (Ctrl+O), Export (Ctrl+E), Generate Preview (Ctrl+Enter or F5), Cancel (Esc), the Page, Completed and
  Tinted versions (Ctrl+1, 2, 3), Zoom In and Out (Ctrl+plus and minus), Fit Page (Ctrl+0) and 100% (Ctrl+Alt+0).
  On macOS they use Cmd and sit in the system's menu bar. A button and its menu item are the same command, enabled
  and disabled together, and the buttons' tooltips name their shortcuts.
- **View > Appearance**: System (following the system's light or dark setting), Light or Dark, kept between runs.
- **Help > About Tessellatum**: the version, and the licenses of the app and of what it bundles.

### Changed

- Under the hood, the picture, its settings and the page made of them live in one document model, apart from the
  window: the groundwork for undo, projects and recovery.

### Fixed

- In dark mode, the empty canvas's "Open an image" text was nearly black on the dark canvas.

## [1.1.0] - 2026-11-24

### Added

- **A new look, light and dark.** The window has a design of its own: calm neutral surfaces, the page on a slightly
  darker canvas so it reads first, and one accent color, **Viridian**, for what's selected, focused or primary
  (Generate Preview). Every text and control color meets WCAG AA contrast in both themes. The theme follows the
  system's light or dark setting live, as it changes, with no restart.
- The interface uses the **Inter** typeface (SIL Open Font License) and carries a set of **Lucide** icons (ISC
  license) drawn in the theme's colors, both bundled with the app with their licenses.
- The progress percentage sits beside a slim progress bar, rather than on it.
- The build self-test also checks that the app carries its font and icons.

### Fixed

- After the system switched from dark to light while the app was open, the panel's section titles were white on the
  light panel and could hardly be read. Every color now comes from the theme and changes with it.
- Three hard-coded grays (the thumbnail's frame, the Screen hint and the PNG size readout) ignored the theme; the
  readouts were too faint in dark mode.
- Without a saved settings file, the two mixing settings started enabled although colors came from the picture.

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

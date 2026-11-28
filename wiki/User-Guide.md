# User guide

The settings sit in the sidebar on the left, one page per category; **Generate Preview** draws the page with them
(a change to the lines, the numbers or the tone draws it again by itself, a moment after the last change), and
**Export…** draws that
same page at print resolution: the regions, colors and legend the preview shows, its lines smoothed as they are
there, and its numbers placed again, as small as the style lets them print. If a setting changed since the preview,
the export makes the page with it first. Every setting has a tooltip; sizes are on the printed page, on the sheet
**Paper** names; the app keeps them between runs, and **Reset** puts them all back (it asks first).

The window follows your system's light or dark appearance, and changes with it while open.

## The workspace

- **Toolbar** (top): Open, Generate Preview (it turns into Cancel while a page is made) and Export; at its right
  end, the buttons that show or hide the sidebar and the inspector, and Focus Mode.
- **Activity bar** (left edge): picks the settings shown in the sidebar:
  - **Source**: the picture, small, with its file name and size, how large it prints on the paper chosen and at how
    many of its pixels an inch, and, after a preview, whether it was read as line art; then the switches for how it is
    read (line art, faces and subject, text);
  - **Generation**: the difficulty and the settings that shape the regions;
  - **Colors**: what the page is colored with, where its colors come from, the palette and mixing;
  - **Document**: the paper, lines and numbers, their tone, and the legend;
  - **Analysis**: the page measured on paper, its issues, and overlays to see them (see [Analysis](#analysis)).

  Clicking the page already shown folds the sidebar away; clicking any page brings it back.
- **Sidebar**: a search field on top (Ctrl+F; see [Finding a setting](#finding-a-setting)), then the category's
  settings in sections, the everyday ones on top and the rest in one folded
  **Advanced** group at the foot of the page (see [Levels of detail](#levels-of-detail)); click a section's title to
  fold or unfold it, and its **Reset**, shown while something in it is changed, to put just that section back.
  Generate Preview and Export stay at its foot on every page.
- **Canvas** (middle): the page, with its own toolbar above it (the version shown, Compare, and whether the page
  shown is up to date) and the zoom under it. On a narrow window, what doesn't fit goes into a toolbar's » menu. The preview's state reads
  **Up to date** (the page the settings make), **Out of date** (a setting that changes the page moved since; its
  tooltip names them), **Generating…**, or **Cancelled** / **Failed** with **· earlier preview** when the page asked
  for didn't come and the one shown is from before. How soon a change shows:
  - the **legend**'s detail: at once, as it changes nothing but the legend;
  - the **lines, numbers and tone** (Document page): the page is drawn again by itself, 0.4 s after the last change
    (a slider dragged draws once, when it stops), from the regions already found, in about a second;
  - **everything else** makes the page again, which takes longer, so it waits for **Generate Preview**, unless
    Preferences > Performance > **Generate automatically** says otherwise: **When it's quick** (while the picture's last
    page took under 2 s) or **Always** make it by itself a second after the last change;
  - a change back to settings whose page is kept shows that page at once, whatever the setting.
- **Inspector** (right): the page's colors, numbered as on its legend (see [The palette](#the-palette)).
- **Status bar** (bottom): what the app is doing, step by step while a page is made ("Choosing the colors… 42%",
  "Building the regions… 60%"), with a progress bar (one that just moves while a PNG is written, which can't say how
  far along it is). The ✕ beside the sidebar's progress bar cancels, as Cancel does; cancelling is said at once, with
  the step it waits for ("Cancelling: it stops after the current step (building the regions)…"), as a page stops
  between steps. A failure says what to try: a page too large for memory suggests a smaller paper, an easier
  difficulty or a lower resolution; a file that can't be written, whether it is open elsewhere or the disk is full.
  And, once a setting that changes the page moves, that the preview is out of
  date and which settings changed (the legend's detail and the export's options don't count; put them back and it is
  ready again); on its right, the page's colors
  and regions, its size in pixels and on paper, the zoom, and the count of warnings and errors in the diagnostics
  (click it to open them).
- **Notifications** (over the page): what just happened, such as an export done, with **Show in Folder**, or a failure,
  with **Details**. They never take the keyboard focus: a success goes by itself, a warning or an error stays until
  you close it.
- **Diagnostics** (View > Diagnostics, Ctrl+Shift+D): a log, under the page, of each step the app ran, how long it took
  and how it ended (a cancelled one, how long it took to stop), and every warning and error. A page made says how many
  of its stages were kept from before, and its details give each stage's time. Select one to see its details; **Copy** puts it on the clipboard,
  for a bug report. It never opens by itself.

Drag the borders between the panels to resize them. **View > Sidebar** (Ctrl+B) and **View > Inspector** (Ctrl+Alt+B)
show or hide them; **Focus Mode** (Ctrl+Shift+F) hides everything around the page, saying how to leave it, and Esc,
the same shortcut, or **Exit Focus Mode** on the bar above the page brings it all back as it was. On a narrow window the inspector folds away by itself, then the sidebar, and they come
back as the window widens. The layout is kept between runs; **View > Reset Layout** puts it back as it starts.

## The views

Every preview is drawn in three versions, and the bar above it switches between them and the picture itself, keeping
the zoom and the scroll:

- **Page**: the coloring page as it prints.
- **Completed**: the page painted in. Every region is filled in its legend color, which covers the lines and numbers,
  as paint covers their grays; what the page prints of the picture itself (line art's ink, the marks in a face, the
  letters of a sign) shows on top, and paper the page leaves bare stays white.
- **Tinted**: the page under a faint wash of its colors, each 40% of the way from white, with the lines, numbers and
  ink showing through as through a translucent paint: a guide to paint from.
- **Original**: the picture itself, in the page's place, so a point of the page is the same point of the picture. It
  is shown, never exported.

### Comparing

**Compare** above the page (also View > Compare) shows two views at once, each named in a corner:

- **Side by Side**: two canvases, exactly as wide as each other, that zoom and move together; zooming or dragging
  either moves both, so a point of the page is at the same place in each.
- **Split Left and Right** or **Top and Bottom**: the view compared with past a divider across the page, in the same
  place; drag the divider to move it.

Compare **With** any version, the **Original**, or the **Previous version**: the page made before this one, of the
same picture, in the version shown, named by what changed since ("Previous version (before: Colors)"). It comes from
the pages kept for undo (Preferences > Performance > **Memory for pages made**), so it may be gone; it is never made
again. Comparing makes nothing: every view is of the page shown, or kept.

### Color vision

**View > Color Vision** shows the canvas as seen with another color vision, to check that two colors a painter must
tell apart still look different to them:

- **Protanopia** and **Deuteranopia**: without red- or green-sensitive cones (each about 1 in 100 men), where reds
  and greens of a similar lightness look alike.
- **Tritanopia**: without blue-sensitive cones (rare).
- **Grayscale**: no color at all, whether the page reads in light and dark alone.

The page, the view compared with and the legend are shown simulated (Machado, Oliveira and Fernandes, 2009, at full
severity: the strongest form of each); the Page view, black on white, looks the same. It is on the screen only: the
inspector's swatches and every export keep the page's colors. While it is on, the bar under the page names it; click
it, or pick **Normal**, to see the page's own colors.

## The palette

The inspector lists the page's colors by their legend numbers, with their names and how much of the page each
paints. That share is measured in the background once the page is made, and fills in a moment later.

- **Sort** by number, coverage, lightness, hue (the grays first) or name. Sorting changes only the order shown: never
  a number, a region's color or the export.
- **Grid** shows swatches with their numbers; click it again for the list.
- **Find** a color by its number ("12") or a word of its name.
- Click a color (or move to it with the arrow keys) to see it in full: its hex and RGB, its recipe if it is mixed, its
  share of the page and its area on paper, and its number of regions. Its regions are highlighted on the canvas, tinted
  and outlined in the accent; **Isolate** dims the rest. The highlight is drawn over the page, never into an export.
  Esc lets go of it.
- **Copy** the hex (Ctrl+C, or the Copy button), or the RGB or name from the button's menu or a color's right-click
  menu.

### A region

Click a region on the canvas, in any view, to pick it: its color is picked in the palette, the region is outlined, and
the inspector shows, under the color, the region measured on paper: its area in mm², its size, the widest brush that
fits in it, and how its number is written (inside it, outside on a leader line, or cramped where lines may cross it),
at what size in points and on how many other sides. Picking another color lets it go; so does a click off the regions.

From the keyboard: press Enter on a color to list its regions, largest first. The first is picked and centered on the
canvas; Up and Down go through the rest, and Esc goes back to the palette. A pick is kept when the page is made again
with the same regions (a change to the lines or the numbers), and let go when the regions change.

## Analysis

The Analysis page of the sidebar (View > Analysis, Ctrl+Shift+5, or the chart button at the foot of the activity bar)
measures the page shown on paper, as the project's own quality benchmark does. It measures in the background while
the page is open, once a page:

- **Measures**: the colors and regions, regions per cm², the smallest region, the narrowest region (the widest brush
  that fits in it), the share of the page thinner than the brush (leaving out corners' points, which a brush's tip
  fills), the numbers written on leader lines, cramped, or smaller than the page's smallest size, and the regions with
  no number.
- **Issues**: the regions behind those measures, largest first: regions nowhere as wide as the brush, smaller than
  the settings keep anywhere, with no number, with a cramped number, a number under the smallest size, or a number on
  a leader line. Choose an issue to highlight all its regions on the canvas; unfold it and choose a region (or move to
  it with the arrow keys) to pick it, centered, with its measures in the inspector.
- **Show on the canvas**: the parts thinner than the brush, the small regions and the numbers on leader lines, each
  drawn over the page in its own color. They are off until asked for, and never reach an export.

## Menus and shortcuts

Every command is in the menu bar, and the most used have a shortcut (Cmd in place of Ctrl on macOS). **Help >
Keyboard Shortcuts** (Ctrl+/) lists them all as your platform writes them, with the keys that work in one place (the
canvas, the palette, a field), and finds one by its name or its keys:

| Command | Shortcut |
|---|---|
| File > Open (a picture or a project) | Ctrl+O |
| File > Save Project, Save Project As | Ctrl+S, Ctrl+Shift+S |
| File > Export | Ctrl+E |
| File > Close | Ctrl+W (Windows: also Ctrl+F4) |
| File > Quit | Ctrl+Q on macOS and Linux (Windows: Alt+F4) |
| Edit > Undo, Redo | Ctrl+Z; Ctrl+Y or Ctrl+Shift+Z |
| Edit > Paste Picture | Ctrl+V |
| Edit > Find Setting | Ctrl+F |
| Edit > Preferences | Ctrl+, (macOS: in the app menu, Cmd+,) |
| Generate > Generate Preview | Ctrl+Enter, F5 |
| Generate > Cancel | Esc |
| View > Page, Completed, Tinted, Original | Ctrl+1, Ctrl+2, Ctrl+3, Ctrl+4 |
| View > Zoom In, Zoom Out | Ctrl+plus, Ctrl+minus |
| View > Fit Page, 100% | Ctrl+0, Ctrl+Alt+0 |
| View > Rulers, Grid | Ctrl+R, Ctrl+' |
| View > Source, Generation, Colors, Document Settings; Analysis | Ctrl+Shift+1 to 4; Ctrl+Shift+5 |
| View > Sidebar, Inspector | Ctrl+B, Ctrl+Alt+B |
| View > Focus Mode | Ctrl+Shift+F (Esc leaves it) |
| View > Diagnostics | Ctrl+Shift+D |
| Help > Keyboard Shortcuts | Ctrl+/ |

No command is a key that types, so typing in a field never triggers one. Esc is Cancel's while a page is being made,
except in a field or a list that uses it: there it clears the search, puts back the value being typed, or lets go of
the color.

**View > Appearance** picks System (the window follows the system's light or dark setting as it changes), Light or
Dark, and the choice is kept. **Edit > Undo** takes back the last change to the settings or the picture, named
for what it did ("Undo Brush width", "Undo Difficulty: Hard", "Undo Reset Detail"), and **Redo** puts it back: a slider
dragged is one step, a preset or a reset is one step however many settings it moves, and opening another picture is
one. The zoom, the panels and the version shown are never undone. A change made after an undo drops what could have
been redone. The pages made are kept, so an undo or a redo shows its page at once, as does putting a setting back
by hand. A page no longer kept (memory is limited, see Preferences) is made again in the background, with the page shown
kept and labeled out of date until it is ready. Settings that never had a page wait for Generate Preview, as any other
change does: the status bar says the preview is out of date. Whether the preview is up to date is apart from whether the settings have changed since they were last saved. **Edit > Reset All Settings** is the Difficulty section's Reset, and asks first; **Help > About Tessellatum** shows
the version and the licenses.

## Keyboard and screen readers

Everything works from the keyboard. **Tab** goes round the window's areas in order: the toolbar's Generate button,
the activity bar, the sidebar, the canvas (its views, the page, the zoom), the inspector, and the status bar's
Diagnostics button. **Shift+Tab** goes the other way. A row of which one is down (the activity bar, the views
above the page) is one stop, and the arrow keys move along it. A control reached in a scrolled page scrolls into
view. The toolbar's other buttons aren't stops: their commands are in the menu bar (**Alt**, or **F10**) and have
shortcuts. In the canvas, the arrow keys scroll the page, and on the palette **Enter** lists a color's regions.

Every control has a name and a description for a screen reader (Narrator, NVDA, VoiceOver, Orca). A color is read
with its number, name, hex value, share of the page and regions. Nothing is said by color alone:
- the preview's state is a word beside its icon;
- a changed setting is marked by a sign, read as "Reset … to …";
- a notification says its level.

**Preferences > Appearance > Motion** turns animations off. The interface has none of its own, so it changes
nothing yet.

## Language and numbers

The interface is in English; it is ready to be translated, and uses the system's language when a translation of it
ships. Numbers follow the system's format, typed and shown: in German a brush is "1,5 mm" and a page
"1.448 × 1.086 px". A right-to-left language mirrors the window; the page itself is never mirrored. What the page
prints (the legend's headings and color names) stays English.

## Getting a picture in

While no picture is open, the canvas shows the welcome page:
- **Open Picture…** and **Open Project…**;
- four example pictures to try (public domain or CC0): one click opens it and makes its page;
- the recent projects and pictures. One no longer where it was is greyed, with **Remove** to take it off the list.

A picture or a project can also be dropped anywhere on the window, or pasted with **Edit > Paste Picture** (Ctrl+V): a
picture copied from another app, or a file (or its path) copied. A picture dropped or pasted into the work replaces the
one there as one step to undo, and the settings stay; a project asks first if the work has unsaved changes. A file that
can't be opened says why, and the work stays as it was. A picture opens up to 200 megapixels, in a file up to 512 MB:
what opens can always be saved in a project and recovered. Ctrl+V in a text field pastes text there, as always.

## Projects

**File > Save Project** keeps the work as a `.tessellatum` file, to open again later, on this computer or another:
- the picture itself, its file as it was opened, byte for byte, so the project doesn't need the picture to stay where it
  was;
- every setting, the paints listed and the export's options;
- a small thumbnail.

Pages aren't kept in it. Opening a project makes its page again from the picture and the settings: on the computer
it was saved on, with the same version, that is the page that was saved; on another platform (Windows, macOS, Linux)
it can differ slightly ([why](Quality-and-Performance#platforms-and-scaling)). To keep a page exactly, export it.

**File > Open…** opens a project or a picture. A picture opened into the work replaces the one there as one step to
undo, and the settings stay. A project replaces the work, so if the work has changes that aren't saved, it asks first
whether to save them. Closing the picture (**File > Close**) and quitting ask the same. **Open Recent** lists the
projects opened or saved last that are still there. The window's title names the project, with a mark while it has
unsaved changes.

Saving is safe: the project is written beside its file, read back and checked, and only then put in its place. A save
that fails (a full disk, a folder that doesn't allow it) says why and leaves the file saved before as it was. A project
that can't be read (damaged, made by a newer Tessellatum, or not a project at all) says why, and the work open stays as
it was.

### Recovery

While the work has changes that aren't saved, Tessellatum keeps a copy of it every 2 minutes (Preferences > Workspace),
in its own folder, never over the project. If it ends without closing properly (a crash, a power cut), the next start
lists what was kept, with a thumbnail and when it was kept:

- **Restore** opens it as it was. It isn't saved, and says where it came from; Save Project keeps it.
- **Discard** removes it for good.
- **Keep for Later** asks again at the next start.

Saving, closing the picture and quitting remove the copy. A second window open at the same time never offers the first
one's work. Copies left go after 14 days.

## Zoom

The preview zooms about the point under the cursor, with the mouse wheel, Ctrl+wheel, a touchpad's pinch, Ctrl+plus
and Ctrl+minus, or the buttons under it. The zoom's field takes a zoom typed in (150, or 37.5%) or picked from its
list, where **100%** (Ctrl+Alt+0) is a pixel of the page as it prints on a pixel of the screen and **Print size** the
page as large as on paper: approximate, by the size the screen reports. **Fit** (Ctrl+0) shows the whole page. A
window resized keeps the same point of the page in its middle.

To move the page, drag it, drag with the middle button, or hold Space and drag (Space only while the canvas has the
keyboard: in a text field it types a space). A click without a drag picks the point under it.

**View > Rulers**, **Grid** and **Page Bounds** are hidden until asked for, and kept as they were left:

- the rulers measure on paper, in millimeters or inches as Preferences > Units says, from the page's top left corner;
- the grid is a line every centimeter (or inch) on paper;
- the bounds are the sheet the page prints on and its margins, the page in the middle of them, as the export lays it
  out; the legend, which prints on a sheet of its own, moves below it.

Further in than the preview's own pixels, the zoom shows the page as it prints: the
export's own drawing, with its numbers where and as small as they print. It is drawn in the background the first time
and kept, so the export is drawn from it without laying the page out again. A new preview of the same picture keeps the zoom and the
place, to compare settings there.

## Paper

**Paper** (A4 by default, or Letter where the system measures in US units, until you choose one; A3, A2 or A1; the US sizes Letter, Legal and Tabloid; the usual kit canvases, 30 × 40,
40 × 50, 50 × 65 and 60 × 75 cm and 12 × 16, 16 × 20 and 18 × 24 in; or **Custom**, at the top of the region
settings, each paper's size in its tooltip) is the sheet the page is made for. With Custom, **Width** and **Height**
under it set the sheet's size, either way round, from 100 mm to 1,189 mm, shown in millimeters or inches as the
other lengths are (Preferences > Units); a size of whole inches is a sheet in inches. Every size
on paper -- the brush, the smallest region, the lines, the numbers -- applies on that sheet, so a larger one prints the
picture larger and makes a finer page of it: more and smaller regions, each still as wide as the brush. The page is
made at more pixels on a larger sheet, the picture scaled up where it has fewer, so that every region still has room
for the brush and its number; scaling up adds no detail the picture doesn't have, so a larger picture still makes a
better page on a larger sheet. A page takes longer to make there: a Medium preview of a 2048 px photograph takes 1.5
to 2 times as long on A3 and 2.3 to 4 times on A2 (the benchmark's lion: 2 s on A4, 4 s on A3, 9 s on A2 and 29 s on
A1; at Realistic, 7 s, 15 s, 34 s and 110 s). Letter takes about as long as A4, Legal a quarter longer, and Tabloid as long as A3.
A1 needs a lot of memory: a page of the lion peaks at 5.8 GB at Medium and 8.7 GB at Realistic (3.4 GB on A2 at
Realistic), so it wants a computer with 16 GB or more, and its PNG is at most 507 dpi (16,384 px along a side). The
PDF's sheets are this paper, the page landscape when the picture is wider than tall and the legend on a portrait
sheet of its own; an A1, canvas or custom page larger than A4 has its legend on A4 instead (Letter for a size in
inches), a sheet to keep beside the canvas rather than a poster. A canvas prints at a print shop, its page inside
10 mm margins; a size smaller than A4 keeps its legend on its own sheets.
A legend always prints at its own sizes, its text 7 pt and more and its swatches 12 mm: one too long for its sheet
(64 colors with their mixes on Letter) goes on to another, under a heading that says what it continues. A PNG
records the resolution that prints it on this paper.

## Coloring on

**Coloring on** (under Paper) says what the page will be colored with.
- **Paper and paint**, the default: paper and paints or pencils can't go darker than their own black, so near-blacks a
  screen shows apart look alike there. A page keeps its colors apart as printed too (see **Darkest black on paper**),
  and gives the colors that frees to the rest of the picture.
- **Screen**, for coloring in an app such as Procreate, where every color is picked exactly by eyedropper or fill:
  there is no darkest black, and the color margin is at most 2.5 ΔE00, a little over what an eye tells apart side by
  side, so more of the picture's own shades stay on the legend.

## Colors from

**Colors from** (under Regions) says where the page's colors come from.
- **The picture**, the default: the colors found in it.
- **My paints**: the paints or pencils you own. The page takes at most as many of them as **Colors** asks for, the
  ones that paint the picture best: each pixel is painted in the nearest of them, and the ones chosen bring the
  picture's pixels nearest, by CIEDE2000, the faces' and the subject's counting twice. Two closer than the color
  margin are never both used (on paper, as printed), so a box of 24 pencils gives at most 24 colors, and often fewer.
  Every legend color is exactly one of your paints, called by its own name; a paint with no name gets the everyday
  name of its color.

**Paints…** opens the list of your paints: a row each, its swatch, its hex code, its name and a ✕ to delete it.
- **Add color…** opens a color picker: pick on the wheel, type a hex code or RGB values, or use **Pick Screen Color**,
  an eyedropper, to take the color from anywhere on the screen (a photo of your swatches, or the picture in the
  preview). The new paint is named with its color's everyday name, selected so that typing replaces it with the
  paint's own. A color already listed is not added twice.
- Click a swatch to pick its color again (a name the list gave it follows the new color; one you typed stays);
  double-click a name to change it.
- **Start from a kit…** offers a few generic sets to start from: **Basic acrylics (12)** (titanium white, mars black,
  lemon and cadmium yellow, yellow ochre, cadmium red, alizarin crimson, ultramarine and phthalo blue, sap green,
  burnt sienna, burnt umber), **Primaries (5)** (white, black, a yellow, a red and a blue: the box mixing is made
  for), **Colored pencils (24)**, **Earth palette (8)** and **Grays (6)** (black to white in even steps of
  lightness). With paints listed, it asks whether the kit **replaces** them (the default) or is **added**. A kit's
  colors are typical of its pigments, not any maker's: real tubes and pencils differ a little, so replace a color with
  your own (picked from a photo of your swatches) where it matters.
- **Load palette file…** adds the colors of a GIMP (`.gpl`), Adobe Swatch Exchange (`.ase`) or Procreate
  (`.swatches`) palette, or a text file, so a palette exported from the app reads back.
- **Edit as text** shows the list as lines of text, to paste a list or type one: a paint a line, a hex code
  (`#5A6B2F`, `5A6B2F`, `#fff`) or three numbers from 0 to 255 (`90 107 47`, `90, 107, 47`), the rest of the line its
  name, either side of the color. Lines starting with `#` and text or `//` are comments; a line with no color is
  pointed out, not guessed at. Clicking it again reads the text back into the list.

Up to 256 paints; the list is kept for next time, and switching back to The picture keeps it. Choosing My paints with
none listed opens the list.

**Mix colors from my paints** (under Colors from, with My paints) lets the page also take colors mixed from two of
your paints, in 1 + 1, 2 + 1 or 3 + 1 parts, where the picture has colors none of your paints comes close to. Your
paints come first, chosen as without mixing; a mix is added only if it earns its place:
- **Mix if closer by** (3 ΔE00 by default): the parts of the picture it would paint come out at least this much
  closer to the picture, on average, than with the colors already taken; about 2.3 is the smallest difference an eye
  sees side by side, so a mix is never added for a change no one would see.
- **and it paints at least** (1% by default): those parts are at least this share of the picture, the faces and the
  subject counting twice, so nothing is mixed for a speck.

There is no fixed number of mixes: a box that covers the picture well gets few or none, a few paints get many, up to
Colors in all. A mix is never within the color margin of a paint you have, nor of another color on the legend. The
legend then has two sections, **Your paints** and **Mixed colors** (numbered on from the paints), each mix with its
everyday name and its recipe: "1 Titanium white", "1 Ivory black", whatever the Legend setting (a mix can't be painted
without it). A palette file names a mix with its recipe too: "12 gray (1 Ivory black + 1 Titanium white)". The mixes'
colors are predicted, by the same model of paint as the recipes (see Legend): your own paints may mix a little
differently, and the painted versions of the page show the predicted colors.

## Preferences

**Edit > Preferences** (on macOS, in the Tessellatum menu) holds how the app looks and works on this computer. A
preference is never part of a page's settings: it isn't undone with Edit > Undo, Reset All Settings leaves it as it is,
and it is kept between runs. Each change applies at once.

| Page | Preference |
|---|---|
| Appearance | **Theme** (System, Light, Dark) and **Motion** (System, Reduced, Full), as in View > Appearance. **Interface scale**, Automatic (the system's) or 90% to 200%: applied when Tessellatum next starts, as the dialog says. The page's zoom is apart from it, so 100% stays a pixel of the page to a pixel of the screen. **Canvas background**: the theme's, a neutral gray to judge the page's colors against, white or black. |
| Units | **Lengths in** millimeters or inches: how lengths and areas are shown and typed. The page's values are kept in millimeters, so switching never changes them. |
| Workspace | **Reset Layout**, as in View. **Keep for recovery**: how often work with unsaved changes is kept to be recovered after a crash, every 1, 2 (the default), 5 or 10 minutes, or Off (see [Recovery](#recovery)). |
| Performance | **Memory for pages made**: how much memory the pages made may take, kept so that Undo, Redo and a setting put back show their page at once (Automatic: a quarter of the memory, at most 2 GB; the page shown and the ones Undo and Redo go to are always kept). **Results kept**: how many pages' intermediate results are kept, so that trying one setting after another makes the next preview faster (Automatic keeps two, a few tens of MB each). **Clear Now** frees both. **Redraw lines and numbers**: how soon a change to the lines, numbers or tone draws the preview again by itself (after 0.2, 0.4 or 1 s), or **Only with Generate**. **Generate automatically**: whether a change to the colors, the difficulty, the paper or what is done with line art, faces and text makes the page by itself, a second after the last change: **Off** (the default; it waits for Generate Preview), **When it's quick** (while the picture's last page took under 2 s), or **Always**. |
| Shortcuts | The keyboard reference, as Help > Keyboard Shortcuts shows it. |

## Difficulty

Beginner, Easy, Medium (the default), Hard and Realistic fill in the settings below them; moving any of them picks
**Custom**. Under the choice, a line says what the level makes: about how many regions it gives on the paper chosen
(measured on the benchmark's pictures on A4, and in proportion to the printable area on the other papers; a busy picture
gives more, a plain one fewer), how many colors, how large and how wide its regions, and its corners. For Custom it
names the level it was moved from and what was changed, e.g. "Hard (modified, 2 changes): Colors 26, Smoothing 0.9."
The five levels differ in four settings:

| Level | Colors | Smallest region | Brush | Smoothing |
|---|---|---|---|---|
| Beginner | 16 | 10 mm² | 1.0 mm | 1.2 |
| Easy | 20 | 8 mm² | 0.8 mm | 1.0 |
| Medium | 24 | 6 mm² | 0.7 mm | 0.9 |
| Hard | 28 | 4 mm² | 0.6 mm | 0.8 |
| Realistic | 32 | 2 mm² | 0.5 mm | 0.7 |

Every level keeps colors 4 ΔE00 apart, on screen and as printed on paper whose darkest black is L* 25, settles ragged edges three times as far as the brush alone would (with the edge
hold at 1 ΔE00, which keeps them closest to the picture's), lets a region in a face or the subject be a quarter of the
smallest, and rounds corners to the brush. Region sizes are areas on paper, so a page's regions are the same size on
paper whatever its pixel count, and a long, narrow picture, which prints smaller, gets fewer regions rather than
smaller ones.

**Presets of your own.** The **Presets** menu beside the difficulty:

- **Back to Hard** (the level it was changed from): its settings again, as one step to undo.
- **Save as Preset…**: the difficulty's settings as they are, under a name of your own, listed after the built-in
  levels from then on. A built-in level's name can't be taken; saving under a name already used asks first, as does
  one that differs only in capitals (two presets are never "Fur" and "fur").
- **Rename Preset…** and **Delete Preset…**, for one of your own while it is chosen. Deleting it leaves the settings
  as they are, as Custom.

A preset, built-in or your own, sets only the difficulty's settings, as one step to undo: the paper, what the page is
colored on, the paints and the page's style stay as they are. Your presets are kept in `presets.json` in the app's
preferences folder; a file that can't be read is left out (the log says why) and kept as `presets.json.bak` when a
preset is next saved. A page set up with a preset since deleted keeps its settings, as Custom; so does a project
or an undo step of a preset saved again since with other values (or one opened on a computer where that preset has
other values): the settings are the ones it was made with, shown as Custom from that preset.

## Settings

The sidebar shows them by category (see [The workspace](#the-workspace)); below they are described by what they
shape.

### Finding a setting

Type in the search field at the top of the sidebar (Ctrl+F puts the focus there, from anywhere) to find a setting by
its name ("brush"), by a word it may be called by ("outline" finds Line width, "pupils" Dark mark contrast), by what
its tooltip says, or by its page or section. Every word typed must match; the best matches come first, each with where
it is and what it is set to. Up and Down move through them; Enter (or a click) goes to the setting: its page opens, its
section unfolds (the Advanced group and its part of it too), the page scrolls to it, and it is outlined for a moment with the focus on
it, to change from the keyboard at once. Esc clears the search.

### Levels of detail

Each page shows its settings at three levels. On top, the choices most pages need (the difficulty, the paper, where
the colors come from), then the settings worth a look (the detail, the palette, the lines and numbers); last, folded
away, one **Advanced** group with the rest, in parts that each fold on their own (folded at first, so opening the
group lists them by name):

| Page | In Advanced |
|---|---|
| Generation | Faces and subject (dark marks, face tones), Line art, Text, Edges, Corners, Resolution |
| Colors | Mixing |
| Document | Tone (the three grays), Lines (line smoothing), Numbers (leaders, gap beside text, necks) |

Open a picture, pick a difficulty, generate and export: none of it takes an Advanced setting. Every section, the
Advanced groups and their parts included, shows **Reset** in its header while something under it is changed: it puts the sliders
back to their preset (the grays to the tone, the rest to their default) and the choices to their defaults, and leaves
the other sections alone. The paints you list stay listed. The folded and unfolded sections are kept between runs.

**Editing a setting.** Drag its slider, or type an exact value in the box beside it and press Enter: "1.2", "1.2 mm",
or "1,2" where a comma is the decimal mark. A value out of range is refused, with the range said under the box; Esc
puts the box back. Up and Down step the value, Page Up and Page Down ten steps, in the box as on the slider. The grays
are typed as their lightness, L* 0 (black) to 100 (white).

**What you changed.** A setting moved off its preset shows a dot after it (the grays: off the tone; the rest: off the
default). Hover the dot to see what it was, click it to put it back, or right-click the setting to reset it to its
preset or to the factory default. When every difficulty setting is its preset's again, the difficulty shows that
preset again. A setting that does nothing with the other choices as they are is turned off, and why is said under it.

**Regions and colors**
- **Colors**: how many colors to look for. Colors closer than the color margin are merged, so a page can keep fewer.
- **Color margin**: how different every two palette colors must be (CIEDE2000; about 1 is the smallest difference an
  eye sees side by side). Lower keeps more, subtler colors.
- **Darkest black on paper**: how dark the printer and the paints or pencils can go, as a lightness (L* 25 by default,
  0 to 40). Colors darker than about that look alike on paper, so two of them must also keep the color margin as
  printed; the slots that frees go to the rest of the picture, where its colors are. The faces and the subject keep
  their dark shades. About 20 for a good laser or pigment inkjet and soft pencils, 30 for a weak printer or cheap
  paper; 0 judges the colors as the screen shows them, as Screen does.
- **Smallest region**: the smallest area a region may have on paper (2 to 500 mm²); smaller ones merge into a neighbor.
- **Brush width**: the narrowest any part of a region may be (0.5 to 6 mm); narrower parts are given to the region
  beside them.
- **Pixels across brush**: how finely a page is made (4 by default, 2 to 8). A 3:2 picture's page is about 1,100 px
  along its long edge at Beginner and 2,200 at Realistic, never fewer than 1,100, nor more than the picture has or
  than 300 dpi. Not part of the difficulty: moving it keeps the preset.
- **Mix if closer by** and **and it paints at least**: when a color mixed from your paints earns its place, with
  Mix colors from my paints (see **Colors from**): 3 ΔE00 closer where it paints (1 to 10) over at least 1% of the
  picture (0.1 to 10%). Not part of the difficulty.
- **Sharpest corner**: corners down to this angle keep their points rather than being rounded to the brush. 180°, the
  default, rounds every corner; 20° keeps a triangle's points. The spikes of fur and foliage are rounded all the same.
- **Corner contrast**: how plainly the picture must show a corner's point for it to be kept.
- **Smoothing**: how much the picture is smoothed, keeping its edges, before its colors are found.
- **Edge settling** and **Edge hold**: how far, and how firmly, the vote that smooths ragged edges in fur, foliage and
  stone reaches.

**Faces and subject**
- **Face and subject detail**: how many times smaller a region may be in the faces and subject found.
- **Dark mark contrast** and **Dark mark length**: how dark and how long a thin mark in a face (a pupil, a lip line)
  must be to be printed rather than painted over.
- **Face tone step**: a region in a face joins its neighbor while that changes its color by less than this, so skin and
  fur paint in a few large tones.

**Lines and numbers**
- **Line width**: 0.1 to 1 mm (0.2 mm by default).
- **Tone**: Light, Medium or Dark sets the page's three grays, shown as a lightness (L*, 0 black to 100 white) on
  their own sliders: **Line gray**, **Number gray** for the numbers inside their regions, and **Leader gray** for the
  numbers written outside their regions, their leaders and the leaders' dots. Medium prints them at 34, 34 and 70: the
  numbers inside their regions in the lines' gray, which a number's shape tells from a line, and a number with a leader
  in a lighter gray of its own, told from them at a glance. Gray rather than black, the numbers read as the page's
  apparatus rather than as writing in the picture. Light and Dark move all three by 15; Light is for pale paints, under
  which darker lines would show. Moving any of the three picks Custom.
- **Legend**: what the legend writes beside each color's swatch, to find, buy or mix the color by. **Name, hex and
  RGB** (the default) gives its everyday name ("dusty rose", "olive green") in 8 pt and under it, in 7 pt, its hex code
  and RGB values (`#C0737A  RGB 192 115 122`): exactly the color the page is painted with, to type into a painting app
  or match a paint or pencil against. **Name** gives the name alone, **Number only** the numbered swatches as before.
  The name is the nearest by CIEDE2000 of about 800 names from the xkcd color survey (CC0), every color on a legend a
  different one. With the details, four colors fit across a portrait A4 sheet, so 64 colors take 243 mm of its 277.
  **Name and recipe** gives the name and under it how to mix the color from the paints you list (**Paints…**, under
  Regions; see **Colors from**): "3 Titanium white", "1 Ultramarine", one to three paints in whole parts adding up to
  10 at most, the mix nearest the color by CIEDE2000, a second or third paint only when it brings the mix at least 1
  ΔE00 nearer. A color no mix comes within 5 ΔE00 of says "closest mix, 7 dE00 off". The page keeps its own colors
  (with Colors from: The picture), so this is how to get a picture's colors from a small box of paints; with My paints
  every recipe is a single paint, and the colors mixed from them (Mix colors from my paints) are listed in a section
  of their own with theirs. Mixing is modeled, not measured: your own tubes of the same colors mix a little
  differently, so a recipe is where to start. With no paints listed it shows hex and RGB instead. Recipes for the
  picture's own colors are headed "Recipes from your paints list: mix each in the parts given", so they don't read as
  the paints the page was made of.
  Changing it redraws the preview's legend at once and applies to the next export, without making the page again.
- **Leader width**: 0.1 mm by default, half the default line, so a leader doesn't read as a boundary. Numbers keep as
  clear of a leader as of a line however thin it is drawn.
- **Line smoothing**: how far along a line its pixel staircase is smoothed.
- **Smallest number**: 3 pt by default, down to 1 pt. Smaller numbers fit inside more small regions, so fewer need a
  leader; under about 3 pt they are for coloring on screen, zoomed in, rather than on paper. Every export prints
  them as small as this asks, and so does the zoom past the preview's own pixels; the preview itself draws no number
  under 10 px. Where a cluster of tiny regions leaves a number no room at all, not even on a leader, an export prints
  it smaller inside its region, down to half this size, rather than across its lines.
- **Leader reach**: how far outside a small region its number may go, on a leader line.
- **Gap beside text**: how far numbers keep from lettering.
- **Split at necks under**: 0.35 mm by default. A region whose paper narrows to less than this between its lines, or
  that its lines close off, reads as two regions there, so each side gets its number. Sides smaller than the smallest
  region don't count; 0 numbers again only the sides the lines close off.

**Line art and text**
- **Paint over ink up to**: on a cartoon or comic, the paint goes over ink lines up to this wide (hatching, fine
  strokes), never over bolder ones.
- **Close line gaps up to**: gaps in ink lines up to this wide are closed, so two areas' paint doesn't run together.
- **Print lettering up to**: lines of text up to this tall are printed; taller lettering is painted as shapes.

**Picture handling** (each on by default; turn one off when it gets a picture wrong)
- **Print line art's own ink**: a cartoon or comic is drawn from its ink lines, which are printed, and the areas they
  enclose; off, it is drawn from its colors like a photograph.
- **More detail on faces and subject**: on a photograph or painting, the faces and subject found get smaller regions,
  a face is painted in a few tones, and its thin dark marks are printed.
- **Print text**: the letters of signs, titles and captions found are printed, their ground left bare, and no number
  goes on them.

A setting that means nothing with the choices made is turned off, and its tooltip says why first: the darkest
black when coloring on a screen, the mixing settings without My paints and Mix, and the faces', line art's and text's
own settings while their switch above is off.

**Abort** stops a page in progress.

## Export

**Export…** (Ctrl+E) opens the export dialog: the options below, a thumbnail of what will be written, and a line on
it (a PNG's pixels and dpi, its size on paper). It warns without stopping you: a PNG under 150 dpi prints soft, a
Procreate palette of more than 30 colors is written as several files, and numbers under 3 pt are hard to read on
paper. If you changed a setting that shapes the page since the preview, it says which, and asks which page to write:
**Generate and export** (the default) makes the page again with the settings as they are now; **Export the previewed
page** writes the page the preview shows, as it was made. The legend's detail and the export's own options never make
the page again. Press **Export…** (or Enter) to pick the file: it is named for the picture and suggested beside it.
If writing it fails, the notification's **Retry** writes the same file again. The options are kept between runs.

Every export is the preview's page laid out once to print, at 300 dpi on its paper -- on A4, a long edge of 2,244 to
3,272 px depending on the picture's shape -- however many pixels the picture has: its regions are the preview's, its
lines and ink redrawn there, its numbers placed there once, as small as **Smallest number** asks. The PDF and the PNG
are both drawn from that layout, so they number the page the same way, each number at the same place on paper.
**Version** picks which of the three versions to export, in either format; the legend comes with each.

- **PDF**: two sheets of vector art, of the page's paper. The page fills the printable area inside 10 mm margins, centered, on a
  landscape sheet when the picture is wider than tall. Its lines are paths of the width asked for and its numbers text,
  so a printer draws them at its own resolution. Line art's ink, the marks in a face and the letters of signs are
  filled outlines, smoothed off the pixel staircase without joining or breaking anything. The painted versions fill
  each region along a path made of the page's own lines. Nothing on the sheet is an image. The legend gets a portrait
  sheet of its own (A4 for an A1 page; more than one where a long legend doesn't fit), its swatches 12 mm squares in
  their exact colors, their names and codes as text a PDF reader can
  search and copy.
- **PNG**: the page above its legend in one image: the PDF's page drawn in pixels, at the **Resolution** asked
  for, which the file records, so it prints as large as the PDF's page.
- **Palette**: the legend's colors as a palette file, for coloring the page in a painting app with exactly its
  colors; the Colors page and the export dialog suggest it when **Coloring on** is Screen. Each color is named by its number and its name on the
  legend ("7 dusty rose"), in the legend's order. **Palette** picks the format:
  - **Procreate** (`.swatches`): a zip holding `Swatches.json`, the colors in HSB, as the MIT-licensed
    procreate-swatches library writes them (Procreate publishes no spec). Procreate keeps the colors but not their
    names, and a palette holds 30 at most, so a larger legend is saved as several files of 30, named for the colors
    each holds ("page 1-30.swatches", "page 31-60.swatches"). Drop them on Procreate's Palettes tab.
  - **Adobe Swatch Exchange** (`.ase`): read by Photoshop, Illustrator, InDesign, Affinity's apps, Procreate and
    others; the colors in RGB, in a group named after the file.
  - **GIMP palette** (`.gpl`): read by GIMP, Krita, Inkscape, Aseprite and others.

  The palette is the preview's legend, saved at once; if the settings changed since the preview, the dialog asks,
  as for the other formats. A palette written as several files asks before replacing any of them that exist.

**Resolution** is the PNG's: in **dpi** on its paper (the buttons set 150, 300 or 600; any other can be typed) or in
**px** along the page's long edge, the other worked out from the paper and the picture's shape and shown below it
with the file's pixel size. 600 dpi is the default: sharp in print and zoomed in on a screen, and an A4 page of the
benchmark's slowest picture exports at it in about 10 s. 300 dpi prints as sharp as a photo print, with a quarter of
the pixels. A PNG is at most 16,384 px along a side, which is also the most a Procreate canvas takes (Procreate also
limits a canvas to 33 to 134 megapixels by iPad, so the readout gives the megapixels); where the paper makes that less
than the dpi asked for, the box stops there and says so, and goes back to the dpi asked for on a paper that takes it.
The PNG is drawn in the background, a few seconds at most, and can't be stopped once it starts. A PDF has no
resolution of its own: it is vector art, and an app that turns it into pixels picks its own.

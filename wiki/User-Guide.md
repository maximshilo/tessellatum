# User guide

The settings sit in the sidebar on the left, one page per category; **Generate Preview** draws the page with them,
and **Export…** draws that
same page at print resolution: the regions, colors and legend the preview shows, its lines smoothed as they are
there, and its numbers placed again, as small as the style lets them print. If a setting changed since the preview,
the export makes the page with it first. Every setting has a tooltip; sizes are on the printed page, on the sheet
**Paper** names; the app keeps them between runs, and **Reset** puts them all back.

The window follows your system's light or dark appearance, and changes with it while open.

## The workspace

- **Toolbar** (top): Open Image, Generate Preview (it turns into Cancel while a page is made) and Export; at its right
  end, the buttons that show or hide the sidebar and the inspector, and Focus Mode.
- **Activity bar** (left edge): picks the settings shown in the sidebar:
  - **Source**: the picture, small, with its file name and size, and how it is read (line art, faces and subject,
    text);
  - **Generation**: the difficulty and the settings that shape the regions;
  - **Colors**: what the page is colored with, where its colors come from, the palette and mixing;
  - **Document**: the paper, lines and numbers, their tone, the legend, and the export's options.

  Clicking the page already shown folds the sidebar away; clicking any page brings it back.
- **Sidebar**: the category's settings in sections; click a section's title to fold or unfold it (sections of
  advanced settings start folded). Generate Preview and Export stay at its foot on every page.
- **Canvas** (middle): the page, with its own toolbar above it: the version shown and the zoom. On a narrow window,
  what doesn't fit goes into the toolbar's » menu.
- **Inspector** (right): the page's colors, numbered as on its legend, with their names; hover one for its hex and
  RGB, and, for a mixed color, its recipe.
- **Status bar** (bottom): what the app is doing, with a progress bar while it works; on its right, the page's colors
  and regions, its size in pixels and on paper, the zoom, and the count of warnings and errors in the diagnostics
  (click it to open them).
- **Notifications** (over the page): what just happened, such as an export done, with **Show in Folder**, or a failure,
  with **Details**. They never take the keyboard focus: a success goes by itself, a warning or an error stays until
  you close it.
- **Diagnostics** (View > Diagnostics, Ctrl+Shift+D): a log, under the page, of each step the app ran, how long it took
  and how it ended, and every warning and error. Select one to see its details; **Copy** puts it on the clipboard,
  for a bug report. It never opens by itself.

Drag the borders between the panels to resize them. **View > Sidebar** (Ctrl+B) and **View > Inspector** (Ctrl+Alt+B)
show or hide them; **Focus Mode** (Ctrl+Shift+F) hides everything around the page, and Esc, or the same shortcut,
brings it all back as it was. On a narrow window the inspector folds away by itself, then the sidebar, and they come
back as the window widens. The layout is kept between runs; **View > Reset Layout** puts it back as it starts.

## The three versions

Every preview is drawn in three versions, and the bar above it switches between them, keeping the zoom and the scroll:

- **Page**: the coloring page as it prints.
- **Completed**: the page painted in. Every region is filled in its legend color, which covers the lines and numbers,
  as paint covers their grays; what the page prints of the picture itself (line art's ink, the marks in a face, the
  letters of a sign) shows on top, and paper the page leaves bare stays white.
- **Tinted**: the page under a faint wash of its colors, each 40% of the way from white, with the lines, numbers and
  ink showing through as through a translucent paint: a guide to paint from.

## Menus and shortcuts

Every command is in the menu bar, and the most used have a shortcut (Cmd in place of Ctrl on macOS):

| Command | Shortcut |
|---|---|
| File > Open Image | Ctrl+O |
| File > Export | Ctrl+E |
| Generate > Generate Preview | Ctrl+Enter, F5 |
| Generate > Cancel | Esc |
| View > Page, Completed, Tinted | Ctrl+1, Ctrl+2, Ctrl+3 |
| View > Zoom In, Zoom Out | Ctrl+plus, Ctrl+minus |
| View > Fit Page, 100% | Ctrl+0, Ctrl+Alt+0 |
| View > Source, Generation, Colors, Document Settings | Ctrl+Shift+1 to 4 |
| View > Sidebar, Inspector | Ctrl+B, Ctrl+Alt+B |
| View > Focus Mode | Ctrl+Shift+F (Esc leaves it) |
| View > Diagnostics | Ctrl+Shift+D |

**View > Appearance** picks System (the window follows the system's light or dark setting as it changes), Light or
Dark, and the choice is kept. **Edit > Reset All Settings** is the Difficulty section's Reset; **Help > About Tessellatum** shows
the version and the licenses.

## Zoom

The preview zooms about the point under the cursor, with the mouse wheel, Ctrl+wheel, a touchpad's pinch, Ctrl+plus
and Ctrl+minus, or the buttons above it, and the page is dragged to move it. **Fit** (Ctrl+0) shows the whole page,
**100%** a pixel of the page as it prints on a pixel of the screen, and **Print size** the page as large as on paper,
by the size the screen reports. Further in than the preview's own pixels, the zoom shows the page as it prints: the
export's own drawing, with its numbers where and as small as they print. It is drawn in the background the first time
and kept, so the export is drawn from it without laying the page out again. A new preview of the same picture keeps the zoom and the
place, to compare settings there.

## Paper

**Paper** (A4 by default, A3 or A2, at the top of the region settings) is the sheet the page is made for. Every size
on paper -- the brush, the smallest region, the lines, the numbers -- applies on that sheet, so a larger one prints the
picture larger and makes a finer page of it: more and smaller regions, each still as wide as the brush. The page is
made at more pixels on a larger sheet, the picture scaled up where it has fewer, so that every region still has room
for the brush and its number; scaling up adds no detail the picture doesn't have, so a larger picture still makes a
better page on a larger sheet. A page takes longer to make there: a Medium preview of a 2048 px photograph takes 1.5
to 2 times as long on A3 and 2.3 to 4 times on A2 (the benchmark's lion: 2 s on A4, 4 s on A3 and 9 s on A2; at
Realistic, 7 s, 15 s and 34 s). The PDF's sheets are this paper, the page landscape when the picture is wider than
tall and the legend on a portrait sheet of its own, and a PNG records the resolution that prints it on this paper.

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

## Difficulty

Beginner, Easy, Medium (the default), Hard and Realistic fill in the settings below them; moving any of them picks
**Custom**. The five levels differ in four settings:

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

## Settings

The sidebar shows them by category (see [The workspace](#the-workspace)); below they are described by what they
shape.

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
  differently, so a recipe is where to start. With no paints listed it shows hex and RGB instead.
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
  sheet of its own, its swatches 12 mm squares in their exact colors, their names and codes as text a PDF reader can
  search and copy.
- **PNG**: the page above its legend in one image: the PDF's page drawn in pixels, at the **Resolution** asked
  for, which the file records, so it prints as large as the PDF's page.
- **Palette**: the legend's colors as a palette file, for coloring the page in a painting app with exactly its
  colors; the panel suggests it when **Coloring on** is Screen. Each color is named by its number and its name on the
  legend ("7 dusty rose"), in the legend's order. **Palette** picks the format:
  - **Procreate** (`.swatches`): a zip holding `Swatches.json`, the colors in HSB, as the MIT-licensed
    procreate-swatches library writes them (Procreate publishes no spec). Procreate keeps the colors but not their
    names, and a palette holds 30 at most, so a larger legend is saved as several files of 30, named for the colors
    each holds ("page 1-30.swatches", "page 31-60.swatches"). Drop them on Procreate's Palettes tab.
  - **Adobe Swatch Exchange** (`.ase`): read by Photoshop, Illustrator, InDesign, Affinity's apps, Procreate and
    others; the colors in RGB, in a group named after the file.
  - **GIMP palette** (`.gpl`): read by GIMP, Krita, Inkscape, Aseprite and others.

  The palette is the preview's legend, saved at once; if the settings changed since the preview, the page is made
  again with them first, as for the other formats.

**Resolution** is the PNG's: in **dpi** on its paper (the buttons set 150, 300 or 600; any other can be typed) or in
**px** along the page's long edge, the other worked out from the paper and the picture's shape and shown below it
with the file's pixel size. 600 dpi is the default: sharp in print and zoomed in on a screen, and an A4 page of the
benchmark's slowest picture exports at it in about 10 s. 300 dpi prints as sharp as a photo print, with a quarter of
the pixels. A PNG is at most 16,384 px along a side, which is also the most a Procreate canvas takes (Procreate also
limits a canvas to 33 to 134 megapixels by iPad, so the readout gives the megapixels); where the paper makes that less
than the dpi asked for, the box stops there and says so, and goes back to the dpi asked for on a paper that takes it.
The PNG is drawn in the background, a few seconds at most, and can't be stopped once it starts. A PDF has no
resolution of its own: it is vector art, and an app that turns it into pixels picks its own.

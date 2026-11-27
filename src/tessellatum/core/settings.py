"""Every setting the app offers: what it means, its unit, its range and its default.

The pipeline takes its settings in three objects: how granular the page is
(``difficulty.DifficultyParams``, which the presets fill in), how its lines
and numbers print (``render.PageStyle``), and what it does with the line art,
faces and text it finds (``pipeline.Handling``). Each setting here is one
field of one of them, by name, with what the app needs to offer it: a label,
a unit, a range, a step and a tooltip. Its default is the field's own, or for
the colors, the smallest region and the smoothing, which have none, the
default preset's: a page drawn at the defaults is the page the pipeline draws
at that preset when given nothing else.

Sizes are on the printed page, on the paper chosen, as everywhere in the
pipeline (see ``print_size``). The defaults are what the quality benchmarks are measured at
(``benchmarks/QUALITY_BENCHMARKS.md``); a setting moved past them -- a finer
brush, closer colors -- can make a page with more detail than those targets
allow, which is the user's call.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Mapping

from tessellatum.core import difficulty
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.legend import DEFAULT_LEGEND_DETAIL, LEGEND_DETAILS, NAME, NAME_HEX_RGB, NAME_RECIPE, NUMBER_ONLY
from tessellatum.core.pipeline import Handling
from tessellatum.core.print_size import PAPERS
from tessellatum.core.render import (
    DEFAULT_TONE,
    LINE_WIDTH_MM_RANGE,
    LINE_WIDTH_MM_STEP,
    PAPER,
    TONES,
    PageStyle,
    lightness,
)

DIFFICULTY = "difficulty"
STYLE = "style"
HANDLING = "handling"

_OWNERS = {DIFFICULTY: DifficultyParams, STYLE: PageStyle, HANDLING: Handling}


@dataclass(frozen=True)
class Setting:
    """One setting: the field ``name`` of the object ``owner`` stands for (``DIFFICULTY``, ``STYLE`` or ``HANDLING``).

    ``minimum`` to ``maximum`` in steps of ``step``, shown with ``unit``;
    ``log`` moves a slider over it in equal ratios rather than equal steps,
    and ``whole`` takes whole numbers only. ``group`` is the part of the panel
    it is shown in. ``shown``, if given, is what a value is shown as, in
    ``unit``, where the field holds it otherwise: a gray as its lightness.

    The registry's description of it (see below): its ``category`` and ``section`` in the new workspace, its
    ``level`` (intent, precise or advanced), its ``unit_kind``, the words it may be searched by (``aliases``), when a
    change is applied (``policy``), and when it means nothing (``enabled_when``: the document's choices by name to
    the reason, or None while it does).
    """

    name: str
    owner: str
    group: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float
    tooltip: str
    log: bool = False
    whole: bool = False
    shown: Callable[[float], float] | None = None
    category: str = ""
    section: str = ""
    level: str = "advanced"
    unit_kind: str = ""
    aliases: tuple = ()
    policy: str = "explicit"
    enabled_when: Callable[[Mapping], str | None] | None = None

    @property
    def preset_owned(self) -> bool:
        """Whether the difficulty presets set it (and moving it makes the difficulty Custom)."""
        return self.owner == DIFFICULTY

    @property
    def key(self) -> str:
        """Where its value is kept between runs."""
        return f"settings/{self.name}"

    @property
    def decimals(self) -> int:
        """How many decimals it is shown with: as many as its step has."""
        if self.whole or self.step >= 1:
            return 0
        return max(0, -math.floor(math.log10(self.step) + 1e-9))

    def check(self, value) -> str | None:
        """Why ``value`` can't be this setting's, or None if it can. Nothing is held within the range here (see
        ``clamp``): a value out of it is refused, with the reason to show."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{value!r} is not a number"
        if math.isnan(number):
            return "not a number"
        if number < self.minimum:
            return f"{self.label} is at least {self.text(self.minimum)}"
        if number > self.maximum:
            return f"{self.label} is at most {self.text(self.maximum)}"
        if self.whole and number != int(number):
            return f"{self.label} takes a whole number"
        return None

    def disabled_reason(self, chosen: Mapping) -> str | None:
        """Why it means nothing with the document's choices ``chosen`` (by name: medium, colors_from, mix_paints and
        the switches), or None while it does."""
        return None if self.enabled_when is None else self.enabled_when(chosen)

    @property
    def default(self) -> float:
        """The value the pipeline uses when given nothing: the field's default, or the default preset's value."""
        if self.owner == DIFFICULTY:
            return getattr(difficulty.params_for_preset(difficulty.DEFAULT_PRESET), self.name)
        return next(f.default for f in dataclasses.fields(_OWNERS[self.owner]) if f.name == self.name)

    def clamp(self, value: float) -> float:
        """``value`` held within the range, rounded to a whole number if the setting takes one."""
        value = min(self.maximum, max(self.minimum, float(value)))
        return int(round(value)) if self.whole else value

    def text(self, value: float) -> str:
        """``value`` as the panel shows it, with its unit."""
        if self.shown is not None:
            return f"{self.shown(value):.0f} {self.unit}".strip()
        number = f"{int(round(value))}" if self.whole else f"{value:.{self.decimals}f}"
        return f"{number} {self.unit}".strip()


REGIONS = "Regions and colors"
FACES = "Faces and subject"
LINES = "Lines and numbers"
LINE_ART = "Line art"
TEXT = "Text"
GROUPS = (REGIONS, FACES, LINES, LINE_ART, TEXT)


def _gray(name: str, label: str, tooltip: str) -> Setting:
    """A gray of the page's ink, 0 black to 255 white, shown as its lightness: the tone sets all three (see
    ``render.TONES``)."""
    return Setting(name, STYLE, LINES, label, "L*", 0, PAPER - 1, 1, tooltip, whole=True, shown=lightness)


def _difficulty(name: str, label: str, unit: str, step: float, tooltip: str, **kw) -> Setting:
    minimum, maximum = difficulty.CUSTOM_RANGES[name]
    group = kw.pop("group", REGIONS)
    return Setting(name, DIFFICULTY, group, label, unit, minimum, maximum, step, tooltip, **kw)


SETTINGS: tuple[Setting, ...] = (
    # -- Regions and colors (the difficulty: the presets set these) --
    _difficulty(
        "num_colors", "Colors", "", 1,
        "How many colors to look for. Colors closer than the color margin are merged, so a page can keep fewer.",
        whole=True,
    ),
    _difficulty(
        "palette_margin_de00", "Color margin", "ΔE00", 0.5,
        "How different every two palette colors must be (CIEDE2000; about 1 is the smallest difference an eye "
        "sees side by side). Closer colors are merged into one. Lower keeps more, subtler colors; 0 keeps every "
        "color found.",
    ),
    _difficulty(
        "paper_black_lstar", "Darkest black on paper", "L*", 1,
        "How dark the printer and the paints or pencils can go, as a lightness (0 is black). Colors darker than "
        "that look alike on paper, so they must also keep the color margin as printed, and the colors that frees go "
        "to the rest of the picture; the faces and the subject keep their dark shades all the same. About 20 for a "
        "good laser or pigment inkjet and soft pencils, 30 for a weak printer or cheap paper; 0 judges the colors as "
        "the screen shows them.",
    ),
    _difficulty(
        "min_region_area_mm2", "Smallest region", "mm²", 1,
        "The smallest area a region may have on the printed page; smaller ones merge into a neighbor. Never "
        "below the brush's own footprint.",
        log=True,
    ),
    _difficulty(
        "min_width_mm", "Brush width", "mm", 0.1,
        "The narrowest any part of a region may be on the printed page: the brush it is painted with. Narrower "
        "parts are given to the region beside them, which also rounds sharp corners unless Sharpest corner keeps "
        "them. A finer brush keeps thinner shapes.",
    ),
    # The page's resolution, from the brush. Not the difficulty's: moving it keeps the preset.
    Setting(
        "brush_px", HANDLING, REGIONS, "Pixels across brush", "px", 2.0, 8.0, 0.5,
        "How finely a page is made: with this many pixels across the brush, so that a finer brush gets more of them. "
        "At 4, a 3:2 picture's page is made at about 1,100 px along its long edge at Beginner and 2,200 at Realistic; "
        "never at fewer than 1,100, nor at more than the picture has or than 300 dpi. More pixels let the regions' "
        "edges follow the picture's more closely and the brush find parts of a region too thin to paint, and take "
        "longer. At 2, every page is made at about 1,100 px.",
    ),
    # When a color mixed from the painter's paints earns its place (Mix colors from my paints; see quantize.MixRule).
    Setting(
        "mix_gain_de00", HANDLING, REGIONS, "Mix if closer by", "ΔE00", 1.0, 10.0, 0.5,
        "With Mix colors from my paints: a color mixed from two of your paints is added only if the parts of the "
        "picture it would paint come out at least this much closer to the picture, on average, than with the colors "
        "already taken (CIEDE2000; about 2.3 is the smallest difference an eye sees side by side). Higher, fewer "
        "mixes and only the ones that clearly matter; lower, more of them.",
    ),
    Setting(
        "mix_min_share_pct", HANDLING, REGIONS, "and it paints at least", "%", 0.1, 10.0, 0.1,
        "With Mix colors from my paints: a mixed color is added only if it would paint at least this share of the "
        "picture (faces and the subject count twice), so that nothing is mixed for a speck. Higher, fewer mixes.",
        log=True,
    ),
    _difficulty(
        "sharpest_corner_deg", "Sharpest corner", "°", 1,
        "The sharpest corner whose point is kept, rather than rounded off where the brush can't reach: a painter "
        "fills it with the brush's tip. 180, the default, rounds every corner to the brush; 20 keeps a triangle's or "
        "a spire's points. Points crowding within a brush's width of each other, the spikes of a ragged edge, are "
        "rounded all the same.",
    ),
    _difficulty(
        "corner_contrast_de00", "Corner contrast", "ΔE00", 0.5,
        "How plainly the picture must show a corner's point for it to be kept: how much closer its colors are, on "
        "average, to its own region's than to its neighbor's. The points of a roof or a flat shape stand well apart; "
        "the spikes of fur and foliage don't, and are rounded off. 0 keeps every corner sharp enough.",
    ),
    _difficulty(
        "blur_sigma", "Smoothing", "", 0.1,
        "How much the picture is smoothed, keeping its edges, before its colors are found. More gives fewer, "
        "larger patches; 0 keeps every grain.",
    ),
    _difficulty(
        "edge_settling", "Edge settling", "×", 0.05,
        "How far the vote that smooths the regions' ragged edges in fur, foliage and stone reaches, as a multiple "
        "of its usual reach (a third of the brush). 0 leaves the edges as the colors fall.",
    ),
    _difficulty(
        "edge_color_step_de00", "Edge hold", "ΔE00", 0.5,
        "How firmly settled edges stay on the picture's own edges: a color this much further from a pixel counts "
        "e times less in its vote. Lower holds the edges to the picture; higher smooths them more.",
    ),
    # -- Faces and subject --
    _difficulty(
        "detail_weight", "Face and subject detail", "×", 1,
        "How many times smaller a region may be in the faces and the subject found in a photograph or painting "
        "than elsewhere. 1 gives them no more detail than the rest.",
        group=FACES, whole=True,
    ),
    Setting(
        "mark_contrast", HANDLING, FACES, "Dark mark contrast", "L*", 2.0, 50.0, 1.0,
        "How much darker than around it a thin mark in a face (a pupil, a lip line) must be to be printed rather "
        "than painted over. Lower prints fainter marks.",
    ),
    Setting(
        "mark_length_mm", HANDLING, FACES, "Dark mark length", "mm", 0.5, 10.0, 0.1,
        "How long a thin dark mark in a face must be to be printed; shorter ones are texture and are left out.",
    ),
    Setting(
        "face_tone_step_de00", HANDLING, FACES, "Face tone step", "ΔE00", 0.0, 5.0, 0.1,
        "A region in a face joins its neighbor while that puts its pixels less than this much further from the "
        "picture, so skin and fur paint in a few large tones. 0 joins none.",
    ),
    # -- Lines and numbers --
    Setting(
        "line_width_mm", STYLE, LINES, "Line width", "mm", LINE_WIDTH_MM_RANGE[0], LINE_WIDTH_MM_RANGE[1],
        LINE_WIDTH_MM_STEP, "How wide the lines print on the page.",
    ),
    _gray("line_gray", "Line gray", "How dark the lines print, as a lightness: 0 is black, 100 white."),
    _gray(
        "label_gray", "Number gray",
        "How dark a number inside its region prints, as a lightness: 0 is black, 100 white. Gray, like the lines, it "
        "reads as part of the page rather than as writing in the picture.",
    ),
    _gray(
        "leader_gray", "Leader gray",
        "How dark a number written outside its region prints, with the leader pointing it in and the leader's dot, as "
        "a lightness: 0 is black, 100 white. A gray of their own tells them at a glance from the numbers inside "
        "their regions.",
    ),
    Setting(
        "leader_width_mm", STYLE, LINES, "Leader width", "mm", 0.05, LINE_WIDTH_MM_RANGE[1], 0.05,
        "How wide the leaders print on the page: the lines pointing a number written outside its region in, its "
        "dot three times as wide. Thinner than the lines, they don't read as boundaries. However thin, the numbers "
        "keep as clear of them as of a line.",
    ),
    Setting(
        "line_smoothing_mm", STYLE, LINES, "Line smoothing", "mm", 0.0, 2.0, 0.05,
        "How far along a line its pixel staircase is smoothed away. More rounds bends and corners; the pixel "
        "steps themselves always go.",
    ),
    Setting(
        "min_label_pt", STYLE, LINES, "Smallest number", "pt", 1.0, 12.0, 0.5,
        "The smallest a region's number prints. Smaller fits more numbers inside small regions, and fewer need a "
        "leader; below 6 pt they get hard to read, and under about 3 pt a number is for coloring on screen, "
        "zoomed in, rather than on paper. An export prints them this small; the preview draws no number under 10 "
        "pixels, 6.5-7 pt on a page made at 1,100 px. Where a number has no room at all, the export prints it "
        "smaller inside its region, down to half this size, rather than across its lines.",
    ),
    Setting(
        "leader_reach_mm", STYLE, LINES, "Leader reach", "mm", 1.0, 20.0, 0.5,
        "How far outside a region too small for its number the number may be written, with a line pointing in.",
    ),
    Setting(
        "text_gap_mm", STYLE, LINES, "Gap beside text", "mm", 0.0, 3.0, 0.1,
        "How far a number keeps from the lettering of a sign or caption, so it doesn't read as part of it.",
    ),
    Setting(
        "split_neck_mm", STYLE, LINES, "Split at necks under", "mm", 0.0, 2.0, 0.05,
        "A region whose paper narrows to less than this between its lines reads as two regions there, so each side "
        "gets its number. Sides smaller than the smallest region don't count. 0 numbers again only the sides the "
        "lines close off completely.",
    ),
    # -- Line art --
    Setting(
        "thin_ink_mm", HANDLING, LINE_ART, "Paint over ink up to", "mm", 0.0, 5.0, 0.1,
        "On a cartoon or comic, the paint goes over its ink lines up to this wide (hatching, fine strokes), and "
        "never over bolder ones.",
    ),
    Setting(
        "ink_gap_mm", HANDLING, LINE_ART, "Close line gaps up to", "mm", 0.0, 3.0, 0.1,
        "Gaps in a cartoon's ink lines up to this wide are closed, so the paint of two areas doesn't run together.",
    ),
    # -- Text --
    Setting(
        "text_max_height_mm", HANDLING, TEXT, "Print lettering up to", "mm", 5.0, 60.0, 1.0,
        "Lines of text up to this tall on the page are printed; taller lettering is big enough to paint as shapes.",
    ),
)

# -- The registry: what each setting is, beyond its range (F4, D-008) ----------------------------------------------
#
# Every setting the user edits, the sliders above and the choices below, with where it belongs (its category and
# section), how deep (its level), what kind of quantity it is, when a change to it is applied (its policy), when it
# means nothing (its enablement, with the reason to show), the words it may be searched by, and the key it is kept
# under between runs. Qt-free: the GUI, the tests and the benchmark read the same entries. The strings the panel shows
# for them live here, in one place.

SOURCE, GENERATION, COLORS, DOCUMENT = "Source", "Generation", "Colors", "Document"
CATEGORIES = (SOURCE, GENERATION, COLORS, DOCUMENT)
INTENT, PRECISE, ADVANCED = "intent", "precise", "advanced"
LEVELS = (INTENT, PRECISE, ADVANCED)
# When a change is applied (D10): at once, as it changes nothing but what is drawn over the page (the legend); a moment
# after the last of a run of changes, as it only draws the page again from the regions the pipeline keeps (the lines,
# the numbers, the tone: the page's style, but for its paper); or with Generate, as it makes the page again.
IMMEDIATE, DEBOUNCED, EXPLICIT = "immediate", "debounced", "explicit"
POLICIES = (IMMEDIATE, DEBOUNCED, EXPLICIT)
LENGTH, AREA, COLOR_DIFFERENCE, LIGHTNESS, ANGLE = "length", "area", "color difference", "lightness", "angle"
PERCENT, RATIO, COUNT, POINTS, PIXELS = "percent", "ratio", "count", "points", "pixels"
UNIT_KINDS = (LENGTH, AREA, COLOR_DIFFERENCE, LIGHTNESS, ANGLE, PERCENT, RATIO, COUNT, POINTS, PIXELS)

# The document's choices (the values of the entries in CHOICES) that settings depend on.
CUSTOM = "Custom"  # the difficulty set by hand, and the grays set by hand (the tone)
CUSTOM_TONE = CUSTOM
DEFAULT_PAPER = "A4"
PICTURE_COLORS = "The picture"
MY_PAINTS = "My paints"
COLORS_FROM = (PICTURE_COLORS, MY_PAINTS)

ONLY_WITH_DETAIL = "Only with More detail on faces and subject."
ONLY_WITH_LINE_ART = "Only with Print line art's own ink."
ONLY_WITH_TEXT = "Only with Print text."


def _unless(choice: str, reason: str) -> Callable[[Mapping], str | None]:
    """Enabled while the switch ``choice`` is on; else ``reason``."""
    return lambda chosen: None if chosen.get(choice, True) else reason


def _on_paper(chosen: Mapping) -> str | None:
    if chosen.get("medium") == difficulty.SCREEN_MEDIUM:
        return "Coloring on a screen: there is no darkest black on paper."
    return None


def _mixing(chosen: Mapping) -> str | None:
    if chosen.get("colors_from") != MY_PAINTS:
        return "Only with Colors from: My paints."
    if not chosen.get("mix_paints"):
        return "Only with Mix colors from my paints."
    return None


# name: (category, section, level, unit kind, aliases, enablement). The policy follows the owner: the style's sliders
# only draw the page again (DEBOUNCED), the rest make it again (EXPLICIT).
_SLIDER_META: dict[str, tuple] = {
    "num_colors": (GENERATION, "Detail", INTENT, COUNT, ("palette size", "number of colors", "how many colors"), None),
    "palette_margin_de00": (COLORS, "Palette", PRECISE, COLOR_DIFFERENCE,
                            ("color margin", "distinct colors", "merge similar colors", "delta E"), None),
    "paper_black_lstar": (COLORS, "Palette", PRECISE, LIGHTNESS,
                          ("darkest black", "black point", "printer", "dark colors"), _on_paper),
    "min_region_area_mm2": (GENERATION, "Detail", PRECISE, AREA,
                            ("region size", "smallest area", "minimum region", "detail"), None),
    "min_width_mm": (GENERATION, "Detail", PRECISE, LENGTH,
                     ("brush", "brush width", "tip", "minimum width", "thin parts"), None),
    "brush_px": (GENERATION, "Advanced · Resolution", ADVANCED, PIXELS,
                 ("pixels across brush", "resolution", "precision", "quality"), None),
    "mix_gain_de00": (COLORS, "Mixing", ADVANCED, COLOR_DIFFERENCE, ("mixing threshold", "mixes"), _mixing),
    "mix_min_share_pct": (COLORS, "Mixing", ADVANCED, PERCENT, ("mix share", "mixes"), _mixing),
    "sharpest_corner_deg": (GENERATION, "Advanced · Corners", ADVANCED, ANGLE,
                            ("corners", "points", "sharp", "triangle"), None),
    "corner_contrast_de00": (GENERATION, "Advanced · Corners", ADVANCED, COLOR_DIFFERENCE, ("corner contrast",), None),
    "blur_sigma": (GENERATION, "Detail", PRECISE, RATIO, ("smoothing", "blur", "simplify", "denoise", "grain"), None),
    "edge_settling": (GENERATION, "Advanced · Edges", ADVANCED, RATIO,
                      ("ragged edges", "fur", "foliage", "texture"), None),
    "edge_color_step_de00": (GENERATION, "Advanced · Edges", ADVANCED, COLOR_DIFFERENCE, ("keep edges",), None),
    "detail_weight": (GENERATION, "Faces and subject", PRECISE, RATIO, ("face detail", "subject detail", "focus"),
                      _unless("detail", ONLY_WITH_DETAIL)),
    "mark_contrast": (GENERATION, "Faces and subject", ADVANCED, LIGHTNESS, ("dark marks", "pupils", "eyes", "lips"),
                      _unless("detail", ONLY_WITH_DETAIL)),
    "mark_length_mm": (GENERATION, "Faces and subject", ADVANCED, LENGTH, ("dark marks",),
                       _unless("detail", ONLY_WITH_DETAIL)),
    "face_tone_step_de00": (GENERATION, "Faces and subject", ADVANCED, COLOR_DIFFERENCE, ("skin tones", "face tones"),
                            _unless("detail", ONLY_WITH_DETAIL)),
    "line_width_mm": (DOCUMENT, "Lines and numbers", PRECISE, LENGTH, ("outline", "stroke", "line thickness"), None),
    "line_gray": (DOCUMENT, "Tone", ADVANCED, LIGHTNESS, ("line darkness", "ink"), None),
    "label_gray": (DOCUMENT, "Tone", ADVANCED, LIGHTNESS, ("number darkness",), None),
    "leader_gray": (DOCUMENT, "Tone", ADVANCED, LIGHTNESS, ("leader darkness",), None),
    "leader_width_mm": (DOCUMENT, "Advanced · Numbers", ADVANCED, LENGTH, ("leader thickness",), None),
    "line_smoothing_mm": (DOCUMENT, "Advanced · Lines", ADVANCED, LENGTH, ("jaggies", "staircase", "smooth lines"),
                          None),
    "min_label_pt": (DOCUMENT, "Lines and numbers", PRECISE, POINTS,
                     ("number size", "font size", "label size", "smallest number"), None),
    "leader_reach_mm": (DOCUMENT, "Advanced · Numbers", ADVANCED, LENGTH, ("leader length",), None),
    "text_gap_mm": (DOCUMENT, "Advanced · Numbers", ADVANCED, LENGTH, ("gap beside text", "lettering clearance"),
                    _unless("text", ONLY_WITH_TEXT)),
    "split_neck_mm": (DOCUMENT, "Advanced · Numbers", ADVANCED, LENGTH,
                      ("necks", "pinched regions", "second number", "sides"), None),
    "thin_ink_mm": (GENERATION, "Line art", ADVANCED, LENGTH, ("hatching", "paint over ink", "thin lines"),
                    _unless("line_art", ONLY_WITH_LINE_ART)),
    "ink_gap_mm": (GENERATION, "Line art", ADVANCED, LENGTH, ("line gaps", "leaks", "close gaps"),
                   _unless("line_art", ONLY_WITH_LINE_ART)),
    "text_max_height_mm": (GENERATION, "Text", ADVANCED, LENGTH, ("lettering", "captions", "signs", "text height"),
                           _unless("text", ONLY_WITH_TEXT)),
}
SETTINGS = tuple(
    dataclasses.replace(
        s, category=m[0], section=m[1], level=m[2], unit_kind=m[3], aliases=m[4], enabled_when=m[5],
        policy=DEBOUNCED if s.owner == STYLE else EXPLICIT,
    )
    for s in SETTINGS for m in (_SLIDER_META[s.name],)
)
_BY_NAME = {setting.name: setting for setting in SETTINGS}

# -- The choices: the settings that aren't sliders ---------------------------------------------------------------
ENUM, BOOL, PAINTS = "enum", "bool", "paints"

TONE_TOOLTIPS = {
    "Light": "Fainter lines, numbers and leaders, which vanish under the palest paints.",
    "Medium": "Gray lines and numbers that don't read as writing in the picture, and the numbers written outside "
    "their regions in a lighter gray of their own, with their leaders.",
    "Dark": "Darker lines, numbers and leaders, easier to follow on paper.",
}
CUSTOM_TONE_TOOLTIP = "Set the three grays yourself. Moving any of them picks Custom, starting from the tone."
CUSTOM_TOOLTIP = "Set every difficulty setting yourself. Moving any of them picks Custom, starting from the preset."
PAPER_TOOLTIP = (
    "The sheet the page is made for: every size on paper (the brush, the smallest region, the lines, the numbers) "
    "applies on it. A larger sheet prints the picture larger, so it makes a finer page, with more and smaller regions, "
    "and takes longer; the PDF's sheets are this size."
)
LEGEND_TOOLTIPS = {
    NUMBER_ONLY: "Each color's swatch with its number in it, and nothing beside it.",
    NAME: "Each color's everyday name beside its swatch (\"dusty rose\", \"olive green\"): the nearest of about 800 names, every color a different one.",
    NAME_HEX_RGB: "Its name, and under it its hex code and RGB values (\"#C0737A  RGB 192 115 122\"), exactly the color the page is painted with: to find it in a painting app, or match a paint or pencil to it.",
    NAME_RECIPE: "Its name, and how to mix it from the paints you list (Paints…): \"3 Titanium white\", \"1 Ultramarine\", parts of one to three of them, mixed as real paint mixes (blue and yellow make green). A model of paint, not of your own tubes: a place to start. A color no mix comes within 5 ΔE00 of says how far off its closest is. With no paints listed, as Name, hex and RGB.",
}
LEGEND_TOOLTIP = "What the legend writes beside each color's swatch, to find, buy or mix it by. " + " ".join(
    f"{name}: {tip}" for name, tip in LEGEND_TOOLTIPS.items()
)
COLORS_FROM_TOOLTIP = (
    "Where the page's colors come from. The picture: the colors it finds in it. My paints: the paints or pencils you "
    "list (Paints…), at most as many as Colors asks for, the ones that paint the picture best, each called by its own "
    "name on the legend. Every color on the legend is then one you have; two closer than the color margin are never "
    "both used."
)
MIX_TOOLTIP = (
    "With My paints: the page may also take colors mixed from two of your paints, in 1 + 1 to 3 + 1 parts, where the "
    "picture has colors none of your paints comes close to. Your paints come first; a mix is added only if it brings "
    "the parts of the picture it paints clearly closer (Mix if closer by) and paints enough of it (and it paints at "
    "least), so a box that covers the picture gets few mixes or none, and a few paints get many. The legend lists "
    "them under Mixed colors, each with its recipe. Mixed as real paint mixes (blue and yellow make green), by a "
    "model: your own paints may mix a little differently."
)
MEDIUM_TOOLTIPS = {
    difficulty.PAPER_MEDIUM: "Colored on paper, with paints or pencils: colors darker than the darkest black on paper "
    "look alike there, so they also keep the color margin as printed.",
    difficulty.SCREEN_MEDIUM: "Colored on a screen, in an app such as Procreate: every color is picked exactly, so "
    "there is no darkest black on paper, and the color margin is at most "
    f"{difficulty.SCREEN_MARGIN_DE00:g} ΔE00, enough to keep two neighboring regions apart. More of the picture's "
    "shades stay on the legend.",
}
MEDIUM_TOOLTIP = "What the page will be colored with. " + " ".join(
    f"{name}: {tip}" for name, tip in MEDIUM_TOOLTIPS.items()
)


@dataclass(frozen=True)
class Choice:
    """A setting that isn't a slider: one of ``options`` (``ENUM``), on or off (``BOOL``), or the painter's paints
    (``PAINTS``, a tuple of (name, (r, g, b))). ``field`` names the pipeline field it sets directly ("Owner.field"),
    if it sets one; the others (the difficulty, the tone, Coloring on, Colors from) set others' values."""

    name: str
    label: str
    kind: str
    default: object
    key: str
    category: str
    section: str
    level: str
    tooltip: str
    options: tuple = ()
    option_tips: Mapping[str, str] = dataclasses.field(default_factory=dict)
    field: str = ""
    policy: str = EXPLICIT
    aliases: tuple = ()
    enabled_when: Callable[[Mapping], str | None] | None = None

    def check(self, value) -> str | None:
        """Why ``value`` can't be this setting's, or None if it can."""
        if self.kind == ENUM:
            return None if value in self.options else f"{value!r} is not one of {', '.join(self.options)}"
        if self.kind == BOOL:
            return None if isinstance(value, bool) else f"{value!r} is not on or off"
        if not isinstance(value, tuple) or not all(
            isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and len(p[1]) == 3 for p in value
        ):
            return "not a list of paints, (name, (r, g, b)) each"
        return None

    def disabled_reason(self, chosen: Mapping) -> str | None:
        """Why it means nothing with the document's choices ``chosen`` (by name), or None while it does."""
        return None if self.enabled_when is None else self.enabled_when(chosen)


CHOICES: tuple[Choice, ...] = (
    Choice("preset", "Difficulty", ENUM, difficulty.DEFAULT_PRESET, "difficulty/preset", GENERATION, "Difficulty",
           INTENT, "How fine the page is: each level sets the difficulty's settings at once.",
           options=tuple(difficulty.preset_names()), option_tips={CUSTOM: CUSTOM_TOOLTIP},
           aliases=("level", "difficulty", "preset")),
    Choice("paper", "Paper", ENUM, DEFAULT_PAPER, "style/paper", DOCUMENT, "Page", INTENT, PAPER_TOOLTIP,
           options=tuple(PAPERS), field="PageStyle.paper", aliases=("sheet", "page size", "A3", "A2")),
    Choice("medium", "Coloring on", ENUM, difficulty.DEFAULT_MEDIUM, "difficulty/medium", COLORS, "Medium", INTENT,
           MEDIUM_TOOLTIP, options=tuple(difficulty.MEDIA), option_tips=MEDIUM_TOOLTIPS,
           aliases=("screen", "procreate", "digital", "paper")),
    Choice("colors_from", "Colors from", ENUM, PICTURE_COLORS, "difficulty/colors_from", COLORS, "Source of colors",
           INTENT, COLORS_FROM_TOOLTIP, options=COLORS_FROM, aliases=("my paints", "pencils", "kit")),
    Choice("paints", "Paints", PAINTS, (), "difficulty/paints", COLORS, "Source of colors", PRECISE,
           "List the paints or pencils you have, or load them from a palette file.", field="Handling.paints",
           aliases=("paints", "pencils", "kit", "palette file")),
    Choice("mix_paints", "Mix colors from my paints", BOOL, False, "difficulty/mix_paints", COLORS, "Mixing", PRECISE,
           MIX_TOOLTIP, field="Handling.mix_paints", aliases=("mixing", "mixes"),
           enabled_when=lambda chosen: None if chosen.get("colors_from") == MY_PAINTS
           else "Only with Colors from: My paints."),
    Choice("tone", "Tone", ENUM, DEFAULT_TONE, "style/tone", DOCUMENT, "Tone", INTENT,
           "How dark the lines, numbers and leaders print: each tone sets the three grays at once.",
           options=(*TONES, CUSTOM_TONE), option_tips={**TONE_TOOLTIPS, CUSTOM_TONE: CUSTOM_TONE_TOOLTIP},
           policy=DEBOUNCED, aliases=("line darkness", "light lines")),
    Choice("legend_detail", "Legend", ENUM, DEFAULT_LEGEND_DETAIL, "style/legend", DOCUMENT, "Legend", INTENT,
           LEGEND_TOOLTIP, options=LEGEND_DETAILS, option_tips=LEGEND_TOOLTIPS, field="PageStyle.legend_detail",
           policy=IMMEDIATE, aliases=("legend", "names", "hex", "recipes")),
    Choice("line_art", "Print line art's own ink", BOOL, True, "handling/line_art", SOURCE, "Picture handling",
           PRECISE, "On a cartoon or a comic, print its ink lines and paint the areas they enclose. Off, it is drawn "
           "from its colors like a photograph.", field="Handling.line_art", aliases=("cartoon", "comic", "ink lines")),
    Choice("detail", "More detail on faces and subject", BOOL, True, "handling/detail", SOURCE, "Picture handling",
           PRECISE, "On a photograph or a painting, regions in the faces and the subject found may be smaller (see "
           "Face and subject detail), a face is painted in a few tones, and its thin dark marks (pupils, lip lines) "
           "are printed.", field="Handling.detail", aliases=("faces", "subject", "saliency")),
    Choice("text", "Print text", BOOL, True, "handling/text", SOURCE, "Picture handling", PRECISE,
           "Print the letters of the signs, titles and captions found in the picture, their ground left bare, and "
           "keep the numbers off them.", field="Handling.text", aliases=("text", "lettering", "OCR")),
)
_CHOICES = {choice.name: choice for choice in CHOICES}
SWITCHES = ("line_art", "detail", "text")

# The pipeline fields no entry sets: worked out from others, or fixed.
INTERNAL_FIELDS = {
    "PageStyle.min_line_width_px": "the thinnest a line is drawn in pixels, from the page's resolution",
    "PageStyle.leader_dot_ratio": "the leader's dot, a fixed multiple of its width",
}


def choice(name: str) -> Choice:
    """The choice ``name``; KeyError if there is none."""
    return _CHOICES[name]


def entry(name: str) -> Setting | Choice:
    """The registry's entry ``name``, slider or choice; KeyError if there is none."""
    return _BY_NAME[name] if name in _BY_NAME else _CHOICES[name]


def entries() -> tuple[Setting | Choice, ...]:
    """Every entry: the sliders, then the choices."""
    return (*SETTINGS, *CHOICES)


def update_policy(names) -> str:
    """How a change to the entries ``names`` is applied, all together: with Generate (``EXPLICIT``) if any of them
    makes the page again; else a moment after the last change (``DEBOUNCED``) if any draws it again; else at once
    (``IMMEDIATE``), as nothing but what is drawn over the page changed, or nothing at all. A name the registry
    doesn't know waits for Generate."""
    policies = set()
    for name in names:
        try:
            policies.add(entry(name).policy)
        except KeyError:
            return EXPLICIT
    for policy in (EXPLICIT, DEBOUNCED):
        if policy in policies:
            return policy
    return IMMEDIATE


def find(text: str) -> list[Setting | Choice]:
    """The entries whose label, name or aliases hold ``text`` (case aside), in registry order."""
    needle = text.strip().lower()
    if not needle:
        return []
    return [e for e in entries() if any(needle in word.lower() for word in (e.label, e.name, *e.aliases))]


def preset_values(preset: str) -> dict[str, float]:
    """The values the difficulty ``preset`` gives the settings it owns."""
    params = difficulty.params_for_preset(preset)
    return {s.name: getattr(params, s.name) for s in SETTINGS if s.preset_owned}


def factory_values() -> dict[str, float]:
    """Every slider's value as the app starts: the default preset's difficulty, the default tone's grays."""
    return {s.name: s.default for s in SETTINGS}



def setting(name: str) -> Setting:
    """The setting for the field ``name``; KeyError if the app offers no such setting."""
    return _BY_NAME[name]


def in_group(group: str) -> list[Setting]:
    """The settings shown in ``group``, in the order the panel shows them."""
    return [s for s in SETTINGS if s.group == group]


def values(params: DifficultyParams, style: PageStyle, handling: Handling) -> dict[str, float]:
    """Every setting's value in the three objects, by name."""
    owners = {DIFFICULTY: params, STYLE: style, HANDLING: handling}
    return {s.name: getattr(owners[s.owner], s.name) for s in SETTINGS}


def apply(
    chosen: Mapping[str, float],
    params: DifficultyParams,
    style: PageStyle = PageStyle(),
    handling: Handling = Handling(),
) -> tuple[DifficultyParams, PageStyle, Handling]:
    """The three objects with the settings in ``chosen`` (name to value) set, each held within its range.

    Raises KeyError for a name that is not a setting.
    """
    changes: dict[str, dict[str, float]] = {DIFFICULTY: {}, STYLE: {}, HANDLING: {}}
    for name, value in chosen.items():
        s = setting(name)
        changes[s.owner][name] = s.clamp(value)
    return (
        dataclasses.replace(params, **changes[DIFFICULTY]),
        dataclasses.replace(style, **changes[STYLE]),
        dataclasses.replace(handling, **changes[HANDLING]),
    )

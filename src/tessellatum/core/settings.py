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

Sizes are on the printed A4 page, as everywhere in the pipeline (see
``print_size``). The defaults are what the quality benchmarks are measured at
(``benchmarks/QUALITY_BENCHMARKS.md``); a setting moved past them -- a finer
brush, closer colors -- can make a page with more detail than those targets
allow, which is the user's call.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Mapping

from tessellatum.core import difficulty
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.pipeline import Handling
from tessellatum.core.render import LINE_WIDTH_MM_RANGE, LINE_WIDTH_MM_STEP, PageStyle

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
    it is shown in.
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
        if self.whole:
            number = f"{int(round(value))}"
        else:
            decimals = max(0, -math.floor(math.log10(self.step) + 1e-9)) if self.step < 1 else 0
            number = f"{value:.{decimals}f}"
        return f"{number} {self.unit}".strip()


REGIONS = "Regions and colors"
FACES = "Faces and subject"
LINES = "Lines and numbers"
LINE_ART = "Line art"
TEXT = "Text"
GROUPS = (REGIONS, FACES, LINES, LINE_ART, TEXT)


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
        "min_region_area_mm2", "Smallest region", "mm²", 1,
        "The smallest area a region may have on the printed A4 page; smaller ones merge into a neighbor. Never "
        "below the brush's own footprint.",
        log=True,
    ),
    _difficulty(
        "min_width_mm", "Brush width", "mm", 0.1,
        "The narrowest any part of a region may be on the printed page: the brush it is painted with. Narrower "
        "parts are given to the region beside them, which also rounds sharp corners unless Sharpest corner keeps "
        "them. A finer brush keeps thinner shapes.",
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
        LINE_WIDTH_MM_STEP, "How wide the lines print on the A4 page.",
    ),
    Setting(
        "line_smoothing_mm", STYLE, LINES, "Line smoothing", "mm", 0.0, 2.0, 0.05,
        "How far along a line its pixel staircase is smoothed away. More rounds bends and corners; the pixel "
        "steps themselves always go.",
    ),
    Setting(
        "min_label_pt", STYLE, LINES, "Smallest number", "pt", 4.0, 12.0, 0.5,
        "The smallest a region's number prints. Smaller fits more numbers inside small regions; below 6 pt they "
        "get hard to read. A preview draws no number under 10 pixels, 6.5-7 pt, so smaller sizes show on an export.",
    ),
    Setting(
        "leader_reach_mm", STYLE, LINES, "Leader reach", "mm", 1.0, 20.0, 0.5,
        "How far outside a region too small for its number the number may be written, with a line pointing in.",
    ),
    Setting(
        "text_gap_mm", STYLE, LINES, "Gap beside text", "mm", 0.0, 3.0, 0.1,
        "How far a number keeps from the lettering of a sign or caption, so it doesn't read as part of it.",
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

_BY_NAME = {setting.name: setting for setting in SETTINGS}


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

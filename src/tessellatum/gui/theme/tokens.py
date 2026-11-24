"""The design tokens: every color, size and duration the interface is drawn with, in one place.

Two color sets, light and dark, share one structure; widgets never name a color, they name a token (through the
palette and the stylesheet built from these, see ``style``). The accent is Viridian, a pigment a painter knows: deep
in light, lighter in dark, so that it keeps its contrast on both. The artwork itself (the page, its versions and
legend) is never themed: it is shown with its own pixels.

Every text color is at least 4.5:1 against the surfaces it sits on, and every boundary or icon that carries meaning
at least 3:1 (WCAG 2.2 AA); ``CONTRAST_PAIRS`` lists them and the tests check each.
"""

from __future__ import annotations

from dataclasses import dataclass

LIGHT, DARK = "light", "dark"
SCHEMES = (LIGHT, DARK)


@dataclass(frozen=True)
class Colors:
    """One scheme's colors, as ``#RRGGBB``."""

    window: str  # the app's background
    chrome: str  # toolbars, the status bar, menus
    sidebar: str  # the settings panel
    raised: str  # buttons
    field: str  # text fields, spin boxes, combo boxes
    canvas: str  # behind the page: a step darker than the panels, so the page reads first
    border: str  # separators and quiet outlines
    border_strong: str  # the outlines of controls (3:1 on fields)
    text: str
    text_secondary: str
    text_hint: str  # hints and readouts beside a control
    text_disabled: str  # exempt from contrast minimums, as WCAG allows for disabled controls
    hover: str
    pressed: str
    accent: str
    accent_hover: str
    accent_pressed: str
    on_accent: str  # text on the accent
    selected: str  # a selected or checked control's fill
    info: str
    success: str
    warning: str
    error: str
    tooltip: str
    tooltip_text: str


def _blend(a: str, b: str, t: float) -> str:
    """``a`` moved ``t`` of the way to ``b`` in sRGB."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


ACCENT_NAME = "Viridian"
_ACCENT = {LIGHT: "#0D7468", DARK: "#43B9A9"}


def _colors(scheme: str) -> Colors:
    if scheme == LIGHT:
        base = dict(
            window="#F3F3F5", chrome="#EBEBEE", sidebar="#F8F8FA", raised="#FFFFFF", field="#FFFFFF",
            canvas="#DDDEE2", border="#D2D3D8", border_strong="#83858E", text="#1C1D20", text_secondary="#50535B",
            text_hint="#666971", text_disabled="#A3A5AC", hover="#E5E6EA", pressed="#D9DADF", info="#2F5BD3",
            success="#1A784A", warning="#8F5B00", error="#B8322A", tooltip="#1C1D20", tooltip_text="#F3F3F5",
        )
        accent, on_accent, toward = _ACCENT[LIGHT], "#FFFFFF", "#000000"
        selected = _blend(base["raised"], accent, 0.14)
    else:
        base = dict(
            window="#1E1F22", chrome="#232428", sidebar="#26272B", raised="#2E3035", field="#1B1C1F",
            canvas="#151618", border="#3A3C42", border_strong="#6E717A", text="#E8E9EC", text_secondary="#A9ACB4",
            text_hint="#9598A0", text_disabled="#62656D", hover="#32343A", pressed="#3A3D44", info="#7397F5",
            success="#55BE8A", warning="#E3A33B", error="#F2766A", tooltip="#E8E9EC", tooltip_text="#1E1F22",
        )
        accent, on_accent, toward = _ACCENT[DARK], "#101114", "#FFFFFF"
        selected = _blend(base["raised"], accent, 0.22)
    return Colors(
        **base, accent=accent, accent_hover=_blend(accent, toward, 0.12), accent_pressed=_blend(accent, toward, 0.24),
        on_accent=on_accent, selected=selected,
    )


COLORS = {scheme: _colors(scheme) for scheme in SCHEMES}

# Type: Inter, in three weights; sizes relative to the platform's own UI size, so a user's text scaling carries.
FONT_FAMILY = "Inter"
FONT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")
WEIGHT_REGULAR, WEIGHT_MEDIUM, WEIGHT_STRONG = 400, 500, 600
TYPE_SCALE = {"caption": 0.9, "body": 1.0, "heading": 1.0, "title": 1.25}  # times the platform's UI point size

# Space, corners and icons, in device-independent pixels.
SPACING = {"xs": 2, "s": 4, "m": 8, "l": 12, "xl": 16, "xxl": 24}
RADIUS = {"s": 3, "m": 4, "l": 8}
ICON_SIZE = {"s": 16, "m": 20, "l": 24}
ICON_STROKE = 1.5  # Lucide's 2 is heavy beside 9 pt text at 16 px
CONTROL_HEIGHT = 26  # buttons, fields and combo boxes, so rows line up

# Motion, in milliseconds; all 0 when the system or the user asks for reduced motion (see ``motion``).
MOTION_MS = {"fast": 120, "normal": 200, "slow": 320}


def contrast(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio of two ``#RRGGBB`` colors, 1 to 21."""

    def luminance(color: str) -> float:
        channels = []
        for i in (1, 3, 5):
            v = int(color[i:i + 2], 16) / 255
            channels.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


TEXT, NON_TEXT = 4.5, 3.0
# (foreground token, background token, minimum ratio): every pairing the interface draws.
CONTRAST_PAIRS = (
    *((fg, bg, TEXT) for fg in ("text", "text_secondary", "text_hint")
      for bg in ("window", "chrome", "sidebar", "raised", "field")),
    ("text", "hover", TEXT),
    ("text", "pressed", TEXT),
    ("text", "selected", TEXT),
    ("on_accent", "accent", TEXT),
    ("on_accent", "accent_hover", TEXT),
    ("on_accent", "accent_pressed", TEXT),
    ("tooltip_text", "tooltip", TEXT),
    *((status, bg, TEXT) for status in ("info", "success", "warning", "error") for bg in ("window", "field", "chrome")),
    *(("accent", bg, NON_TEXT) for bg in ("window", "chrome", "sidebar", "field", "raised")),
    *(("border_strong", bg, NON_TEXT) for bg in ("field", "sidebar", "window")),
    ("text", "canvas", TEXT),
)

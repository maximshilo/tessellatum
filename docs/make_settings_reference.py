"""Regenerate ``wiki/Settings-Reference.md``: every setting the app offers, from the registry itself (``core.settings``):
where it is, its unit, range, step and default, what each difficulty preset sets it to, when a change to it is applied,
the words that find it, and what it does.

    .venv\\Scripts\\python docs\\make_settings_reference.py          # rewrite the page
    .venv\\Scripts\\python docs\\make_settings_reference.py --check  # fail if the page is out of date

``tests/test_settings_reference.py`` runs the check, so the page can't fall behind the registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tessellatum.core import difficulty, settings  # noqa: E402

PAGE = ROOT / "wiki" / "Settings-Reference.md"
LEVEL_NAMES = {settings.INTENT: "Shown first", settings.PRECISE: "Worth a look", settings.ADVANCED: "Advanced"}
APPLIED = {
    settings.IMMEDIATE: "at once",
    settings.DEBOUNCED: "drawn again by itself",
    settings.EXPLICIT: "with Generate",
}
KINDS = {settings.ENUM: "one of", settings.BOOL: "on or off", settings.PAINTS: "a list of paints"}


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _value(setting: settings.Setting, value: float) -> str:
    return setting.text(value).replace(" ", " ")  # a value and its unit kept together


def _choice_default(choice: settings.Choice) -> str:
    if choice.kind == settings.BOOL:
        return "on" if choice.default else "off"
    if choice.kind == settings.PAINTS:
        return "none"
    return str(choice.default)


def page() -> str:
    presets = list(difficulty.PRESETS)
    lines = [
        "# Settings reference",
        "",
        "Every setting in the sidebar, by page and section, as the app defines it. This page is generated from the "
        "settings registry (`src/tessellatum/core/settings.py`) by `docs/make_settings_reference.py`, and a test fails "
        "when it falls behind; what each setting is *for*, with examples, is in the [User guide](User-Guide#settings).",
        "",
        "- **Level**: *Shown first* (the choices most pages need), *Worth a look*, or *Advanced* (folded away in the "
        "page's Advanced group). See [Levels of detail](User-Guide#levels-of-detail).",
        "- **Applied**: *with Generate* (it makes the page again), *drawn again by itself* (it only draws the page "
        "again from its regions, a moment after the last change; Preferences sets how soon), or *at once* (it changes "
        "only what is drawn over the page).",
        "- **Sizes** are on the printed page, on the paper chosen. Colors differ in CIEDE2000 (ΔE00: about 1 is the "
        "smallest difference an eye sees side by side); grays are a lightness, L* 0 (black) to 100 (white).",
        "- **Presets**: the difficulty presets set the settings under Generation > Detail, Faces and subject, Line art "
        "and Text, and the Colors page's palette. Moving one makes the difficulty Custom.",
        "",
    ]
    for category in settings.CATEGORIES:
        lines += [f"## {category}", ""]
        choices = [c for c in settings.CHOICES if c.category == category]
        if choices:
            lines += ["| Choice | Section | Level | Takes | Default | Applied | Found by | What it does |",
                      "|---|---|---|---|---|---|---|---|"]
            for c in choices:
                takes = KINDS.get(c.kind, c.kind)
                if c.options:
                    takes += ": " + ", ".join(str(o) for o in c.options)
                lines.append("| " + " | ".join(_cell(x) for x in (
                    f"**{c.label}** (`{c.name}`)", c.section, LEVEL_NAMES[c.level], takes, _choice_default(c),
                    APPLIED[c.policy], ", ".join(c.aliases), c.tooltip)) + " |")
            lines.append("")
        sliders = [s for s in settings.SETTINGS if s.category == category]
        if sliders:
            lines += ["| Setting | Section | Level | Range (step) | Default | " + " | ".join(presets)
                      + " | Applied | Found by | What it does |",
                      "|---|---|---|---|---|" + "---|" * len(presets) + "---|---|---|"]
            for s in sliders:
                values = settings.preset_values
                by_preset = [_value(s, values(p)[s.name]) if s.preset_owned else "" for p in presets]
                range_ = f"{_value(s, s.minimum)} to {_value(s, s.maximum)} ({s.text(s.step) if not s.shown else 1})"
                lines.append("| " + " | ".join(_cell(x) for x in (
                    f"**{s.label}** (`{s.name}`)", s.section, LEVEL_NAMES[s.level], range_, _value(s, s.default),
                    *by_preset, APPLIED[s.policy], ", ".join(s.aliases), s.tooltip)) + " |")
            lines.append("")
    lines += [
        "## Where they are kept",
        "",
        "The settings shown are kept between runs, each under its key (`settings/<name>` for the sliders; the choices' "
        "keys are `difficulty/…`, `style/…` and `handling/…`), in the platform's settings store: the registry on "
        "Windows, a `.plist` on macOS, an `.ini` file on Linux. A project file keeps them with its picture instead "
        "(see [Projects](User-Guide#projects)).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the page is out of date")
    args = parser.parse_args()
    text = page()
    if args.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
        if current != text:
            print(f"{PAGE.relative_to(ROOT)} is out of date: run docs/make_settings_reference.py")
            return 1
        print(f"{PAGE.relative_to(ROOT)} is current")
        return 0
    PAGE.write_text(text, encoding="utf-8", newline="\n")
    print("written", PAGE.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

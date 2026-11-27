"""Collect the interface's words into Qt translation files, and compile them for the app.

    python packaging/update_translations.py                 # list what is collected, by context
    python packaging/update_translations.py de fr           # write or update translations/tessellatum_de.ts, _fr.ts
    python packaging/update_translations.py --release       # compile every .ts into the app's .qm files

The interface's words come from two places, and both are collected (``catalog``):

- The code: every ``translate("context", "text")`` and ``QT_TRANSLATE_NOOP("context", "text")`` in ``gui/``, read
  from the source (strings split over lines are joined as Python joins them), and every ``tr("text")`` in the core's
  modules that phrase sentences for the interface (``CORE_PHRASES``), under ``settings``.
- The data: the commands (their texts, tips and menus, under ``actions``), and the core's words the interface shows
  through ``gui.i18n.term``: the settings' labels, tooltips, sections, choices and reasons, the versions' names, the
  stages' names, under ``settings``.

A ``.ts`` file is updated in place: a string still in the code keeps its translation, a new one is added unfinished,
and one no longer in the code is marked obsolete (kept, so a translation is not lost to a rename). Translators edit
the ``.ts`` files (Qt Linguist, or any XLIFF-aware tool after conversion); ``--release`` runs ``pyside6-lrelease`` to
make the ``.qm`` files ``gui.i18n.install`` loads from ``resources/translations``.
"""

from __future__ import annotations

import ast
import itertools
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tessellatum"
TS_DIR = ROOT / "translations"
QM_DIR = SRC / "resources" / "translations"
SETTINGS_CONTEXT = "settings"
# The core's modules that phrase sentences for the interface with a ``tr`` passed in (the core has no Qt).
CORE_PHRASES = ("core/difficulty.py",)
# The core's modules whose words (constants, and the registry's entries) the interface shows through ``term``.
CORE_WORDS = ("tessellatum.core.settings", "tessellatum.core.difficulty", "tessellatum.core.painting",
              "tessellatum.core.pipeline", "tessellatum.core.swatches", "tessellatum.app.jobs")
_KEY = re.compile(r"[a-z0-9_./:-]*")  # a key, a name in code or a path: not words to translate


def _const(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def from_code() -> set[tuple[str, str]]:
    """(context, text) for every literal translate/QT_TRANSLATE_NOOP call in ``gui/``, and tr call in the core's
    phrasing modules."""
    found = set()
    for path in sorted((SRC / "gui").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            context, text = _const(node.args[0]), _const(node.args[1])
            if name in ("translate", "QT_TRANSLATE_NOOP") and context and text:
                found.add((context, text))
    for name in CORE_PHRASES:
        for node in ast.walk(ast.parse((SRC / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "tr" and node.args:
                text = _const(node.args[0])
                if text:
                    found.add((SETTINGS_CONTEXT, text))
    return found


def _words(value, seen: set[int] | None = None):
    """Every string with words in it in ``value``, through tuples, lists, dicts (keys and values) and enums."""
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, str):
        if re.search(r"[A-Za-z]", value) and not _KEY.fullmatch(value):
            yield value
    elif isinstance(value, dict):
        for item in itertools.chain(value.keys(), value.values()):
            yield from _words(item, seen)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _words(item, seen)


def from_data() -> set[tuple[str, str]]:
    """(context, text) for the commands and the core's words the interface shows (see the module's docstring)."""
    sys.path.insert(0, str(ROOT / "src"))
    import importlib

    from tessellatum.core import settings
    from tessellatum.core.painting import Version
    from tessellatum.gui import actions

    found = set()
    for command in actions.COMMANDS:
        for text in (command.text, command.tip, command.submenu):
            if text:
                found.add(("actions", text))
    found.update(("actions", title) for title in actions.MENUS)
    words = set()
    for name in CORE_WORDS:
        module = importlib.import_module(name)
        for attr, value in vars(module).items():
            if attr.isupper() or attr.startswith("_") and attr[1:].isupper():
                words.update(_words(value))
    for entry in (*settings.SETTINGS, *settings.CHOICES):
        for field in ("label", "tooltip", "section", "category", "unit"):
            words.update(_words(getattr(entry, field, None)))
        words.update(_words(getattr(entry, "options", ())))
        words.update(_words(getattr(entry, "option_tips", {})))
    words.update(_reasons(settings))
    for version in Version:
        words.update((version.value, version.description))
    # Words the interface derives: a palette format's name without its file pattern, and an advanced section's name
    # without its "Advanced · " (the title of its part of the page's Advanced group).
    from tessellatum.core import swatches

    words.update(title.split(" (")[0] for title in swatches.FORMATS.values())
    words.update(word.replace("Advanced · ", "") for word in list(words) if word.startswith("Advanced · "))
    found.update((SETTINGS_CONTEXT, text) for text in words)
    return found


def _reasons(settings) -> set[str]:
    """Every reason a setting or choice can be turned off with, over every combination of the choices it reads."""
    values = {
        "medium": [*settings.choice("medium").options],
        "colors_from": list(settings.COLORS_FROM),
        "mix_paints": [False, True],
        **{switch: [False, True] for switch in settings.SWITCHES},
    }
    names = list(values)
    found = set()
    for combination in itertools.product(*(values[name] for name in names)):
        chosen = dict(zip(names, combination))
        for entry in (*settings.SETTINGS, *settings.CHOICES):
            reason = entry.disabled_reason(chosen)
            if reason:
                found.add(reason)
    return found


def catalog() -> dict[str, set[str]]:
    """Every string the interface can show, by context."""
    by_context = defaultdict(set)
    for context, text in from_code() | from_data():
        by_context[context].add(text)
    return dict(by_context)


def write_ts(path: Path, language: str, messages: dict[str, set[str]]) -> tuple[int, int, int]:
    """Write or update ``path`` for ``language`` with ``messages``; (kept, added, made obsolete)."""
    old: dict[tuple[str, str], ET.Element] = {}
    if path.exists():
        for context in ET.parse(path).getroot().iter("context"):
            name = context.findtext("name")
            for message in context.iter("message"):
                old[(name, message.findtext("source"))] = message.find("translation")
    root = ET.Element("TS", version="2.1", language=language)
    kept = added = 0
    wanted = {(context, text) for context, texts in messages.items() for text in texts}
    for context_name in sorted({c for c, _ in wanted} | {c for c, _ in old}):
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = context_name
        for text in sorted({t for c, t in wanted | set(old) if c == context_name}):
            message = ET.SubElement(context, "message")
            ET.SubElement(message, "source").text = text
            translation = ET.SubElement(message, "translation")
            previous = old.get((context_name, text))
            if previous is not None:
                translation.text = previous.text
                translation.attrib.update(previous.attrib)
            else:
                translation.set("type", "unfinished")
            if (context_name, text) not in wanted:
                translation.set("type", "vanished")
            elif previous is not None:
                kept += 1
                if previous.get("type") == "vanished":
                    translation.attrib.pop("type")
            else:
                added += 1
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ET.tostring(root, encoding="unicode")
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n' + body + "\n", encoding="utf-8")
    obsolete = sum(1 for key in old if key not in wanted)
    return kept, added, obsolete


def release() -> int:
    """Compile every ``translations/*.ts`` into ``resources/translations``; how many."""
    QM_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for ts in sorted(TS_DIR.glob("tessellatum_*.ts")):
        tool = Path(sys.executable).with_name("pyside6-lrelease")
        subprocess.run([str(tool), str(ts), "-qm", str(QM_DIR / ts.with_suffix(".qm").name)], check=True)
        count += 1
    return count


def main(argv: list[str]) -> int:
    if "--release" in argv:
        print(f"compiled {release()} translation(s) into {QM_DIR.relative_to(ROOT)}")
        return 0
    messages = catalog()
    languages = [arg for arg in argv if not arg.startswith("-")]
    if not languages:
        for context in sorted(messages):
            print(f"{context:20} {len(messages[context]):5}")
        print(f"{'total':20} {sum(len(texts) for texts in messages.values()):5}")
        return 0
    for language in languages:
        path = TS_DIR / f"tessellatum_{language}.ts"
        kept, added, obsolete = write_ts(path, language, messages)
        print(f"{path.relative_to(ROOT)}: {kept} kept, {added} new, {obsolete} obsolete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

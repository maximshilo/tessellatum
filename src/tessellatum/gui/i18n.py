"""The interface's words, ready to be translated; no translation ships yet.

Every string the interface shows goes through ``translate(context, text)``: Qt's ``QCoreApplication.translate``, the
context being the module the string is in. A string kept in a constant is marked with ``QT_TRANSLATE_NOOP`` where it
is written and translated where it is shown. The core's words (the settings' labels, tooltips, sections, choices and
reasons; the versions' and the stages' names) come from a layer with no Qt and are translated as they are shown, with
``term``, under the context ``settings``; a sentence the core phrases takes a ``tr`` to phrase it with. A choice in a
combo box or a segmented control is shown translated and known to the code by its value (``fields.ChoiceCombo``).
``packaging/update_translations.py`` collects all of them into a ``.ts`` file per language and compiles those.

Placeholders are named, ``{count} colors``, and filled after the translation, so a language can move them; the
string ``translate`` returns writes numbers the locale's way as it fills them (``Text``: "1.448" and "2,5" in German).
A count is said by one string for one and another for more, chosen in the code (English's two forms); a language
with more plural forms would move those to Qt's ``%n`` forms, with an English catalog for English's.

``install`` loads ``tessellatum_<language>.qm`` from ``resources/translations`` for the system's language, if one is
there, and Qt's own translation of its words with it. ``TESSELLATUM_PSEUDO_LOCALE=1`` installs a pseudo-translation
instead: every string lengthened by about 40% and accented, between brackets, to find a string that isn't translated
(it shows without them) and a label that clips when a language is longer than English. Qt's own ``-reverse``
argument lays the window out right to left.

What the page prints (the legend's headings and color names in the PDF and PNG) stays English: it is drawn by the
pipeline, which has no Qt.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

TRANSLATIONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "translations"
PSEUDO_ENV = "TESSELLATUM_PSEUDO_LOCALE"
OPEN, CLOSE = "[", "]"  # around a pseudo-translated string

_installed: list[QTranslator] = []  # kept alive while installed


class Text(str):
    """A translated string whose ``format`` writes numbers the locale's way (see ``LocaleFormatter``)."""

    def format(self, *args, **kwargs) -> str:  # noqa: A003 - str's own name
        return _FORMATTER.vformat(self, args, kwargs)


class LocaleFormatter(string.Formatter):
    """``str.format`` with numbers written by the default ``QLocale``: ``{0:,}``, ``{0:.1f}``, ``{0:,.0f}`` and
    ``{0:g}`` give "1.448", "2,5" and "0,5" in German where Python gives "1,448", "2.5" and "0.5". A comma in the spec
    asks for the locale's group separator, as it asks for Python's; any other spec is Python's."""

    _SPEC = re.compile(r",?(?:\.\d+)?[fgd]?")

    def format_field(self, value, format_spec: str) -> str:
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not format_spec or not numeric or not self._SPEC.fullmatch(format_spec):
            return super().format_field(value, format_spec)
        return _localized(format(value, format_spec))


_FORMATTER = LocaleFormatter()


def _localized(text: str) -> str:
    """A number as Python writes it ("1,448.5") with the default locale's group separator and decimal point."""
    locale = QLocale()
    return text.translate({ord(","): "\x00", ord("."): locale.decimalPoint()}).replace("\x00", locale.groupSeparator())


def number(value: float, decimals: int | None = None, grouped: bool = True) -> str:
    """``value`` written the default locale's way, with ``decimals`` decimals (a whole number as it is, a float as
    Python writes it, if None) and the group separator if ``grouped``: "1,448.5" in English, "1.448,5" in German."""
    spec = ("," if grouped else "") + (f".{decimals}f" if decimals is not None else "")
    return _localized(format(value, spec))


def translate(context: str, text: str, disambiguation: str | None = None, n: int = -1) -> Text:
    """``text`` in the interface's language: ``QCoreApplication.translate``. ``n`` picks a plural form (``%n``). Its
    ``format`` writes numbers the locale's way (``Text``)."""
    return Text(QCoreApplication.translate(context, text, disambiguation, n))


# -- The pseudo-locale --------------------------------------------------------------------------
_ACCENTED = str.maketrans("aeiouyAEIOUYcnsCNS", "àéîõüýÀÉÎÕÜÝçñšÇÑŠ")
# What a translation keeps as it is: placeholders ({name}, {0:.1f}, %n, %1), markup, entities and mnemonics' "&".
_KEPT = re.compile(r"\{[^{}]*\}|%n|%L?\d|<[^<>]*>|&[a-zA-Z]+;|&&|&")
_VOWELS = set("aeiouyAEIOUY")


def pseudo(text: str) -> str:
    """``text`` pseudo-translated: accented, each vowel doubled (about 40% longer, as German or French often is), in
    brackets; its placeholders, markup and mnemonics kept. Empty or blank text stays as it is."""
    if not text.strip():
        return text
    out, last = [], 0
    for match in _KEPT.finditer(text):
        out.append(_stretch(text[last:match.start()]))
        out.append(match.group())
        last = match.end()
    out.append(_stretch(text[last:]))
    return OPEN + "".join(out) + CLOSE


def _stretch(part: str) -> str:
    return "".join(c + c if c in _VOWELS else c for c in part).translate(_ACCENTED)


def is_pseudo(text: str) -> bool:
    return text.startswith(OPEN) and text.endswith(CLOSE)


class PseudoTranslator(QTranslator):
    """Translates every string into the pseudo-locale (see ``pseudo``)."""

    def translate(self, context, source_text, disambiguation=None, n=-1):  # noqa: D401 - Qt's signature
        return pseudo(source_text) if source_text else source_text

    def isEmpty(self) -> bool:
        return False


def install(app: QCoreApplication | None = None, locale: QLocale | None = None) -> str | None:
    """Install the translation for ``locale`` (the system's by default), or the pseudo-locale if asked for; what was
    installed ("pseudo", the .qm file's name), or None."""
    app = app or QCoreApplication.instance()
    if app is None:
        return None
    if os.environ.get(PSEUDO_ENV, "") not in ("", "0"):
        translator = PseudoTranslator(app)
        app.installTranslator(translator)
        _installed.append(translator)
        return "pseudo"
    locale = locale or QLocale()
    translator = QTranslator(app)
    if not translator.load(locale, "tessellatum", "_", str(TRANSLATIONS_DIR)):
        return None
    # Qt's own words (its dialogs' buttons, a line edit's menu) in the same language, where Qt ships them.
    qt = QTranslator(app)
    if qt.load(locale, "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(qt)
        _installed.append(qt)
    app.installTranslator(translator)
    _installed.append(translator)
    return Path(translator.filePath()).name


def uninstall(app: QCoreApplication | None = None) -> None:
    """Take every translation ``install`` installed off again (for tests)."""
    app = app or QCoreApplication.instance()
    while _installed:
        translator = _installed.pop()
        if app is not None:
            app.removeTranslator(translator)


SETTINGS_CONTEXT = "settings"  # the registry's words (``core.settings``, ``core.difficulty``, ``core.painting``)


def term(text: str | None) -> str | None:
    """A word or sentence from the core (a setting's label, tooltip, section or reason; a choice; a version's name),
    in the interface's language. The core has no Qt, so its strings are translated here, as they are shown;
    ``packaging/update_translations.py`` collects them into the catalog under ``SETTINGS_CONTEXT``."""
    return translate(SETTINGS_CONTEXT, text) if text else text

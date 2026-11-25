"""A document's settings, as one immutable value: what the page is made with, and how it is exported.

The panel edits it, the document keeps it, and the pipeline's three settings objects (``DifficultyParams``,
``PageStyle``, ``Handling``) are worked out from it, so every view of the settings agrees. Qt-free.

``values`` holds every setting of ``core.settings`` by name, as the panel shows it: for a difficulty other than
Custom, its preset's values (which ``difficulty_params`` takes from the preset itself), and the tone's grays unless
the tone is Custom.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from tessellatum.core import difficulty, settings, swatches
from tessellatum.core.legend import DEFAULT_LEGEND_DETAIL
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import Handling
from tessellatum.core.render import DEFAULT_TONE, TONE_FIELDS, TONES, PageStyle
from tessellatum.core.settings import (  # noqa: F401 - the registry's names for the choices' values, kept here too
    COLORS_FROM,
    CUSTOM,
    CUSTOM_TONE,
    DEFAULT_PAPER,
    MY_PAINTS,
    PICTURE_COLORS,
    SWITCHES,
)

PNG, PDF, PALETTE = "PNG", "PDF", "Palette"
FORMATS = (PNG, PDF, PALETTE)
DPI, PIXELS = "dpi", "px"
DEFAULT_DPI = 600


@dataclass(frozen=True)
class ExportOptions:
    """How the next export is saved."""

    version: Version = Version.PAGE
    format: str = PNG
    palette_format: str = next(iter(swatches.FORMATS))
    unit: str = DPI
    dpi: int = DEFAULT_DPI
    pixels: int | None = None  # None until first asked for in pixels


def default_values(preset: str = difficulty.DEFAULT_PRESET, tone: str = DEFAULT_TONE) -> dict[str, float]:
    """Every setting's value as the panel starts: the preset's difficulty, the tone's grays, the rest their defaults."""
    values = {s.name: s.default for s in settings.SETTINGS}
    if preset != CUSTOM:
        params = difficulty.params_for_preset(preset)
        values.update({s.name: getattr(params, s.name) for s in settings.SETTINGS if s.owner == settings.DIFFICULTY})
    if tone in TONES:
        values.update(zip(TONE_FIELDS, TONES[tone]))
    return values


# Each choice's field here, and its entry in the registry.
_CHOICE_FIELDS = {
    "preset": "preset", "tone": "tone", "paper": "paper", "medium": "medium", "legend": "legend_detail",
    "colors_from": "colors_from", "paints": "paints", "mix": "mix_paints", "line_art": "line_art",
    "detail": "detail", "text": "text",
}


@dataclass(frozen=True)
class DocumentSettings:
    preset: str = difficulty.DEFAULT_PRESET
    values: Mapping[str, float] = field(default_factory=lambda: MappingProxyType(default_values()))
    tone: str = DEFAULT_TONE
    paper: str = DEFAULT_PAPER
    medium: str = difficulty.DEFAULT_MEDIUM
    legend: str = DEFAULT_LEGEND_DETAIL
    colors_from: str = PICTURE_COLORS
    paints: tuple = ()  # the paints listed, (name, (r, g, b)) each, kept whatever the colors come from
    mix: bool = False
    line_art: bool = True
    detail: bool = True
    text: bool = True
    export: ExportOptions = ExportOptions()

    def __post_init__(self):
        if not isinstance(self.values, MappingProxyType):
            object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        missing = {s.name for s in settings.SETTINGS} - set(self.values)
        if missing:
            raise ValueError(f"no value for {', '.join(sorted(missing))}")
        for name, entry in _CHOICE_FIELDS.items():
            if name == "preset" and isinstance(self.preset, str) and self.preset.strip():
                continue  # a built-in preset, Custom, or a preset of one's own (``app.presets``): any name
            reason = settings.choice(entry).check(getattr(self, name))
            if reason is not None:
                raise ValueError(f"{name}: {reason}")

    def chosen(self) -> dict:
        """The choices, by their registry names: what the registry's enablement reads (``disabled_reason``)."""
        return {entry: getattr(self, name) for name, entry in _CHOICE_FIELDS.items()}

    def __eq__(self, other):
        if not isinstance(other, DocumentSettings):
            return NotImplemented
        return all(
            (dict(getattr(self, f.name)) == dict(getattr(other, f.name))) if f.name == "values"
            else getattr(self, f.name) == getattr(other, f.name)
            for f in dataclasses.fields(self)
        )

    def __hash__(self):
        return hash(tuple(sorted(self.values.items())) + tuple(
            getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "values"
        ))

    def replace(self, **changes) -> DocumentSettings:
        return dataclasses.replace(self, **changes)

    def with_values(self, **values: float) -> DocumentSettings:
        unknown = set(values) - set(self.values)
        if unknown:
            raise KeyError(f"no setting {', '.join(sorted(unknown))}")
        return dataclasses.replace(self, values={**self.values, **values})

    # -- What the pipeline is asked for ---------------------------------------
    def _owned(self, owner: str) -> dict[str, float]:
        return {s.name: self.values[s.name] for s in settings.SETTINGS if s.owner == owner}

    def difficulty_params(self) -> difficulty.DifficultyParams:
        if self.preset in difficulty.PRESETS:
            params = difficulty.params_for_preset(self.preset)
        else:  # Custom, or a preset of one's own (``app.presets``), whose values are the document's
            params = difficulty.custom_params(**self._owned(settings.DIFFICULTY))
        return difficulty.for_medium(params, self.medium)

    def page_style(self) -> PageStyle:
        return PageStyle(**self._owned(settings.STYLE), paper=self.paper, legend_detail=self.legend)

    def paints_used(self) -> tuple:
        """The paints the page takes its colors from: none unless Colors from is My paints."""
        return self.paints if self.colors_from == MY_PAINTS else ()

    def handling(self) -> Handling:
        paints = self.paints_used()
        switches = {name: getattr(self, name) for name in SWITCHES}
        return Handling(**switches, **self._owned(settings.HANDLING), paints=paints, mix_paints=bool(paints) and self.mix)

    def changes_from(self, made: DocumentSettings) -> tuple[str, ...]:
        """The settings, by label, that differ from ``made`` (a page's) and change the page: not the legend's detail
        nor the export's options, which an export applies without making the page again. Empty if the page made with
        ``made`` is the page these would make."""
        if made.generation_key() == self.generation_key():
            return ()
        # Another difficulty is named as such, not as the settings it moved.
        preset = made.preset != self.preset
        labels = [settings.choice("preset").label] if preset else []
        labels += [
            s.label for s in settings.SETTINGS
            if made.values[s.name] != self.values[s.name] and not (preset and s.owner == settings.DIFFICULTY)
        ]
        for name, entry in _CHOICE_FIELDS.items():
            if entry not in ("legend", "preset") and getattr(made, name) != getattr(self, name):
                labels.append(settings.choice(entry).label)
        return tuple(labels) or ("the settings",)

    def change_text(self, before: DocumentSettings) -> str:
        """What changed from ``before``, in a few words, as Edit > Undo names it: the one setting moved ("Brush
        width"), the difficulty or the tone picked ("Difficulty: Hard"), or how many settings changed."""
        values = [s.label for s in settings.SETTINGS if before.values[s.name] != self.values[s.name]]
        choices = [
            settings.choice(entry).label for name, entry in _CHOICE_FIELDS.items()
            if entry not in ("preset", "tone") and getattr(before, name) != getattr(self, name)
        ]
        if len(values) == 1 and not choices:
            return values[0]
        if before.preset != self.preset and self.preset != CUSTOM:
            return f"Difficulty: {self.preset}"
        if before.tone != self.tone and self.tone != CUSTOM_TONE:
            return f"Tone: {self.tone}"
        names = values + choices + (["Export options"] if before.export != self.export else [])
        if len(names) == 1:
            return names[0]
        return f"{len(names)} settings" if names else "Settings"

    def generation_key(self) -> tuple:
        """What the page made with these settings depends on: equal keys make the same page. The legend's detail and
        the export options aren't in it, as changing them doesn't make the page again."""
        style = dataclasses.replace(self.page_style(), legend_detail=DEFAULT_LEGEND_DETAIL)
        return self.difficulty_params(), style, self.handling()

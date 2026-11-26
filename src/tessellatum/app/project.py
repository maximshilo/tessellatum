"""Project files: a document saved whole, to be opened again on any computer.

A project (``.tessellatum``) is a zip archive holding:

- ``manifest.json``: the format and its version, the app's version, the settings by their registry names (the
  registry's names are its stable IDs), the paints listed, the export's options, and the picture's file name, size and
  SHA-256;
- the picture's original bytes, as they were read (``source`` plus its suffix), so the project carries the picture
  itself, wherever it came from;
- ``thumbnail.png``: the picture, small, for the lists of recent work.

Pages are not stored: they are made again from the picture and the settings when the project is opened, and making a
page is deterministic, so the page is the one that was saved.

**Saving** is atomic: the project is written to a temporary file in the same folder, read back and checked, and only
then put in the place of the file, so the file saved before stays whole until the new one is complete, and a save that
fails leaves it as it was.

**Reading** never trusts the file: the archive is read in memory, never extracted; its members' names are checked
(a path out of the archive is refused) and their sizes bounded, read up to the bound whatever the archive claims; the
manifest is validated field by field; the picture is checked against its hash and its size before it is decoded. A
file that fails any check raises ``ProjectError`` with what is wrong, and nothing of it is used.

**Versions:** ``FORMAT_VERSION`` is this app's. A project of an older version is migrated as it is read
(``_MIGRATIONS``); one of a newer version is refused with a message saying a newer Tessellatum made it. Settings the
project lacks take their defaults; settings it names that this version doesn't know are ignored.

Qt-free.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from tessellatum import __version__
from tessellatum.app.snapshot import (
    DPI,
    FORMATS,
    PIXELS,
    DocumentSettings,
    ExportOptions,
    default_values,
)
from tessellatum.core import settings, swatches
from tessellatum.core.painting import Version

SUFFIX = ".tessellatum"
FORMAT = "tessellatum-project"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
THUMBNAIL = "thumbnail.png"
SOURCE_STEM = "source"
THUMBNAIL_EDGE = 256  # px, the thumbnail's long edge

# Bounds on what is read, in bytes (and pixels for the picture): a file past them is refused rather than read.
MAX_MANIFEST = 1024 ** 2
MAX_SOURCE = 512 * 1024 ** 2
MAX_THUMBNAIL = 4 * 1024 ** 2
MAX_MEMBERS = 16
MAX_PIXELS = 200_000_000
IMAGE_FORMATS = {"PNG", "JPEG", "BMP", "WEBP", "TIFF", "GIF"}  # what Open Image reads (``OPEN_IMAGE_FILTER``)

_CHOICES = ("preset", "tone", "paper", "medium", "legend", "colors_from")  # DocumentSettings' text choices
_SWITCHES = ("mix", "line_art", "detail", "text")


class ProjectError(Exception):
    """A project that can't be read (or written), with what is wrong in words."""


@dataclass(frozen=True)
class Project:
    """What a project holds: the settings, and the picture as its original bytes and decoded (BGR)."""

    settings: DocumentSettings
    source_bytes: bytes
    source_name: str
    image: np.ndarray
    app_version: str = __version__
    format_version: int = FORMAT_VERSION


# -- The picture ---------------------------------------------------------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    """``data``, a picture file's bytes, as a BGR array, exactly as Open Image reads a file (``load_image_bgr``), after
    checking that it is a picture Open Image takes and no larger than ``MAX_PIXELS``."""
    from tessellatum.core import pipeline

    try:
        with Image.open(io.BytesIO(data)) as probe:
            kind, (width, height) = probe.format, probe.size
    except Exception as exc:  # noqa: BLE001 - whatever Pillow raises for bytes it can't read
        raise ProjectError(f"the picture in it can't be read ({exc})") from exc
    if kind not in IMAGE_FORMATS:
        raise ProjectError(f"the picture in it is a {kind or 'file of an unknown kind'}, not a picture Open Image takes")
    if width * height > MAX_PIXELS:
        raise ProjectError(f"the picture in it is {width:,} × {height:,} px, larger than Tessellatum opens")
    try:
        return pipeline.load_image_bgr(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ProjectError(f"the picture in it can't be read ({exc})") from exc


def thumbnail_png(image_bgr: np.ndarray, edge: int = THUMBNAIL_EDGE) -> bytes:
    """A small PNG of the picture, its long edge ``edge``."""
    rgb = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
    rgb.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    rgb.save(out, "PNG", optimize=True)
    return out.getvalue()


# -- The manifest --------------------------------------------------------------------------------------------------------
def settings_to_json(snapshot: DocumentSettings) -> dict:
    """``snapshot`` as the manifest keeps it: every setting by its registry name."""
    options = snapshot.export
    return {
        "choices": {name: getattr(snapshot, name) for name in _CHOICES},
        "switches": {name: getattr(snapshot, name) for name in _SWITCHES},
        "values": {s.name: snapshot.values[s.name] for s in settings.SETTINGS},
        "paints": [{"name": name, "rgb": list(rgb)} for name, rgb in snapshot.paints],
        "export": {
            "version": options.version.value, "format": options.format, "palette_format": options.palette_format,
            "unit": options.unit, "dpi": options.dpi, "pixels": options.pixels,
        },
    }


def _expect(condition: bool, what: str) -> None:
    if not condition:
        raise ProjectError(what)


def settings_from_json(data) -> DocumentSettings:
    """The settings a manifest keeps (see ``settings_to_json``), every one checked; a setting missing takes its
    default, one this version doesn't know is ignored."""
    _expect(isinstance(data, dict), "its settings aren't a table of settings")
    defaults = DocumentSettings()
    choices = data.get("choices", {})
    switches = data.get("switches", {})
    _expect(isinstance(choices, dict) and isinstance(switches, dict), "its settings aren't a table of settings")
    fields = {}
    for name in _CHOICES:
        value = choices.get(name, getattr(defaults, name))
        _expect(isinstance(value, str) and 0 < len(value) <= 200, f"its setting {name!r} isn't a name")
        fields[name] = value
    for name in _SWITCHES:
        value = switches.get(name, getattr(defaults, name))
        _expect(isinstance(value, bool), f"its setting {name!r} isn't on or off")
        fields[name] = value
    values = dict(default_values())
    kept = data.get("values", {})
    _expect(isinstance(kept, dict), "its settings' values aren't a table")
    known = {s.name: s for s in settings.SETTINGS}
    for name, value in kept.items():
        setting = known.get(name)
        if setting is None:
            continue  # a setting of a newer version: ignored
        _expect(isinstance(value, (int, float)) and not isinstance(value, bool), f"{setting.label} isn't a number")
        reason = setting.check(value)
        _expect(reason is None, f"{setting.label}: {reason}")
        values[name] = value  # JSON keeps a number exactly as it was written
    paints = data.get("paints", [])
    _expect(isinstance(paints, list) and len(paints) <= 1000, "its paints aren't a list")
    listed = []
    for paint in paints:
        _expect(isinstance(paint, dict), "a paint in it isn't a paint")
        name, rgb = paint.get("name"), paint.get("rgb")
        _expect(isinstance(name, str) and 0 < len(name) <= 200, "a paint in it has no name")
        _expect(isinstance(rgb, list) and len(rgb) == 3
                and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in rgb),
                f"the paint {name!r} has no color")
        listed.append((name, tuple(rgb)))
    export = data.get("export", {})
    _expect(isinstance(export, dict), "its export options aren't a table")
    base = ExportOptions()
    try:
        version = Version(export.get("version", base.version.value))
    except ValueError:
        raise ProjectError("its export's version isn't one there is") from None
    fmt = export.get("format", base.format)
    palette_format = export.get("palette_format", base.palette_format)
    unit = export.get("unit", base.unit)
    dpi = export.get("dpi", base.dpi)
    pixels = export.get("pixels", base.pixels)
    _expect(fmt in FORMATS, "its export's format isn't one there is")
    _expect(palette_format in swatches.FORMATS, "its palette format isn't one there is")
    _expect(unit in (DPI, PIXELS), "its export's unit isn't one there is")
    _expect(isinstance(dpi, int) and not isinstance(dpi, bool) and 1 <= dpi <= 10_000, "its export's dpi is wrong")
    _expect(pixels is None or (isinstance(pixels, int) and not isinstance(pixels, bool) and 1 <= pixels <= 1_000_000),
            "its export's size is wrong")
    try:
        return DocumentSettings(
            **fields, values=values, paints=tuple(listed),
            export=ExportOptions(version, fmt, palette_format, unit, dpi, pixels),
        )
    except ValueError as exc:
        raise ProjectError(f"its settings are wrong: {exc}") from exc


def manifest(snapshot: DocumentSettings, source_name: str, source_bytes: bytes, image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "app_version": __version__,
        "settings": settings_to_json(snapshot),
        "source": {
            "name": source_name, "member": _source_member(source_name), "size": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(), "width": width, "height": height,
        },
        "thumbnail": THUMBNAIL,
    }


def _source_member(source_name: str) -> str:
    suffix = PurePosixPath(source_name.replace("\\", "/")).suffix.lower()
    if not (suffix[1:].isalnum() and len(suffix) <= 6):
        suffix = ""
    return SOURCE_STEM + suffix


# Each older format version's manifest, made the next version's: (version, step). None yet: version 1 is the first.
_MIGRATIONS: dict[int, object] = {}


def _migrate(data: dict) -> dict:
    version = data.get("format_version")
    _expect(isinstance(version, int) and not isinstance(version, bool) and version >= 1,
            "it has no format version")
    if version > FORMAT_VERSION:
        made = data.get("app_version")
        by = f" (Tessellatum {made})" if isinstance(made, str) and len(made) < 40 else ""
        raise ProjectError(f"it was saved by a newer Tessellatum{by}, in project format {version}; this version reads "
                           f"format {FORMAT_VERSION} and older. Update Tessellatum to open it.")
    while version < FORMAT_VERSION:
        data = _MIGRATIONS[version](data)
        version = data["format_version"]
    return data


# -- Writing ---------------------------------------------------------------------------------------------------------
def project_bytes(snapshot: DocumentSettings, source_name: str, source_bytes: bytes, image: np.ndarray) -> bytes:
    """The project file's bytes."""
    info = manifest(snapshot, source_name, source_bytes, image)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(MANIFEST, json.dumps(info, indent=2, ensure_ascii=False), zipfile.ZIP_DEFLATED)
        archive.writestr(info["source"]["member"], source_bytes, zipfile.ZIP_STORED)  # already compressed, mostly
        archive.writestr(THUMBNAIL, thumbnail_png(image), zipfile.ZIP_STORED)
    return out.getvalue()


def save_project(path: Path, snapshot: DocumentSettings, source_name: str, source_bytes: bytes,
                 image: np.ndarray) -> None:
    """Write the project to ``path`` atomically (see the module's docstring): a file there is replaced only once the
    new one is written and reads back whole. Raises ``OSError`` if it can't be written, ``ProjectError`` if what was
    written doesn't read back."""
    path = Path(path)
    data = project_bytes(snapshot, source_name, source_bytes, image)
    handle, temp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        read_project(Path(temp))  # whole, and as it was meant
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


# -- Reading ---------------------------------------------------------------------------------------------------------
def _safe_name(name: str) -> bool:
    """Whether ``name`` stays inside the archive: relative, no parent steps, no drive, no backslashes."""
    if not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name or "\x00" in name:
        return False
    return ".." not in PurePosixPath(name).parts


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int, what: str) -> bytes:
    if info.file_size > limit:
        raise ProjectError(f"its {what} is {info.file_size:,} bytes, more than a project holds ({limit:,})")
    try:
        with archive.open(info) as member:
            data = member.read(limit + 1)
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, zlib.error) as exc:
        raise ProjectError(f"its {what} is damaged ({exc})") from exc
    if len(data) > limit:
        raise ProjectError(f"its {what} is larger than a project holds ({limit:,} bytes)")
    return data


def read_project(path: Path) -> Project:
    """The project at ``path``, checked whole (see the module's docstring); raises ``ProjectError`` saying what is
    wrong with it, or ``OSError`` if it can't be read at all."""
    path = Path(path)
    data = path.read_bytes()
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ProjectError("it isn't a Tessellatum project (not a zip archive, or a damaged one)") from exc
    with archive:
        members = archive.infolist()
        _expect(len(members) <= MAX_MEMBERS, "it holds more files than a project does")
        by_name = {}
        for info in members:
            _expect(_safe_name(info.filename), f"it holds a file named {info.filename!r}, outside the project")
            by_name[info.filename] = info
        _expect(MANIFEST in by_name, "it isn't a Tessellatum project (no manifest)")
        try:
            raw = _read_member(archive, by_name[MANIFEST], MAX_MANIFEST, "manifest")
            info = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectError(f"its manifest is damaged ({exc})") from exc
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as exc:
            raise ProjectError(f"it is damaged ({exc})") from exc
        _expect(isinstance(info, dict) and info.get("format") == FORMAT, "it isn't a Tessellatum project")
        info = _migrate(info)
        snapshot = settings_from_json(info.get("settings"))
        source = info.get("source")
        _expect(isinstance(source, dict), "it holds no picture")
        name, member, digest = source.get("name"), source.get("member"), source.get("sha256")
        _expect(isinstance(name, str) and 0 < len(name) <= 255, "its picture has no name")
        _expect(isinstance(member, str) and _safe_name(member) and member in by_name, "its picture is missing")
        _expect(isinstance(digest, str) and len(digest) == 64, "its picture has no checksum")
        try:
            source_bytes = _read_member(archive, by_name[member], MAX_SOURCE, "picture")
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as exc:
            raise ProjectError(f"its picture is damaged ({exc})") from exc
    _expect(hashlib.sha256(source_bytes).hexdigest() == digest, "its picture is damaged (it doesn't match its checksum)")
    image = decode_image(source_bytes)
    return Project(snapshot, source_bytes, PurePosixPath(name.replace("\\", "/")).name or "picture", image,
                   app_version=str(info.get("app_version", "")), format_version=FORMAT_VERSION)


def read_thumbnail(path: Path) -> bytes | None:
    """The project's thumbnail (PNG bytes), or None if it has none or can't be read."""
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(THUMBNAIL)
            return _read_member(archive, info, MAX_THUMBNAIL, "thumbnail")
    except (OSError, KeyError, zipfile.BadZipFile, ProjectError, EOFError, RuntimeError):
        return None

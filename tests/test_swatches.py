"""Palette files: a legend saved for a painting app, read back by a small reader of each format."""

import colorsys
import json
import struct
import zipfile

import pytest

from tessellatum.core import swatches
from tessellatum.core.color_names import name_colors

PALETTE = [(192, 115, 122), (90, 107, 47), (0, 0, 0), (255, 255, 255), (13, 200, 251)]


def _read_gpl(data: bytes):
    lines = data.decode("utf-8").splitlines()
    assert lines[0] == "GIMP Palette"
    name = lines[1].removeprefix("Name: ")
    columns = int(lines[2].removeprefix("Columns: "))
    colors = []
    for line in lines[3:]:
        if not line or line.startswith("#"):
            continue
        r, g, b, label = line.split(maxsplit=3)
        colors.append((label, (int(r), int(g), int(b))))
    return name, columns, colors


def _read_ase(data: bytes):
    assert data[:4] == b"ASEF"
    major, minor, count = struct.unpack(">HHI", data[4:12])
    assert (major, minor) == (1, 0)
    at, blocks = 12, []
    for _ in range(count):
        kind, length = struct.unpack(">HI", data[at : at + 6])
        body = data[at + 6 : at + 6 + length]
        at += 6 + length
        if kind == 0xC002:
            assert body == b""
            blocks.append(("end", None, None))
            continue
        units = struct.unpack(">H", body[:2])[0]
        name = body[2 : 2 + 2 * units].decode("utf-16-be")
        assert name.endswith("\x00")
        rest = body[2 + 2 * units :]
        if kind == 0xC001:
            assert rest == b""
            blocks.append(("group", name[:-1], None))
        else:
            assert kind == 0x0001 and rest[:4] == b"RGB " and len(rest) == 4 + 12 + 2
            values = struct.unpack(">3f", rest[4:16])
            assert struct.unpack(">H", rest[16:])[0] == 2  # a normal color
            blocks.append(("color", name[:-1], tuple(round(v * 255) for v in values)))
    assert at == len(data)
    return blocks


def _read_procreate(data: bytes, tmp_path):
    path = tmp_path / "read.swatches"
    path.write_bytes(data)
    with zipfile.ZipFile(path) as package:
        assert package.namelist() == ["Swatches.json"]
        document = json.loads(package.read("Swatches.json").decode("utf-8"))
    colors = []
    for swatch in document["swatches"]:
        assert set(swatch) == {"hue", "saturation", "brightness", "alpha", "colorSpace"}
        assert all(0 <= swatch[k] <= 1 for k in ("hue", "saturation", "brightness"))
        assert (swatch["alpha"], swatch["colorSpace"]) == (1, 0)
        rgb = colorsys.hsv_to_rgb(swatch["hue"], swatch["saturation"], swatch["brightness"])
        colors.append(tuple(round(v * 255) for v in rgb))
    return document["name"], colors


def test_each_color_is_named_by_its_number_and_its_legend_name():
    names = name_colors(PALETTE)
    assert swatches.entries(PALETTE) == [(f"{i + 1} {names[i]}", rgb) for i, rgb in enumerate(PALETTE)]


def test_a_gimp_palette_reads_back_exactly(tmp_path):
    written = swatches.save_palette(tmp_path / "lion.gpl", PALETTE)
    assert written == [tmp_path / "lion.gpl"]
    name, columns, colors = _read_gpl(written[0].read_bytes())
    assert (name, columns) == ("lion", swatches.GPL_COLUMNS)
    assert colors == swatches.entries(PALETTE)


def test_an_adobe_swatch_exchange_file_reads_back_exactly(tmp_path):
    written = swatches.save_palette(tmp_path / "lion.ase", PALETTE, title="Lion")
    blocks = _read_ase(written[0].read_bytes())
    assert blocks[0] == ("group", "Lion", None) and blocks[-1] == ("end", None, None)
    assert [(name, rgb) for _, name, rgb in blocks[1:-1]] == swatches.entries(PALETTE)


def test_a_procreate_palette_reads_back_exactly(tmp_path):
    written = swatches.save_palette(tmp_path / "lion.swatches", PALETTE)
    name, colors = _read_procreate(written[0].read_bytes(), tmp_path)
    assert name == "lion" and colors == PALETTE


def test_a_legend_of_more_than_thirty_colors_is_saved_as_procreate_palettes_of_thirty(tmp_path):
    palette = [(i * 4, 255 - i * 3, (i * 37) % 256) for i in range(64)]
    written = swatches.save_palette(tmp_path / "lion.swatches", palette)
    assert [p.name for p in written] == ["lion 1-30.swatches", "lion 31-60.swatches", "lion 61-64.swatches"]
    read = [_read_procreate(p.read_bytes(), tmp_path) for p in written]
    assert [name for name, _ in read] == ["lion 1-30", "lion 31-60", "lion 61-64"]
    assert [rgb for _, colors in read for rgb in colors] == palette
    with pytest.raises(ValueError):
        swatches.procreate_bytes("too many", swatches.entries(palette))
    # The other formats hold any number of colors in one file.
    assert len(_read_gpl(swatches.save_palette(tmp_path / "lion.gpl", palette)[0].read_bytes())[2]) == 64


def test_the_same_legend_gives_the_same_bytes(tmp_path):
    for suffix in swatches.FORMATS:
        first = swatches.save_palette(tmp_path / f"a{suffix}", PALETTE, title="t")[0].read_bytes()
        second = swatches.save_palette(tmp_path / f"b{suffix}", PALETTE, title="t")[0].read_bytes()
        assert first == second


def test_a_name_with_non_ascii_letters_and_a_line_break_is_kept_on_one_line(tmp_path):
    written = swatches.save_palette(tmp_path / "x.gpl", PALETTE, title="Café\nau lait")
    assert _read_gpl(written[0].read_bytes())[0] == "Café au lait"
    blocks = _read_ase(swatches.save_palette(tmp_path / "x.ase", PALETTE, title="Café")[0].read_bytes())
    assert blocks[0][1] == "Café"


def test_only_palette_formats_are_saved(tmp_path):
    with pytest.raises(ValueError):
        swatches.save_palette(tmp_path / "lion.png", PALETTE)

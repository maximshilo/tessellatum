"""The painter's own paints: read from text or a palette file, the palette chosen from them, called by their names."""

import cv2
import numpy as np
import pytest

from tessellatum.core import difficulty, pipeline, quantize, swatches
from tessellatum.core.color import bgr_to_lab, ciede2000
from tessellatum.core.color_names import legend_names, name_colors
from tessellatum.core.legend import NAME_PT, _wrap, plain, text_width_mm
from tessellatum.core.paints import MOST_PAINTS, distinct, paints_text, parse_paints, read_palette_file
from tessellatum.core.pipeline import Handling

MM_PER_PT = 25.4 / 72


@pytest.mark.parametrize(
    "line, paint",
    [
        ("#2000B1  Ultramarine", ("Ultramarine", (0x20, 0x00, 0xB1))),
        ("Ultramarine #2000b1", ("Ultramarine", (0x20, 0x00, 0xB1))),
        ("Sap green - #5A6B2F", ("Sap green", (90, 107, 47))),
        ("90 107 47 Sap green", ("Sap green", (90, 107, 47))),
        ("90, 107, 47\tSap green", ("Sap green", (90, 107, 47))),
        ("Sap green: 90;107;47", ("Sap green", (90, 107, 47))),
        ("Polychromos 120 90 107 47", ("Polychromos 120", (90, 107, 47))),  # the color ends the line
        ("5A6B2F sap", ("sap", (90, 107, 47))),  # a hex code without its #
        ("#fff", ("", (255, 255, 255))),
        ("  #000000  ", ("", (0, 0, 0))),
    ],
)
def test_a_paint_is_a_color_and_the_rest_of_its_line_its_name(line, paint):
    assert parse_paints(line).paints == (paint,)


def test_blank_lines_and_comments_are_skipped_and_a_line_with_no_color_is_reported():
    text = "# my pencils\n\n// box of 24\n#2000B1 Ultramarine\nburnt umber\n300 20 20 too bright\n#Faber notes"
    parsed = parse_paints(text)
    assert parsed.paints == (("Ultramarine", (0x20, 0x00, 0xB1)),)
    assert parsed.rejected == ((5, "burnt umber"), (6, "300 20 20 too bright"))


def test_paints_written_as_text_read_back_and_a_color_counts_once():
    paints = (("Ultramarine", (32, 0, 177)), ("", (90, 107, 47)), ("Ultramarine again", (32, 0, 177)))
    assert parse_paints(paints_text(paints)).paints == paints
    assert distinct(paints) == paints[:2]
    many = tuple((f"p{i}", (i % 256, i // 256, 7)) for i in range(400))
    assert len(distinct(many)) == MOST_PAINTS


@pytest.mark.parametrize("suffix", [".gpl", ".ase", ".swatches"])
def test_a_palette_file_the_app_saves_reads_back_as_paints(tmp_path, suffix):
    palette = [(192, 115, 122), (90, 107, 47), (0, 0, 0), (255, 255, 255)]
    names = ["Rose", "Sap green", "Ivory black", "Titanium white"]
    path = swatches.save_palette(tmp_path / f"box{suffix}", palette, names=names)[0]
    paints = read_palette_file(path)
    assert [rgb for _, rgb in paints] == palette
    if suffix != ".swatches":  # Procreate keeps no names
        assert [name for name, _ in paints] == [f"{i + 1} {n}" for i, n in enumerate(names)]


def test_a_text_file_reads_as_paints_and_a_broken_palette_says_so(tmp_path):
    (tmp_path / "box.txt").write_text("#2000B1 Ultramarine\n", encoding="utf-8")
    assert read_palette_file(tmp_path / "box.txt") == [("Ultramarine", (32, 0, 177))]
    (tmp_path / "bad.ase").write_bytes(b"not a palette")
    with pytest.raises(ValueError, match="bad.ase"):
        read_palette_file(tmp_path / "bad.ase")
    (tmp_path / "bad.gpl").write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_palette_file(tmp_path / "bad.gpl")


def _flat(colors_bgr, size=(120, 160)) -> np.ndarray:
    """A picture of vertical stripes, one of each color, the first twice as wide."""
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    edges = np.linspace(0, w, len(colors_bgr) + 2).astype(int)
    edges = np.concatenate([[0], edges[2:]])
    for color, (x0, x1) in zip(colors_bgr, zip(edges[:-1], edges[1:])):
        image[:, x0:x1] = color
    return image


def _samples(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def test_the_paints_the_picture_is_painted_in_are_taken_over_the_rest():
    picture = [(40, 60, 200), (200, 120, 30), (60, 180, 60)]  # BGR
    distractors = [(250, 0, 250), (0, 250, 250), (128, 128, 128), (20, 20, 20), (240, 240, 240)]
    paints = np.array(distractors + picture, dtype=np.uint8)
    image = _flat(picture)
    labels, chosen = quantize.paint_palette(_samples(image), paints, num_colors=3, min_de00=4.0)
    assert sorted(chosen.tolist()) == [5, 6, 7]
    lab = bgr_to_lab(paints[chosen])
    assert list(chosen) == [chosen[i] for i in np.argsort(lab[:, 0], kind="stable")]  # darkest first
    painted = paints[chosen][labels].reshape(image.shape)
    assert (painted == image).all()


def test_no_more_paints_than_asked_for_and_none_that_doesn_t_help():
    picture = [(40, 60, 200), (200, 120, 30), (60, 180, 60)]
    paints = np.array(picture + [(250, 0, 250)], dtype=np.uint8)
    _labels, two = quantize.paint_palette(_samples(_flat(picture)), paints, num_colors=2, min_de00=4.0)
    assert len(two) == 2 and 0 in two  # the widest stripe's
    _labels, enough = quantize.paint_palette(_samples(_flat(picture)), paints, num_colors=10, min_de00=4.0)
    assert sorted(enough.tolist()) == [0, 1, 2]  # magenta brings no pixel nearer


def test_two_paints_closer_than_the_margin_are_never_both_taken():
    picture = [(40, 60, 200), (40, 60, 215), (200, 120, 30)]  # two reds 3 ΔE00 apart
    paints = np.array(picture, dtype=np.uint8)
    _labels, chosen = quantize.paint_palette(_samples(_flat(picture)), paints, num_colors=3, min_de00=4.0)
    lab = bgr_to_lab(paints[chosen])
    assert len(chosen) == 2 and ciede2000(lab[0], lab[1]) >= 4.0
    _labels, all_three = quantize.paint_palette(_samples(_flat(picture)), paints, num_colors=3, min_de00=2.0)
    assert len(all_three) == 3


def test_near_blacks_paper_shows_alike_are_not_both_taken_but_in_the_subject():
    picture = [(12, 12, 12), (30, 26, 24), (200, 200, 200)]
    paints = np.array(picture, dtype=np.uint8)
    samples = _samples(_flat(picture))
    _labels, screen = quantize.paint_palette(samples, paints, 3, 4.0, paper_black_lstar=0.0)
    _labels, paper = quantize.paint_palette(samples, paints, 3, 4.0, paper_black_lstar=25.0)
    kept = np.ones(len(samples), dtype=bool)
    _labels, subject = quantize.paint_palette(samples, paints, 3, 4.0, paper_black_lstar=25.0, kept=kept)
    assert (len(screen), len(paper), len(subject)) == (3, 2, 3)


def test_the_swaps_improve_on_the_greedy_choice():
    # Greedy takes the middle gray first, which a pair of a dark and a light gray serves better.
    picture = [(30, 30, 30), (225, 225, 225)]
    paints = np.array([(128, 128, 128), (30, 30, 30), (225, 225, 225)], dtype=np.uint8)
    samples = _samples(_flat(picture))
    _labels, chosen = quantize.paint_palette(samples, paints, num_colors=2, min_de00=4.0)
    assert sorted(chosen.tolist()) == [1, 2]


def test_a_picture_made_with_paints_is_painted_in_them_alone_and_called_by_their_names(sample_image_bgr):
    rng = np.random.default_rng(3)
    paints = tuple((f"Paint {i}", tuple(int(v) for v in rng.integers(0, 256, 3))) for i in range(48))
    params = difficulty.params_for_preset("Medium")
    result = pipeline.generate(sample_image_bgr, params, 600, handling=Handling(paints=paints))
    owned = dict((rgb, name) for name, rgb in paints)
    assert result.palette_rgb and all(rgb in owned for rgb in result.palette_rgb)
    assert result.color_names == [owned[rgb] for rgb in result.palette_rgb]
    assert len(result.palette_rgb) <= params.num_colors
    # Without paints, the same picture's own colors, named from the list.
    free = pipeline.generate(sample_image_bgr, params, 600)
    assert free.color_names == name_colors(free.palette_rgb)


def test_a_paint_with_no_name_is_named_like_any_color_and_names_are_plain_ascii():
    paints = [("Céruléen", (42, 82, 190)), ("", (192, 115, 122))]
    assert legend_names([(42, 82, 190), (192, 115, 122)], paints) == ["Céruléen", "dusty rose"]
    assert plain("Céruléen  bleu\n") == "Ceruleen bleu" and plain("赤") == "?"


def test_a_name_too_long_for_its_two_lines_is_cut_at_a_word_with_dots():
    name = "Faber-Castell Polychromos 151 Helioblue reddish extra long name"
    lines = _wrap(name, NAME_PT * MM_PER_PT, 30.5, 2)
    assert len(lines) == 2 and lines[-1].endswith(" ...")
    assert all(text_width_mm(line, NAME_PT * MM_PER_PT) <= 30.5 for line in lines)
    assert " ".join(lines)[: -len(" ...")] in name

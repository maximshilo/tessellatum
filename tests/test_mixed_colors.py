"""Mixed colors: the palette may take colors mixed from two of the painter's paints, each where it earns its place,
listed after the paints with its recipe on the legend, the PDF and the palette files."""

import cv2
import numpy as np
import pytest

from tessellatum.core import difficulty, export, mixing, paint_kits, pipeline, quantize
from tessellatum.core.color import bgr_to_lab, ciede2000, printed_de00
from tessellatum.core.legend import MIXED_HEADING, PAINTS_HEADING, mixed_from, render_legend
from tessellatum.core.pipeline import Handling
from tessellatum.core.quantize import MixRule

import pdf_reading

WHITE, BLACK = (250, 250, 250), (20, 20, 20)
RULE = MixRule(3.0, 0.01)


def _flat(colors_bgr, widths, height=60) -> np.ndarray:
    """Vertical stripes of ``colors_bgr``, ``widths`` pixels wide."""
    image = np.zeros((height, sum(widths), 3), dtype=np.uint8)
    x = 0
    for color, width in zip(colors_bgr, widths):
        image[:, x : x + width] = color
        x += width
    return image


def _samples(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _lab(rgb):
    return bgr_to_lab(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1]))


def test_pair_mixes_are_each_color_once_simplest_parts_first_and_mix_as_paint_does():
    paints = np.array([(250, 220, 20), (20, 60, 190), WHITE], dtype=np.uint8)  # yellow, blue, white (RGB)
    pairs, parts, rgb = mixing.pair_mixes(paints)
    assert len(pairs) == len(parts) == len(rgb) and len({tuple(c) for c in rgb.tolist()}) == len(rgb)
    assert parts.sum(axis=1).tolist() == sorted(parts.sum(axis=1).tolist())
    assert {tuple(p) for p in parts.tolist()} <= set(mixing.MIX_PARTS)
    one_to_one = next(i for i, (pair, part) in enumerate(zip(pairs.tolist(), parts.tolist())) if pair == [0, 1])
    assert parts[one_to_one].tolist() == [1, 1]
    r, g, b = rgb[one_to_one].tolist()
    assert g > r and g > b  # yellow and blue make green, not gray
    assert rgb[one_to_one].tolist() == list(mixing.mixed_rgb(paints[[0, 1]], [1, 1]))
    weighed = mixing.pair_mixes(paints, [0, 2])[0]
    assert weighed.tolist() and all(pair == [0, 2] for pair in weighed.tolist())


def test_a_mix_is_taken_where_the_paints_leave_the_picture_far_and_named_by_its_recipe():
    gray = mixing.mixed_rgb(np.array([BLACK, WHITE]), [1, 1])
    picture = [WHITE[::-1], BLACK[::-1], gray[::-1]]  # BGR
    paints = np.array([WHITE[::-1], BLACK[::-1]], dtype=np.uint8)
    samples = _samples(_flat(picture, [60, 60, 60]))
    labels, palette, recipes = quantize.mixed_palette(samples, paints, 24, 4.0, RULE)
    assert recipes[:2] == (None, None) and len(recipes) == len(palette) == 3
    assert recipes[2] == ((0, 1), (1, 1))  # 1 + 1, the paint listed first first
    assert tuple(palette[2][::-1]) == gray and (labels.reshape(60, 180)[:, 120:] == 2).all()
    assert [tuple(c) for c in palette[:2]] == [tuple(paints[1]), tuple(paints[0])]  # the paints darkest first


def test_no_mix_where_the_paints_do_it_or_for_a_speck_or_without_room():
    gray = mixing.mixed_rgb(np.array([BLACK, WHITE]), [1, 1])
    paints = np.array([WHITE[::-1], BLACK[::-1]], dtype=np.uint8)
    covered = _samples(_flat([WHITE[::-1], BLACK[::-1]], [90, 90]))
    assert quantize.mixed_palette(covered, paints, 24, 4.0, RULE)[2] == (None, None)
    speck = _samples(_flat([WHITE[::-1], BLACK[::-1], gray[::-1]], [100, 79, 1]))  # 1 of 180 columns: 0.6%
    assert quantize.mixed_palette(speck, paints, 24, 4.0, RULE)[2] == (None, None)
    assert quantize.mixed_palette(speck, paints, 24, 4.0, MixRule(3.0, 0.005))[2][2] is not None
    some = _samples(_flat([WHITE[::-1], BLACK[::-1], gray[::-1]], [60, 60, 60]))
    assert quantize.mixed_palette(some, paints, 2, 4.0, RULE)[2] == (None, None)  # Colors: 2


def test_a_mix_must_be_clearly_closer_where_it_paints():
    # The picture's third stripe is 2.5 ΔE00 from a paint: a mix nearer it gains too little to be worth mixing.
    paints_rgb = [WHITE, BLACK, (120, 120, 120)]
    near = (126, 126, 126)
    assert ciede2000(_lab(near)[0], _lab(paints_rgb[2])[0]) < 3.0
    paints = np.array([c[::-1] for c in paints_rgb], dtype=np.uint8)
    samples = _samples(_flat([WHITE[::-1], BLACK[::-1], near[::-1]], [60, 60, 60]))
    assert all(r is None for r in quantize.mixed_palette(samples, paints, 24, 4.0, RULE)[2])
    loose = quantize.mixed_palette(samples, paints, 24, 4.0, MixRule(1.0, 0.01))[2]
    assert all(r is None for r in loose)  # nor within the margin of a paint, whatever the gain asked for


@pytest.mark.parametrize("key", [kit.key for kit in paint_kits.KITS])
def test_every_mix_on_a_photo_meets_the_rule_keeps_the_margin_and_follows_the_paints(sample_image_bgr, key):
    kit = paint_kits.kit(key)
    paints = np.array([rgb[::-1] for _name, rgb in kit.paints], dtype=np.uint8)
    image = cv2.resize(sample_image_bgr, (240, 180), interpolation=cv2.INTER_AREA)
    samples = _samples(image)
    kept = np.zeros(len(samples), dtype=bool)
    kept[: len(samples) // 3] = True
    labels, palette, recipes = quantize.mixed_palette(samples, paints, 24, 4.0, RULE, 25.0, kept)
    again = quantize.mixed_palette(samples, paints, 24, 4.0, RULE, 25.0, kept)
    assert (labels == again[0]).all() and (palette == again[1]).all() and recipes == again[2]  # deterministic
    assert len(palette) <= 24
    paint_count = sum(r is None for r in recipes)
    assert all(r is None for r in recipes[:paint_count]) and all(r is not None for r in recipes[paint_count:])
    if key == "Primaries":
        assert paint_count < len(palette)  # a few paints far from the picture's colors: mixes earn their place
    _labels, chosen = quantize.paint_palette(samples, paints, 24, 4.0, 25.0, kept)
    assert (palette[:paint_count] == paints[chosen]).all()  # the paints exactly as without mixing
    lab = bgr_to_lab(np.ascontiguousarray(palette))
    for section in (lab[:paint_count], lab[paint_count:]):
        assert (np.diff(section[:, 0]) >= 0).all()  # each section darkest first
    paint_lab = bgr_to_lab(paints)
    for m in range(paint_count, len(palette)):
        (a, n), (b, k) = recipes[m]
        assert n >= k and {n, k} <= {1, 2, 3}
        assert tuple(palette[m][::-1]) == mixing.mixed_rgb(paints[[a, b]][:, ::-1], [n, k])
        assert ciede2000(lab[m][None], paint_lab).min() >= 4.0  # no mix within the margin of any paint
        others = np.delete(lab, m, axis=0)
        assert ciede2000(lab[m][None], others).min() >= 4.0
    # The mixes earn their place: each against all the other colors, over the pixels it is nearest.
    coarse, coarse_lab, weight = quantize._binned(samples, kept, quantize._PAINT_BIN)
    distance = ciede2000(coarse_lab[:, None, :], lab[None, :, :])
    for m in range(paint_count, len(palette)):
        without = np.delete(distance, m, axis=1).min(axis=1)
        gain = np.maximum(without - distance[:, m], 0)
        used = float(weight @ (gain > 0))
        assert used >= RULE.min_share * weight.sum() - 1e-6
        assert float(weight @ gain) >= RULE.gain_de00 * used - 1e-6


def test_mixes_keep_the_printed_margin_from_near_black_paints():
    paints_rgb = [(30, 30, 30), WHITE, (200, 40, 40)]
    paints = np.array([c[::-1] for c in paints_rgb], dtype=np.uint8)
    pairs, parts, rgb = mixing.pair_mixes(paints[:, ::-1])
    picture = [c[::-1] for c in paints_rgb] + [tuple(c[::-1]) for c in rgb[:6].tolist()]
    samples = _samples(_flat(picture, [20] * len(picture)))
    _labels, palette, recipes = quantize.mixed_palette(samples, paints, 24, 4.0, MixRule(2.0, 0.01), 25.0)
    lab = bgr_to_lab(np.ascontiguousarray(palette))
    apart = np.minimum(ciede2000(lab[:, None], lab[None]), printed_de00(lab[:, None], lab[None], 25.0))
    np.fill_diagonal(apart, np.inf)
    assert apart.min() >= 4.0 and any(r is not None for r in recipes)


def test_quantize_returns_the_recipes_only_when_asked_to_mix():
    image = _flat([WHITE[::-1], BLACK[::-1], (128, 128, 128)], [60, 60, 60])
    paints = np.array([WHITE[::-1], BLACK[::-1]], dtype=np.uint8)
    plain = quantize.quantize(image, 24, 0.0, paints_bgr=paints)
    assert len(plain) == 2
    mixed = quantize.quantize(image, 24, 0.0, paints_bgr=paints, mix=RULE)
    assert len(mixed) == 3 and mixed[2][-1] is not None and len(mixed[1]) == 3


def test_mixing_is_off_unless_asked_and_needs_paints(sample_image_bgr):
    paints = paint_kits.kit("primaries").paints
    assert Handling(paints=paints).mix_rule() is None and Handling(mix_paints=True).mix_rule() is None
    assert Handling(paints=paints, mix_paints=True, mix_gain_de00=4.0, mix_min_share_pct=2.0).mix_rule() == MixRule(
        4.0, 0.02
    )
    params = difficulty.params_for_preset("Medium")
    unmixed = pipeline.generate(sample_image_bgr, params, 600, handling=Handling(paints=paints))
    assert unmixed.mixes == [] and unmixed.model.mixes == []


def test_a_page_with_mixes_lists_them_after_the_paints_with_their_recipes(sample_image_bgr, tmp_path):
    paints = paint_kits.kit("primaries").paints
    params = difficulty.params_for_preset("Medium")
    handling = Handling(paints=paints, mix_paints=True)
    page = pipeline.generate(sample_image_bgr, params, 600, handling=handling)
    plain_page = pipeline.generate(sample_image_bgr, params, 600, handling=Handling(paints=paints))
    assert len(page.mixes) == len(page.palette_rgb) == len(page.color_names)
    start = mixed_from(page.mixes)
    assert start is not None and start > 0
    owned = {rgb for _name, rgb in paints}
    assert all(rgb in owned for rgb in page.palette_rgb[:start])
    names = {name for name, _rgb in paints}
    for rgb, lines in zip(page.palette_rgb[start:], page.mixes[start:]):
        assert rgb not in owned and len(lines) == 2
        assert all(line.split(" ", 1)[0] in "123" and line.split(" ", 1)[1] in names for line in lines)
    assert page.legend.height > plain_page.legend.height  # the headings and the mixed section's rows
    # The page painted in, with the mixes' predicted colors.
    painting = page.painting
    assert painting.palette_rgb[: len(page.palette_rgb)] == page.palette_rgb
    # Drawn again larger, the same mixes and legend sections.
    drawn = pipeline.draw_page(page, sample_image_bgr, 900)
    assert drawn.mixes == page.mixes and drawn.legend.height > plain_page.legend.height

    # The PDF's legend sheet: the two headings, and each mix's recipe, as text.
    export.save_pdf(page.drawing, page.palette_rgb, tmp_path / "page.pdf", names=page.color_names, mixes=page.mixes)
    sheet = pdf_reading.sheets((tmp_path / "page.pdf").read_bytes())[1]["content"]
    assert f"({PAINTS_HEADING})" in sheet and f"({MIXED_HEADING})" in sheet
    for lines in page.mixes[start:]:
        assert all(f"({line})" in sheet for line in lines)


def test_the_legend_s_sections_number_on_and_a_mix_never_comes_before_a_paint():
    palette_bgr = np.array([(20, 20, 20), (240, 240, 240), (128, 128, 128), (90, 90, 160)], dtype=np.uint8)
    mixes = [None, None, ["1 Black", "1 White"], ["2 Black", "1 Red"]]
    plain = render_legend(palette_bgr, 2000, 8.0)
    sectioned = render_legend(palette_bgr, 2000, 8.0, mixes=mixes)
    assert sectioned.height > plain.height and sectioned.width == plain.width
    assert render_legend(palette_bgr, 2000, 8.0, mixes=[None] * 4).tobytes() == plain.tobytes()
    assert mixed_from([None, ["1 A"]]) == 1 and mixed_from([None, None]) is None and mixed_from(None) is None
    with pytest.raises(ValueError):
        mixed_from([["1 A"], None])


def test_64_colors_half_of_them_mixed_fit_the_a4_legend_sheet_unscaled(tmp_path):
    from tessellatum.core.print_size import PT_PER_INCH, MM_PER_INCH
    from tessellatum.core.render import PageStyle, render_page

    ids = np.zeros((300, 400), dtype=np.int32)
    drawing = render_page((400, 300), [], ids, PageStyle()).drawing
    palette = [(i * 4, 100, 255 - i * 4) for i in range(64)]
    mixes = [None] * 33 + [["3 Titanium white", "1 Ultramarine blue"]] * 31
    export.save_pdf(drawing, palette, tmp_path / "legend.pdf", mixes=mixes)
    sheet = pdf_reading.sheets((tmp_path / "legend.pdf").read_bytes())[1]["content"]
    pt_per_mm = PT_PER_INCH / MM_PER_INCH
    assert sheet.startswith(f"q {pt_per_mm:.8f} 0 0 -{pt_per_mm:.8f} ")  # not scaled down
    assert sheet.count("(3 Titanium white)") == 31 and sheet.count("(65)") == 0 and "(64)" in sheet

"""Text: PP-OCRv6-small finds the lines of text in a picture, looking twice where it finds any (D-049)."""

import hashlib
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import difficulty, faces, pipeline, text
from tessellatum.core import ink as ink_module
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.print_size import print_scale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_case  # noqa: E402
import bench_manifest  # noqa: E402
import bench_metrics as bm  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_images"
WORDS = ("PAINT BY NUMBERS", "Hello world 2026", "quiet river")


def _lettering(scale: float = 0.7, size: tuple[int, int] = (900, 600)) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Three lines of dark lettering on a pale page, and each line's (x, y, width, height) box."""
    picture = np.full((size[1], size[0], 3), 235, dtype=np.uint8)
    boxes = []
    for i, word in enumerate(WORDS):
        (width, height), baseline = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        x, y = 60, 120 + 140 * i
        cv2.putText(picture, word, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA)
        boxes.append((x, y - height, width, height + baseline))
    return picture, boxes


def _rect_line(center, size, angle=0.0, score=0.9) -> text.TextLine:
    corners = cv2.boxPoints((center, size, angle))
    return text.TextLine(quad=tuple((float(x), float(y)) for x, y in corners), score=score)


def test_the_model_is_the_file_models_md_lists():
    notes = (faces.RESOURCES_DIR / "MODELS.md").read_text(encoding="utf-8")
    model = (faces.RESOURCES_DIR / text.MODEL).read_bytes()
    assert hashlib.sha256(model).hexdigest() in notes
    assert text.MODEL in notes


@pytest.mark.parametrize(
    "picture",
    [
        np.zeros((30, 40, 3), dtype=np.float32),  # not 8-bit
        np.zeros((30, 40), dtype=np.uint8),  # gray
        np.zeros((30, 40, 4), dtype=np.uint8),  # with alpha
        np.zeros((0, 40, 3), dtype=np.uint8),  # empty
    ],
)
def test_finding_text_takes_8_bit_color_pictures(picture):
    with pytest.raises(ValueError):
        text.find_text(picture)
    with pytest.raises(ValueError):
        text.find_text(np.zeros((30, 40, 3), dtype=np.uint8), picture)


def test_lines_of_lettering_are_found_one_box_each_and_the_same_every_time():
    picture, boxes = _lettering()
    lines = text.find_text(picture)
    assert len(lines) == len(WORDS)
    found = text.mask(lines, picture.shape[1::-1])
    for x, y, w, h in boxes:
        assert found[y : y + h, x : x + w].mean() > 0.95
    # Each box holds one line, top to bottom: its middle lies in that line's box.
    for line, (x, y, w, h) in zip(lines, boxes):
        assert text.mask([line], picture.shape[1::-1])[y + h // 2, x + w // 2]
    assert found.mean() < 0.15  # and not the page
    assert text.find_text(picture) == lines


@pytest.mark.parametrize(
    "turn", [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE], ids=["180", "90cw", "90ccw"]
)
def test_lettering_turned_or_upside_down_is_found_too(turn):
    picture, _ = _lettering()
    turned = cv2.rotate(picture, turn)
    lines = text.find_text(turned)
    assert len(lines) == len(WORDS)
    tall = turn != cv2.ROTATE_180
    for line in lines:
        quad = np.asarray(line.quad)
        extent = quad.max(axis=0) - quad.min(axis=0)
        assert (extent[1] > extent[0]) == tall  # turned a quarter, a line runs down the page


def test_pictures_without_lettering_have_no_text():
    rng = np.random.default_rng(0)
    plain = np.full((600, 900, 3), 128, dtype=np.uint8)
    noise = rng.integers(0, 256, (600, 900, 3), dtype=np.uint8)
    disc = cv2.circle(np.full((600, 900, 3), 200, dtype=np.uint8), (450, 300), 120, (20, 40, 180), -1)
    stripes = np.repeat(((np.arange(900) // 12 % 2) * 200 + 30).astype(np.uint8)[None, :, None], 600, 0).repeat(3, 2)
    for picture in (plain, noise, disc, stripes):
        assert text.find_text(picture) == []


def test_lettering_big_enough_to_paint_is_not_text():
    # Lettering whose box is over MAX_BOX_HEIGHT_MM tall on paper is shapes to paint, not text to keep.
    picture, _ = _lettering(scale=2.0)
    limit = print_scale(picture.shape[1::-1]).mm_to_px(text.MAX_BOX_HEIGHT_MM)
    cv2.putText(picture, "Hi", (60, 120), cv2.FONT_HERSHEY_DUPLEX, 2.0, (30, 30, 30), 2, cv2.LINE_AA)
    assert all(line.sides[0] <= limit for line in text.find_text(picture))
    assert text.find_text(_lettering(scale=2.0)[0]) == []


class _Network:
    """Stands in for the network: records what it is shown, and answers with a map made from the view."""

    def __init__(self, answer):
        self.views = []
        self.answer = answer

    def __call__(self, view):
        self.views.append(view.copy())
        return self.answer(view)


def _blank_map(view):
    return np.zeros(view.shape[:2], dtype=np.float32)


def _core_map(view, rows=(0.4, 0.42), cols=(0.2, 0.6), value=0.9):
    """A map with one rectangular core, at the same share of the view whatever its size."""
    probability = np.zeros(view.shape[:2], dtype=np.float32)
    h, w = probability.shape
    probability[int(rows[0] * h) : int(rows[1] * h), int(cols[0] * w) : int(cols[1] * w)] = value
    return probability


def test_a_picture_without_text_at_half_its_preview_size_is_looked_at_once(monkeypatch):
    network = _Network(_blank_map)
    monkeypatch.setattr(text, "_probability", network)
    picture = np.full((600, 900, 3), 200, dtype=np.uint8)
    assert text.find_text(picture) == []
    assert len(network.views) == 1
    assert network.views[0].shape[:2] == (288, 448)  # half of each side, to the nearest multiple of 32


def test_where_text_is_found_the_second_look_sees_the_source_at_twice_the_preview_s_size(monkeypatch):
    network = _Network(_core_map)
    monkeypatch.setattr(text, "_probability", network)
    rng = np.random.default_rng(1)
    picture = rng.integers(0, 256, (600, 900, 3), dtype=np.uint8)
    large = rng.integers(0, 256, (1500, 2250, 3), dtype=np.uint8)  # larger than twice the picture: shrunk
    small = rng.integers(0, 256, (750, 1125, 3), dtype=np.uint8)  # smaller: stretched
    for source, interpolation in ((large, cv2.INTER_AREA), (small, cv2.INTER_LINEAR)):
        network.views.clear()
        assert text.find_text(picture, source)
        assert len(network.views) == 2
        np.testing.assert_array_equal(network.views[0], cv2.resize(picture, (448, 288)))
        np.testing.assert_array_equal(network.views[1], cv2.resize(source, (1792, 1216), interpolation=interpolation))
    network.views.clear()
    text.find_text(picture)  # without a source, the picture itself
    np.testing.assert_array_equal(network.views[1], cv2.resize(picture, (1792, 1216)))


def test_a_core_is_grown_out_to_its_line_and_given_in_the_picture_s_pixels(monkeypatch):
    # A core of 40 x 8 of the map's pixels at (100, 300): grown by its area times UNCLIP over its perimeter on every
    # side, and scaled from the map to the picture.
    def answer(view):
        probability = np.zeros((640, 960), dtype=np.float32)
        probability[300:308, 100:140] = 0.9
        return probability

    monkeypatch.setattr(text, "_probability", answer)
    lines = text._lines(np.zeros((640, 960, 3), dtype=np.uint8), (1920, 1280))
    assert len(lines) == 1
    grow = 40 * 8 * 2.0 / (2 * (40 + 8))  # UNCLIP = 2.0 (D-049)
    height, length = lines[0].sides
    assert height == pytest.approx(2 * (8 + 2 * grow), abs=1e-3)
    assert length == pytest.approx(2 * (40 + 2 * grow), abs=1e-3)
    center = np.asarray(lines[0].quad).mean(axis=0)
    np.testing.assert_allclose(center, (2 * 120, 2 * 304), atol=1e-3)
    assert lines[0].score == pytest.approx(0.9 * 40 * 8 / (41 * 9), abs=1e-6)  # the box filled as PaddleOCR fills it


def test_the_network_is_fed_the_picture_as_paddleocr_feeds_it(monkeypatch):
    # BGR, scaled to [0, 1] and standardized by ImageNet's statistics channel by channel, as the model's own
    # inference.yml has it (D-049); not RapidOCR's (x - 0.5) / 0.5.
    seen = []

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "x"})()]

        def run(self, outputs, feed):
            seen.append(feed["x"])
            return [np.zeros((1, 1) + feed["x"].shape[2:], dtype=np.float32)]

    monkeypatch.setattr(text, "_session", Session())
    rng = np.random.default_rng(3)
    view = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    text._probability(view)
    expected = (view.astype(np.float64) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(seen[0][0].transpose(1, 2, 0), expected, atol=1e-5)
    assert seen[0].dtype == np.float32 and seen[0].shape == (1, 3, 64, 96)


def test_the_session_gives_its_memory_back_and_uses_the_pipeline_s_threads(monkeypatch):
    # ONNX Runtime's arena would keep ~1 GB after a few pictures of different sizes (D-049).
    import onnxruntime

    from tessellatum.core import parallel

    made = []
    real = onnxruntime.InferenceSession

    def session(model, options, providers):
        made.append(options)
        return real(model, options, providers=providers)

    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    monkeypatch.setattr(text, "_session", None)
    monkeypatch.setenv("TESSELLATUM_THREADS", "3")
    text._probability(np.zeros((32, 32, 3), dtype=np.uint8))
    (options,) = made
    assert options.enable_cpu_mem_arena is False
    assert options.intra_op_num_threads == parallel.worker_count() == 3


def test_the_core_is_where_the_map_is_above_its_threshold(monkeypatch):
    # A faint ring at 0.25 round a strong core is part of it at THRESHOLD = 0.2: the core is 12 x 44, not 8 x 40.
    def answer(view):
        probability = np.zeros((640, 960), dtype=np.float32)
        probability[298:310, 98:142] = 0.25
        probability[300:308, 100:140] = 0.9
        return probability

    monkeypatch.setattr(text, "_probability", answer)
    (line,) = text._lines(np.zeros((640, 960, 3), dtype=np.uint8), (960, 640))
    grow = 44 * 12 * 2.0 / (2 * (44 + 12))
    assert line.sides == pytest.approx((12 + 2 * grow, 44 + 2 * grow), abs=1e-3)


@pytest.mark.parametrize(
    "core, value, kept",
    [
        ((slice(300, 308), slice(100, 140)), 0.9, True),
        ((slice(300, 308), slice(100, 140)), 0.55, True),  # a middling core: its mean probability 0.48, over MIN_SCORE
        ((slice(300, 308), slice(100, 140)), 0.3, False),  # a faint core: its mean probability under MIN_SCORE
        ((slice(300, 302), slice(100, 140)), 0.9, False),  # a core 2 px tall is noise
        ((slice(300, 312), slice(100, 112)), 0.9, False),  # as tall as it is long, if short enough: not a line
        ((slice(300, 312), slice(100, 120)), 0.9, False),  # grown to 27 x 35, 1.3 times as long as tall: not a line
        ((slice(300, 314), slice(100, 500)), 0.9, True),  # grown to 41 px, 11.8 mm on paper: a line
        ((slice(300, 320), slice(100, 500)), 0.9, False),  # grown to 58 px, 16.8 mm: over MAX_BOX_HEIGHT_MM
        ((slice(200, 290), slice(100, 600)), 0.9, False),  # far over it: shapes to paint
    ],
    ids=["line", "middling", "faint", "thin", "square", "stubby", "under-15mm", "over-15mm", "tall"],
)
def test_only_cores_that_look_like_a_line_of_text_are_kept(monkeypatch, core, value, kept):
    def answer(view):
        probability = np.zeros((640, 960), dtype=np.float32)
        probability[core] = value
        return probability

    monkeypatch.setattr(text, "_probability", answer)
    assert bool(text._lines(np.zeros((640, 960, 3), dtype=np.uint8), (960, 640))) == kept


def test_a_line_s_sides_are_its_box_s_height_and_length():
    line = _rect_line((50, 40), (30, 10), angle=30)
    assert line.sides == pytest.approx((10, 30), abs=1e-4)


def test_scaled_lines_follow_the_page():
    line = _rect_line((50, 40), (30, 10), angle=15)
    (scaled,) = text.scaled([line], (100, 80), (250, 160))
    np.testing.assert_allclose(np.asarray(scaled.quad), np.asarray(line.quad) * (2.5, 2.0))
    assert scaled.score == line.score


def test_the_mask_holds_the_pixels_whose_middle_lies_in_a_box():
    rng = np.random.default_rng(2)
    for _ in range(200):
        line = _rect_line(
            tuple(rng.uniform(-5, 45, 2)), tuple(rng.uniform(0.5, 30, 2)), angle=float(rng.uniform(0, 180))
        )
        got = text.mask([line], (40, 30))
        quad = np.asarray(line.quad)
        ys, xs = np.mgrid[0:30, 0:40] + 0.5
        sides = [
            (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) for a, b in zip(quad, np.roll(quad, -1, axis=0))
        ]
        inside = np.all([s >= -1e-9 for s in sides], axis=0) | np.all([s <= 1e-9 for s in sides], axis=0)
        mismatched = got != inside
        assert mismatched.sum() <= 1  # a middle within rounding of an edge may go either way
    # A box of whole pixels covers exactly the pixels it spans, whichever way round its corners go, and one off the page
    # none.
    corners = ((2.0, 3.0), (6.0, 3.0), (6.0, 8.0), (2.0, 8.0))
    expected = np.zeros((10, 10), dtype=bool)
    expected[3:8, 2:6] = True
    for quad in (corners, corners[::-1]):
        np.testing.assert_array_equal(text.mask([text.TextLine(quad=quad, score=1.0)], (10, 10)), expected)
    off = text.TextLine(quad=((-9.0, -9.0), (-2.0, -9.0), (-2.0, -2.0), (-9.0, -2.0)), score=1.0)
    assert not text.mask([off], (10, 10)).any()
    # A box of no area holds no pixel, though every middle lies on the same side of all its edges: a point, and a flat
    # box along a row of middles (whose line runs past its ends).
    point = text.TextLine(quad=((5.0, 5.0),) * 4, score=1.0)
    flat = text.TextLine(quad=((2.0, 5.5), (8.0, 5.5), (8.0, 5.5), (2.0, 5.5)), score=1.0)
    assert not text.mask([point], (10, 10)).any() and not text.mask([flat], (10, 10)).any()
    # A middle on a box's edge is in it: a box through the middles of pixels 2 and 6 holds both.
    edges = text.TextLine(quad=((2.5, 3.5), (6.5, 3.5), (6.5, 7.5), (2.5, 7.5)), score=1.0)
    expected = np.zeros((10, 10), dtype=bool)
    expected[3:8, 2:7] = True
    np.testing.assert_array_equal(text.mask([edges], (10, 10)), expected)


def test_two_threads_find_what_one_finds():
    pictures = [_lettering()[0], cv2.rotate(_lettering(0.6)[0], cv2.ROTATE_180)]
    alone = [text.find_text(p) for p in pictures]
    found = [None, None]

    def find(i: int) -> None:
        for _ in range(2):
            found[i] = text.find_text(pictures[i])

    threads = [threading.Thread(target=find, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert found == alone


# --- the lettering ------------------------------------------------------------------------------------------------


def _strokes(ground: int, letters: int, size: tuple[int, int] = (80, 30)) -> tuple[np.ndarray, np.ndarray]:
    """A gray picture of ``ground`` with upright strokes of ``letters``, 2 px wide every 6 px, and where they are."""
    width, height = size
    picture = np.full((height, width, 3), ground, dtype=np.uint8)
    strokes = np.zeros((height, width), dtype=bool)
    for x in range(10, width - 10, 6):
        strokes[10:20, x : x + 2] = True
    picture[strokes] = letters
    return picture, strokes


_LINE = _rect_line((40, 15), (70, 20))  # a box over the strokes, x 5-75 and y 5-25


def _lightness(gray: int) -> float:
    """CIE L* of a gray, as the lettering reads it."""
    return float(cv2.cvtColor(np.full((1, 1, 3), gray / 255, dtype=np.float32), cv2.COLOR_BGR2Lab)[0, 0, 0])


def test_dark_lettering_prints_as_ink_on_paper():
    picture, strokes = _strokes(ground=220, letters=30)
    ink, area = text.lettering(picture, [_LINE])
    np.testing.assert_array_equal(area, text.mask([_LINE], (80, 30)))
    assert ink.dtype == np.uint8
    assert (ink[strokes] == 255).all()  # the letters, solid
    assert (ink[area & ~strokes] == 0).all()  # their ground, bare paper
    assert (ink[~area] == 0).all()  # and nothing outside the line


def test_light_lettering_prints_as_paper_letters_in_its_dark_ground():
    # As it looks: no guess at which side is the lettering.
    picture, strokes = _strokes(ground=30, letters=220)
    ink, area = text.lettering(picture, [_LINE])
    assert (ink[strokes] == 0).all()
    assert (ink[area & ~strokes] == 255).all()
    assert (ink[~area] == 0).all()


def test_the_lightness_is_stretched_from_the_box_s_98th_percentile_to_its_2nd():
    # Specks darker or lighter than the lettering and its ground, 1% of the box each, don't stretch the scale: the
    # letters are still solid, the ground still paper, and a tone halfway between prints halfway.
    picture, strokes = _strokes(ground=200, letters=60)
    box = text.mask([_LINE], (80, 30))
    ground = np.argwhere(box & ~strokes)
    specks = len(np.argwhere(box)) // 100
    for (y, x) in ground[:specks]:
        picture[y, x] = 0
    for (y, x) in ground[-specks:]:
        picture[y, x] = 255
    half = ground[len(ground) // 2]
    lightness = (_lightness(200) + _lightness(60)) / 2
    gray = min(range(256), key=lambda g: abs(_lightness(g) - lightness))
    picture[half[0], half[1]] = gray
    ink, _ = text.lettering(picture, [_LINE])
    assert (ink[strokes] == 255).all()
    rest = box & ~strokes
    rest[half[0], half[1]] = False
    for (y, x) in ground[:specks]:
        rest[y, x] = False
    assert (ink[rest] == 0).all()
    assert all(ink[y, x] == 255 for y, x in ground[:specks])  # darker than the letters: solid all the same
    expected = 255 * (_lightness(200) - _lightness(gray)) / (_lightness(200) - _lightness(60))
    assert abs(int(ink[half[0], half[1]]) - expected) <= 1


def test_between_its_two_percentiles_the_ink_follows_the_box_s_lightness():
    # A box of every gray from black to white: its 2nd percentile of L* and darker print solid, its 98th and lighter bare
    # paper, and in between in proportion.
    picture = np.tile(np.linspace(0, 255, 80).round().astype(np.uint8)[None, :, None], (30, 1, 3))
    ink, area = text.lettering(picture, [_LINE])
    lightness = cv2.cvtColor(picture.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float64)
    low, high = np.percentile(lightness[area], [2, 98])
    expected = np.clip((high - lightness) / (high - low), 0, 1) * 255
    assert np.abs(ink[area] - expected[area]).max() <= 0.5 + 1e-3
    assert (ink[area & (lightness <= low)] == 255).all() and (ink[area & (lightness >= high)] == 0).all()
    assert ((ink[area] > 0) & (ink[area] < 255)).mean() > 0.8


def test_lettering_fainter_than_the_contrast_floor_prints_nothing():
    # The floor is 20 L* (D-050): a box of two tones stands as far apart as they are.
    ground = 200
    faint = min(g for g in range(ground) if _lightness(ground) - _lightness(g) < 20.0)
    strong = faint - 1
    assert 19.0 < _lightness(ground) - _lightness(faint) < 20.0
    assert 20.0 <= _lightness(ground) - _lightness(strong) < 21.0
    picture, strokes = _strokes(ground=ground, letters=faint)
    ink, area = text.lettering(picture, [_LINE])
    assert area.any() and not ink.any()  # the line is still there; it prints nothing
    picture, strokes = _strokes(ground=ground, letters=strong)
    ink, _ = text.lettering(picture, [_LINE])
    assert (ink[strokes] == 255).all()


def test_a_box_all_one_tone_but_a_few_specks_prints_nothing():
    # The specks stand far apart from the rest, but there is no lettering to stretch: its two percentiles are one tone.
    picture = np.full((30, 80, 3), 200, dtype=np.uint8)
    box = np.argwhere(text.mask([_LINE], (80, 30)))
    for y, x in box[:: len(box) // 10][:10]:
        picture[y, x] = 0
    ink, area = text.lettering(picture, [_LINE])
    assert area.any() and not ink.any()


def test_where_two_lines_overlap_the_one_inking_a_pixel_more_does():
    rng = np.random.default_rng(3)
    picture = rng.integers(0, 256, (40, 90, 3), dtype=np.uint8)
    picture[:, 45:] //= 3  # a darker right half, so the two boxes stretch differently
    first, second = _rect_line((35, 20), (60, 18)), _rect_line((55, 22), (60, 14), angle=10)
    ink_first, area_first = text.lettering(picture, [first])
    ink_second, area_second = text.lettering(picture, [second])
    ink, area = text.lettering(picture, [first, second])
    np.testing.assert_array_equal(area, area_first | area_second)
    np.testing.assert_array_equal(ink, np.maximum(ink_first, ink_second))
    assert ((ink_first != ink_second) & area_first & area_second).any()  # the overlap does tell them apart


def test_no_lines_print_nothing():
    picture, _ = _strokes(ground=220, letters=30)
    ink, area = text.lettering(picture, [])
    assert ink.shape == area.shape == (30, 80) and not ink.any() and not area.any()
    point = text.TextLine(quad=((40.0, 15.0),) * 4, score=1.0)
    ink, area = text.lettering(picture, [point])
    assert not ink.any() and not area.any()
    with pytest.raises(ValueError):
        text.lettering(picture.astype(np.float32), [_LINE])


def test_the_contrast_is_the_gap_between_the_means_of_otsu_s_two_classes():
    rng = np.random.default_rng(4)
    for _ in range(50):
        values = rng.normal(rng.uniform(0, 100), rng.uniform(1, 30), int(rng.integers(2, 60)))
        ordered = np.sort(values)
        best, gap = -1.0, 0.0
        for k in range(1, len(ordered)):  # every split, by brute force
            low, high = ordered[:k], ordered[k:]
            between = len(low) * len(high) * (low.mean() - high.mean()) ** 2
            if between > best + 1e-9:
                best, gap = between, high.mean() - low.mean()
        assert text._otsu_contrast(values) == pytest.approx(gap)
    assert text._otsu_contrast(np.array([5.0])) == 0.0
    assert text._otsu_contrast(np.array([10.0, 30.0])) == pytest.approx(20.0)


def _recall(name: str) -> tuple[float, list]:
    """Text found recall on a sample picture at preview size, against its manifest, and the lines found."""
    source = pipeline.load_image_bgr(SAMPLES / name)
    picture = pipeline.resize_to_long_edge(source, pipeline.PREVIEW_LONG_EDGE)
    lines = text.find_text(picture, source)
    info = bench_manifest.find_image(SAMPLES / name).scaled_to(picture.shape[1::-1])
    boxes = [(b.box.x, b.box.y, b.box.w, b.box.h) for b in info.text]
    match = bm.found_text_match([line.quad for line in lines], boxes, picture.shape[:2])
    return match["text_found_recall"], lines


def test_the_text_of_times_square_is_found_and_the_cat_has_none():
    # Q32's target, on the benchmark's own annotations at preview size: text found recall at least 0.9, no text where
    # the manifest has none.
    recall, lines = _recall("l-photo-times-square.jpg")
    assert recall >= 0.9 and len(lines) > 8
    cat = pipeline.resize_to_long_edge(pipeline.load_image_bgr(SAMPLES / "l-photo-cats-face.jpg"), 1100)
    assert text.find_text(cat) == []


def test_text_is_found_once_per_picture_and_the_analysis_changes_nothing(monkeypatch):
    # The page prints the lettering (T6.2), so every page looks for it; a preview, an export and the analysis share one
    # look at the picture.
    pipeline.clear_cache()
    calls = []
    real = text.find_text

    def counting(picture, source=None):
        calls.append((picture.shape, None if source is None else source.shape))
        return real(picture, source)

    monkeypatch.setattr(text, "find_text", counting)
    picture, _ = _lettering(size=(1400, 933))
    params = params_for_preset("Easy")
    plain = pipeline.generate(picture, params, pipeline.PREVIEW_LONG_EDGE)
    preview = pipeline.generate(picture, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    export = pipeline.generate(picture, params, pipeline.EXPORT_LONG_EDGE, collect_analysis=True)
    assert calls == [((733, 1100, 3), (933, 1400, 3))]  # once, on the picture at preview size, with its source
    assert plain.page.tobytes() == preview.page.tobytes()
    assert len(preview.analysis.text) == len(WORDS)
    expected = text.scaled(preview.analysis.text, preview.page.size, export.page.size)
    for got, want in zip(export.analysis.text, expected):
        np.testing.assert_allclose(np.asarray(got.quad), np.asarray(want.quad))


def test_line_art_has_its_text_found_too():
    pipeline.clear_cache()
    comic = pipeline.load_image_bgr(SAMPLES / "m-comics-upside-downs-writing-pig.jpg")
    page = pipeline.generate(comic, params_for_preset("Easy"), pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert page.analysis.line_art.is_line_art
    assert len(page.analysis.text) > 20


def _without_text(monkeypatch, picture: np.ndarray, params, long_edge: int):
    """The page of ``picture`` with no text found, and the text found put back after."""
    with monkeypatch.context() as patched:
        patched.setattr(text, "find_text", lambda picture, source=None: [])
        pipeline.clear_cache()
        page = pipeline.generate(picture, params, long_edge, collect_analysis=True)
    pipeline.clear_cache()
    return page


def test_the_page_prints_the_lettering_found_and_leaves_the_regions_as_they_are(monkeypatch):
    picture, _ = _lettering(size=(1400, 933))
    params = params_for_preset("Medium")
    pipeline.clear_cache()
    page = pipeline.generate(picture, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    analysis = page.analysis
    size = page.page.size
    resized = pipeline.resize_to_long_edge(picture, pipeline.PREVIEW_LONG_EDGE)
    assert len(analysis.text) == len(WORDS)

    ink, area = text.lettering(resized, analysis.text)
    assert ink.any()
    np.testing.assert_array_equal(analysis.lettering_area, area)
    np.testing.assert_array_equal(analysis.lettering_area, text.mask(analysis.text, size))
    np.testing.assert_array_equal(analysis.lettering, 255 - ink)
    # The lettering darker than halfway is printed ink -- here all of it, with no face on the page -- in the gray of
    # the picture under it.
    printed = ink >= 128
    np.testing.assert_array_equal(analysis.printed_ink, printed)
    assert analysis.ink_gray == ink_module.ink_gray(resized, printed)
    # The page shows it in that gray, as it looks, darkened only where a line runs through.
    gray = np.asarray(page.page.convert("L")).astype(np.float64)
    tone = np.rint(255 - ink * ((255 - analysis.ink_gray) / 255))
    assert (gray[area] <= tone[area]).all()
    assert (gray[area] == tone[area]).mean() > 0.9
    assert (gray[ink == 255] <= analysis.ink_gray).all()

    plain = _without_text(monkeypatch, picture, params, pipeline.PREVIEW_LONG_EDGE)
    assert not plain.analysis.lettering_area.any() and (plain.analysis.lettering == 255).all()
    assert not plain.analysis.printed_ink.any()
    np.testing.assert_array_equal(analysis.region_id_map, plain.analysis.region_id_map)
    np.testing.assert_array_equal(analysis.region_color, plain.analysis.region_color)
    assert analysis.legend_size == plain.analysis.legend_size


def test_light_lettering_prints_its_ground_and_leaves_its_letters_paper():
    picture, _ = _lettering(size=(1400, 933))
    picture = 255 - picture  # pale words on a dark page
    pipeline.clear_cache()
    page = pipeline.generate(picture, params_for_preset("Easy"), pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    analysis = page.analysis
    assert len(analysis.text) == len(WORDS)
    resized = pipeline.resize_to_long_edge(picture, pipeline.PREVIEW_LONG_EDGE)
    # The cores of the letters and of their ground; the letters' anti-aliased edges print part ink.
    letters = analysis.lettering_area & (resized[:, :, 0] > 200)
    ground = analysis.lettering_area & (resized[:, :, 0] < 25)
    assert letters.sum() > 500 and ground.sum() > letters.sum()
    assert (analysis.lettering[letters] > 220).mean() > 0.9  # the letters bare paper, or nearly
    assert (analysis.lettering[ground] < 10).mean() > 0.95  # their ground solid, or nearly
    assert analysis.printed_ink[ground].all() and not analysis.printed_ink[letters].any()


def test_on_line_art_no_number_clears_lettering_and_the_ink_keeps_the_artwork_s_gray(monkeypatch):
    comic = pipeline.load_image_bgr(SAMPLES / "m-comics-upside-downs-writing-pig.jpg")
    params = params_for_preset("Hard")
    calls = []
    real = pipeline.render_page

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "render_page", recording)
    pipeline.clear_cache()
    page = pipeline.generate(comic, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert calls and page.analysis.lettering_area.any()
    for kwargs in calls:
        assert kwargs["clearable"].any()  # the comic's hatching, which a number may clear
        assert not (kwargs["clearable"] & kwargs["lettering_area"]).any()  # but never inside a line of text
    # Inside the lines, the ink lying in a region is the lettering darker than halfway and no other, as the page prints
    # it: the comic's own ink there, pale where the lettering is, is not printed ink any more. Its ink in no region is.
    analysis = page.analysis
    lettering_ink = 255 - analysis.lettering.astype(np.int64)
    in_region = analysis.lettering_area & (analysis.region_id_map >= 0)
    np.testing.assert_array_equal(analysis.printed_ink[in_region], lettering_ink[in_region] >= 128)
    plain = _without_text(monkeypatch, comic, params, pipeline.PREVIEW_LONG_EDGE)
    assert (plain.analysis.printed_ink & in_region & (lettering_ink < 128)).sum() > 1000  # it was before
    walls = analysis.lettering_area & (analysis.region_id_map < 0) & plain.analysis.printed_ink
    assert walls.any() and analysis.printed_ink[walls].all()
    assert page.analysis.ink_gray == plain.analysis.ink_gray
    np.testing.assert_array_equal(page.analysis.region_id_map, plain.analysis.region_id_map)


def test_after_a_merge_no_number_clears_lettering_either(monkeypatch):
    # On line art a region whose number found no room joins the area beside it and the page is drawn again, with its
    # hatching found again: lettering is still none of it. The comic takes that path at no preset, so a number is made
    # to find no room, and the merge changes nothing.
    comic = pipeline.load_image_bgr(SAMPLES / "m-comics-upside-downs-writing-pig.jpg")
    calls = []
    real = pipeline.render_page

    def cramping(*args, **kwargs):
        calls.append(kwargs)
        rendered = real(*args, **kwargs)
        if len(calls) == 1:
            rendered.labels[0].cramped = True
        return rendered

    monkeypatch.setattr(pipeline, "render_page", cramping)
    monkeypatch.setattr(pipeline, "merge_cramped", lambda region_id_map, *args: region_id_map.copy())
    pipeline.clear_cache()
    pipeline.generate(comic, params_for_preset("Hard"), pipeline.PREVIEW_LONG_EDGE)
    assert len(calls) == 2  # drawn again after the merge
    assert calls[1]["clearable"].any() and calls[1]["lettering_area"].any()
    assert not (calls[1]["clearable"] & calls[1]["lettering_area"]).any()
    pipeline.clear_cache()


def test_a_picture_without_text_prints_no_lettering():
    pipeline.clear_cache()
    ramp = np.tile(np.linspace(40, 220, 600).astype(np.uint8)[None, :, None], (400, 1, 3))
    page = pipeline.generate(ramp, params_for_preset("Easy"), pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert page.analysis.text == []
    assert not page.analysis.lettering_area.any() and (page.analysis.lettering == 255).all()


# --- the harness -------------------------------------------------------------------------------------------------


def test_text_found_recall_is_the_share_of_the_annotated_boxes_in_text_found():
    shape = (20, 30)
    found = [((0, 0), (10, 0), (10, 10), (0, 10))]  # pixels 0-9 by 0-9
    boxes = [(5, 0, 10, 10), (20, 15, 5, 5)]  # half in, and out
    match = bm.found_text_match(found, boxes, shape)
    assert match["covers"] == [0.5, 0.0]
    assert match["text_found_recall"] == pytest.approx(50 / 125)
    assert match["stray_text_lines"] is None
    # Overlapping boxes count their shared pixels once.
    assert bm.found_text_match(found, [(0, 0, 10, 10), (0, 0, 20, 10)], shape)["text_found_recall"] == 0.5
    # A line's box turned to it covers what it covers, not its bounding box: a diamond 10 across holds half of it.
    diamond = [((10, 5), (15, 10), (10, 15), (5, 10))]
    assert bm.found_text_match(diamond, [(5, 5, 10, 10)], shape)["text_found_recall"] == pytest.approx(0.5, abs=0.05)


def test_the_report_holds_text_found_recall_to_0_9_and_stray_lines_to_none():
    import bench_report

    recall, stray = bench_report.METRICS_BY_KEY["text_found_recall"], bench_report.METRICS_BY_KEY["stray_text_lines"]
    assert bench_report.misses_target(recall, {"text_found_recall": 0.899}) is True
    assert bench_report.misses_target(recall, {"text_found_recall": 0.9}) is False
    assert bench_report.misses_target(recall, {"text_found_recall": None}) is None
    assert bench_report.misses_target(stray, {"stray_text_lines": 1}) is True
    assert bench_report.misses_target(stray, {"stray_text_lines": 0}) is False
    assert recall.job is None and stray.job is None  # the finding itself, outside the scorecard


def test_lines_found_without_annotated_text_are_stray():
    lines = [((0, 0), (10, 0), (10, 4), (0, 4)), ((0, 8), (10, 8), (10, 12), (0, 12))]
    match = bm.found_text_match(lines, [], (20, 20))
    assert match == {"covers": [], "text_found_recall": None, "stray_text_lines": 2}
    assert bm.found_text_match([], [], (20, 20))["stray_text_lines"] == 0


def test_the_case_runner_scores_the_text_found_and_lists_it():
    # A page 40 wide and 20 tall; the line covers the left half of the block, which reaches the page's right edge.
    line = _rect_line((10, 5), (20, 10))
    page = bench_case.PageData(
        source="analysis",
        region_id_map=np.zeros((20, 40), dtype=np.int32),
        region_color=np.zeros(1, dtype=np.int32),
        palette_bgr=np.zeros((1, 3), dtype=np.uint8),
        legend_bgr=np.zeros((1, 3), dtype=np.uint8),
        min_region_area_px=1,
        regions=[],
        labeled_region_ids=set(),
        label_font_sizes_px=[],
        label_boxes=[],
        strokes=[],
        outlines=None,
        leaders=None,
        leader_labels=0,
        text=[line],
    )
    block = bench_manifest.TextBlock(box=bench_manifest.Box(0, 0, 40, 10), string="HELLO")
    quality, found, matches = bench_case.found_text_scores(page, [block])
    assert quality == {"text_lines_found": 1, "text_found_recall": 0.5, "stray_text_lines": None}
    assert found == [{"quad": [list(p) for p in line.quad], "score": 0.9}]
    assert matches == [{"string": "HELLO", "cover": 0.5}]
    quality, _, matches = bench_case.found_text_scores(page, [])
    assert quality == {"text_lines_found": 1, "text_found_recall": None, "stray_text_lines": 1} and matches is None
    page.text = None  # a version that doesn't look for text
    assert bench_case.found_text_scores(page, [block]) == (dict.fromkeys(bench_case.FOUND_TEXT_KEYS), None, None)


@pytest.mark.parametrize(
    "name, preset",
    [("m-comics-upside-downs-writing-pig.jpg", "Hard"), ("l-photo-times-square.jpg", "Max")],
)
def test_every_number_keeps_half_a_millimeter_from_the_lines_of_text_found(name, preset):
    # On v0.1.41 a number on the comic sat 0.22 mm from a caption, and two on Times Square inside lines found (D-051).
    pipeline.clear_cache()
    picture = pipeline.load_image_bgr(SAMPLES / name)
    params = bench_case.preset_params(difficulty, preset)
    page = pipeline.generate(picture, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    analysis = page.analysis
    assert analysis.lettering_area.any()
    gap = print_scale(page.page.size).mm_to_px(0.5)
    away = cv2.distanceTransform((~analysis.lettering_area).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    assert len(analysis.labels) == len(analysis.regions) and not any(label.cramped for label in analysis.labels)
    for label in analysis.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert away[max(0, y0) : y1, max(0, x0) : x1].min() > gap, label

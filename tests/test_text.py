"""Text: PP-OCRv6-small finds the lines of text in a picture, looking twice where it finds any (D-049)."""

import hashlib
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import faces, pipeline, text
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


def test_a_picture_without_text_at_preview_size_is_looked_at_once(monkeypatch):
    network = _Network(_blank_map)
    monkeypatch.setattr(text, "_probability", network)
    picture = np.full((600, 900, 3), 200, dtype=np.uint8)
    assert text.find_text(picture) == []
    assert len(network.views) == 1
    assert network.views[0].shape[:2] == (608, 896)  # each side the nearest multiple of 32


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
        np.testing.assert_array_equal(network.views[0], cv2.resize(picture, (896, 608)))
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


def test_text_is_found_once_per_picture_for_the_analysis_only(monkeypatch):
    # Nothing on the page uses it yet (T6.2 will): the page is the same, and the app doesn't pay for it.
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
    assert calls == []
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

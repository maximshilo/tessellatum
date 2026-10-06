"""Faces: YuNet for photographed and painted faces, an LBP cascade for drawn ones, and what the pipeline reports."""

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tessellatum.core import faces, pipeline, subject
from tessellatum.core.difficulty import params_for_preset
from tessellatum.core.print_size import print_scale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_manifest  # noqa: E402
import bench_metrics as bm  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_images"
CASCADE = faces.load_cascade(faces.RESOURCES_DIR / faces.CASCADE_MODEL)
# The SHA-1 of scene.png's page at Medium, at preview size, as v0.1.33 drew it (benchmark result set T4.1-candidate).
SCENE_MEDIUM_PAGE_SHA1 = "e8c9a24bb35bb3bc6d1d0b5e099c8d7200ae56af"


def gray_input(name: str, factor: int, pad: int) -> np.ndarray:
    """A sample picture in gray, shrunk ``factor``-fold by block means and padded with its edge, in integers only.

    Built without OpenCV, so that the pinned OpenCV 4 results below hold
    whatever OpenCV version the tests run with.
    """
    with Image.open(SAMPLES / name) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int64)
    gray = (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114 + 500) // 1000
    h, w = gray.shape[0] // factor * factor, gray.shape[1] // factor * factor
    blocks = gray[:h, :w].reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
    return np.pad(((blocks + factor * factor // 2) // (factor * factor)).astype(np.uint8), pad, mode="edge")


def _lcg(seed: int):
    state = seed

    def draw(k: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) % 2**31
        return (state >> 8) % k

    return draw


def random_cascade_xml(seed: int) -> str:
    """An OpenCV LBP cascade of 3 stages over a 6 x 6 window, whose first stage passes about three windows in four.

    On noise it passes hundreds of windows at every size, where the real
    cascade passes a handful: so every rule about which windows are searched
    shows in what it finds.
    """
    draw = _lcg(seed)
    rects = []
    for _ in range(6):
        w, h = 1 + draw(2), 1 + draw(2)
        rects.append((draw(6 - 3 * w + 1), draw(6 - 3 * h + 1), w, h))
    stages = []
    for stumps, threshold in ((2, 0.0), (3, -1.0), (3, -1.0)):
        weak = []
        for _ in range(stumps):
            words = []
            for _ in range(8):
                word = (draw(65536) << 16) | draw(65536)
                words.append(word - 2**32 if word >= 2**31 else word)
            leaves = [(-1.0, 1.0), (1.0, -1.0), (0.5, -1.0), (-0.5, 1.0)][draw(4)]
            weak.append(
                f"<_><internalNodes>0 -1 {draw(6)} {' '.join(map(str, words))}</internalNodes>"
                f"<leafValues>{leaves[0]} {leaves[1]}</leafValues></_>"
            )
        stages.append(
            f"<_><maxWeakCount>{stumps}</maxWeakCount><stageThreshold>{threshold}</stageThreshold>"
            f"<weakClassifiers>{''.join(weak)}</weakClassifiers></_>"
        )
    features = "".join(f"<_><rect>{x} {y} {w} {h}</rect></_>" for x, y, w, h in rects)
    return (
        '<?xml version="1.0"?>\n<opencv_storage><cascade><stageType>BOOST</stageType><featureType>LBP</featureType>'
        "<height>6</height><width>6</width><stageParams><boostType>GAB</boostType><minHitRate>0.995</minHitRate>"
        "<maxFalseAlarm>0.5</maxFalseAlarm><weightTrimRate>0.95</weightTrimRate><maxDepth>1</maxDepth>"
        "<maxWeakCount>100</maxWeakCount></stageParams><featureParams><maxCatCount>256</maxCatCount>"
        f"<featSize>1</featSize></featureParams><stageNum>3</stageNum><stages>{''.join(stages)}</stages>"
        f"<features>{features}</features></cascade></opencv_storage>\n"
    )


def edge_cascade_xml() -> str:
    """``random_cascade_xml(1)`` cut to one stage of one stump whose leaves are both its threshold as OpenCV lowers it.

    Every window's sum is then exactly the threshold, which passes: the
    windows it reports are exactly the windows searched.
    """
    xml = random_cascade_xml(1)
    leaf = repr(float(np.float32(0) - np.float32(1e-5)))
    head = xml.split("<stages>")[0].replace("<stageNum>3</stageNum>", "<stageNum>1</stageNum>")
    stage = (
        "<stages><_><maxWeakCount>1</maxWeakCount><stageThreshold>0</stageThreshold><weakClassifiers>"
        f"<_><internalNodes>0 -1 0 -1 -1 -1 -1 -1 -1 -1 -1</internalNodes><leafValues>{leaf} {leaf}</leafValues></_>"
        "</weakClassifiers></_></stages>"
    )
    return head + stage + "<features>" + xml.split("<features>")[1]


def noise_image(seed: int, size: tuple[int, int] = (48, 36)) -> np.ndarray:
    """A gray gradient with noise, from an LCG."""
    draw = _lcg(seed)
    w, h = size
    return np.array([[(5 * x + 3 * y + draw(60)) % 256 for x in range(w)] for y in range(h)], dtype=np.uint8)


def _digest(boxes) -> str:
    return hashlib.sha1(repr(sorted(tuple(map(int, box)) for box in boxes)).encode()).hexdigest()[:16]


def windows(seed: int, count: int) -> list[list[int]]:
    """Square windows around a few centers, two of them inside two others, from a small LCG."""
    state = seed

    def draw(k: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) % 2**31
        return (state >> 8) % k

    centers = [(draw(300), draw(300), 20 + draw(80)) for _ in range(5)]
    centers += [(x + size // 4, y + size // 4, size // 2) for x, y, size in centers[:2]]
    out = []
    for _ in range(count):
        x, y, size = centers[draw(len(centers))]
        side = size + draw(9) - 4
        out.append([x + draw(7) - 3, y + draw(7) - 3, side, side])
    return out


# OpenCV 4.14's CascadeClassifier("lbpcascade_animeface.xml").detectMultiScale(gray_input(name, factor, pad), 1.1,
# min_neighbors, minSize=(min_size, min_size)), sorted: (name, factor, pad, min_size, min_neighbors): boxes.
CASCADE_EXPECTED = {
    ("m-cartoon-bold-lines-girl.png", 4, 60, 0, 0): [
        (43, 65, 260, 260), (48, 60, 286, 286), (49, 79, 236, 236), (49, 89, 236, 236), (54, 54, 260, 260),
        (54, 65, 260, 260), (59, 79, 236, 236), (59, 89, 236, 236), (65, 76, 260, 260), (69, 89, 236, 236),
        (73, 114, 195, 195), (81, 122, 195, 195),
    ],
    ("m-cartoon-bold-lines-girl.png", 4, 60, 0, 1): [(59, 82, 241, 241)],
    ("m-cartoon-bold-lines-girl.png", 4, 60, 0, 3): [(59, 82, 241, 241)],
    ("m-cartoon-bold-lines-girl.png", 4, 60, 0, 5): [(59, 82, 241, 241)],
    ("m-cartoon-bold-lines-girl.png", 4, 60, 60, 0): [
        (43, 65, 260, 260), (48, 60, 286, 286), (49, 79, 236, 236), (49, 89, 236, 236), (54, 54, 260, 260),
        (54, 65, 260, 260), (59, 79, 236, 236), (59, 89, 236, 236), (65, 76, 260, 260), (69, 89, 236, 236),
        (73, 114, 195, 195), (81, 122, 195, 195),
    ],
    ("m-cartoon-bold-lines-girl.png", 4, 60, 60, 3): [(59, 82, 241, 241)],
    ("m-cartoon-bold-lines-girl.png", 3, 0, 0, 0): [(0, 26, 315, 315), (12, 60, 286, 286), (13, 39, 315, 315)],
    ("m-cartoon-bold-lines-girl.png", 3, 0, 0, 1): [(8, 42, 305, 305)],
    ("m-cartoon-bold-lines-girl.png", 3, 0, 0, 3): [],
    # A box reaching a pixel past the image, from rounding a shrunk window back up, is clipped.
    ("m-cartoon-bold-lines-girl.png", 6, 0, 0, 0): [
        (0, 7, 161, 161), (0, 13, 161, 161), (0, 18, 147, 147), (6, 18, 147, 147), (6, 33, 133, 133),
        (7, 13, 159, 161), (11, 33, 133, 133), (12, 24, 147, 147), (15, 40, 121, 121),
    ],
    ("m-cartoon-bold-lines-girl.png", 6, 0, 0, 3): [(6, 22, 146, 146)],
    ("m-cartoon-bold-lines-reaper.png", 4, 0, 0, 0): [],
    ("m-comics-upside-downs-writing-pig.jpg", 2, 0, 0, 0): [
        (14, 328, 83, 83), (16, 336, 75, 75), (19, 352, 75, 75), (21, 342, 62, 62), (226, 455, 75, 75),
        (236, 86, 62, 62), (236, 88, 62, 62), (247, 334, 51, 51), (276, 183, 83, 83), (282, 185, 75, 75),
        (282, 188, 75, 75), (288, 200, 57, 57), (311, 76, 83, 83), (311, 79, 83, 83), (314, 79, 83, 83),
        (400, 116, 24, 24), (402, 118, 24, 24),
    ],
    ("m-comics-upside-downs-writing-pig.jpg", 2, 0, 0, 1): [
        (17, 335, 73, 73), (236, 87, 62, 62), (280, 185, 78, 78), (312, 78, 83, 83), (401, 117, 24, 24),
    ],
    ("m-comics-upside-downs-writing-pig.jpg", 2, 0, 0, 3): [],
    ("scene.png", 1, 0, 0, 0): [],
}

# OpenCV 4.14's groupRectangles(windows(seed, count), min_neighbors, 0.2): (seed, count, min_neighbors): (box, windows).
GROUP_EXPECTED = {
    (1, 40, 1): [((130, 293, 57, 57), 7), ((195, 179, 88, 88), 10), ((239, 0, 89, 89), 7), ((243, 136, 82, 82), 2)],
    (1, 40, 2): [((130, 293, 57, 57), 7), ((195, 179, 88, 88), 10), ((239, 0, 89, 89), 7)],
    (2, 80, 3): [
        ((10, 211, 63, 63), 11), ((153, 226, 85, 85), 12), ((165, 181, 95, 95), 17), ((289, 28, 73, 73), 10),
        ((296, 175, 22, 22), 13),
    ],
    (3, 120, 5): [((56, 49, 87, 87), 17), ((80, 216, 30, 30), 24), ((125, 243, 73, 73), 27), ((138, 53, 42, 42), 14)],
    (7, 60, 3): [
        ((72, 72, 43, 43), 12), ((142, 168, 90, 90), 11), ((270, 15, 67, 67), 10), ((291, 145, 80, 80), 12),
        ((292, 80, 72, 72), 8),
    ],
}

# OpenCV 4.14's CascadeClassifier on random_cascade_xml(seed) and noise_image(seed + 10), scale step 1.1:
# (seed, min_neighbors, min_size): (how many boxes, _digest of them).
SYNTHETIC_EXPECTED = {
    (1, 0, 0): (361, "ed79a4788e41396e"),
    (1, 0, 10): (166, "fd7c74b59cf959d3"),
    (1, 2, 0): (10, "bae6789e2bd4739f"),
    (2, 0, 0): (810, "2f31fdf53b0e8fd5"),
    (2, 0, 10): (346, "e9994a4d1632c325"),
    (2, 2, 0): (29, "96c59d2027c9d7d5"),
    (3, 0, 0): (618, "3bc971ccbd2573de"),
    (3, 0, 10): (295, "26bb1f13e7a2ea79"),
    (3, 2, 0): (18, "71d62e7d8795d842"),
}
# The same for random_cascade_xml(1) on noise_image(11) with min_neighbors 0 and a min_size no window, or some, reaches.
# At (38, 6) the sizes 21 and 23 are equally near: the first is searched.
MIN_SIZE_EXPECTED = {
    (100, 100): (0, "97d170e1550eee4a"),
    (37, 5): (7, "312fe68bad5c9be0"),
    (38, 6): (7, "312fe68bad5c9be0"),
    (13, 13): (120, "884fc71e896b1d86"),
}
# random_cascade_xml(1) with its window 55 px wide, on noise_image(11, (90, 40)), min_neighbors 0, by min_size: 55 x 1.1
# is 60.500000000000014, a 61 px window in double precision but 60 in the single precision OpenCV scales in.
WIDE_EXPECTED = {(61, 0): (44, "23592f281542c5a9"), (0, 0): (126, "137f6c103d7e3429")}
# OpenCV 4.14's CascadeClassifier on edge_cascade_xml() and noise_image(5, size), min_neighbors 0: the windows it searches.
EDGE_EXPECTED = {
    (8, 8): (5, "7d4ebf4ab3fd86f9"),
    (9, 9): (9, "350efe0cf43fab27"),
    (40, 7): (48, "d96763271aa30990"),
    (70, 13): (519, "20075efff7b327ab"),
}

_BIG = [[100 + d, 100 + d, 100, 100] for d in range(-2, 3)] * 2  # ten windows of one face
# OpenCV 4.14's groupRectangles on groups nested in others, with min_neighbors 1 and 3 alike.
NESTED_EXPECTED = [
    # Four windows inside a group of ten: the ten outnumber them, so they are dropped.
    (_BIG + [[130 + d, 130, 40, 40] for d in range(4)], [((100, 100, 100, 100), 10)]),
    # Twelve inside ten: they outnumber the ten, so they stay.
    (_BIG + [[130 + d % 4, 130 + d // 4, 40, 40] for d in range(12)], [((100, 100, 100, 100), 10), ((132, 131, 40, 40), 12)]),
]


def _picture(name: str) -> np.ndarray:
    return pipeline.resize_to_long_edge(pipeline.load_image_bgr(SAMPLES / name), pipeline.PREVIEW_LONG_EDGE)


def test_the_bundled_models_are_the_files_recorded_with_their_licenses():
    notes = (faces.RESOURCES_DIR / "MODELS.md").read_text(encoding="utf-8")
    onnx = (faces.RESOURCES_DIR / faces.YUNET_MODEL).read_bytes()
    assert hashlib.sha256(onnx).hexdigest() in notes
    # Git may check the XML out with CRLF line endings; its hash is recorded as downloaded, with LF.
    xml = (faces.RESOURCES_DIR / faces.CASCADE_MODEL).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(xml).hexdigest() in notes
    assert b"The MIT License" in xml[:1000]
    assert notes.count("MIT License") >= 2


def test_the_cascade_loads_as_opencv_reads_it():
    assert CASCADE.window == (24, 24)
    assert CASCADE.stage_ends.size == 20 and CASCADE.stage_ends[-1] == CASCADE.features.size == 771
    assert CASCADE.fx.shape == CASCADE.fy.shape == (642, 4)
    assert CASCADE.subsets.shape == (771, 8) and CASCADE.subsets.dtype == np.uint32
    # The file's first stage threshold, lowered by 1e-5 in single precision.
    text = (faces.RESOURCES_DIR / faces.CASCADE_MODEL).read_text(encoding="ascii")
    first = text.split("<stageThreshold>")[1].split("</stageThreshold>")[0]
    assert CASCADE.thresholds[0] == np.float32(first) - np.float32(1e-5)


def test_a_cascade_of_another_kind_is_refused(tmp_path):
    path = tmp_path / "haar.xml"
    path.write_text(
        "<opencv_storage><cascade><stageType>BOOST</stageType><featureType>HAAR</featureType></cascade></opencv_storage>"
    )
    with pytest.raises(ValueError):
        faces.load_cascade(path)


@pytest.mark.parametrize(
    "broken",
    [
        ("<rect>0 0 2 2</rect>", "<rect>1 0 2 2</rect>"),  # three 2-px blocks from x = 1 reach past a 6-px window
        ("<rect>0 0 2 2</rect>", "<rect>0 0 0 2</rect>"),  # blocks of no width
        ("<internalNodes>0 -1 0 ", "<internalNodes>0 -1 6 "),  # a seventh feature, of six
        ("<leafValues>1.0 -1.0</leafValues>", "<leafValues>1.0</leafValues>"),
    ],
    ids=["past the window", "empty blocks", "no such feature", "one leaf"],
)
def test_a_cascade_the_search_could_read_outside_of_is_refused(broken, tmp_path):
    xml = random_cascade_xml(1)
    old, new = broken
    assert old in xml
    path = tmp_path / "cascade.xml"
    path.write_text(xml.replace(old, new, 1))
    with pytest.raises(ValueError):
        faces.load_cascade(path)
    path.write_text(xml)
    faces.load_cascade(path)  # the cascade as written loads


def test_an_image_whose_integral_overflows_32_bits_is_refused():
    faces.detect_cascade(np.zeros((2048, 4112), dtype=np.uint8), _one_stump_cascade(0), min_neighbors=0)  # 8,421,376 px
    with pytest.raises(ValueError):
        faces.detect_cascade(np.zeros((2048, 4113), dtype=np.uint8), _one_stump_cascade(0), min_neighbors=0)


def _one_stump_cascade(code: int) -> faces.Cascade:
    """A 3 x 3 window, one feature of 1 x 1 blocks, passing only windows whose LBP code is ``code``."""
    subset = np.zeros((1, 8), dtype=np.uint32)
    subset[0, code >> 5] = np.uint32(1) << np.uint32(code & 31)
    edges = np.arange(4, dtype=np.int32).reshape(1, 4)
    return faces.Cascade(
        window=(3, 3),
        stage_ends=np.array([1], dtype=np.int32),
        thresholds=np.array([0.0], dtype=np.float32),
        features=np.array([0], dtype=np.int32),
        subsets=subset,
        leaves=np.array([[1.0, -1.0]], dtype=np.float32),
        fx=edges,
        fy=edges.copy(),
    )


def test_an_lbp_code_has_a_bit_for_each_outer_block_at_least_as_bright_as_the_middle():
    # Clockwise from the top left, the top left the highest bit: 200 >= 100 (128), 50 (no), 100 = 100 (32), 99 (no),
    # 101 (8), 0 (no), 255 (2), 100 (1).
    gray = np.array([[200, 50, 100], [100, 100, 99], [255, 0, 101]], dtype=np.uint8)
    code = 128 | 32 | 8 | 2 | 1
    found = faces.detect_cascade(gray, _one_stump_cascade(code), min_neighbors=0)
    assert found and {box for box, _count in found} == {(0, 0, 3, 3)}
    for other in (code ^ 1, code ^ 16, code ^ 128):
        assert faces.detect_cascade(gray, _one_stump_cascade(other), min_neighbors=0) == []


@pytest.mark.parametrize("case", sorted(CASCADE_EXPECTED), ids=lambda case: "-".join(map(str, case)))
def test_the_cascade_finds_what_opencv_4_finds(case):
    name, factor, pad, min_size, min_neighbors = case
    found = faces.detect_cascade(gray_input(name, factor, pad), CASCADE, 1.1, min_neighbors, (min_size, min_size))
    assert sorted(box for box, _count in found) == CASCADE_EXPECTED[case]


@pytest.mark.parametrize("case", sorted(SYNTHETIC_EXPECTED), ids=lambda case: "-".join(map(str, case)))
def test_a_cascade_passing_hundreds_of_windows_finds_what_opencv_4_finds(case, tmp_path):
    seed, min_neighbors, min_size = case
    path = tmp_path / "cascade.xml"
    path.write_text(random_cascade_xml(seed))
    found = faces.detect_cascade(noise_image(seed + 10), faces.load_cascade(path), 1.1, min_neighbors, (min_size, min_size))
    assert (len(found), _digest(box for box, _count in found)) == SYNTHETIC_EXPECTED[case]


@pytest.mark.parametrize("min_size", sorted(MIN_SIZE_EXPECTED))
def test_a_min_size_no_window_reaches_searches_the_nearest_size(min_size, tmp_path):
    path = tmp_path / "cascade.xml"
    path.write_text(random_cascade_xml(1))
    found = faces.detect_cascade(noise_image(11), faces.load_cascade(path), 1.1, 0, min_size)
    assert (len(found), _digest(box for box, _count in found)) == MIN_SIZE_EXPECTED[min_size]
    if min_size in ((37, 5), (38, 6)):  # no window is 37 wide: the size nearest, 21 x 21, is searched alone
        assert {box[2] for box, _count in found} <= {20, 21}


@pytest.mark.parametrize("min_size", sorted(WIDE_EXPECTED))
def test_window_sizes_are_judged_against_min_size_in_single_precision(min_size, tmp_path):
    path = tmp_path / "cascade.xml"
    path.write_text(random_cascade_xml(1).replace("<width>6</width>", "<width>55</width>"))
    found = faces.detect_cascade(noise_image(11, (90, 40)), faces.load_cascade(path), 1.1, 0, min_size)
    assert (len(found), _digest(box for box, _count in found)) == WIDE_EXPECTED[min_size]
    assert (60 in {box[2] for box, _count in found}) == (min_size == (0, 0))


@pytest.mark.parametrize("size", sorted(EDGE_EXPECTED))
def test_the_windows_searched_are_opencv_4s_and_a_sum_at_the_threshold_passes(size, tmp_path):
    path = tmp_path / "cascade.xml"
    path.write_text(edge_cascade_xml())
    found = faces.detect_cascade(noise_image(5, size), faces.load_cascade(path), 1.1, 0)
    assert (len(found), _digest(box for box, _count in found)) == EDGE_EXPECTED[size]


def test_opencv_stripes_leave_the_last_row_of_windows_unsearched_where_they_divide_exactly():
    # Three positions at a two-pixel step, in one stripe: rows at 0 and 2, but the stripe is one step tall.
    assert faces._searched_rows(3, 2, 1) == 1
    assert faces._searched_rows(4, 2, 1) == 2 and faces._searched_rows(3, 1, 1) == 3
    # 101 positions, 3 stripes of 17 steps: all 51 rows.
    assert faces._searched_rows(101, 2, 3) == 51
    assert faces._searched_rows(0, 2, 1) == 0


def test_the_cascade_counts_each_window_alone_with_no_neighbors_asked():
    found = faces.detect_cascade(gray_input("m-cartoon-bold-lines-girl.png", 4, 60), CASCADE, 1.1, 0)
    assert [count for _box, count in found] == [1] * 12


@pytest.mark.parametrize("case", sorted(GROUP_EXPECTED), ids=lambda case: "-".join(map(str, case)))
def test_windows_group_as_opencv_4_groups_them(case):
    seed, count, min_neighbors = case
    assert sorted(faces.group_windows(windows(seed, count), min_neighbors)) == GROUP_EXPECTED[case]


@pytest.mark.parametrize("min_neighbors", [1, 3])
@pytest.mark.parametrize("nested, expected", NESTED_EXPECTED, ids=["outnumbered", "outnumbering"])
def test_a_group_inside_a_bigger_one_is_dropped_where_the_bigger_one_has_more_windows(nested, expected, min_neighbors):
    assert sorted(faces.group_windows(nested, min_neighbors)) == expected


def test_a_pair_inside_a_pair_is_dropped_as_a_group_of_fewer_than_three():
    pairs = [[300, 300, 100, 100], [301, 300, 100, 100], [330, 330, 40, 40], [331, 330, 40, 40]]
    assert faces.group_windows(pairs, 1) == [((300, 300, 100, 100), 2)]
    assert faces.group_windows(pairs, 3) == []


def _annotated(name: str, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    info = bench_manifest.find_image(SAMPLES / name).scaled_to(size)
    return [(face.box.x, face.box.y, face.box.w, face.box.h) for face in info.faces]


@pytest.mark.parametrize("name", sorted(p.name for p in SAMPLES.iterdir() if p.suffix in (".png", ".jpg")))
def test_the_annotated_faces_are_found_and_nothing_else(name):
    picture = _picture(name)
    found = faces.find_faces(picture)
    annotated = _annotated(name, picture.shape[1::-1])
    match = bm.found_faces_match([face.box for face in found], annotated)
    assert match["stray_faces"] == 0
    if annotated:
        assert match["face_found_recall"] == 1.0
        # Photographs and paintings by YuNet, the drawing by the cascade.
        expected = "cascade" if name.startswith("m-cartoon") else "yunet"
        assert {found[m["found_index"]].detector for m in match["matches"]} == {expected}
    else:
        assert found == []


def test_yunet_gives_the_eyes_above_the_nose_above_the_mouth_inside_the_face():
    found = faces.find_faces(_picture("l-painting-face-girl-with-a-pearl-earring.jpg"))
    assert [face.detector for face in found] == ["yunet"]
    (x, y, w, h), points = found[0].box, found[0].landmarks
    assert len(points) == 5 and all(x <= px <= x + w and y <= py <= y + h for px, py in points)
    right_eye, left_eye, nose, right_mouth, left_mouth = points
    assert max(right_eye[1], left_eye[1]) < nose[1] < min(right_mouth[1], left_mouth[1])
    assert right_eye[0] < left_eye[0] and right_mouth[0] < left_mouth[0]  # the face's right is the picture's left


def test_a_face_narrower_than_the_floor_is_left_out(monkeypatch):
    picture = _picture("l-photo-lion.jpg")
    widths_mm = [face.box[2] / print_scale(picture.shape[1::-1]).px_per_mm for face in faces.find_faces(picture)]
    assert len(widths_mm) == 2 and min(widths_mm) >= faces.MIN_FACE_WIDTH_MM
    monkeypatch.setattr(faces, "MIN_FACE_WIDTH_MM", (min(widths_mm) + max(widths_mm)) / 2)
    assert len(faces.find_faces(picture)) == 1
    monkeypatch.setattr(faces, "MIN_FACE_WIDTH_MM", max(widths_mm) + 1)
    assert faces.find_faces(picture) == []


def test_the_agreed_values():
    # D-042, Q24: faces under five brush widths are left out.
    assert faces.MIN_FACE_WIDTH_MM == 15.0
    # D-042: YuNet at 640 px, where the benchmark's faces sit in its range, keeping scores from 0.5.
    assert (faces.YUNET_LONG_EDGE, faces.YUNET_MIN_SCORE, faces.YUNET_MAX_OVERLAP) == (640, 0.5, 0.3)
    # D-042, Q25: the cascade at 480 px padded 15%, as nagadomi's own example runs it but for the padding and 3 neighbors.
    assert (faces.CASCADE_LONG_EDGE, faces.CASCADE_PAD, faces.CASCADE_SCALE_STEP) == (480, 0.15, 1.1)
    assert (faces.CASCADE_MIN_NEIGHBORS, faces.GROUP_EPS) == (3, 0.2)
    # D-042, Q25: found when nine tenths of the face lies in one face found, of which it is a third.
    assert (bm.FACE_MIN_COVER, bm.FACE_MIN_SHARE) == (0.9, 1 / 3)


def test_boxes_are_clipped_to_the_picture():
    picture = _picture("l-photo-lion.jpg")  # YuNet frames the lion's mane from above the picture's top edge
    raw = faces._yunet_faces(picture)
    assert min(face.box[1] for face in raw) < 0
    h, w = picture.shape[:2]
    for face in faces.find_faces(picture):
        x, y, bw, bh = face.box
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h


def test_scaled_moves_boxes_and_landmarks_with_the_page():
    face = faces.Face(box=(10.0, 20.0, 30.0, 40.0), score=0.9, detector="yunet", landmarks=((15.0, 25.0),))
    (moved,) = faces.scaled([face], (100, 200), (150, 100))
    assert moved.box == (15.0, 10.0, 45.0, 20.0) and moved.landmarks == ((22.5, 12.5),)
    assert moved.score == 0.9 and moved.detector == "yunet"


def test_the_pipeline_reports_the_faces_on_the_page_and_finds_them_once_per_picture(monkeypatch):
    pipeline.clear_cache()
    calls = []
    find = faces.find_faces
    monkeypatch.setattr(faces, "find_faces", lambda picture: calls.append(picture.shape) or find(picture))
    image = pipeline.load_image_bgr(SAMPLES / "m-cartoon-bold-lines-girl.png")
    params = params_for_preset("Easy")
    preview = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    export = pipeline.generate(image, params, pipeline.export_long_edge(image), collect_analysis=True)
    assert calls == [(1100, 685, 3)]  # once, on the picture at preview size
    assert len(preview.analysis.faces) == 1 and preview.analysis.faces[0].detector == "cascade"
    sx, sy = export.page.width / preview.page.width, export.page.height / preview.page.height
    assert export.analysis.faces[0].box == pytest.approx(
        tuple(v * s for v, s in zip(preview.analysis.faces[0].box, (sx, sy, sx, sy)))
    )
    pipeline.clear_cache()


def test_line_art_looks_for_faces_only_when_analysis_is_collected_and_its_page_never_uses_them(monkeypatch):
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "m-cartoon-bold-lines-girl.png")
    params = params_for_preset("Easy")
    with_faces = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert len(with_faces.analysis.faces) == 1
    assert not with_faces.analysis.detail.any()
    pipeline.clear_cache()

    def refuse(picture):
        raise AssertionError("looked for faces")

    monkeypatch.setattr(faces, "find_faces", refuse)
    without = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE)
    assert without.page.tobytes() == with_faces.page.tobytes()
    pipeline.clear_cache()


def _no_subject(picture):
    """``subject.find_subject`` on a picture with no subject: the face rule alone (the subject's own is D-048's)."""
    return np.zeros((subject.INPUT_SIZE, subject.INPUT_SIZE), dtype=np.float32)


def test_a_picture_drawn_from_its_colors_looks_for_faces_on_the_page_s_path_and_details_them(monkeypatch):
    # D-043, Q26: inside the faces found a region may be half the difficulty's smallest, on photographs and paintings.
    pipeline.clear_cache()
    monkeypatch.setattr(subject, "find_subject", _no_subject)
    calls = []
    find = faces.find_faces
    monkeypatch.setattr(faces, "find_faces", lambda picture: calls.append(picture.shape) or find(picture))
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-cats-face.jpg")
    params = params_for_preset("Medium")
    page = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE)
    assert len(calls) == 1  # without analysis, on the page's path
    analysis = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    assert len(calls) == 1  # and once per picture
    size = (analysis.region_id_map.shape[1], analysis.region_id_map.shape[0])
    assert len(analysis.faces) == 1
    np.testing.assert_array_equal(analysis.detail, faces.mask(analysis.faces, size))

    # Every region another touches counts at least the smallest region, its pixels in the face twice; some count only
    # because of that.
    ids, area = analysis.region_id_map, analysis.min_region_area_px
    assert bm.count_undersized(ids, area, analysis.detail) == 0
    assert bm.count_undersized(ids, area) > 0

    pipeline.clear_cache()
    monkeypatch.setattr(faces, "find_faces", lambda picture: [])
    plain = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert not plain.analysis.detail.any()
    assert plain.page.tobytes() != page.page.tobytes()
    assert plain.num_regions < page.num_regions
    pipeline.clear_cache()


def test_a_picture_without_faces_is_drawn_as_before(monkeypatch):
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "scene.png")
    # The regions' edges are settled since v0.1.37 (see ``texture``), its subject has more detail since v0.1.39 (see
    # ``subject``), and its corners keep their points since v0.1.50 (see ``regions.CornerRule``), none of them about
    # faces: they are left out here.
    params = dataclasses.replace(params_for_preset("Medium"), sharpest_corner_deg=180.0)
    monkeypatch.setattr(pipeline, "smooth_regions", lambda picture, ids, colors, *rest: (ids, colors))
    monkeypatch.setattr(subject, "find_subject", _no_subject)
    page = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert page.analysis.faces == [] and not page.analysis.detail.any()
    # Its page as v0.1.33 drew it, before faces were given more detail.
    assert hashlib.sha1(page.page.tobytes()).hexdigest() == SCENE_MEDIUM_PAGE_SHA1
    pipeline.clear_cache()


def test_the_mask_of_whole_pixel_boxes_is_the_pixels_they_span():
    face = faces.Face(box=(2.0, 1.0, 3.0, 2.0), score=1.0, detector="yunet")
    other = faces.Face(box=(6.0, 3.0, 1.0, 1.0), score=1.0, detector="cascade")
    expected = np.zeros((5, 8), dtype=bool)
    expected[1:3, 2:5] = True
    expected[3, 6] = True
    np.testing.assert_array_equal(faces.mask([face, other], (8, 5)), expected)
    assert not faces.mask([], (8, 5)).any()


def test_the_mask_of_a_fractional_box_is_the_pixels_whose_middle_it_holds():
    # From x = 1.4 to 3.6 the middles 1.5, 2.5 and 3.5 are inside; from y = 0.6 to 2.5, 1.5 only (2.5 is its far edge).
    face = faces.Face(box=(1.4, 0.6, 2.2, 1.9), score=1.0, detector="yunet")
    expected = np.zeros((4, 6), dtype=bool)
    expected[1, 1:4] = True
    np.testing.assert_array_equal(faces.mask([face], (6, 4)), expected)
    # From x = 1.6 to 3.8, 1.5 is left out; from y = 1.7 to 3.4, only 2.5.
    face = faces.Face(box=(1.6, 1.7, 2.2, 1.7), score=1.0, detector="yunet")
    expected = np.zeros((4, 6), dtype=bool)
    expected[2, 2:4] = True
    np.testing.assert_array_equal(faces.mask([face], (6, 4)), expected)
    # From x = 1.6 to 3.3, only 2.5 (3.3 falls short of 3.5).
    face = faces.Face(box=(1.6, 1.7, 1.7, 1.7), score=1.0, detector="yunet")
    expected = np.zeros((4, 6), dtype=bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(faces.mask([face], (6, 4)), expected)


def test_the_mask_stops_at_the_picture_s_edge():
    face = faces.Face(box=(-3.0, -2.0, 5.0, 10.0), score=1.0, detector="yunet")
    expected = np.zeros((4, 6), dtype=bool)
    expected[:, :2] = True
    np.testing.assert_array_equal(faces.mask([face], (6, 4)), expected)

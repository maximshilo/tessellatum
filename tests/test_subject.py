"""The subject: U²-Net-p finds what a picture is of, and the region stage gives it more detail (D-048)."""

import hashlib
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import faces, pipeline, subject
from tessellatum.core.difficulty import params_for_preset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_manifest  # noqa: E402
import bench_metrics as bm  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_images"


def _disk(center: tuple[int, int], radius: int = 60, size: tuple[int, int] = (400, 300)) -> np.ndarray:
    picture = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    cv2.circle(picture, center, radius, (40, 60, 200), -1)
    return picture


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


def _outline(name: str, size: tuple[int, int]) -> np.ndarray:
    """The subject's pixels as the manifest outlines them, on a page of ``size`` (width, height)."""
    info = bench_manifest.find_image(SAMPLES / name).scaled_to(size)
    return bm.outline_pixels((size[1], size[0]), [s.outline for s in info.subjects])


def test_the_model_is_the_file_models_md_lists():
    notes = (faces.RESOURCES_DIR / "MODELS.md").read_text(encoding="utf-8")
    model = (faces.RESOURCES_DIR / subject.MODEL).read_bytes()
    assert hashlib.sha256(model).hexdigest() in notes
    assert subject.MODEL in notes


@pytest.mark.parametrize(
    "picture",
    [
        np.zeros((30, 40, 3), dtype=np.float32),  # not 8-bit
        np.zeros((30, 40), dtype=np.uint8),  # gray
        np.zeros((30, 40, 4), dtype=np.uint8),  # with alpha
        np.zeros((0, 40, 3), dtype=np.uint8),  # empty
    ],
)
def test_finding_the_subject_takes_an_8_bit_color_picture(picture):
    with pytest.raises(ValueError):
        subject.find_subject(picture)


def test_the_map_is_the_network_s_square_view_of_the_whole_picture_and_the_same_every_time():
    picture = _disk((140, 160))
    first = subject.find_subject(picture)
    assert first.shape == (subject.INPUT_SIZE, subject.INPUT_SIZE) and first.dtype == np.float32
    assert 0.0 <= first.min() and first.max() <= 1.0
    np.testing.assert_array_equal(subject.find_subject(picture), first)


def test_the_network_sees_a_picture_scaled_to_its_brightest_value():
    # As U²-Net's own code feeds it: a dim picture is seen as the same picture at full brightness. At the network's own
    # size nothing is resampled, and halving even values is exact.
    full = _disk((160, 150), radius=70, size=(subject.INPUT_SIZE, subject.INPUT_SIZE))
    full[:20] = 254
    np.testing.assert_array_equal(subject.find_subject(full // 2), subject.find_subject(full))


def test_two_threads_find_what_one_finds():
    pictures = [_disk((140, 160)), _disk((260, 120), radius=40)]
    alone = [subject.find_subject(p) for p in pictures]
    found = [None, None]

    def find(i: int) -> None:
        for _ in range(3):
            found[i] = subject.find_subject(pictures[i])

    threads = [threading.Thread(target=find, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for got, expected in zip(found, alone):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("center", [(200, 150), (110, 90), (300, 210)])
def test_a_shape_on_a_plain_ground_is_the_subject_wherever_it_lies(center):
    # A salient object detector, not a center prior: the disk is found where it is.
    picture = _disk(center)
    found = subject.mask(subject.find_subject(picture), (400, 300))
    disk = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(disk, center, 60, 1, -1)
    assert _iou(found, disk.astype(bool)) > 0.9


def test_a_plain_or_noisy_picture_has_no_subject():
    rng = np.random.default_rng(0)
    for picture in (np.full((300, 400, 3), 128, dtype=np.uint8), rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)):
        assert not subject.mask(subject.find_subject(picture), (400, 300)).any()


def test_the_mask_stretches_the_map_over_the_page_and_holds_its_threshold():
    probability = np.zeros((4, 4), dtype=np.float32)
    probability[:, :2] = 1.0
    for size in ((40, 10), (7, 30)):
        found = subject.mask(probability, size)
        assert found.shape == (size[1], size[0])
        assert found[:, 0].all() and not found[:, -1].any()
    # At the threshold is in, under it out.
    level = np.full((2, 2), subject.MIN_PROBABILITY, dtype=np.float32)
    assert subject.mask(level, (5, 3)).all()
    assert not subject.mask(np.nextafter(level, np.float32(0)), (5, 3)).any()
    for size in ((0, 3), (3, 0)):
        with pytest.raises(ValueError):
            subject.mask(probability, size)
    with pytest.raises(ValueError):
        subject.mask(np.zeros(4, dtype=np.float32), (4, 4))


def test_the_cat_is_found_as_its_outline_and_times_square_has_no_subject():
    # The benchmark's own annotations (the manifest's outlines), at preview size.
    cat = pipeline.resize_to_long_edge(pipeline.load_image_bgr(SAMPLES / "l-photo-cats-face.jpg"), 1100)
    size = cat.shape[1::-1]
    found, outline = subject.mask(subject.find_subject(cat), size), _outline("l-photo-cats-face.jpg", size)
    assert found[outline].mean() > 0.95 and found[~outline].mean() < 0.02
    street = pipeline.resize_to_long_edge(pipeline.load_image_bgr(SAMPLES / "l-photo-times-square.jpg"), 1100)
    assert not subject.mask(subject.find_subject(street), street.shape[1::-1]).any()


def test_a_picture_drawn_from_its_colors_details_its_subject(monkeypatch):
    # D-048, Q31: inside the subject a region may be half the difficulty's smallest, as inside a face. Palermo has no face.
    pipeline.clear_cache()
    calls = []
    find = subject.find_subject
    monkeypatch.setattr(subject, "find_subject", lambda picture: calls.append(picture.shape) or find(picture))
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-palermo-castle.jpg")
    params = params_for_preset("Hard")
    page = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE)
    assert calls == [(736, 1100, 3)]  # on the page's path, on the picture at preview size
    analysis = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    export = pipeline.generate(image, params, pipeline.export_long_edge(image), collect_analysis=True).analysis
    assert len(calls) == 1  # once per picture, for the export too
    size = analysis.region_id_map.shape[::-1]
    resized = pipeline.resize_to_long_edge(image, pipeline.PREVIEW_LONG_EDGE)
    np.testing.assert_array_equal(analysis.subject, subject.mask(find(resized), size))
    assert analysis.faces == [] and 0.1 < analysis.subject.mean() < 0.3
    np.testing.assert_array_equal(analysis.detail, analysis.subject)
    # The export's subject is the preview's map stretched over the export's page.
    np.testing.assert_array_equal(export.subject, subject.mask(find(resized), export.region_id_map.shape[::-1]))

    # Every region another touches counts at least the smallest region, its pixels in the subject twice; some count only
    # because of that.
    ids, area = analysis.region_id_map, analysis.min_region_area_px
    assert bm.count_undersized(ids, area, analysis.detail) == 0
    assert bm.count_undersized(ids, area) > 0

    pipeline.clear_cache()
    monkeypatch.setattr(subject, "find_subject", lambda picture: np.zeros((4, 4), dtype=np.float32))
    plain = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert not plain.analysis.detail.any() and not plain.analysis.subject.any()
    assert plain.page.tobytes() != page.page.tobytes()
    assert plain.num_regions < page.num_regions
    # The subject is painted closer to the picture.
    outline = _outline("l-photo-palermo-castle.jpg", size)
    nearer = bm.subject_fidelity(resized, bm.paint(ids, analysis.region_color, analysis.palette_bgr), outline)
    p = plain.analysis
    farther = bm.subject_fidelity(resized, bm.paint(p.region_id_map, p.region_color, p.palette_bgr), outline)
    assert nearer["subject_de00_mean"] < farther["subject_de00_mean"]
    pipeline.clear_cache()


def test_a_face_and_the_subject_are_both_detail():
    # Only the face's tones are settled and only its marks printed (see test_tones and test_marks).
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "l-photo-cats-face.jpg")
    params = params_for_preset("Medium")
    analysis = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True).analysis
    in_faces = faces.mask(analysis.faces, analysis.region_id_map.shape[::-1])
    assert in_faces.any() and analysis.subject.any()
    np.testing.assert_array_equal(analysis.detail, in_faces | analysis.subject)
    pipeline.clear_cache()


def test_line_art_never_looks_for_its_subject(monkeypatch):
    pipeline.clear_cache()
    image = pipeline.load_image_bgr(SAMPLES / "m-cartoon-bold-lines-girl.png")
    params = params_for_preset("Easy")

    def refuse(picture):
        raise AssertionError("looked for the subject")

    monkeypatch.setattr(subject, "find_subject", refuse)
    result = pipeline.generate(image, params, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    assert result.analysis.line_art.is_line_art
    assert not result.analysis.subject.any() and not result.analysis.detail.any()
    pipeline.clear_cache()

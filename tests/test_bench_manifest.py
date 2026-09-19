"""The benchmark image manifest: schema checks, scaling, category selection, and the sample-image manifest."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench  # noqa: E402
import bench_manifest as manifest  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"


def _entry(**fields):
    return {"size": [200, 100], "categories": ["photo"], **fields}


def _parse(images: dict) -> dict:
    return manifest.parse_manifest({"schema": 1, "images": images})


def _write_manifest(path: Path, images: dict) -> None:
    path.write_text(json.dumps({"schema": 1, "images": images}), encoding="utf-8")


def test_parses_every_kind_of_annotation():
    info = _parse(
        {
            "a.png": _entry(
                categories=["cartoon", "face", "text", "gradient"],
                faces=[
                    {
                        "kind": "cartoon",
                        "box": [10, 10, 80, 60],
                        "features": [{"part": "eye", "box": [20, 20, 10, 10]}, {"part": "mouth", "box": [30, 50, 20, 10]}],
                    }
                ],
                text=[{"box": [100, 10, 90, 20], "string": "HELLO\nWORLD", "rotation": 90}],
                flat_colors=["#ffffff", "#FF0000"],
                ink_colors=["#000000"],
                areas=[{"kind": "gradient", "box": [0, 70, 200, 30]}],
                notes="synthetic",
            )
        }
    )["a.png"]

    assert info.primary_category == "cartoon"
    assert info.faces[0].features[1] == manifest.Feature("mouth", manifest.Box(30, 50, 20, 10))
    assert (info.text[0].string, info.text[0].rotation) == ("HELLO\nWORLD", 90)
    assert info.flat_colors == ((255, 255, 255), (255, 0, 0))
    assert info.ink_colors == ((0, 0, 0),)
    assert info.areas_of("gradient") == (manifest.Area("gradient", manifest.Box(0, 70, 200, 30)),)
    assert info.missing_annotations() == []
    assert not info.exact_colors  # unless the entry says so, colors are cluster centers of printed colors


def test_exact_colors_are_the_files_own_and_stay_exact_at_any_size():
    info = _parse({"a.png": _entry(flat_colors=["#ffffff", "#ff0000"], ink_colors=["#000000"], exact_colors=True)})["a.png"]

    assert info.exact_colors
    assert info.scaled_to((100, 50)).exact_colors


@pytest.mark.parametrize(
    "entry, message",
    [
        (_entry(categories=["painting"]), "'painting' is not one of"),
        (_entry(categories=[]), "categories must not be empty"),
        (_entry(categories=["photo", "photo"]), "categories must not repeat"),
        ({"categories": ["photo"]}, "missing key(s) size"),
        (_entry(colour="#ffffff"), "unknown key(s) colour"),
        (_entry(text=[{"box": [190, 0, 20, 10], "string": "x"}]), "outside the 200x100 image"),
        (_entry(text=[{"box": [0, 0, 0, 10], "string": "x"}]), "is empty or outside"),
        (_entry(text=[{"box": [0, 0, 10.5, 10], "string": "x"}]), "[x, y, width, height] in integer pixels"),
        (_entry(text=[{"box": [0, 0, 10, 10], "string": " "}]), "expected non-empty text"),
        (_entry(text=[{"box": [0, 0, 10, 10], "string": "x", "rotation": 360}]), "expected one of 0, 90, 180, 270 degrees"),
        (_entry(text=[{"box": [0, 0, 10, 10], "string": "x", "rotation": 45}]), "expected one of 0, 90, 180, 270 degrees"),
        (
            _entry(faces=[{"kind": "human", "box": [0, 0, 50, 50], "features": [{"part": "eye", "box": [40, 40, 20, 20]}]}]),
            "is not inside the face box",
        ),
        (_entry(faces=[{"kind": "human", "box": [0, 0, 50, 50], "features": [{"part": "ear", "box": [0, 0, 5, 5]}]}]), "'ear' is not one of"),
        (_entry(faces=[{"kind": "robot", "box": [0, 0, 50, 50]}]), "'robot' is not one of"),
        (_entry(areas=[{"kind": "sky", "box": [0, 0, 50, 50]}]), "'sky' is not one of"),
        (_entry(flat_colors=["#fff"]), "expected '#rrggbb'"),
        (_entry(flat_colors=["#000000", "#000000"]), "colors must not repeat"),
        (_entry(flat_colors=["#000000"], ink_colors=["#000000"]), "both a flat color and an ink color"),
        (_entry(flat_colors=["#000000"], exact_colors="yes"), "exact_colors must be true or false"),
        (_entry(exact_colors=True), "exact_colors says the colors are exact, but there are none"),
    ],
)
def test_rejects_entries_that_break_the_schema(entry, message):
    with pytest.raises(manifest.ManifestError, match=re.escape(message)):
        _parse({"a.png": entry})


def test_rejects_wrong_schema_version_and_duplicate_image_names(tmp_path):
    with pytest.raises(manifest.ManifestError, match="schema must be 1"):
        manifest.parse_manifest({"schema": 2, "images": {}})

    path = tmp_path / "manifest.json"
    entry = '{"size": [1, 1], "categories": ["photo"]}'
    path.write_text(f'{{"schema": 1, "images": {{"a.png": {entry}, "a.png": {entry}}}}}', encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="duplicate key"):
        manifest.load_manifest(path)


def test_lists_the_annotations_each_category_still_needs():
    info = _parse(
        {
            "a.png": _entry(
                categories=["face", "text", "cartoon", "gradient", "texture"],
                faces=[{"kind": "human", "box": [0, 0, 50, 50]}],
                flat_colors=["#ffffff"],
            )
        }
    )["a.png"]

    assert info.missing_annotations() == [
        "face: faces[0] has no features",
        "text: no text blocks",
        "cartoon: fewer than 2 flat colors",
        "cartoon: no ink colors",
        "gradient: no gradient areas",
        "texture: no texture areas",
    ]


def test_scaled_annotations_round_outward_and_stay_inside_the_image():
    info = _parse(
        {
            "a.png": _entry(
                size=[2000, 1000],
                text=[{"box": [101, 50, 999, 949], "string": "x"}],
                faces=[{"kind": "animal", "box": [0, 0, 2000, 1000], "features": [{"part": "nose", "box": [1998, 998, 2, 2]}]}],
                areas=[{"kind": "texture", "box": [3, 3, 1, 1]}],
            )
        }
    )["a.png"]

    small = info.scaled_to((1100, 550))

    assert small.size == (1100, 550)
    assert small.text[0].box == manifest.Box(55, 27, 550, 523)
    assert small.faces[0].box == manifest.Box(0, 0, 1100, 550)
    assert small.faces[0].features[0].box == manifest.Box(1098, 548, 2, 2)
    assert small.areas[0].box == manifest.Box(1, 1, 2, 2)  # 1.65..2.2 covers pixels 1 and 2
    assert info.scaled_to((20, 10)).areas[0].box == manifest.Box(0, 0, 1, 1)  # never shrinks to nothing
    assert np.zeros((550, 1100))[small.text[0].box.slices].shape == (523, 550)


def test_local_manifest_adds_and_replaces_entries(tmp_path):
    _write_manifest(tmp_path / manifest.MANIFEST_NAME, {"a.png": _entry(), "b.png": _entry()})
    _write_manifest(
        tmp_path / manifest.LOCAL_MANIFEST_NAME,
        {"b.png": _entry(categories=["text"], text=[{"box": [0, 0, 10, 10], "string": "B"}]), "local.png": _entry()},
    )

    entries = manifest.load_directory(tmp_path)

    assert sorted(entries) == ["a.png", "b.png", "local.png"]
    assert entries["b.png"].categories == ("text",)
    assert manifest.find_image(tmp_path / "local.png") is not None
    assert manifest.find_image(tmp_path / "unknown.png") is None
    assert manifest.load_directory(tmp_path / "no-such-dir") == {}


def test_run_benchmarks_only_the_requested_categories(tmp_path, monkeypatch, capsys):
    for name in ("face.png", "text.png", "unlisted.png"):
        Image.new("RGB", (200, 100)).save(tmp_path / name)
    _write_manifest(
        tmp_path / manifest.MANIFEST_NAME,
        {"face.png": _entry(categories=["photo", "face"]), "text.png": _entry(categories=["text"])},
    )
    calls = []

    def fake_run_case(args, src_dir, env, image, preset, long_edge, case_dir, categories):
        calls.append((image.name, preset, categories))
        return {"case": case_dir.name, "image": image.name, "categories": categories, "status": "timeout", "timeout_s": 1.0}

    monkeypatch.setattr(bench, "_run_case", fake_run_case)
    common = ["--images", str(tmp_path), "--presets", "Easy", "--results-dir", str(tmp_path / "results")]

    assert bench.main(["run", "WORKTREE=faces", "--category", "face", *common]) == 0
    assert calls == [("face.png", "Easy", ["photo", "face"])]
    assert "unlisted.png" in capsys.readouterr().err
    saved = json.loads((tmp_path / "results" / "faces" / "results.json").read_text(encoding="utf-8"))
    assert [case["categories"] for case in saved["cases"]] == [["photo", "face"]]

    assert bench.main(["run", "WORKTREE=gradients", "--category", "gradient", *common]) == 1
    assert "No images in categories gradient" in capsys.readouterr().err
    assert len(calls) == 1


def test_drawing_outlines_boxes_without_covering_them_and_adds_swatches(tmp_path):
    import draw_annotations

    Image.new("RGB", (200, 100), (128, 128, 128)).save(tmp_path / "a.png")
    info = _parse(
        {
            "a.png": _entry(
                categories=["cartoon", "face", "text"],
                faces=[{"kind": "cartoon", "box": [20, 20, 60, 60], "features": [{"part": "eye", "box": [30, 30, 10, 10]}]}],
                text=[{"box": [120, 20, 60, 20], "string": "HI\nTHERE", "rotation": 180}],
                flat_colors=["#ffffff", "#ff0000"],
                ink_colors=["#000000"],
            )
        }
    )["a.png"]

    out = np.asarray(draw_annotations.draw(tmp_path / "a.png", info))

    assert out.shape == (140, 200, 3)  # one row of 25 x 40 px swatches under the image
    assert tuple(out[50, 17]) == draw_annotations.FACE_COLOR  # just outside the face box
    assert tuple(out[50, 20]) == (128, 128, 128)  # the box's own pixels stay visible
    assert tuple(out[130, 37]) == (255, 0, 0)  # second swatch: flat red
    assert tuple(out[130, 62]) == (0, 0, 0)  # third swatch: the ink color comes after the flat colors


# -- the committed sample images ----------------------------------------------
def _sample_images() -> list[Path]:
    return sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in bench.IMAGE_SUFFIXES)


def test_every_sample_image_has_a_complete_manifest_entry():
    entries = manifest.load_directory(SAMPLE_DIR)
    problems = []
    for path in _sample_images():
        info = entries.get(path.name)
        if info is None:
            problems.append(f"{path.name}: no manifest entry")
            continue
        with Image.open(path) as image:
            if image.size != info.size:
                problems.append(f"{path.name}: manifest size {info.size}, file is {image.size}")
        problems += [f"{path.name}: {missing}" for missing in info.missing_annotations()]
    stale = sorted(set(entries) - {p.name for p in _sample_images()})
    problems += [f"{name}: manifest entry for a file that doesn't exist" for name in stale]

    assert problems == []


def test_sample_manifest_covers_every_category():
    categories = {c for info in manifest.load_directory(SAMPLE_DIR).values() for c in info.categories}

    assert categories == set(manifest.CATEGORIES)

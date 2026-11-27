"""The stage cache keeps a page's regions: a change to its lines or numbers draws it again from them, as it would be
drawn from scratch; and each stage is announced as it starts and ends (``pipeline.StageEvent``)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tessellatum.core import pipeline
from tessellatum.core.pipeline import Handling, StageEvent, generate
from tessellatum.core.render import PageStyle

from difficulty_levels import MIDDLE

SAMPLES = Path(__file__).parent / "sample_images"


def _hatched_sheet() -> np.ndarray:
    """Line art whose hatching a number may clear: what it may clear depends on the line's width."""
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    for x in range(0, 400, 4):
        image[6:, x : x + 2] = 0
    return image


def _same(a: pipeline.GeneratedPage, b: pipeline.GeneratedPage) -> None:
    assert np.array_equal(np.asarray(a.page), np.asarray(b.page))
    assert np.array_equal(np.asarray(a.legend), np.asarray(b.legend))
    assert a.palette_rgb == b.palette_rgb and a.num_regions == b.num_regions
    assert np.array_equal(a.model.region_id_map, b.model.region_id_map)
    assert np.array_equal(a.model.printed_ink, b.model.printed_ink)
    assert (a.model.clearable is None) == (b.model.clearable is None)
    if a.model.clearable is not None:
        assert np.array_equal(a.model.clearable, b.model.clearable)
    assert [label.box for label in a.model.labels] == [label.box for label in b.model.labels]


def _counting(monkeypatch) -> list:
    calls = []
    real = pipeline.build_regions

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_regions", counting)
    return calls


RENDER_CHANGES = [
    {"line_width_mm": 0.45},
    {"min_label_pt": 9.0},
    {"line_gray": 120, "label_gray": 90, "leader_gray": 160},
    {"leader_width_mm": 0.3, "leader_reach_mm": 4.0},
    {"line_smoothing_mm": 0.0},
    {"split_neck_mm": None},
    {"text_gap_mm": 3.0},
]


@pytest.mark.parametrize("change", RENDER_CHANGES, ids=lambda c: "+".join(c))
@pytest.mark.parametrize("picture", ["photo", "hatching"])
def test_a_change_to_the_lines_or_numbers_draws_the_page_from_the_regions_kept_as_from_scratch(
    picture, change, sample_image_bgr, monkeypatch
):
    image = sample_image_bgr if picture == "photo" else _hatched_sheet()
    changed = dataclasses.replace(PageStyle(), **change)
    pipeline.clear_cache()
    fresh = generate(image, MIDDLE, long_edge=400, style=changed)

    pipeline.clear_cache()
    generate(image, MIDDLE, long_edge=400)
    calls = _counting(monkeypatch)
    kept = generate(image, MIDDLE, long_edge=400, style=changed)

    assert calls == []  # the regions came from the cache
    _same(kept, fresh)


@pytest.fixture(scope="module")
def comic():
    return pipeline.load_image_bgr(SAMPLES / "m-comics-upside-downs-writing-pig.jpg")


def test_line_art_with_lettering_drawn_again_from_its_regions_is_the_page_drawn_from_scratch(comic, monkeypatch):
    # Line art whose lines of text print their letters, and whose regions with no room for a number merge: both after
    # the regions kept, so both are done again with the new lines and numbers.
    changed = dataclasses.replace(PageStyle(), line_width_mm=0.4, min_label_pt=8.0)
    pipeline.clear_cache()
    fresh = generate(comic, MIDDLE, long_edge=700, style=changed)
    assert fresh.model.clearable is not None and not fresh.model.drawn_from_colors

    pipeline.clear_cache()
    generate(comic, MIDDLE, long_edge=700)
    calls = _counting(monkeypatch)
    kept = generate(comic, MIDDLE, long_edge=700, style=changed)

    assert calls == []
    _same(kept, fresh)


@pytest.mark.parametrize("what", ["paper", "difficulty", "handling", "size"])
def test_a_change_to_the_paper_difficulty_handling_or_size_builds_the_regions_again(what, sample_image_bgr, monkeypatch):
    style, params, handling, edge = PageStyle(), MIDDLE, Handling(), 400
    pipeline.clear_cache()
    generate(sample_image_bgr, params, edge, style=style, handling=handling)
    calls = _counting(monkeypatch)
    if what == "paper":
        style = dataclasses.replace(style, paper="A3")
    elif what == "difficulty":
        params = dataclasses.replace(params, edge_settling=params.edge_settling / 2)
    elif what == "handling":
        handling = dataclasses.replace(handling, detail=False)
    else:
        edge = 380
    generate(sample_image_bgr, params, edge, style=style, handling=handling)
    assert len(calls) == 1


def test_the_regions_kept_are_read_only(sample_image_bgr):
    pipeline.clear_cache()
    page = generate(sample_image_bgr, MIDDLE, long_edge=300)
    for array in (page.model.region_id_map, page.model.printed_ink):
        with pytest.raises(ValueError):
            array[0, 0] = array[0, 0]


def test_the_cache_counts_each_thread_s_hits_and_misses(sample_image_bgr):
    pipeline.clear_cache()
    hits, misses = pipeline._cache.counts()
    generate(sample_image_bgr, MIDDLE, long_edge=300)
    after_cold = pipeline._cache.counts()
    generate(sample_image_bgr, MIDDLE, long_edge=300)
    after_warm = pipeline._cache.counts()
    assert after_cold[1] > misses
    assert after_warm[1] == after_cold[1]  # the second page found everything it looked up
    assert after_warm[0] > after_cold[0]


def _events(image, **kwargs) -> list[StageEvent]:
    events: list[StageEvent] = []
    generate(image, MIDDLE, stage_callback=events.append, **kwargs)
    return events


def test_each_stage_is_announced_as_it_starts_and_ends_in_order(sample_image_bgr):
    pipeline.clear_cache()
    events = _events(sample_image_bgr, long_edge=300)

    starts = [e.stage for e in events if not e.done]
    assert starts == ["resize", "ink", "detect", "quantize", "text", "regions", "contours", "render"]
    # Each one ends before the next starts, with its time.
    assert [e.stage for e in events] == [stage for stage in starts for _ in (0, 1)]
    assert [e.done for e in events] == [False, True] * len(starts)
    assert all(e.seconds >= 0 for e in events if e.done)
    assert all(e.label == pipeline.STAGES[e.stage] for e in events)


def test_a_stage_the_cache_had_says_so(sample_image_bgr):
    pipeline.clear_cache()
    cold = {e.stage: e for e in _events(sample_image_bgr, long_edge=300) if e.done}
    warm = {e.stage: e for e in _events(sample_image_bgr, long_edge=300, style=PageStyle(line_width_mm=0.4)) if e.done}

    assert not any(e.cached for e in cold.values())
    assert all(warm[stage].cached for stage in ("resize", "ink", "detect", "quantize", "text", "regions"))
    assert not warm["contours"].cached and not warm["render"].cached


def test_a_picture_without_faces_handling_or_text_skips_those_stages(sample_image_bgr):
    pipeline.clear_cache()
    events = _events(sample_image_bgr, long_edge=300, handling=Handling(detail=False, text=False))
    assert [e.stage for e in events if not e.done] == ["resize", "ink", "quantize", "regions", "contours", "render"]


def test_a_page_drawn_larger_announces_its_own_stages_after_the_page_it_is_drawn_from(sample_image_bgr):
    pipeline.clear_cache()
    page = generate(sample_image_bgr, MIDDLE, long_edge=300)
    events: list[StageEvent] = []
    pipeline.draw_page(page, sample_image_bgr, 600, stage_callback=events.append)
    assert [e.stage for e in events if not e.done] == ["resize", "text", "scale", "render"]
    assert events[-1].done and events[-1].stage == "render"
    # Without text, no stage for it.
    page = generate(sample_image_bgr, MIDDLE, long_edge=300, handling=Handling(text=False))
    events.clear()
    pipeline.draw_page(page, sample_image_bgr, 600, stage_callback=events.append)
    assert [e.stage for e in events if not e.done] == ["resize", "scale", "render"]


def test_the_stages_change_nothing_on_the_page(sample_image_bgr):
    pipeline.clear_cache()
    plain = generate(sample_image_bgr, MIDDLE, long_edge=300)
    pipeline.clear_cache()
    told = generate(sample_image_bgr, MIDDLE, long_edge=300, stage_callback=lambda _event: None)
    _same(told, plain)


def test_the_warm_up_leaves_the_stages_kept_for_the_picture_being_worked_on(sample_image_bgr, monkeypatch):
    # The app warms the pipeline up on a background thread as it starts, on a tiny picture of its own: if that runs
    # after the first preview, it must not throw away the preview's stages, which the next redraw is made from.
    pipeline.clear_cache()
    generate(sample_image_bgr, MIDDLE, long_edge=400)
    pipeline.warm_up()
    calls = _counting(monkeypatch)
    generate(sample_image_bgr, MIDDLE, long_edge=400, style=PageStyle(line_width_mm=0.4))
    assert calls == []

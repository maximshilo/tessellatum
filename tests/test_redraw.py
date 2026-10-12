"""An export is its preview's page drawn at print size (D-006, D-008): the same regions, closed by the same lines."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tessellatum.core import labels, pipeline, redraw, regions
from tessellatum.core.boundaries import region_outlines
from tessellatum.core.difficulty import PRESETS, finest_params
from tessellatum.core.ink_outline import ink_outline, traced_rings
from tessellatum.core.render import render_page

from difficulty_levels import FINE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_metrics as bm  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_images"


def _shapes(size: tuple[int, int] = (1700, 1200)) -> np.ndarray:
    """A picture larger than a preview: discs, a diamond and a ring on a gradient, with a little noise, BGR."""
    width, height = size
    rng = np.random.default_rng(7)
    ramp = np.linspace(60, 200, width, dtype=np.float64)
    picture = np.dstack([np.tile(ramp, (height, 1)), np.full((height, width), 120.0), np.tile(ramp[::-1], (height, 1))])
    cv2.circle(picture, (400, 400), 250, (30, 40, 200), -1)
    cv2.circle(picture, (1200, 800), 300, (200, 180, 40), -1)
    cv2.circle(picture, (1200, 800), 150, (40, 160, 40), -1)
    diamond = np.array([[850, 150], [1050, 350], [850, 550], [650, 350]], dtype=np.int32)
    cv2.fillPoly(picture, [diamond], (220, 220, 220))
    cv2.ellipse(picture, (450, 950), (300, 120), 20, 0, 360, (90, 30, 90), 40)
    picture += rng.normal(0, 4, picture.shape)
    return np.clip(picture, 0, 255).astype(np.uint8)


def _scaled(preview_ids: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return redraw.scaled_up(preview_ids, size)


@pytest.fixture(scope="module")
def pages():
    pipeline.clear_cache()
    picture = _shapes()
    preview = pipeline.generate(picture, FINE, pipeline.PREVIEW_LONG_EDGE, collect_analysis=True)
    export = pipeline.generate(picture, FINE, pipeline.export_long_edge(picture), collect_analysis=True)
    yield picture, preview, export
    pipeline.clear_cache()


def test_an_export_has_its_preview_s_regions_colors_and_legend(pages):
    _picture, preview, export = pages
    assert export.page.size == (1700, 1200)
    assert export.num_regions == preview.num_regions and export.palette_rgb == preview.palette_rgb
    p, e = preview.analysis, export.analysis
    assert set(np.unique(e.region_id_map).tolist()) == set(np.unique(p.region_id_map).tolist())
    np.testing.assert_array_equal(e.region_color, p.region_color)
    # Each region where the preview's is, scaled, but along its edges, smoothed as the lines are.
    agree = (e.region_id_map == _scaled(p.region_id_map, export.page.size)).mean()
    assert agree > 0.97
    assert sorted(label.text for label in e.labels) == sorted(label.text for label in p.labels)


def test_an_export_s_regions_are_closed_by_its_lines_and_touch_only_where_the_preview_s_do(pages):
    _picture, preview, export = pages
    e = export.analysis
    assert bm.enclosure(e.region_id_map, e.outlines)["unenclosed_area_fraction"] == 0
    p = preview.analysis
    assert redraw.contacts(e.region_id_map) <= redraw.contacts(p.region_id_map)
    assert redraw.contacts(e.region_id_map, corners=True) <= redraw.contacts(p.region_id_map, corners=True)
    assert bm.same_color_boundary_share(e.region_id_map, e.region_color) in (0, None)


def test_an_export_draws_its_preview_s_lines_and_places_its_numbers_again(pages):
    picture, preview, export = pages
    sx, sy = export.page.width / preview.page.width, export.page.height / preview.page.height
    drawn = [line for line in export.analysis.strokes if len(line) > 2 or np.ptp(line, axis=0).any()]
    for line, own in zip(drawn, preview.analysis.strokes):
        np.testing.assert_allclose(line, redraw.scaled_points(own, sx, sy))
    # The numbers are placed at the export's size: none smaller than the style lets them print there, and each clear of
    # the lines and the ink the export draws.
    floor = labels.min_font_size(export.page.size, preview.model.style.min_label_pt)
    assert all(label.font_size >= floor for label in export.analysis.labels)
    for label in export.analysis.labels:
        x0, y0, x1, y1 = (int(v) for v in label.box)
        assert not label.cramped and (export.analysis.outlines[y0:y1, x0:x1] == 255).all()


def test_drawing_a_preview_is_the_export_generate_makes(pages):
    picture, preview, export = pages
    drawn = pipeline.draw_page(preview, picture, pipeline.export_long_edge(picture))
    assert drawn.page.tobytes() == export.page.tobytes()
    assert drawn.model is preview.model
    with pytest.raises(ValueError, match="no smaller"):
        pipeline.draw_page(preview, picture, 800)


def test_a_page_is_made_at_its_working_size_and_an_export_drawn_from_it():
    # D-012 (T8): Easy's 0.8 mm brush is 4 px across at 1385 px on a 3:2 picture's page.
    pipeline.clear_cache()
    picture = _shapes((1800, 1200))
    params = PRESETS["Easy"]
    assert pipeline.working_long_edge(picture, params) == 1385
    preview = pipeline.generate(picture, params, 1385)
    assert preview.page.size == (1385, 923) and preview.model.region_id_map.shape == (923, 1385)
    export = pipeline.generate(picture, params, pipeline.export_long_edge(picture))
    assert export.page.size == (1800, 1200)
    np.testing.assert_array_equal(export.model.region_id_map, preview.model.region_id_map)  # made there, not again
    assert export.page.tobytes() == pipeline.draw_page(preview, picture, 1800).page.tobytes()
    # With fewer pixels across the brush, the page is made at the preview's 1100 px and drawn from there.
    coarse = pipeline.generate(picture, params, 1800, handling=pipeline.Handling(brush_px=2.0))
    assert coarse.page.size == (1800, 1200) and coarse.model.region_id_map.shape == (733, 1100)
    pipeline.clear_cache()


def test_a_picture_no_larger_than_a_preview_is_made_at_its_own_size():
    pipeline.clear_cache()
    picture = _shapes((1000, 700))
    exported = pipeline.generate(picture, FINE, 2400)
    made = pipeline._make_page(picture, FINE, 1000)
    assert exported.page.tobytes() == made.page.tobytes()
    pipeline.clear_cache()


def test_an_export_whose_numbers_move_comes_out_as_if_its_second_render_drew_everything_again(monkeypatch):
    pipeline.clear_cache()
    picture = pipeline.load_image_bgr(SAMPLES / "l-photo-palermo-castle.jpg")
    preview = pipeline.generate(picture, PRESETS["Hard"], pipeline.PREVIEW_LONG_EDGE)
    long_edge = pipeline.export_long_edge(picture)
    real = pipeline.render_page
    reused = []

    def render(*args, reuse=None, **kwargs):
        reused.append(reuse is not None)
        return real(*args, reuse=reuse, **kwargs)

    monkeypatch.setattr(pipeline, "render_page", render)
    drawn = pipeline.draw_page(preview, picture, long_edge)
    assert reused == [False, True]  # a number found no room drawn larger: the page was rendered again, reusing the first
    monkeypatch.setattr(pipeline, "render_page", lambda *args, reuse=None, **kwargs: real(*args, **kwargs))
    fresh = pipeline.draw_page(preview, picture, long_edge)
    assert drawn.page.tobytes() == fresh.page.tobytes()
    assert drawn.drawing.labels == fresh.drawing.labels
    pipeline.clear_cache()


@pytest.mark.parametrize(
    "name, preset",
    [("m-cartoon-bold-lines-girl.png", "Medium"), ("l-photo-palermo-castle.jpg", "Hard"), ("scene.png", "Max")],
)
def test_a_benchmark_image_s_export_is_its_preview_drawn_larger(name, preset):
    pipeline.clear_cache()
    picture = pipeline.load_image_bgr(SAMPLES / name)
    params = finest_params() if preset == "Max" else PRESETS[preset]
    preview = pipeline.generate(picture, params, pipeline.working_long_edge(picture, params), collect_analysis=True).analysis
    export = pipeline.generate(picture, params, pipeline.export_long_edge(picture), collect_analysis=True).analysis
    assert set(np.unique(export.region_id_map).tolist()) == set(np.unique(preview.region_id_map).tolist())
    np.testing.assert_array_equal(export.palette_bgr[: export.legend_size], preview.palette_bgr[: preview.legend_size])
    assert bm.enclosure(export.region_id_map, export.outlines)["unenclosed_area_fraction"] == 0
    pipeline.clear_cache()


def test_scaled_regions_fill_each_region_inside_its_smoothed_outline():
    ids = np.zeros((40, 60), dtype=np.int32)
    ids[10:30, 10:30] = 1
    ids[12:28, 35:55] = 2
    ids[0:5, 0:5] = -1  # in no region
    drawn = redraw.scaled_regions(ids, (150, 100), smoothing_px=2.0)
    assert drawn.shape == (100, 150)
    assert set(np.unique(drawn).tolist()) == {-1, 0, 1, 2}
    assert (redraw.scaled_up(ids, (150, 100)) == drawn).mean() > 0.97
    # A square's corners are rounded as its line is: its staircase gone, and nothing it didn't touch.
    assert redraw.contacts(drawn) <= redraw.contacts(ids)
    assert redraw.contacts(drawn, corners=True) <= redraw.contacts(ids, corners=True)


def test_two_regions_meeting_at_a_corner_never_come_to_touch():
    ids = np.full((20, 20), -1, dtype=np.int32)
    ids[:10, :10] = 0
    ids[10:, 10:] = 1  # 0 and 1 meet at a corner only, -1 round them
    ids[:10, 10:] = 2
    ids[10:, :10] = 3
    drawn = redraw.scaled_regions(ids, (55, 55), smoothing_px=3.0)
    assert (0, 1) not in redraw.contacts(drawn) and (2, 3) not in redraw.contacts(drawn)
    assert redraw.contacts(drawn) <= redraw.contacts(ids)
    assert redraw.contacts(drawn, corners=True) <= redraw.contacts(ids, corners=True)


def test_a_region_alone_in_its_ink_meets_no_other_even_at_a_corner():
    ids = np.full((30, 30), 1, dtype=np.int32)
    ids[8:22, 8:22] = -1  # ink, a pixel wide at the corners where smoothing cuts in
    ids[9:21, 9:21] = 0
    drawn = redraw.scaled_regions(ids, (77, 77), smoothing_px=4.0)
    assert (0, 1) not in redraw.contacts(drawn, corners=True)


def test_a_mask_s_outline_filled_is_the_mask_and_scaled_is_the_mask_scaled_up():
    rng = np.random.default_rng(3)
    mask = np.zeros((60, 80), dtype=np.uint8)
    for _ in range(25):
        x, y = (int(v) for v in rng.integers(0, 80, size=2))
        cv2.line(mask, (x, y % 60), (x + int(rng.integers(-30, 30)), (y + int(rng.integers(-30, 30))) % 60), 1,
                 int(rng.integers(1, 3)))
    mask = mask.astype(bool)
    np.testing.assert_array_equal(redraw.filled(traced_rings(mask), (80, 60), 1.0, 1.0), mask)
    np.testing.assert_array_equal(redraw.filled(ink_outline(mask), (80, 60), 1.0, 1.0), mask)  # its margins hold
    np.testing.assert_array_equal(redraw.filled(traced_rings(mask), (240, 180), 3.0, 3.0), redraw.scaled_up(mask, (240, 180)))


def test_regions_filled_side_by_side_fill_each_pixel_once():
    ids = np.kron(np.arange(12, dtype=np.int32).reshape(3, 4), np.ones((7, 9), dtype=np.int32))
    ids[5:12, 10:20] = 3
    outlines = region_outlines(ids, 2.5)
    counts = np.zeros((63, 108), dtype=np.int32)
    for rings in outlines.values():
        one = np.zeros((63, 108), dtype=np.int32)
        redraw.fill([rings], [1], one, 3.0, 3.0)
        counts += one
    assert (counts == 1).all()


def test_gaps_are_the_white_pixels_beside_another_region_s_white():
    ids = np.zeros((4, 6), dtype=np.int32)
    ids[:, 3:] = 1
    white = np.ones((4, 6), dtype=bool)
    white[:, 3] = False  # a line down the boundary, but for one pixel
    white[2, 3] = True
    found = redraw.gaps(ids, white)
    assert found.sum() == 1 and found[2, 2]


def test_scaled_ink_keeps_a_region_s_white_whole_where_smoothing_would_pinch_it():
    ids = np.zeros((40, 40), dtype=np.int32)
    printed = np.zeros((40, 40), dtype=bool)
    printed[18:22, :19] = True  # two bars of ink, with a neck of paper a pixel wide between them
    printed[18:22, 20:] = True
    drawn = redraw.scaled_regions(ids, (120, 120), smoothing_px=2.0)
    ink = redraw.scaled_ink(printed, drawn, radius=4.0)
    assert regions.white_areas(drawn, ink, 4.0)[0] == regions.white_areas(ids, printed, 4.0 / 3)[0] == 1


def test_render_page_draws_the_lines_and_numbers_it_is_given(pages):
    _picture, preview, _export = pages
    a = preview.analysis
    again = render_page(
        preview.page.size, a.regions, a.region_id_map, preview.model.style, ink=preview.model.printed_ink,
        ink_gray=preview.model.ink_gray, strokes=list(a.strokes), labels=list(a.labels),
    )
    assert again.image.tobytes() == preview.page.tobytes()

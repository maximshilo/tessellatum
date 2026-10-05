"""The page painted in: the completed version, every region in its paint under the picture's ink, and the tinted one,
the page under a faint wash of its paint."""

import numpy as np
import pytest
from PIL import Image

from tessellatum.core.painting import TINT_OPACITY, Painting, Version, completed, render_version, tint_rgb, tinted
from tessellatum.core.render import LABEL_GRAY, LINE_GRAY, PAPER

PALETTE = [(200, 40, 40), (40, 160, 60), (20, 20, 120)]


def _regions() -> np.ndarray:
    """Four regions in a 12 x 8 map, and a patch in none."""
    ids = np.zeros((8, 12), dtype=np.int32)
    ids[:, 6:] = 1
    ids[4:, :6] = 2
    ids[4:, 6:] = 3
    ids[1:3, 1:3] = -1
    return ids


def _painting(ink: np.ndarray | None = None, ink_gray: int = 0) -> Painting:
    ids = _regions()
    return Painting(
        region_id_map=ids,
        region_color=np.array([0, 1, 2, 0]),  # regions 0 and 3 take the same paint
        palette_rgb=PALETTE,
        ink=np.zeros(ids.shape, dtype=np.uint8) if ink is None else ink,
        ink_gray=ink_gray,
    )


def test_the_completed_version_fills_every_region_with_its_paint_and_leaves_paper_bare_in_none():
    painting = _painting()

    image = np.asarray(completed(painting))

    ids = painting.region_id_map
    assert image.shape == (8, 12, 3)
    for region, paint in enumerate(painting.region_color):
        assert (image[ids == region] == PALETTE[paint]).all()
    assert (image[ids < 0] == PAPER).all()


def test_the_completed_version_prints_the_picture_s_ink_over_the_paint_as_thickly_as_it_lies():
    ink = np.zeros((8, 12), dtype=np.uint8)
    ink[6, 1] = 255  # solid, in region 2
    ink[6, 2] = 128  # half, as at the edge of a letter
    ink[1, 1] = 255  # on the bare paper in no region
    painting = _painting(ink, ink_gray=30)

    image = np.asarray(completed(painting)).astype(int)

    assert (image[6, 1] == 30).all() and (image[1, 1] == 30).all()
    paint = np.array(PALETTE[2])
    np.testing.assert_array_equal(image[6, 2], np.rint(paint + (30 - paint) * 128 / 255))
    assert (image[6, 3] == paint).all()  # where no ink lies, the paint


def test_the_wash_is_the_paint_taken_part_of_the_way_from_white():
    assert tint_rgb((PAPER, PAPER, PAPER)) == (PAPER, PAPER, PAPER)
    assert tint_rgb((0, 0, 0)) == (round(PAPER * (1 - TINT_OPACITY)),) * 3
    assert tint_rgb((200, 40, 40)) == tuple(round(PAPER + TINT_OPACITY * (v - PAPER)) for v in (200, 40, 40))


def test_the_tinted_version_is_the_page_multiplied_by_a_wash_of_each_region_s_paint():
    painting = _painting()
    page = np.full((8, 12, 3), PAPER, dtype=np.uint8)
    page[:, 5] = LINE_GRAY  # a line down region 0 and region 2
    page[1, 1] = LABEL_GRAY  # a number on the paper in no region

    image = np.asarray(tinted(Image.fromarray(page), painting)).astype(int)

    ids = painting.region_id_map
    for region, paint in enumerate(painting.region_color):
        wash = np.array(tint_rgb(PALETTE[paint]))
        white = (ids == region) & (page[:, :, 0] == PAPER)
        assert (image[white] == wash).all()
        lined = (ids == region) & (page[:, :, 0] == LINE_GRAY)
        assert (image[lined] == np.rint(LINE_GRAY * wash / PAPER)).all()
    np.testing.assert_array_equal(image[ids < 0], page[ids < 0])  # no paint, no wash


def test_a_wash_keeps_the_numbers_as_much_darker_than_what_is_round_them_as_on_paper():
    # A wash multiplies the page: a number's gray keeps its ratio to the paint round it, on the darkest paint too, where
    # it is about as dark as the page's lines are on paper.
    darkest = tint_rgb((0, 0, 0))[0]
    number = round(LABEL_GRAY * darkest / PAPER)
    assert number / darkest == pytest.approx(LABEL_GRAY / PAPER, abs=0.005)
    assert abs(number - LINE_GRAY) <= 0x08


def test_the_page_version_is_the_page_itself_and_a_painted_one_needs_what_it_is_painted_with():
    page = Image.new("RGB", (12, 8), "white")

    assert render_version(Version.PAGE, page, None) is page
    painted = render_version(Version.COMPLETED, page, _painting())
    np.testing.assert_array_equal(np.asarray(painted), np.asarray(completed(_painting())))
    with pytest.raises(ValueError, match="completed version"):
        render_version(Version.COMPLETED, page, None)
    assert all(version.description for version in Version)

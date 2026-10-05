"""Top-level orchestration: image -> quantize -> regions -> render -> legend."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, TypeVar

import cv2
import numpy as np
from PIL import Image

from tessellatum.core import faces, ink, kernels, marks, subject, text, tones
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.legend import render_legend
from tessellatum.core.painting import Painting, Version, render_version
from tessellatum.core.print_size import MIN_PAINTABLE_WIDTH_MM, MIN_REGION_AREA_MM2, long_edge_at_dpi, print_scale
from tessellatum.core.quantize import quantize
from tessellatum.core.regions import (
    Region,
    build_regions,
    detail_ink,
    extract_regions,
    join_ink,
    leave_pockets,
    look_through_hatching,
    merge_cramped,
    paint_over_thin_ink,
    settle_enclosed,
    split_areas,
)
from tessellatum.core.render import PAPER, Label, PageDrawing, PageStyle, render_page
from tessellatum.core.texture import smooth_regions

PREVIEW_LONG_EDGE = 1100

# A pixel the letters ink at least this much (of 255: halfway, where their outline runs) is printed ink: it ends the
# lines crossing it, and no label point falls on it.
_LETTERING_PRINTED = 128

# On line art, how many times at most the regions whose numbers found no room are merged and the page drawn again. Once
# was always enough until the letters printed alone (T7.5): then a region merged beside a line of text could still
# find none, at Max on the postcard.
_MERGE_ROUNDS = 3

# Rough share of total generation time each stage takes, used to report real
# (if coarse-grained) percentage progress rather than a fake animation.
_STAGE_PROGRESS = {
    "resize": 2,
    "ink": 10,
    "quantize": 60,
    "regions": 80,
    "contours": 90,
    "render": 100,
}

T = TypeVar("T")


class PipelineCancelled(Exception):
    """Raised internally to unwind ``generate`` when the caller cancels it."""


@dataclass
class PageAnalysis:
    """What a generated page is made of, for measuring it (see ``generate``).

    Colors are indices into ``palette_bgr``. Its first ``legend_size`` colors
    are the legend's, in legend order, so color ``i`` is numbered ``i + 1`` on
    the page. The rest are quantized colors no drawn region has.

    A region in ``region_id_map`` with no entry in ``regions`` has an outline
    that encloses no area (e.g. it is one pixel wide), so it gets no number.
    Its boundaries are still drawn: the lines come from the region map, not
    from the regions.

    On line art, the pixels in no region (-1) are the bold printed ink, the
    seam down the middle of a thin line between two regions (or between two
    white areas of one color, kept apart), the bits of bare paper the ink
    encloses that are too small to paint, and the pockets no brush reaches
    against ink it may not go over, left as paper (see
    ``regions.leave_pockets``). The thin ink otherwise belongs to the regions
    whose paint goes over it (see ``regions.look_through_hatching`` and
    ``regions.paint_over_thin_ink``); it is printed all the same.
    """

    region_id_map: np.ndarray  # HxW int32: each pixel's region id, -1 for none
    region_color: np.ndarray  # color of each region id (merged-away ids keep an entry)
    palette_bgr: np.ndarray  # Kx3 uint8, legend colors first
    legend_size: int
    min_region_area_px: int  # merge threshold: smaller regions merge into a neighbor, if they have one
    # HxW bool: where a region may be half as large, each of its pixels there counting twice towards
    # min_region_area_px (see ``regions.build_regions``): the faces found and the subject, on a picture drawn from its
    # colors. The regions lying mostly in the faces are the ones whose tones are settled (see ``tones``). All False on
    # line art, and on a picture with neither.
    detail: np.ndarray
    # HxW bool: the picture's subject, found on it at preview size (see ``subject``), on a picture drawn from its colors;
    # all False on line art, where it isn't looked for.
    subject: np.ndarray
    min_paintable_width_px: float  # brush width: narrower parts of a region are given to a neighbor
    regions: list[Region]  # regions drawn on the page, in region-id order
    labels: list[Label]  # numbers drawn on the page
    # HxW uint8: the ink the lines and the printed ink put on the page, 0 = solid ink, 255 = bare paper.
    outlines: np.ndarray
    # Every line drawn, in drawing order: Nx2 float64 (x, y) points with pixel centers at integer coordinates.
    # One line per boundary between two regions, traced along the pixel cracks and smoothed off them
    # by at most boundaries.MAX_SHIFT_PX.
    # A closed line repeats its first point at the end.
    strokes: list[np.ndarray]
    # HxW uint8: the ink the leader lines of numbers written outside their regions put on the page, as in outlines.
    leaders: np.ndarray
    # Whether the picture is line art, decided on it at preview size (see ``ink``), and HxW bool: the pixels on its
    # ink lines, all False unless it is.
    line_art: ink.LineArt
    ink_lines: np.ndarray
    # HxW bool: the ink printed on the page, solid, in the gray ``ink_gray`` (0 black, 255 white): line art's ink lines,
    # and the patches in the ink's own color taken for it where it runs wider than a line (see ``regions.join_ink`` and
    # ``regions.settle_enclosed``), less the hatching cleared behind numbers written on it. Bold printed ink is in no
    # region, nor is bare paper the ink encloses too small to paint; thin printed ink is in the regions whose paint goes
    # over it. On a picture drawn from its colors, the detail marks printed in its faces, in their own gray (see
    # ``marks``), which lie in the regions around them; all False without faces. On every picture, inside the lines of
    # text found, the printed ink lying in a region is the letters, the pixels they ink at least halfway (see
    # ``lettering``), and no other; the printed ink in no region there is kept. On a picture drawn from its colors,
    # ``ink_gray`` is the gray of the picture under all of it, the faces' marks and the lines of text alike -- under a
    # line's darker side, which is its ground where its letters are light.
    printed_ink: np.ndarray
    ink_gray: int
    # The faces in the picture (see ``faces``), found on it at preview size and given in the page's pixels. On a picture
    # drawn from its colors, they are ``detail``; line art's page doesn't use them.
    faces: list[faces.Face]
    # The lines of text in the picture (see ``text``), found on it once and given in the page's pixels, on every picture.
    text: list[text.TextLine]
    # HxW bool: the pixels inside the lines of text found, where the page prints their letters in ``ink_gray``, their
    # ground bare paper, the lines running through, in place of the printed ink lying in a region (see
    # ``text.lettering`` and ``render.render_page``; printed ink in no region prints solid there too); and HxW uint8,
    # the ink the letters put on the page as ``outlines`` gives it, 0 = solid, 255 = bare paper (255 away from the
    # letters). The pixels they ink at least halfway are printed ink too: in ``printed_ink``, lying in the regions
    # around them, which are painted round them.
    lettering_area: np.ndarray
    lettering: np.ndarray


@dataclass(frozen=True)
class Handling:
    """What the pipeline does with what it finds in a picture. Each is on by default.

    Turned off, the page is drawn as if nothing of the kind had been found.
    The subject and text are then not looked for, nor faces but by the
    analysis (as on line art); whether the picture is line art is still
    decided, but not used.

    ``line_art``: a picture found to be line art is drawn from its own ink, printed, and its regions are the areas the
    ink encloses (see ``ink``); off, it is drawn from its colors like any other picture. ``detail``: on a picture drawn
    from its colors, the subject and the faces found get more detail, a face's tones are settled and its thin dark marks
    printed (see ``subject``, ``faces``, ``tones`` and ``marks``). ``text``: the letters in the lines of text found are
    printed, and no number goes on them (see ``text``).
    """

    line_art: bool = True
    detail: bool = True
    text: bool = True


@dataclass
class GeneratedPage:
    page: Image.Image
    legend: Image.Image
    palette_rgb: list[tuple[int, int, int]]
    num_colors_used: int
    num_regions: int
    analysis: PageAnalysis | None = None  # only with generate(..., collect_analysis=True)
    drawing: PageDrawing | None = None  # what ``page`` was drawn from, to draw it again off the pixel grid (see ``export``)
    painting: Painting | None = None  # what ``page`` is painted with, for its painted versions (see ``painting``)
    _images: dict[Version, Image.Image] = field(default_factory=dict, init=False, repr=False, compare=False)

    def image(self, version: Version = Version.PAGE) -> Image.Image:
        """``version`` of the page (see ``painting``): the page itself, or it painted in. Each is drawn once, when first
        asked for, and kept, so ``page`` and ``painting`` are not to be changed after that."""
        if version not in self._images:
            self._images[version] = render_version(version, self.page, self.painting)
        return self._images[version]


class _StageCache:
    """Recent stage results for the most recently used source image.

    Tuning difficulty re-runs the pipeline on one image over and over, and
    the expensive early stages depend on only some parameters (smoothing and
    k-means don't care about the minimum region size, resizing only about the
    output size). Entries are tied to the image *object* through a weak
    reference, so a different or garbage-collected image never gets stale
    results. Image arrays must not be modified in place after being passed in.
    """

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._image_ref: weakref.ref | None = None
        self._entries: OrderedDict[Hashable, object] = OrderedDict()
        self._max_entries = max_entries

    def get_or_compute(self, image: np.ndarray, key: Hashable, compute: Callable[[], T]) -> T:
        with self._lock:
            if self._image_ref is None or self._image_ref() is not image:
                self._image_ref = weakref.ref(image)
                self._entries.clear()
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]  # type: ignore[return-value]

        value = compute()  # outside the lock: this is the slow part

        with self._lock:
            if self._image_ref() is image:
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._image_ref = None
            self._entries.clear()


_cache = _StageCache(max_entries=8)


def clear_cache() -> None:
    """Forget cached intermediate results (see ``_StageCache``)."""
    _cache.clear()


def _paintable_limits(params: DifficultyParams, size: tuple[int, int]) -> tuple[int, float]:
    """The region stage's limits for a page of ``size`` (width, height) in pixels.

    The difficulty sets the smallest region and the narrowest part of one on
    the printed page, and the printed page sets how small either may get at
    any difficulty: the brush, and its footprint (see ``print_size``). The
    printed page's size follows the image's shape rather than its pixel
    count, so a preview and an export of one image are held to the same
    physical sizes.
    """
    scale = print_scale(size)
    min_area_px = max(4, int(round(scale.mm2_to_px(max(params.min_region_area_mm2, MIN_REGION_AREA_MM2)))))
    return min_area_px, scale.mm_to_px(max(params.min_width_mm, MIN_PAINTABLE_WIDTH_MM))


def export_long_edge(image_bgr: np.ndarray) -> int:
    """The long edge an export of ``image_bgr`` renders at: what prints at 300 dpi on A4, by the picture's shape (see
    ``print_size``), or the picture's own long edge where that is smaller, since the pipeline never upscales."""
    h, w = image_bgr.shape[:2]
    return min(max(h, w), long_edge_at_dpi((w, h)))


def load_image_bgr(path: Path) -> np.ndarray:
    """Load any image format via Pillow, return an OpenCV-style BGR array."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def resize_to_long_edge(image_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    current_long_edge = max(h, w)
    if current_long_edge <= long_edge:
        return image_bgr
    scale = long_edge / current_long_edge
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def detect_ink(image_bgr: np.ndarray, resized: np.ndarray, long_edge: int) -> tuple[ink.LineArt, np.ndarray]:
    """Whether the picture is line art, and the ink lines of ``resized``, its page at ``long_edge``.

    The decision is made once per picture, on it at preview size, so a
    preview and an export always agree; at preview size the picture is
    measured once for both. Both are cached per image object, as the other
    stages are.
    """

    def find() -> tuple[ink.LineArt, np.ndarray]:
        picture = _cache.get_or_compute(
            image_bgr, ("resize", PREVIEW_LONG_EDGE), lambda: resize_to_long_edge(image_bgr, PREVIEW_LONG_EDGE)
        )
        if picture is resized:
            decision, lines = ink.find_ink(resized)
            return _cache.get_or_compute(image_bgr, ("line art",), lambda: decision), lines
        decision = _cache.get_or_compute(image_bgr, ("line art",), lambda: ink.line_art(picture))
        return decision, ink.ink_lines(resized) if decision.is_line_art else np.zeros(resized.shape[:2], dtype=bool)

    return _cache.get_or_compute(image_bgr, ("ink", long_edge), find)


def detect_faces(image_bgr: np.ndarray, resized: np.ndarray) -> list[faces.Face]:
    """The faces in the picture, in the pixels of ``resized``, its page.

    They are found once per picture, on it at preview size, so a preview and
    an export always agree; that is cached per image object, as the other
    stages are.
    """
    picture = _cache.get_or_compute(
        image_bgr, ("resize", PREVIEW_LONG_EDGE), lambda: resize_to_long_edge(image_bgr, PREVIEW_LONG_EDGE)
    )
    found = _cache.get_or_compute(image_bgr, ("faces",), lambda: faces.find_faces(picture))
    return faces.scaled(found, picture.shape[1::-1], resized.shape[1::-1])


def detect_subject(image_bgr: np.ndarray, resized: np.ndarray) -> np.ndarray:
    """The picture's subject, as the pixels of ``resized``, its page (see ``subject``).

    It is found once per picture, on it at preview size, so a preview and an
    export always agree; that is cached per image object, as the other stages
    are.
    """
    picture = _cache.get_or_compute(
        image_bgr, ("resize", PREVIEW_LONG_EDGE), lambda: resize_to_long_edge(image_bgr, PREVIEW_LONG_EDGE)
    )
    probability = _cache.get_or_compute(image_bgr, ("subject",), lambda: subject.find_subject(picture))
    return subject.mask(probability, resized.shape[1::-1])


def detect_text(image_bgr: np.ndarray, resized: np.ndarray) -> list[text.TextLine]:
    """The lines of text in the picture, in the pixels of ``resized``, its page (see ``text``).

    They are found once per picture -- on it at half its preview size, and
    where that finds text, again at twice its preview size from the source's
    own pixels -- so a preview and an export always agree; that is cached per
    image object, as the other stages are.
    """
    picture = _cache.get_or_compute(
        image_bgr, ("resize", PREVIEW_LONG_EDGE), lambda: resize_to_long_edge(image_bgr, PREVIEW_LONG_EDGE)
    )
    found = _cache.get_or_compute(image_bgr, ("text",), lambda: text.find_text(picture, image_bgr))
    return text.scaled(found, picture.shape[1::-1], resized.shape[1::-1])


def warm_up() -> None:
    """Pay one-time start-up costs before the first real generation.

    Loads the compiled region kernels (compiling them if this is the first run
    since install, which takes a few seconds) and runs the whole pipeline once
    on a tiny synthetic image. Meant for a background thread at app start.
    """
    kernels.warm_up()
    tiny = np.zeros((48, 64, 3), dtype=np.uint8)
    tiny[:, 32:] = (40, 160, 220)
    tiny[12:36, 8:24] = (200, 60, 60)
    generate(tiny, DifficultyParams(num_colors=4, min_region_area_mm2=500.0, blur_sigma=1.0), long_edge=64)


def generate(
    image_bgr: np.ndarray,
    params: DifficultyParams,
    long_edge: int,
    progress_callback: Callable[[int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    collect_analysis: bool = False,
    style: PageStyle = PageStyle(),
    handling: Handling = Handling(),
) -> GeneratedPage:
    """Run the full pipeline on ``image_bgr`` and produce a coloring page + legend.

    ``progress_callback``, if given, is called after each pipeline stage with
    a 0-100 percentage reflecting real work completed (not a fake animation).
    ``should_cancel``, if given, is polled between stages; when it returns
    True, ``PipelineCancelled`` is raised and no more work is done.
    ``collect_analysis`` also returns what the page is made of in
    ``GeneratedPage.analysis`` (see ``PageAnalysis``), for benchmarks and
    tests, with the faces in the picture (on line art, looked for only then).
    The page itself is the same either way. ``style`` says how the page
    is drawn -- line width and the tone of the ink (see ``PageStyle``); it
    changes nothing about which regions the page has. ``handling`` can turn
    off what the pipeline does with the line art, the subject and faces, and
    the text it finds (see ``Handling``): the page is then drawn as if it had
    found nothing of the kind.

    Line art (see ``ink``) is drawn from its own ink: the ink is printed, in
    the artwork's own tone, and the regions are the areas it encloses, colored
    from the fills without the ink or its anti-aliased edge. Their paint goes
    over the ink's thin parts -- hatching, and fine lines as far as their
    middle -- but never over bold ink. Every other picture is drawn from its
    colors alone. Its regions' edges are settled by a vote that smooths them
    where the picture is textured and keeps them on its own edges (see
    ``texture``), and its subject and faces get more detail: a region inside
    the subject or a face the pipeline finds may be half the difficulty's
    smallest (see ``subject``, ``faces`` and ``regions.build_regions``). A
    face's regions are painted in the palette
    colors nearest them, those a faint step in tone from a neighbor joined to
    it (see ``tones``), and the thin dark marks there that no region keeps --
    pupils, eyelid and lip lines, whisker dots -- are printed, in their own
    tone (see ``marks``). The brush is the same everywhere.

    On every picture the letters in the lines of text found -- signs, titles,
    captions -- are printed solid in the ink's tone, and nothing else in their
    lines (see ``text``): dark letters and light letters alike as ink on bare
    paper. The regions are left as they are, and painted round them, and no
    number is written on them or right beside them (see ``labels``).

    Resizing, finding the ink and quantization results are cached per image
    object, so regenerating the same image with a different minimum region
    size, or going back to earlier settings, skips straight to the region
    stages.
    """

    def report(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(_STAGE_PROGRESS[stage])

    def check_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise PipelineCancelled()

    check_cancelled()
    resized = _cache.get_or_compute(
        image_bgr, ("resize", long_edge), lambda: resize_to_long_edge(image_bgr, long_edge)
    )
    h, w = resized.shape[:2]
    report("resize")

    check_cancelled()
    line_art, ink_lines = detect_ink(image_bgr, resized, long_edge)
    ink_mask = ink_lines if handling.line_art and line_art.is_line_art and ink_lines.any() else None
    # An anti-aliased edge is at least the pixels right beside the ink, however fine the page.
    halo_px = max(1.0, print_scale((w, h)).mm_to_px(ink.HALO_MM))
    report("ink")

    check_cancelled()
    labels, palette_bgr = _cache.get_or_compute(
        image_bgr,
        # Line art's colors are taken around its ink, unless line art is turned off (see ``Handling``).
        ("quantize", long_edge, params.num_colors, params.blur_sigma, ink_mask is not None),
        lambda: quantize(resized, params.num_colors, params.blur_sigma, ink=ink_mask, halo_px=halo_px),
    )
    report("quantize")

    check_cancelled()
    min_area_px, min_width_px = _paintable_limits(params, (w, h))
    ink_gray = 0
    region_labels = labels
    printed_ink = labels >= len(palette_bgr)  # the ink quantize gave no color: all False unless the picture is line art
    if ink_mask is not None:
        ink_gray = ink.ink_gray(resized, printed_ink)
        off_edge = ~ink.near(ink_mask, halo_px)  # a fill's own colors, away from the ink's anti-aliased edge
        labels = join_ink(labels, len(palette_bgr), resized, (ink_gray,) * 3, min_area_px, off_edge)  # a new map
        # The regions are built through hatching, a hatched patch one run of its gaps' colors, but never through a
        # line between two areas a painter sees as two.
        thin_px = print_scale((w, h)).mm_to_px(ink.THIN_INK_MM)
        region_labels, corners = look_through_hatching(labels, len(palette_bgr), thin_px, min_width_px)
        printed_ink = (labels >= len(palette_bgr)) | corners
    # The palette can be shorter than the difficulty asked for: colors too
    # close to tell apart are merged (see ``quantize``). Passing the count the
    # difficulty asked for gives the same regions -- the labeling only needs an
    # upper bound -- but not the same meaning, and it sizes its arrays for
    # colors that do not exist.
    # A picture drawn from its colors spends more detail on its subject and its faces. Line art's are drawn by its own
    # ink.
    detail = in_faces = None
    in_subject = np.zeros((h, w), dtype=bool)
    if ink_mask is None and handling.detail:
        found = detect_faces(image_bgr, resized)
        if found:
            in_faces = faces.mask(found, (w, h))
        in_subject = detect_subject(image_bgr, resized)
        detail = in_subject | in_faces if in_faces is not None else in_subject
        if not detail.any():
            detail = None
    region_id_map, region_color = build_regions(region_labels, len(palette_bgr), min_area_px, min_width_px, detail)
    if ink_mask is None:
        # Fur, foliage and stone leave the regions ragged edges no brush can follow: they are settled by a vote of
        # the page around each pixel, held to the picture's own edges. Line art's edges are its ink.
        region_id_map, region_color = smooth_regions(
            resized, region_id_map, region_color, palette_bgr, min_area_px, min_width_px, detail
        )
    if in_faces is not None:
        # A face's skin or fur is painted in a few large tones: each region there in the color nearest it, and the ones
        # a faint step from a neighbor joined to it.
        region_id_map, region_color = tones.settle_tones(resized, in_faces, region_id_map, region_color, palette_bgr)
        # The thin dark marks in a face that no brush can paint -- pupils, eyelid and lip lines, whisker dots -- merge into
        # the regions around them. They are printed instead, in their own tone, and the regions' paint goes round them.
        printed_ink = marks.detail_marks(
            resized, in_faces, region_id_map, region_color, palette_bgr, print_scale((w, h))
        )
        ink_gray = ink.ink_gray(resized, printed_ink)
    clearable = None
    if ink_mask is not None:
        region_id_map, inked = settle_enclosed(
            region_id_map, region_color, resized, (ink_gray,) * 3, min_width_px, off_edge
        )
        printed_ink |= inked
        # A brush goes over thin ink, which the regions beside it share; a number on hatching may clear it.
        region_id_map = paint_over_thin_ink(
            region_id_map, region_color, printed_ink, thin_px, resized, palette_bgr, off_edge
        )
        # What a brush still can't reach against ink it may not cross is left unpainted, and every white area a brush
        # fits in is a region of its own, with its own number.
        region_id_map = leave_pockets(region_id_map, region_color, printed_ink, min_width_px)
        region_id_map, region_color, corners = split_areas(
            region_id_map, region_color, printed_ink, min_width_px, min_area_px, len(palette_bgr)
        )
        printed_ink |= corners
        # A number may clear hatching, but no line: not half of one between two areas, nor the ink left where two
        # regions meet under the strokes, a line's width of it.
        clearable = detail_ink(region_id_map, printed_ink, thin_px / 2, style.line_width_px((w, h)))
    # The letters of signs, titles and captions are printed, and nothing else in their lines; the regions are painted
    # round them, and no number may clear them.
    check_cancelled()
    found_text = detect_text(image_bgr, resized) if handling.text else []
    letters = text.lettering(resized, found_text) if found_text else None
    if letters is not None:
        # Inside a line of text the letters take the place of the ink lying in a region, as the page prints them: what
        # is printed there is the letters, and the ink in no region, which keeps two regions apart.
        in_no_region = printed_ink & (region_id_map < 0)
        printed_ink = np.where(letters.area, in_no_region | (letters.ink >= _LETTERING_PRINTED), printed_ink)
        if clearable is not None:
            clearable = clearable & ~letters.area
        if ink_mask is None:
            # Line art prints in the artwork's own ink. Elsewhere the ink's gray is the picture's under the printed ink,
            # and in the lines of text under their darker side: light letters print in the tone of their dark ground.
            ink_gray = ink.ink_gray(resized, np.where(letters.area, in_no_region | letters.dark, printed_ink))
    report("regions")

    def numbered(region_id_map: np.ndarray) -> tuple[list[Region], list[int], dict[int, int]]:
        regions = extract_regions(
            region_id_map, region_color, printed=printed_ink if ink_mask is not None or printed_ink.any() else None
        )
        # Quantizing to more colors than the image actually has can leave some
        # k-means clusters with no (or a merged-away) region. Drop those from the
        # legend and renumber the rest contiguously so "1..N" always matches what
        # is actually drawn on the page.
        used_color_indices = sorted({r.color_index for r in regions})
        remap = {old: new for new, old in enumerate(used_color_indices)}
        for region in regions:
            region.color_index = remap[region.color_index]
        return regions, used_color_indices, remap

    check_cancelled()
    regions, used_color_indices, remap = numbered(region_id_map)
    report("contours")

    check_cancelled()
    printing = dict(ink=printed_ink, ink_gray=ink_gray, lettering=letters)
    rendered = render_page((w, h), regions, region_id_map, style, clearable=clearable, **printing)
    for _ in range(_MERGE_ROUNDS):
        # On line art, a region whose number found no room anywhere joins the area beside it, and the page is drawn
        # again -- until every number has room, no region moves, or _MERGE_ROUNDS are done. Every other picture is
        # drawn once.
        cramped = sorted({label.region_id for label in rendered.labels if label.cramped})
        if ink_mask is None:
            cramped = []
        merged = merge_cramped(region_id_map, region_color, printed_ink, cramped, min_width_px) if cramped else None
        if merged is None:
            break
        region_id_map = merged
        clearable = detail_ink(region_id_map, printed_ink, thin_px / 2, style.line_width_px((w, h)))
        if letters is not None:
            clearable = clearable & ~letters.area
        regions, used_color_indices, remap = numbered(region_id_map)
        rendered = render_page((w, h), regions, region_id_map, style, clearable=clearable, **printing)
    if rendered.printed_ink is not None:
        printed_ink = rendered.printed_ink  # less the hatching cleared behind numbers
    used_palette_bgr = palette_bgr[used_color_indices]
    legend = render_legend(used_palette_bgr, width=w, px_per_mm=print_scale((w, h)).px_per_mm)
    report("render")

    # Legend colors first, so a color index means the same color in region_color as in the renumbered regions.
    order = used_color_indices + [i for i in range(len(palette_bgr)) if i not in remap]
    new_index = np.empty(len(order), dtype=np.int32)
    new_index[order] = np.arange(len(order), dtype=np.int32)
    painting = Painting(
        region_id_map=region_id_map,
        region_color=new_index[region_color],
        palette_rgb=[(int(b[2]), int(b[1]), int(b[0])) for b in palette_bgr[order]],
        ink=rendered.picture_ink,
        ink_gray=rendered.drawing.ink_gray,
    )
    palette_rgb = painting.palette_rgb[: len(used_color_indices)]

    analysis = None
    if collect_analysis:
        analysis = PageAnalysis(
            region_id_map=region_id_map,
            region_color=painting.region_color,
            palette_bgr=palette_bgr[order],  # a copy: palette_bgr belongs to the stage cache
            legend_size=len(used_color_indices),
            min_region_area_px=min_area_px,
            detail=detail if detail is not None else np.zeros((h, w), dtype=bool),
            subject=in_subject,
            min_paintable_width_px=min_width_px,
            regions=regions,
            labels=rendered.labels,
            outlines=np.asarray(rendered.outlines),
            strokes=rendered.strokes,
            leaders=np.asarray(rendered.leaders),
            line_art=line_art,
            ink_lines=ink_lines.copy(),  # a copy: the mask belongs to the stage cache
            printed_ink=printed_ink,
            ink_gray=ink_gray,
            faces=detect_faces(image_bgr, resized),
            text=found_text,
            lettering_area=letters.area if letters is not None else np.zeros((h, w), dtype=bool),
            lettering=PAPER - letters.ink if letters is not None else np.full((h, w), PAPER, dtype=np.uint8),
        )

    return GeneratedPage(
        page=rendered.image,
        legend=legend,
        palette_rgb=palette_rgb,
        num_colors_used=len(used_color_indices),
        num_regions=len(regions),
        analysis=analysis,
        drawing=rendered.drawing,
        painting=painting,
    )

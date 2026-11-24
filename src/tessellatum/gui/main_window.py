"""Top-level application window."""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tessellatum.core import export, mixing, pipeline, swatches
from tessellatum.core.legend import NAME_RECIPE, render_legend
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import GeneratedPage
from tessellatum.core.print_size import MM_PER_INCH, print_scale
from tessellatum.gui.controls_panel import DPI, PALETTE, ControlsPanel
from tessellatum.gui.preview_widget import PreviewWidget, VersionBar, ZoomBar
from tessellatum.gui.worker import PipelineWorker, TaskWorker

OPEN_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)"
# The preview's legend is drawn at least this many pixels a millimeter on paper (300 dpi), finer than a preview's
# page, so that its names and codes stay sharp zoomed in.
PREVIEW_LEGEND_PX_PER_MM = 300 / MM_PER_INCH


class MainWindow(QMainWindow):
    def __init__(self, store: QSettings | None = None):
        """``store``, if given, is where the panel keeps its settings between runs (see ``ControlsPanel``)."""
        super().__init__()
        self.setWindowTitle("Tessellatum")

        self.controls = ControlsPanel(store=store)
        self.preview = PreviewWidget()
        # Every version of a page is drawn with it, so the bar switches between them at once.
        self.version_bar = VersionBar()
        self.version_bar.setEnabled(False)
        self.zoom_bar = ZoomBar(self.preview)
        bars = QHBoxLayout()
        bars.addWidget(self.version_bar, 1)
        bars.addWidget(self.zoom_bar)
        preview_pane = QWidget()
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(bars)
        preview_layout.addWidget(self.preview)

        # The panel scrolls rather than holding the window taller than a small screen.
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(self.controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # As wide as the panel needs with the scroll bar beside it, so that nothing is cut off when it shows.
        controls_scroll.setMinimumWidth(
            self.controls.minimumSizeHint().width() + controls_scroll.verticalScrollBar().sizeHint().width()
        )

        splitter = QSplitter()
        splitter.addWidget(controls_scroll)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 880])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open an image to get started.")

        self.current_image_bgr: np.ndarray | None = None
        self.current_page: GeneratedPage | None = None
        self._previewed_image: np.ndarray | None = None  # the picture current_page is of
        self._worker: PipelineWorker | None = None
        self._pending_export_path: Path | None = None
        self._pending_export_format: str | None = None
        self._pending_export_version: Version | None = None
        self._pending_export_resolution: tuple[str, int] | None = None  # the PNG's, as the panel asked for it
        self._raster_worker: TaskWorker | None = None  # drawing the PNG in pixels
        self._pending_export_drawn = False  # whether the export waits for the preview's page drawn at print size
        self._ready_message = ""  # the status of the preview shown
        # Whether _worker is making a page whose end the window hasn't handled yet. (Its last signal can come while Qt
        # still reports it running.)
        self._making = False
        # The preview's page drawn at print size: what the zoom shows past the preview's own pixels, and what the
        # export saves. Drawn in the background when either first asks for it, and dropped with the preview.
        self._print_page: GeneratedPage | None = None
        self._print_worker: PipelineWorker | None = None
        self._retired: list[QThread] = []  # workers stopped but maybe not finished yet: kept until they are

        self.controls.open_image_requested.connect(self.open_image)
        self.controls.generate_requested.connect(self.generate_preview)
        self.controls.export_requested.connect(self.export_page)
        self.controls.abort_requested.connect(self.abort_generation)
        self.controls.legend_detail_changed.connect(self._show_legend)
        self.controls.paints_changed.connect(self._show_legend)
        self._recipes_for: tuple | None = None  # what the recipes kept were mixed for, and them (see _recipes)
        self.version_bar.version_changed.connect(self.preview.show_version)
        self.preview.detail_wanted.connect(self._draw_print_page)

        self.resize(1280, 840)

        # Get one-time costs (loading/compiling the compiled kernels, first
        # OpenCV/Pillow calls) out of the way while the user picks an image,
        # so the first preview is as fast as later ones.
        threading.Thread(target=_warm_up_pipeline, name="pipeline-warm-up", daemon=True).start()

    # -- Open -----------------------------------------------------------
    def open_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Image", "", OPEN_IMAGE_FILTER)
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.current_image_bgr = pipeline.load_image_bgr(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open image", str(exc))
            return

        self.controls.set_thumbnail(_bgr_to_pixmap(self.current_image_bgr))
        height, width = self.current_image_bgr.shape[:2]
        self.controls.set_picture_size((width, height))
        self.current_page = None
        self._drop_print_page()
        self.controls.set_export_enabled(False)
        self.version_bar.setEnabled(False)  # the preview left showing is the last picture's, until the next one
        self.statusBar().showMessage(f"Loaded {path.name}. Click \"Generate Preview\".")

    # -- Preview ----------------------------------------------------------
    def generate_preview(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if (self._worker is not None and self._worker.isRunning()) or self._pending_export_path is not None:
            return
        if self._raster_worker is not None:
            return

        self._drop_print_page()  # the zoom asks for the new page's
        self._making = True
        self.controls.set_busy(True)
        self.statusBar().showMessage("Generating preview…")

        # The page at its working size, where its level's brush is a few pixels across: what an export draws larger.
        long_edge = pipeline.working_long_edge(
            self.current_image_bgr, self.controls.get_difficulty_params(), self.controls.get_handling(),
            self.controls.get_page_style().paper_format,
        )
        self._worker = self._make_worker(long_edge, versions=tuple(Version))
        self._worker.succeeded.connect(self._on_preview_ready)
        self._worker.start()

    def abort_generation(self) -> None:
        # An export of the preview's page waits for it to be drawn at print size.
        worker = self._print_worker if self._pending_export_drawn else self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            self.statusBar().showMessage("Cancelling…")

    def _on_worker_cancelled(self) -> None:
        self._making = False
        self.controls.set_busy(False)
        self._clear_pending_export()
        self.statusBar().showMessage("Cancelled.")

    def _on_preview_ready(self, page: GeneratedPage) -> None:
        self._making = False
        # A new page of the picture shown is shown as far in and in the same place, to compare the two.
        same_picture = self.current_page is not None and self._previewed_image is self.current_image_bgr
        self._drop_print_page()  # any drawn of the last page, for a zoom while this one was made
        self.current_page = page
        self._previewed_image = self.current_image_bgr
        self.controls.set_busy(False)
        self.controls.set_export_enabled(True)
        width, height = page.page.size
        self._ready_message = (
            f"Preview ready: {page.num_colors_used} colors, {page.num_regions} regions, made at {width} × {height} px."
        )
        self.statusBar().showMessage(self._ready_message)
        versions = {version: page.image(version) for version in Version}
        long_edge = self._print_long_edge()
        print_scale = 1.0 if long_edge is None else long_edge / max(page.page.size)
        legend, legend_scale = self._preview_legend(page)
        self.preview.show_page(
            versions, legend, self.version_bar.version(), print_scale, keep_view=same_picture,
            paper=page.drawing.style.paper_format, legend_scale=legend_scale,
        )
        self.version_bar.setEnabled(True)

    def _preview_legend(self, page: GeneratedPage):
        """The legend of ``page`` as the panel asks for it now, drawn finer than the page, and how many of its pixels go
        to one of the page's (see ``PreviewWidget.show_page``)."""
        paper = page.drawing.style.paper_format
        page_px_per_mm = print_scale(page.page.size, paper).px_per_mm
        legend_scale = max(1.0, PREVIEW_LEGEND_PX_PER_MM / page_px_per_mm)
        palette_bgr = np.asarray(page.palette_rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
        legend = render_legend(
            palette_bgr, round(page.page.width * legend_scale), page_px_per_mm * legend_scale,
            self.controls.get_legend_detail(), names=page.color_names or None, recipes=self._recipes(page),
            mixes=page.mixes or None,
        )
        return legend, legend_scale

    def _recipes(self, page: GeneratedPage) -> list[list[str]] | None:
        """How to mix each of ``page``'s legend colors from the paints listed, as the legend writes it, if the legend
        shows recipes; the last ones found are kept, as the legend is drawn again and again."""
        paints = self.controls.listed_paints()
        if self.controls.get_legend_detail() != NAME_RECIPE or not paints:
            return None
        key = (tuple(page.palette_rgb), paints)
        if self._recipes_for is None or self._recipes_for[0] != key:
            found = mixing.recipes(page.palette_rgb, [rgb for _name, rgb in paints])
            self._recipes_for = (key, [mixing.recipe_lines(recipe, paints) for recipe in found])
        return self._recipes_for[1]

    def _show_legend(self, _detail: str = "") -> None:
        """Show the page's legend as the panel now asks for it: the page itself is the same."""
        if self.current_page is not None and self.current_page.drawing is not None:
            self.preview.set_legend(*self._preview_legend(self.current_page))

    def _on_worker_failed(self, message: str) -> None:
        self._making = False
        self.controls.set_busy(False)
        self._clear_pending_export()
        QMessageBox.critical(self, "Generation failed", message)
        self.statusBar().showMessage("Generation failed.")

    # -- Export -----------------------------------------------------------
    def export_page(self) -> None:
        if self.current_image_bgr is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        if (self._worker is not None and self._worker.isRunning()) or self._raster_worker is not None:
            return

        fmt = self.controls.get_output_format()
        version = self.controls.get_export_version()
        if fmt == PALETTE:
            suffix = self.controls.get_palette_format()
            file_filter = swatches.FORMATS[suffix]
            name = "coloring_page_palette"
        else:
            suffix = ".png" if fmt == "PNG" else ".pdf"
            file_filter = "PNG image (*.png)" if fmt == "PNG" else "PDF document (*.pdf)"
            name = "coloring_page" if version is Version.PAGE else f"coloring_page_{version.value.lower()}"
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Coloring Page", f"{name}{suffix}", file_filter)
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)

        self._pending_export_path = path
        self._pending_export_format = fmt
        self._pending_export_version = version
        self._pending_export_resolution = self.controls.get_export_resolution()
        self.controls.set_busy(True)
        self.statusBar().showMessage(f"Rendering the {fmt} at print resolution…")

        # Every export is drawn from the page's print layout: the preview's page laid out at 300 dpi on its paper, its
        # numbers placed once, as small as the style asks (see ``pipeline.draw_page``), drawn once for the zoom and
        # the export alike; made first if the settings changed since. The PDF draws it as vectors, the PNG draws the
        # same vectors in pixels.
        long_edge = pipeline.export_long_edge(self.current_image_bgr, self.controls.get_page_style().paper_format)
        preview = self._preview_to_draw(long_edge)
        if preview is not None and fmt == PALETTE:
            self._on_export_ready(preview)  # the preview's legend: nothing to draw
            return
        if preview is not None:
            if self._print_page is not None:
                self._on_export_ready(self._print_page)
            else:
                self._pending_export_drawn = True
                self._draw_print_page()  # saved once drawn (see _on_print_page_ready)
            return
        self._making = True
        self._worker = self._make_worker(long_edge, vector=True)
        self._worker.succeeded.connect(self._on_export_ready)
        self._worker.start()

    def _on_export_ready(self, page: GeneratedPage) -> None:
        self._making = False
        self.controls.set_busy(False)
        path = self._pending_export_path
        fmt = self._pending_export_format
        version = self._pending_export_version
        resolution = self._pending_export_resolution
        self._clear_pending_export()
        if path is None or fmt is None or version is None:
            return
        # The legend as the panel asks for it now: changing it doesn't make the page again (see _preview_to_draw).
        drawing = page.drawing
        style = dataclasses.replace(drawing.style, legend_detail=self.controls.get_legend_detail())
        page = dataclasses.replace(page, drawing=dataclasses.replace(drawing, style=style))
        if fmt == PALETTE:
            try:
                written = swatches.save_palette(path, page.palette_rgb, names=_palette_names(page))
            except Exception as exc:  # noqa: BLE001
                self._on_export_failed(str(exc))
                return
            self._on_palette_saved(written)
            return
        if fmt == "PDF":
            try:
                export.save_pdf(
                    page.drawing, page.palette_rgb, path, version=version, painting=page.painting,
                    names=page.color_names or None, recipes=self._recipes(page), mixes=page.mixes or None,
                )
            except Exception as exc:  # noqa: BLE001
                self._on_export_failed(str(exc))
                return
            self._on_export_saved(page, path, page.page.size)
            return

        # The vector page drawn in pixels, as the PDF prints it (see ``export.rasterize``), at the resolution asked
        # for: in the background, since at 600 dpi on A2 it takes seconds. It can't be stopped once started.
        unit, value = resolution
        printed_long_mm = max(print_scale(page.page.size, page.drawing.style.paper_format).printed_size_mm)
        dpi = value if unit == DPI else value * MM_PER_INCH / printed_long_mm
        self.controls.set_busy(True, cancellable=False)
        self.statusBar().showMessage(f"Drawing the PNG at {dpi:.0f} dpi…")
        recipes = self._recipes(page)
        worker = TaskWorker(
            lambda: export.export_png(
                page.drawing, page.palette_rgb, path, dpi, version, page.painting, names=page.color_names or None,
                recipes=recipes, mixes=page.mixes or None,
            )
        )
        worker.succeeded.connect(lambda size: self._on_export_saved(page, path, size))
        worker.failed.connect(self._on_export_failed)
        self._raster_worker = worker
        worker.start()

    def _retire_raster_worker(self) -> None:
        # Its signal can come before its thread ends: kept until it has, so it isn't destroyed while running.
        worker, self._raster_worker = self._raster_worker, None
        self._retired = [retired for retired in self._retired if not retired.isFinished()]
        if worker is not None and not worker.isFinished():
            self._retired.append(worker)

    def _on_export_saved(self, page: GeneratedPage, path: Path, size: tuple[int, int]) -> None:
        self._retire_raster_worker()
        self.controls.set_busy(False)
        if self.preview.wants_detail():
            self._draw_print_page()  # asked for by a zoom while the export was made
        paper = page.drawing.style.paper_format
        width, height = size
        dpi = print_scale(size, paper).dpi
        printed = f"{width:,} × {height:,} px, {dpi:.0f} dpi on {paper.name}"
        self.statusBar().showMessage(f"Exported to {path} ({printed})")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}\n\n{printed}")

    def _on_palette_saved(self, written: list[Path]) -> None:
        self.controls.set_busy(False)
        if self.preview.wants_detail():
            self._draw_print_page()  # asked for by a zoom while the export was made
        files = "\n".join(str(path) for path in written)
        note = f"{len(written)} palettes of at most {swatches.PROCREATE_MAX_SWATCHES} colors" if len(written) > 1 else ""
        self.statusBar().showMessage(f"Exported the palette to {written[0]}" + (f" ({note})" if note else ""))
        QMessageBox.information(self, "Export complete", f"Saved to:\n{files}" + (f"\n\n{note}" if note else ""))

    def _on_export_failed(self, message: str) -> None:
        self._retire_raster_worker()
        self.controls.set_busy(False)
        QMessageBox.critical(self, "Export failed", message)
        self.statusBar().showMessage("Export failed.")

    def _clear_pending_export(self) -> None:
        self._pending_export_path = None
        self._pending_export_format = None
        self._pending_export_version = None
        self._pending_export_resolution = None
        self._pending_export_drawn = False

    def _preview_to_draw(self, long_edge: int) -> GeneratedPage | None:
        """The page previewed, if an export at ``long_edge`` is drawn from it, or is it: it is no larger, and made with
        the panel's settings as they are now. None if not, and the export is made of the picture itself."""
        page = self.current_page
        if page is None or page.model is None or long_edge < max(page.page.size):
            return None
        model = page.model
        # What the legend writes doesn't change the page: the export writes the panel's (see _on_export_ready).
        style = dataclasses.replace(self.controls.get_page_style(), legend_detail=model.style.legend_detail)
        current = (self.controls.get_difficulty_params(), style, self.controls.get_handling())
        return page if (model.params, model.style, model.handling) == current else None

    def _make_worker(
        self,
        long_edge: int,
        versions: tuple[Version, ...] = (),
        drawn_from: GeneratedPage | None = None,
        vector: bool = False,
    ) -> PipelineWorker:
        """A worker generating the current image at ``long_edge`` with the panel's settings, wired to its progress, and
        drawing ``versions`` of the page too: drawing ``drawn_from`` there, the preview made with those settings, if
        given; with ``vector``, laid out to print (see ``PipelineWorker``)."""
        worker = PipelineWorker(
            self.current_image_bgr,
            self.controls.get_difficulty_params(),
            long_edge,
            style=self.controls.get_page_style(),
            handling=self.controls.get_handling(),
            versions=versions,
            drawn_from=drawn_from,
            vector=vector,
        )
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.progress.connect(self.controls.set_progress)
        return worker

    # -- The preview's page at print size -------------------------------------
    def _print_long_edge(self) -> int | None:
        """The long edge the preview's page is laid out at to print (see ``pipeline.export_long_edge``); None if there
        is no page."""
        page = self.current_page
        if page is None or page.model is None or self._previewed_image is None:
            return None
        return max(pipeline.export_long_edge(self._previewed_image, page.drawing.style.paper_format), *page.page.size)

    def _draw_print_page(self) -> None:
        """Lay the preview's page out to print, in the background, unless it is drawn or being drawn: for the zoom,
        which shows it past the preview's own pixels, and for the export, which draws it. Not while a page is being
        made: a preview, which will replace this one, or an export with other settings."""
        long_edge = self._print_long_edge()
        running = self._print_worker is not None and self._print_worker.isRunning()
        if long_edge is None or self._print_page is not None or running or self._making:
            return
        page = self.current_page
        model = page.model
        worker = PipelineWorker(
            self._previewed_image,
            model.params,
            long_edge,
            style=model.style,
            handling=model.handling,
            versions=tuple(Version),
            drawn_from=page,
            vector=True,
        )
        worker.succeeded.connect(self._on_print_page_ready)
        worker.failed.connect(self._on_print_page_failed)
        worker.cancelled.connect(self._on_print_page_cancelled)
        worker.progress.connect(self._on_print_page_progress)
        self._print_worker = worker
        if self._pending_export_path is None:
            self.statusBar().showMessage("Drawing the page as it prints, for the zoom…")
        worker.start()

    def _drop_print_page(self) -> None:
        """Forget the preview's page at print size, and stop drawing it: the preview is about to change."""
        self._print_page = None
        worker = self._print_worker
        self._print_worker = None
        self._retired = [retired for retired in self._retired if not retired.isFinished()]
        if worker is not None and not worker.isFinished():
            worker.requestInterruption()
            self._retired.append(worker)

    # Each of these is the export's, if it waits for the page; else the zoom's alone, telling nothing over an export
    # with other settings that may be under way.
    def _on_print_page_progress(self, percent: int) -> None:
        if self.sender() is self._print_worker and self._pending_export_drawn:
            self.controls.set_progress(percent)

    def _on_print_page_ready(self, drawn: GeneratedPage) -> None:
        page = self.current_page
        if self.sender() is not self._print_worker or page is None or drawn.model is not page.model:
            return  # drawn from a preview since replaced
        self._print_page = drawn
        self.preview.show_detail({version: drawn.image(version) for version in Version})
        if self._pending_export_drawn:
            self._on_export_ready(drawn)
        elif self._pending_export_path is None:
            self.statusBar().showMessage(self._ready_message)

    def _on_print_page_failed(self, message: str) -> None:
        if self.sender() is not self._print_worker:
            return
        if self._pending_export_drawn:
            self._on_worker_failed(message)
        elif self._pending_export_path is None:
            self.statusBar().showMessage(f"The page could not be drawn as it prints: {message}")

    def _on_print_page_cancelled(self) -> None:
        if self.sender() is not self._print_worker:
            return
        if self._pending_export_drawn:
            self._on_worker_cancelled()
        elif self._pending_export_path is None:
            self.statusBar().showMessage(self._ready_message)

    def closeEvent(self, event) -> None:
        # Stop what is still running, rather than leave a thread to be destroyed with the window.
        for worker in (self._worker, self._print_worker, self._raster_worker, *self._retired):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait()
        super().closeEvent(event)


def _warm_up_pipeline() -> None:
    try:
        pipeline.warm_up()
    except Exception:  # noqa: BLE001 - best effort; a real failure resurfaces on Generate
        pass


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = image_bgr[:, :, ::-1].copy()
    h, w, _ = rgb.shape
    qimage = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


def _palette_names(page: GeneratedPage) -> list[str] | None:
    """The names a palette file gives ``page``'s colors: the legend's, a mixed color's with its recipe after it,
    "gray (1 Ivory black + 1 Titanium white)"."""
    if not page.color_names:
        return None
    if not page.mixes:
        return list(page.color_names)
    return [
        name if lines is None else f"{name} ({' + '.join(lines)})" for name, lines in zip(page.color_names, page.mixes)
    ]

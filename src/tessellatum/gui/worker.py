"""Background thread that runs the (potentially slow) image pipeline off the UI thread."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QThread, Signal

from tessellatum.core import pipeline
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.painting import Version
from tessellatum.core.pipeline import GeneratedPage, Handling, PipelineCancelled
from tessellatum.core.render import PageStyle


class PipelineWorker(QThread):
    succeeded = Signal(object)  # GeneratedPage
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int)  # 0-100

    def __init__(
        self,
        image_bgr: np.ndarray,
        params: DifficultyParams,
        long_edge: int,
        style: PageStyle = PageStyle(),
        handling: Handling = Handling(),
        versions: Sequence[Version] = (),
        drawn_from: GeneratedPage | None = None,
    ):
        """``versions`` are the versions of the page (see ``painting``) to draw here too, off the UI thread, before the
        page is handed over: ``GeneratedPage.image`` keeps them. With ``drawn_from``, a page of the same picture made
        with these settings at a smaller size, the page is that one drawn at ``long_edge`` (see
        ``pipeline.draw_page``) rather than made again."""
        super().__init__()
        self._image_bgr = image_bgr
        self._params = params
        self._long_edge = long_edge
        self._style = style
        self._handling = handling
        self._versions = tuple(versions)
        self._drawn_from = drawn_from

    def run(self) -> None:
        try:
            if self._drawn_from is not None:
                result: GeneratedPage = pipeline.draw_page(
                    self._drawn_from,
                    self._image_bgr,
                    self._long_edge,
                    progress_callback=self.progress.emit,
                    should_cancel=self.isInterruptionRequested,
                )
            else:
                result = pipeline.generate(
                    self._image_bgr,
                    self._params,
                    self._long_edge,
                    progress_callback=self.progress.emit,
                    should_cancel=self.isInterruptionRequested,
                    style=self._style,
                    handling=self._handling,
                )
            for version in self._versions:
                if self.isInterruptionRequested():
                    raise PipelineCancelled()
                result.image(version)
        except PipelineCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)

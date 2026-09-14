"""Background thread that runs the (potentially slow) image pipeline off the UI thread."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from tessellatum.core import pipeline
from tessellatum.core.difficulty import DifficultyParams
from tessellatum.core.pipeline import GeneratedPage, PipelineCancelled


class PipelineWorker(QThread):
    succeeded = Signal(object)  # GeneratedPage
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int)  # 0-100

    def __init__(self, image_bgr: np.ndarray, params: DifficultyParams, long_edge: int):
        super().__init__()
        self._image_bgr = image_bgr
        self._params = params
        self._long_edge = long_edge

    def run(self) -> None:
        try:
            result: GeneratedPage = pipeline.generate(
                self._image_bgr,
                self._params,
                self._long_edge,
                progress_callback=self.progress.emit,
                should_cancel=self.isInterruptionRequested,
            )
        except PipelineCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)

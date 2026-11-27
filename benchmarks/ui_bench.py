"""How the app feels: generation, edits, undo and redo, panel switching, pan and zoom, cancelling, and memory, measured
through the real window.

Run from the repo root, alone (no other heavy process; the numbers are wall-clock):

    .venv\\Scripts\\python benchmarks\\ui_bench.py OUT_DIR [--reps N] [--cycles N]

The window is drawn on the native platform with real fonts but kept off screen, and keeps no settings (no store: the
user's settings and recovery folder are never touched). Writes OUT_DIR/ui.json: every sample, and the median and 95th
percentile of each measure. ``--reps`` repeats each timing (default 5), ``--cycles`` the memory cycles (default 50).

What "cold" and "warm" mean here: cold is a page made with the pipeline's stage cache and the window's result cache
empty (the numba kernels warm); warm is the same page made again with the stage cache full and the result cache empty
(what Generate Preview costs after a change that misses both caches' later stages). A restore is undo or redo to a
state whose page is kept: no page is made.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

T_PROCESS = time.perf_counter()

from PySide6 import __version__ as PYSIDE_VERSION  # noqa: E402
from PySide6.QtCore import Qt, qVersion  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "sample_images"
IMAGES = {
    "lion (photo)": "l-photo-lion.jpg",
    "pearl (painting, face)": "l-painting-face-girl-with-a-pearl-earring.jpg",
    "writing pig (line art, text)": "m-comics-upside-downs-writing-pig.jpg",
}
LION = "lion (photo)"


# -- Measuring --------------------------------------------------------------------------------------------------------
def memory_mb() -> float:
    """The process's resident memory (working set on Windows), in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 2**20
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    import resource  # macOS: the peak, in bytes

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def summary(samples_s: list[float]) -> dict:
    """Median, 95th percentile and max of ``samples_s`` (seconds), in ms, with the samples."""
    ordered = sorted(samples_s)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"n": len(ordered), "median_ms": round(statistics.median(ordered) * 1000, 2),
            "p95_ms": round(p95 * 1000, 2), "max_ms": round(ordered[-1] * 1000, 2),
            "samples_ms": [round(s * 1000, 2) for s in samples_s]}


class Bench:
    def __init__(self, out_dir: Path, reps: int, cycles: int):
        from tessellatum.gui import main_window, theme
        from tessellatum.gui.main_window import MainWindow

        # Closing the document lets its changes go: the window would ask in a modal dialog.
        MainWindow._ask_unsaved = lambda _self, _name: main_window.DISCARD_CHANGES

        self.out_dir, self.reps, self.cycles = out_dir, reps, cycles
        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        theme.install(self.app, None)
        # No store: the window keeps nothing between runs and reads none of the user's settings.
        self.window = MainWindow()
        self.window.setAttribute(Qt.WA_DontShowOnScreen)  # drawn with the platform's fonts, never on the screen
        self.window.resize(1920, 1080)
        self.window.show()
        self.idle()
        self.result: dict = {"startup_to_shown_s": round(time.perf_counter() - T_PROCESS, 3)}
        # The window warms the pipeline up in a thread: its kernels are compiled before anything is timed.
        for thread in threading.enumerate():
            if thread.name == "pipeline-warm-up":
                thread.join()

    # -- Waiting ----------------------------------------------------------------------------------------------------
    def idle(self, timeout: float = 300.0) -> None:
        """Run the event loop until no job runs and no redraw or recording waits."""
        window = self.window
        deadline = time.perf_counter() + timeout
        while True:
            self.app.processEvents()
            if window.jobs.is_idle() and not window._redraw_timer.isActive() and not window._auto_timer.isActive():
                self.app.processEvents()
                if window.jobs.is_idle() and not window._redraw_timer.isActive():
                    return
            if time.perf_counter() > deadline:
                raise TimeoutError("the window didn't settle")
            time.sleep(0.001)

    def paint(self) -> None:
        """Draw what changed now, as the next frame would."""
        self.window.preview.viewport().repaint()

    def open(self, name: str) -> None:
        self.window.open_path(SAMPLES / IMAGES[name])
        self.idle()

    def clear_caches(self, stages: bool = True) -> None:
        from tessellatum.core import pipeline

        if stages:
            pipeline.clear_cache()
        self.window.results.clear()

    def shown(self, timeout: float = 300.0) -> None:
        """Run the event loop until the page the settings make is shown (the inspectors may still be measuring it in
        the background), and draw it."""
        window = self.window
        deadline = time.perf_counter() + timeout
        while (window.jobs.is_running("preview") or window._redraw_timer.isActive() or window.current_page is None
               or window.freshness.state() != "current"):
            self.app.processEvents()
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no page shown: {window.freshness.state()}")
            time.sleep(0.0005)
        self.paint()

    def generate(self) -> tuple[float, float]:
        """Generate Preview: the seconds until its page is shown, and until the window is idle again (the inspectors'
        measures of the page done)."""
        start = time.perf_counter()
        self.window.generate_preview()
        self.shown()
        shown = time.perf_counter() - start
        self.idle()
        return shown, time.perf_counter() - start

    # -- The measures -----------------------------------------------------------------------------------------------
    def generation(self) -> None:
        """Generate Preview at Medium, cold and warm, on each picture."""
        rows = []
        for name in IMAGES:
            self.open(name)
            self.window.controls.preset_combo.setCurrentText("Medium")
            self.idle()
            cold, warm, cold_idle, warm_idle = [], [], [], []
            for _ in range(self.reps):
                self.clear_caches()
                shown, settled = self.generate()
                cold.append(shown)
                cold_idle.append(settled)
                self.clear_caches(stages=False)
                shown, settled = self.generate()
                warm.append(shown)
                warm_idle.append(settled)
            page = self.window.current_page
            rows.append({"image": name, "level": "Medium", "size": list(page.page.size), "regions": page.num_regions,
                         "cold": summary(cold), "warm": summary(warm), "cold_until_idle": summary(cold_idle),
                         "warm_until_idle": summary(warm_idle)})
            print("generation", name, rows[-1]["cold"]["median_ms"], rows[-1]["warm"]["median_ms"], flush=True)
        self.result["generation"] = rows

    def edits(self) -> None:
        """On the lion's Medium page: an edit that only draws the page again (a line width: the redraw waits for the
        preference's delay, then draws it with the stage cache's regions), and the legend's detail (drawn at once)."""
        window = self.window
        self.open(LION)
        window.controls.preset_combo.setCurrentText("Medium")
        self.idle()
        self.generate()
        delay_s = window.preferences.redraw_after_ms() / 1000
        total, after_delay = [], []
        field = window.controls.controls["line_width_mm"]
        for i in range(self.reps * 2):
            window.results.clear()  # undone edits would come back from it: each edit draws its page
            start = time.perf_counter()
            field.step(1 if i % 2 == 0 else -1)
            while window._redraw_timer.isActive() or window._record_scheduled:
                self.app.processEvents()
                time.sleep(0.0005)
            fired = time.perf_counter()
            self.shown()
            total.append(time.perf_counter() - start)
            after_delay.append(time.perf_counter() - fired)
            self.idle()
        self.result["redraw_edit"] = {"setting": "line_width_mm", "delay_s": delay_s,
                                      "edit_to_page": summary(total), "after_the_delay": summary(after_delay)}
        legend = []
        for detail in ("Name", "Number only", "Name, hex and RGB") * self.reps:
            start = time.perf_counter()
            window.controls.legend_combo.setCurrentText(detail)
            self.idle()
            self.paint()
            legend.append(time.perf_counter() - start)
        self.result["legend_detail_change"] = summary(legend)
        print("edits", self.result["redraw_edit"]["after_the_delay"]["median_ms"], flush=True)

    def undo_redo(self) -> None:
        """Undo and redo between two Medium pages of the lion that differ in their colors, both kept: no page made."""
        window = self.window
        self.open(LION)
        window.controls.preset_combo.setCurrentText("Medium")
        self.idle()
        self.generate()
        window.controls.controls["num_colors"].step(1)
        self.idle()
        self.generate()
        times, inside, settled = [], [], []
        for i in range(self.reps * 4):
            start = time.perf_counter()
            (window.undo if i % 2 == 0 else window.redo)()
            self.shown()
            times.append(time.perf_counter() - start)
            inside.append(window.last_restore_ms / 1000)
            self.idle()
            settled.append(time.perf_counter() - start)
        self.result["cached_undo_redo"] = {"undo_or_redo_to_page": summary(times), "restore_itself": summary(inside),
                                           "until_idle": summary(settled)}
        # A state whose page is no longer kept: undo shows the old page labeled out of date and makes it again.
        window.results.clear()
        start = time.perf_counter()
        window.undo()
        self.app.processEvents()
        answered = time.perf_counter() - start
        self.idle()
        self.result["uncached_undo"] = {"answered_ms": round(answered * 1000, 2),
                                        "page_made_again_s": round(time.perf_counter() - start, 3)}
        print("undo", self.result["cached_undo_redo"]["undo_or_redo_to_page"]["median_ms"], flush=True)

    def switching(self) -> None:
        """Activities (each settings page and the analysis), the canvas's versions, and the focus mode, each drawn."""
        from tessellatum.core.painting import Version
        from tessellatum.gui.actions import ACTIVITIES

        window = self.window
        activities = []
        for _ in range(self.reps):
            for category in ACTIVITIES:
                start = time.perf_counter()
                window.show_activity(category)
                self.app.processEvents()
                window.repaint()
                activities.append(time.perf_counter() - start)
        self.idle()  # the analysis measures the page in the background: not part of the switch
        window.show_activity(ACTIVITIES[0])
        versions = []
        for version in (Version.COMPLETED, Version.TINTED, Version.PAGE) * self.reps:
            start = time.perf_counter()
            window.version_bar.buttons[version].click()
            self.paint()
            versions.append(time.perf_counter() - start)
        focus = []
        for i in range(self.reps * 2):
            start = time.perf_counter()
            window.set_focus_mode(i % 2 == 0)
            self.app.processEvents()
            window.repaint()
            focus.append(time.perf_counter() - start)
        window.set_focus_mode(False)
        self.idle()
        self.result["switching"] = {"activity": summary(activities), "version": summary(versions),
                                    "focus_mode": summary(focus)}
        print("switching", self.result["switching"]["activity"]["median_ms"], flush=True)

    def pan_zoom(self) -> None:
        """Frame times on the lion's Medium page: one zoom or pan step and a synchronous repaint."""
        window, view = self.window, self.window.preview
        n = max(30, self.reps * 6)

        def frames(step, settle=False):
            times = []
            for i in range(n):
                start = time.perf_counter()
                step(i)
                if settle:
                    view._sharpen()
                self.paint()
                times.append(time.perf_counter() - start)
            return summary(times)

        view.fit_to_window()
        zoom = frames(lambda i: view.zoom_by(1.25 if (i // 10) % 2 == 0 else 0.8))
        settled = frames(lambda i: view.zoom_by(1.25 if (i // 10) % 2 == 0 else 0.8), settle=True)
        view.zoom_to(1.0)
        bar = view.horizontalScrollBar()
        pan = frames(lambda i: bar.setValue(bar.value() + (20 if (i // 15) % 2 == 0 else -20)))
        start = time.perf_counter()
        view.zoom_to(2.5)  # past the preview's pixels: the page as it prints, drawn in the background
        while window._print_page is None:
            self.app.processEvents()
            time.sleep(0.001)
        layout_s = time.perf_counter() - start
        view._sharpen()
        pan_print = frames(lambda i: bar.setValue(bar.value() + (20 if (i // 15) % 2 == 0 else -20)))
        zoom_print = frames(lambda i: view.zoom_by(1.1 if (i // 10) % 2 == 0 else 1 / 1.1), settle=True)
        view.fit_to_window()
        self.idle()
        self.result["pan_zoom"] = {"zoom": zoom, "zoom_settled": settled, "pan": pan,
                                   "print_layout_for_zoom_s": round(layout_s, 3), "pan_print_detail": pan_print,
                                   "zoom_print_detail_settled": zoom_print}
        print("pan/zoom", zoom["p95_ms"], pan["p95_ms"], flush=True)

    def cancelling(self) -> None:
        """Stop while the lion's Realistic page is made: how soon the window says so, and when it is idle again."""
        window = self.window
        self.open(LION)
        window.controls.preset_combo.setCurrentText("Realistic")
        self.idle()
        rows = []
        for delay in (0.3, 0.8, 1.5) * self.reps:
            self.clear_caches()
            window.generate_preview()
            start = time.perf_counter()
            while time.perf_counter() - start < delay:
                self.app.processEvents()
                time.sleep(0.001)
            if not window.jobs.is_running("preview"):
                rows.append({"after_s": delay, "finished_first": True})
                continue
            pressed = time.perf_counter()
            window.abort_generation()
            window.status.repaint()
            acknowledged = time.perf_counter() - pressed
            said = window.status.text()
            self.idle()
            rows.append({"after_s": delay, "acknowledged_ms": round(acknowledged * 1000, 2),
                         "stopped_s": round(time.perf_counter() - pressed, 3), "said": said[:60],
                         "state": window.freshness.state()})
        acks = [r["acknowledged_ms"] / 1000 for r in rows if "acknowledged_ms" in r]
        stops = [r["stopped_s"] for r in rows if "stopped_s" in r]
        self.result["cancel"] = {"runs": rows, "acknowledged": summary(acks), "stopped": summary(stops)}
        window.controls.preset_combo.setCurrentText("Medium")
        self.idle()
        cancel = self.result["cancel"]
        print("cancel", cancel["acknowledged"]["median_ms"], cancel["stopped"]["median_ms"],
              flush=True)

    def memory(self) -> None:
        """Memory over ``cycles`` generations, each with other settings (no page comes back from a cache), then
        ``cycles`` undo/redo steps through them (pages kept come back, the others are made again)."""
        window = self.window
        self.open(LION)
        window.controls.preset_combo.setCurrentText("Medium")
        self.idle()
        self.clear_caches()
        colors = window.controls.controls["num_colors"]
        generated = [memory_mb()]
        for i in range(self.cycles):
            colors.step(1 if (i // 8) % 2 == 0 else -1)  # up and down: 9 values, the stage cache's quantize hits
            window.controls.controls["min_width_mm"].step(1 if i % 2 == 0 else -1)  # and never the same settings
            self.idle()
            self.generate()
            generated.append(memory_mb())
        moved = [memory_mb()]
        steps = []
        for i in range(self.cycles):
            start = time.perf_counter()
            (window.undo if (i // 10) % 2 == 0 else window.redo)()
            self.shown()
            steps.append(time.perf_counter() - start)
            self.idle()
            moved.append(memory_mb())
        window.close_document()
        self.idle()
        self.clear_caches()
        self.app.processEvents()
        self.result["memory"] = {
            "unit": "resident MB (working set on Windows)", "cycles": self.cycles,
            "generate_cycles_mb": [round(m) for m in generated], "undo_redo_cycles_mb": [round(m) for m in moved],
            "undo_redo_step": summary(steps), "after_closing_mb": round(memory_mb()),
            "result_cache_budget_mb": round(window.results.budget() / 2**20),
        }
        print("memory", round(generated[0]), round(max(generated)), round(moved[-1]), flush=True)

    def run(self) -> None:
        from tessellatum import __version__

        self.result["environment"] = {
            "app_version": __version__, "os": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(), "qt": qVersion(),
            "pyside": PYSIDE_VERSION, "platform_plugin": self.app.platformName(),
            "window": [self.window.width(), self.window.height()], "reps": self.reps,
        }
        self.result["memory_at_start_mb"] = round(memory_mb())
        for measure in (self.generation, self.edits, self.undo_redo, self.switching, self.pan_zoom, self.cancelling,
                        self.memory):
            measure()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "ui.json"
        path.write_text(json.dumps(self.result, indent=1, ensure_ascii=False), encoding="utf-8")
        print("written", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=50)
    args = parser.parse_args()
    bench = Bench(args.out_dir, args.reps, args.cycles)
    bench.run()
    bench.window.close()
    sys.stdout.flush()
    os._exit(0)  # Qt's teardown of a window with jobs' threads is no part of the measure


if __name__ == "__main__":
    main()

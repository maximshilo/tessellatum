"""The difficulty settings the tests are written at: the presets before 0.1.52.

Most tests need a page at some difficulty, not at a preset, so they take one of
these and stay where they were written whatever the presets become. Each paints
with the 3 mm brush and keeps colors 10 ΔE00 apart.
"""

from tessellatum.core.difficulty import DifficultyParams

COARSE = DifficultyParams(num_colors=6, min_region_area_mm2=300.0, blur_sigma=9.0)  # Easy until 0.1.51
MIDDLE = DifficultyParams(num_colors=12, min_region_area_mm2=125.0, blur_sigma=5.0)  # Medium until 0.1.51
FINE = DifficultyParams(num_colors=20, min_region_area_mm2=40.0, blur_sigma=2.5)  # Hard until 0.1.51

BY_OLD_NAME = {"Easy": COARSE, "Medium": MIDDLE, "Hard": FINE}

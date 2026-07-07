import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import HeatmapPanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=40, nc=60, t_step=2.0):
    rng = np.random.RandomState(3)
    counts = rng.poisson(30, size=(ns, nc)).astype(np.int64)
    counts[ns // 2, nc // 2] += 2000        # горячая точка -> выраженные изолинии
    counts[:, nc // 3] += 400               # вертикальная полоса
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_contours_off_by_default(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    assert h._contours_on is False
    assert h._iso_items == []                   # без включения изолиний нет


def test_contours_enable_creates_isocurves(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_contours_enabled(True)
    assert len(h._iso_items) > 0
    assert all(isinstance(it, pg.IsocurveItem) for it in h._iso_items)
    # ровно столько изолиний, сколько вернул расчёт уровней
    levels = h._contour_level_values(h._scaled_image())
    assert len(h._iso_items) == len(levels)
    assert len(h._iso_items) <= h._contour_levels


def test_contour_levels_monotone_in_range(app):
    h = HeatmapPanel()
    sg = _make_sg()
    h.set_spectrogram(sg)
    data = np.asarray(h._scaled_image(), dtype=np.float64)
    levels = h._contour_level_values(data)
    assert levels.size >= 1
    assert np.all(np.diff(levels) > 0)          # строго возрастают
    assert levels.min() >= float(data.min())
    assert levels.max() < float(data.max())     # граничный максимум отброшен


def test_contour_level_count_changes(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_contours_enabled(True)
    h.set_contour_levels(9)
    assert h._contour_levels == 9
    assert len(h._iso_items) <= 9
    levels = h._contour_level_values(h._scaled_image())
    assert len(h._iso_items) == len(levels)


def test_contours_disable_clears(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_contours_enabled(True)
    assert len(h._iso_items) > 0
    h.set_contours_enabled(False)
    assert h._iso_items == []                    # выключение убирает все изолинии


def test_contours_refresh_on_zscale_no_leak(app):
    # смена Z-шкалы при включённых изолиниях пересоздаёт их без накопления/утечки
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_contours_enabled(True)
    h.set_z_scale("linear")
    n1 = len(h._iso_items)
    levels1 = h._contour_level_values(h._scaled_image())
    assert n1 == len(levels1)
    h.set_z_scale("sqrt")
    levels2 = h._contour_level_values(h._scaled_image())
    assert len(h._iso_items) == len(levels2)     # ровно перестроены, не задвоены


def test_contour_data_transposed_to_match_image(app):
    # IsocurveItem получает data.T (форма (cols, rows)) — совпадение осей с row-major картой
    h = HeatmapPanel()
    sg = _make_sg(ns=40, nc=60)
    h.set_spectrogram(sg)
    h.set_contours_enabled(True)
    iso = h._iso_items[0]
    assert iso.data.shape == (h._disp_cols, h._disp_rows)


def test_contour_levels_empty_on_constant(app):
    # на константной карте уровней нет (всё на границе) -> изолиний не создаём, без падения
    h = HeatmapPanel()
    ns, nc = 10, 10
    counts = np.full((ns, nc), 5, dtype=np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64)
    sg = Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                     real_time_s=np.ones(ns), live_time_s=np.ones(ns))
    h.set_spectrogram(sg)
    h.set_contours_enabled(True)
    assert h._iso_items == []
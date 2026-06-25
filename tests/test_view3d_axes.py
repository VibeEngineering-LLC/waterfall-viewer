import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=40, t_step=2.0):
    counts = np.random.RandomState(1).poisson(50, size=(ns, nc)).astype(np.int64)
    counts[:, nc // 2] += 800  # выраженный пик -> ненулевой диапазон высоты
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_nice_ticks_round_and_in_range():
    ticks = Waterfall3DView._nice_ticks(0.0, 58.0, 5)
    assert ticks.size >= 3
    assert ticks.min() >= 0.0 and ticks.max() <= 58.0
    assert np.all(np.diff(ticks) > 0)          # строго возрастают
    step = np.diff(ticks)[0]
    assert step in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)  # «круглый» шаг 1/2/5×10^k


def test_nice_ticks_degenerate_empty():
    assert Waterfall3DView._nice_ticks(5.0, 5.0).size == 0
    assert Waterfall3DView._nice_ticks(10.0, 1.0).size == 0


def test_axis_labels_created_after_load(app):
    v = Waterfall3DView()
    assert v._axis_items == []                 # до загрузки подписей нет
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    # подписи по трём осям + три заголовка осей -> заведомо несколько GLTextItem
    assert len(v._axis_items) >= 6
    import pyqtgraph.opengl as gl
    assert all(isinstance(it, gl.GLTextItem) for it in v._axis_items)


def test_axis_labels_toggle_off_on(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    n = len(v._axis_items)
    assert n > 0
    v.set_axis_labels_visible(False)
    assert v._axis_items == []                 # скрытие удаляет все элементы
    v.set_axis_labels_visible(True)
    assert len(v._axis_items) == n             # восстановление воспроизводит тот же набор


def test_count_axis_height_monotonic(app):
    # ось счёта строит высоту через монотонную пару (counts -> height): высота не убывает
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=40, nc=60), max_time=400, max_chan=512)
    cflat = np.asarray(v._z_counts, dtype=float).ravel()
    hflat = np.asarray(v._z_surface, dtype=float).ravel()
    order = np.argsort(cflat)
    hs = hflat[order]
    assert np.all(np.diff(hs) >= -1e-6)        # неубывающая зависимость высоты от счёта


def test_axis_labels_rebuild_on_zscale_change(app):
    # смена Z-шкалы пересоздаёт поверхность и подписи (без накопления/утечки элементов)
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    n0 = len(v._axis_items)
    v.set_z_scale("linear")
    assert len(v._axis_items) == n0            # ровно перестроены, не задвоены
    v.set_z_scale("sqrt")
    assert len(v._axis_items) == n0
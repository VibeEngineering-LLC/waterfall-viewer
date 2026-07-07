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
    texts = [it for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    teeth = [it for it in v._axis_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(texts) >= 6                     # подписи делений + заголовки осей
    assert len(teeth) == 0                     # Задача #80: зубцы шкалы энергий убраны
    # в _axis_items теперь только подписи — ничего постороннего
    assert len(texts) == len(v._axis_items)


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


def test_floor_shift_linear_baseline_and_peaks():
    """Задача #168 (итер.3): поканальный floor. Колонка-«игла» (0-бин среди квантов 5.0):
    квант→Z=0, 0-бин→Z=0 — провал исчезает. Колонка без нулей с пиком: полка 5→0, пик 100→95."""
    from awf.ui.view3d import _floor_shift_linear
    col_holey = [0.0] + [5.0] * 9          # «дырявая»: 0-бин + уровень 1 отсчёта
    col_peak = [5.0] * 9 + [100.0]         # без нулей: полка-квант + настоящий пик
    z = np.array([col_holey, col_peak], dtype=np.float32).T   # (10, 2)
    out = _floor_shift_linear(z)
    assert (out[:, 0] == 0.0).all()                    # иглы нет: и 0, и 1 отсчёт на Z=0
    assert (out[:9, 1] == 0.0).all()
    assert np.allclose(out[9, 1], 95.0)                # пик поднялся над плоскостью


def test_floor_shift_linear_lowE_continuum_preserved():
    """Итер.3: колонка плотного НЭ-континуума (без 0-бинов, min >> квант) сдвигается лишь
    на cap (квант «дырявых» колонок), НЕ проседает к нулю; форма сохраняется. Пустая
    временная строка в хвосте не превращает все колонки в «дырявые»."""
    from awf.ui.view3d import _floor_shift_linear
    col_holey = [0.0, 5.0, 5.0, 5.0, 5.0]              # квант 5 → cap=5
    col_dense = [50.0, 60.0, 50.0, 60.0, 50.0]          # континуум, min=50 >> квант
    z = np.array([col_holey, col_dense], dtype=np.float32).T
    z = np.vstack([z, np.zeros((1, 2), dtype=np.float32)])   # пустая строка-хвост
    out = _floor_shift_linear(z)
    assert np.allclose(out[:5, 1], [45.0, 55.0, 45.0, 55.0, 45.0])   # сдвиг на cap=5
    assert (out[:, 0] == 0.0).all() and (out >= 0.0).all()


def test_floor_shift_linear_degenerate_no_crash():
    """Все-нули, пустой массив, 1-D вход (не 2-D): не крашится, проходит насквозь."""
    from awf.ui.view3d import _floor_shift_linear
    assert _floor_shift_linear(np.zeros((4, 3), dtype=np.float32)).sum() == 0
    assert _floor_shift_linear(np.zeros((0, 0), dtype=np.float32)).size == 0
    one_d = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(_floor_shift_linear(one_d), one_d)
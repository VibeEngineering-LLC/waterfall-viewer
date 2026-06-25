"""Тесты доработок Группы IV по замечаниям оператора IV-R1..IV-R5."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyqtgraph.opengl as gl
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.zscale import smooth_counts, DEFAULT_SMOOTH
from awf.ui.view3d import Waterfall3DView
from awf.ui.panels import HeatmapPanel, SlicePanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=50, t_step=2.0):
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- IV-R5: не отображать последний канал ----------
def test_trim_drops_last_channel():
    sg = _make_sg(ns=10, nc=40)
    sg2 = sg.trimmed_channels(1)
    assert sg2.n_channels == 39
    assert sg2.n_slices == 10
    assert np.array_equal(sg2.counts, sg.counts[:, :-1])


def test_trim_keeps_calibration_and_time():
    sg = _make_sg(ns=8, nc=20)
    sg2 = sg.trimmed_channels(1)
    # энергии оставшихся каналов не меняются (та же калибровка)
    assert np.array_equal(sg2.energies(), sg.energies()[:-1])
    assert np.array_equal(sg2.time_offsets_s, sg.time_offsets_s)
    assert sg2.t0_iso == sg.t0_iso and sg2.source_path == sg.source_path


def test_trim_zero_returns_self():
    sg = _make_sg(ns=5, nc=10)
    assert sg.trimmed_channels(0) is sg


def test_trim_guard_too_many():
    sg = _make_sg(ns=5, nc=2)
    with pytest.raises(ValueError):
        sg.trimmed_channels(2)   # после обрезки 0 каналов — запрещено


def test_trim_default_is_one():
    sg = _make_sg(ns=5, nc=12)
    assert sg.trimmed_channels().n_channels == 11


# ---------- IV-R4: регулируемое усреднение спектра ----------
def test_smooth_radius0_identity():
    a = np.array([0, 0, 10, 0, 0], dtype=float)
    assert np.allclose(smooth_counts(a, 0), a)
    assert DEFAULT_SMOOTH == 0


def test_smooth_moving_average_value():
    a = np.array([0, 0, 10, 0, 0], dtype=float)
    s = smooth_counts(a, 1)             # окно 3
    assert s[2] == pytest.approx(10.0 / 3.0, rel=1e-5)
    assert s[0] == pytest.approx(0.0)
    assert s.sum() == pytest.approx(10.0, rel=1e-5)   # края нулевые -> сумма сохранена


def test_smooth_reduces_variance():
    rng = np.random.RandomState(1)
    a = rng.poisson(100, size=200).astype(float)
    assert smooth_counts(a, 5).var() < a.var()


def test_smooth_2d_axis_preserves_shape():
    m = np.random.RandomState(2).poisson(50, (6, 8)).astype(float)
    assert smooth_counts(m, 2, axis=1).shape == (6, 8)
    assert smooth_counts(m, 2, axis=0).shape == (6, 8)


def test_smooth_view3d_setter(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg())
    v.set_smoothing(4)
    assert v._smooth == 4
    assert v._surface is not None


def test_smooth_heatmap_setter(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_smoothing(3)
    assert h._smooth == 3


def test_smooth_slice_setter_uses_raw_cache(app):
    s = SlicePanel()
    s.set_spectrogram(_make_sg())
    assert s._raw_spec is not None         # сырой спектр закэширован
    s.set_smoothing(2)
    assert s._smooth == 2


# ---------- IV-R2: шкала энергий в «конец по времени» + вертикальные отрезки ----------
def test_energy_scale_has_vertical_teeth(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    teeth = [it for it in v._axis_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(teeth) >= 2     # вертикальные отрезки-зубцы шкалы энергий появились


def test_energy_teeth_cleared_when_axes_hidden(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_axis_labels_visible(False)
    teeth = [it for it in v._axis_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(teeth) == 0     # при скрытых осях зубцы тоже убраны


# ---------- IV-R3: показывать только объём между плоскостями ----------
def test_single_plane_no_clip(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)      # видима только одна плоскость оси
    i0, i1, j0, j1, z_lo, z_hi, ca = v._clip_windows()
    assert (i0, i1) == (0, v._nt - 1)      # окно = весь диапазон оси времени
    assert not ca


def test_both_planes_clip_window(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)
    v.set_plane("time", 1, 0.6, True)
    i0, i1, j0, j1, z_lo, z_hi, ca = v._clip_windows()
    assert i0 > 0 and i1 < v._nt - 1       # окно уже полного диапазона
    assert i0 <= i1


def test_both_planes_rebuild_surface(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("energy", 0, 0.25, True)
    v.set_plane("energy", 1, 0.75, True)
    assert v._surface is not None          # обрезанная поверхность пересоздана без ошибок
    j0, j1 = v._clip_windows()[2], v._clip_windows()[3]
    assert j0 > 0 and j1 < v._nc - 1


def test_counts_planes_activate_height_clip(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("counts", 0, 0.2, True)
    v.set_plane("counts", 1, 0.8, True)
    *_, z_lo, z_hi, ca = v._clip_windows()
    assert ca and z_hi > z_lo              # активна высотная обрезка по счёту
    assert v._surface is not None


def test_disable_one_plane_restores_full(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)
    v.set_plane("time", 1, 0.6, True)
    v.set_plane("time", 1, 0.6, False)     # снимаем одну -> обрезка снимается
    i0, i1, *_ = v._clip_windows()
    assert (i0, i1) == (0, v._nt - 1)


# ---------- IV-R1: серая градиентная схема оформления ----------
def test_app_qss_nonempty():
    from awf.ui.style import APP_QSS
    assert isinstance(APP_QSS, str) and len(APP_QSS) > 100
    assert "qlineargradient" in APP_QSS     # градиентная схема


def test_main_window_applies_style(app):
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    assert "qlineargradient" in w.styleSheet()
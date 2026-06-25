import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyqtgraph.opengl as gl
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=60, t_step=2.0):
    counts = np.random.RandomState(3).poisson(40, size=(ns, nc)).astype(np.int64)
    counts[:, 30] += 900  # пик на канале 30 (=30 кэВ при cal [0,1])
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV -> диапазон 0..59 кэВ
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_energy_lines_stored_before_load(app):
    v = Waterfall3DView()
    v.set_energy_lines([(30.0, "#ff0000", "Cs")])
    assert len(v._energy_lines) == 1
    assert v._ray_items == []          # нет поверхности -> луч не строится


def test_rays_built_after_load_in_range_only(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    # 30 кэВ — в диапазоне; 1000 кэВ — вне (0..59) -> пропущена
    v.set_energy_lines([(30.0, "#ff0000", "A"), (1000.0, "#00ff00", "B")])
    assert len(v._ray_items) == 1
    assert all(isinstance(it, gl.GLLinePlotItem) for it in v._ray_items)


def test_ray_y_matches_energy_channel(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(nc=60), max_time=400, max_chan=512)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    pos = np.asarray(v._ray_items[0].pos)
    # центрировано: канал 30 при nc=60 -> Y = 30 - 30 = 0
    assert pos[0, 1] == pytest.approx(0.0, abs=1.0)
    # вертикальный: оба конца на одной (x,y), z от 0 до вершины
    assert pos[0, 0] == pos[1, 0] and pos[0, 1] == pos[1, 1]
    assert pos[0, 2] == pytest.approx(0.0) and pos[1, 2] > 0.0


def test_rays_cleared_on_empty_and_replaced(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_energy_lines([(10.0, "#fff", "a"), (40.0, "#0ff", "b")])
    assert len(v._ray_items) == 2
    v.set_energy_lines([])              # снятие выбора -> лучи исчезают
    assert v._ray_items == []
    v.set_energy_lines([(20.0, "#f0f", "c")])
    assert len(v._ray_items) == 1


def test_rays_rebuilt_on_zscale_change(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    n = len(v._ray_items)
    v.set_z_scale("linear")            # пересоздание поверхности перестраивает лучи
    assert len(v._ray_items) == n      # не задвоены, не потеряны
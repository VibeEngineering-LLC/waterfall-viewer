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
    assert v._ray_items == []             # Задача #85: рёберных лучей нет
    assert v._plane_nuclide_items == []   # нет поверхности/плоскостей -> маркеров нет


def test_no_edge_rays_markers_only_on_planes(app):
    # Задача #85: после загрузки и выбора линий рёберных лучей нет; маркеры появляются
    # только при видимой секущей плоскости Времени
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    assert v._ray_items == []                # рёберные лучи убраны
    assert v._plane_nuclide_items == []      # плоскость не видима -> маркеров нет
    v.set_plane("time", 0, 0.5, True)        # включаем секущую плоскость
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 1                   # маркер появился на плоскости


def test_plane_marker_y_matches_energy_channel(app):
    # Задача #85: маркер на плоскости — вертикальный отрезок на позиции энергии
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(nc=60), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    pos = np.asarray(v._plane_nuclide_items[0].pos)
    # центрировано: канал 30 при nc=60 -> Y = 30 - 30 = 0
    assert pos[0, 1] == pytest.approx(0.0, abs=1.0)
    # вертикальный: оба конца на одной (x,y), z от 0 до вершины
    assert pos[0, 0] == pos[1, 0] and pos[0, 1] == pos[1, 1]
    assert pos[0, 2] == pytest.approx(0.0) and pos[1, 2] > 0.0


def test_plane_markers_cleared_on_empty_and_replaced(app):
    # Задача #85: маркеры на видимой плоскости снимаются при пустом выборе и появляются вновь
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#fff", "a"), (40.0, "#0ff", "b")])
    assert len(v._plane_nuclide_items) == 2
    v.set_energy_lines([])              # снятие выбора -> маркеры исчезают
    assert v._plane_nuclide_items == []
    v.set_energy_lines([(20.0, "#f0f", "c")])
    assert len(v._plane_nuclide_items) == 1


def test_plane_marker_depth_tested_not_additive(app):
    # Задача #95: маркеры изотопов рисуются с depth-тестом (glOptions='opaque'), чтобы рельеф
    # перекрывал их сзади. Дефолтный 'additive' у GLLinePlotItem (depth-тест выкл) просвечивал.
    from OpenGL.GL import GL_DEPTH_TEST
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    item = v._plane_nuclide_items[0]
    opts = item._GLGraphicsItem__glOpts
    assert opts.get(GL_DEPTH_TEST) is True   # перекрывается рельефом, не «additive»-поверх


def test_plane_markers_rebuilt_on_zscale_change(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(30.0, "#ff0000", "A")])
    n = len(v._plane_nuclide_items)
    v.set_z_scale("linear")            # пересоздание поверхности перестраивает маркеры
    assert len(v._plane_nuclide_items) == n   # не задвоены, не потеряны
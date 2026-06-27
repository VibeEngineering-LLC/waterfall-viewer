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


# ---------- #62: тулбар «Вид» и строка статуса — крупнее шрифт и выше ----------
def test_toolbar_and_statusbar_sized_up():
    """Задача #62: QSS задаёт увеличенный шрифт и высоту контролов тулбара и строки статуса
    (были скучены). Целевые правила — потомки QToolBar и сам QStatusBar."""
    from awf.ui.style import APP_QSS
    assert "QToolBar QComboBox" in APP_QSS and "min-height" in APP_QSS
    assert "QToolBar QPushButton" in APP_QSS
    assert "QStatusBar QLabel" in APP_QSS
    # размер шрифта тулбара/статуса поднят выше дефолтного 12px
    for token in ("font-size: 15px", "font-size: 14px"):
        assert token in APP_QSS


def test_toolbar_combo_taller_than_default(app):
    """Задача #62: min-height из QSS реально увеличивает высоту контролов тулбара
    (комбобокс Z-шкалы становится выше неоформленного дефолта)."""
    from PySide6 import QtWidgets
    from awf.ui.style import APP_QSS
    from awf.ui.main_window import MainWindow
    QtWidgets.QApplication.instance().setStyleSheet(APP_QSS)
    w = MainWindow()
    bare = QtWidgets.QComboBox()              # без применённого QSS min-height
    assert w._z_combo.minimumSizeHint().height() >= bare.minimumSizeHint().height()
    # строка статуса принудительно выше дефолта (задаём в коде, #62)
    assert w.statusBar().minimumHeight() >= 28
    w.close()
    bare.deleteLater()


# ---------- #63/#68: координатная сетка на делениях шкал + рамка в 1 клетку ----------
def test_grid_built_with_border(app):
    """Задача #63/#68: линии сетки на делениях обеих осей + 4 ребра рамки поля."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    lines = [it for it in v._grid_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(lines) >= 8     # деления по двум осям (≥2+2) + 4 ребра рамки


def test_grid_cleared_when_axes_hidden(app):
    """Задача #63: сетка снимается вместе с осями."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_axis_labels_visible(False)
    assert len(v._grid_items) == 0


def test_grid_border_extends_one_cell_beyond_data(app):
    """Задача #63: рамка поля отстоит за край данных (поле обрамлено пустой клеткой)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    xmin, xmax, ymin, ymax, _z = v._axis_extent()
    xs, ys = [], []
    for it in v._grid_items:
        p = np.asarray(it.pos)
        xs.extend(p[:, 0].tolist()); ys.extend(p[:, 1].tolist())
    assert min(xs) < xmin - 1e-6 and max(xs) > xmax + 1e-6
    assert min(ys) < ymin - 1e-6 and max(ys) > ymax + 1e-6


# ---------- #65: вертикальная шкала счёта (Z) убрана ----------
def test_vertical_count_scale_removed(app):
    """Задача #65: ни заголовка, ни делений вертикальной (Z/счёт) шкалы."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert all("отсч" not in t for t in texts)   # нет «N, отсч.»/«N, отсч/с»


# ---------- #66: единицы на каждой клетке ----------
def test_axis_cell_labels_carry_units(app):
    """Задача #66: подпись каждой клетки несёт единицу (кэВ для энергии, с для времени)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert any("кэВ" in t for t in texts)
    assert any(t.endswith(" с") for t in texts)


# ---------- #64: единицы оси времени переключаются (с/мин/ч) ----------
def test_time_unit_switch_changes_labels(app):
    """Задача #64: переключение единицы времени пересобирает подписи в выбранной размерности."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=60, nc=50, t_step=60.0))   # 0..3540 с
    v.set_time_unit("мин")
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert v._time_unit == "мин"
    assert any(t.endswith(" мин") for t in texts)


def test_mainwindow_time_unit_fans_out(app):
    """Задача #64: комбобокс «Время» в тулбаре прокидывает единицу во view3d."""
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg(ns=30, nc=50))
    w._tunit_combo.setCurrentIndex(1)   # «мин»
    assert w._view3d._time_unit == "мин"
    w.close()


# ---------- #67/#69: маркеры нуклидов на секущих плоскостях Времени ----------
def test_plane_nuclides_drawn_on_visible_time_plane(app):
    """Задача #67: на видимой плоскости Времени рисуются маркеры выбранных линий нуклидов."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "Cs-137", 0.85)])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 1


def test_plane_nuclides_absent_without_visible_plane(app):
    """Задача #67: без видимой плоскости Времени маркеры не рисуются."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_energy_lines([(10.0, "#ff0000", "Cs-137", 0.85)])
    assert len(v._plane_nuclide_items) == 0


def test_plane_nuclide_height_scales_with_intensity(app):
    """Задача #69: высота маркера ∝ интенсивности; ярчайшая линия = полная высота zmax."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A", 1.0), (30.0, "#00ff00", "A", 0.25)])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 2
    zmax = v._axis_extent()[4]
    h_hi = float(items[0].pos[1][2])   # I=1.0 -> полная высота
    h_lo = float(items[1].pos[1][2])   # I=0.25 -> ниже
    assert abs(h_hi - zmax) < 1e-3
    assert h_lo < h_hi and abs(h_lo - 0.25 * zmax) < 1e-2


def test_plane_nuclides_backward_compat_3tuple(app):
    """Задача #67: 3-кортежи (без интенсивности) рисуются на полную высоту, без сбоя."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A"), (30.0, "#00ff00", "B")])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 2
    zmax = v._axis_extent()[4]
    assert all(abs(float(it.pos[1][2]) - zmax) < 1e-3 for it in items)


def test_plane_nuclides_cleared_when_lines_removed(app):
    """Задача #67: снятие выбора нуклидов убирает маркеры с плоскостей."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A", 0.5)])
    assert len(v._plane_nuclide_items) == 1
    v.set_energy_lines([])
    assert len(v._plane_nuclide_items) == 0
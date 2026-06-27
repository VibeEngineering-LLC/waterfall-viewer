import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyqtgraph as pg
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView
from awf.ui.zscale import apply_z_scale
from awf.ui.panels import HeatmapPanel, SlicePanel
from awf.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=40, t_step=2.0):
    counts = np.random.RandomState(1).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- #44: модель — единицы и делители ----------

def test_counts_in_unit_counts_is_identity():
    sg = _make_sg()
    assert np.array_equal(sg.counts_in_unit("counts"), sg.counts.astype(np.float64))


def test_counts_in_unit_cps_divides_per_slice():
    sg = _make_sg(t_step=2.0)
    cps = sg.counts_in_unit("cps")
    assert np.allclose(cps, sg.counts.astype(np.float64) / 2.0)


def test_counts_in_unit_cps_dead_slice_zero():
    sg = _make_sg(ns=5, nc=4)
    sg.live_time_s[2] = 0.0           # «мёртвый» срез -> деление на inf -> 0
    cps = sg.counts_in_unit("cps")
    assert np.all(cps[2] == 0.0)


def test_live_time_total_window():
    sg = _make_sg(ns=10, t_step=2.0)
    assert sg.live_time_total() == pytest.approx(20.0)
    assert sg.live_time_total(0, 3) == pytest.approx(6.0)
    assert sg.live_time_total(8, 8) == 0.0


def test_downsample_data_param_uses_given_matrix():
    sg = _make_sg(ns=12, nc=12)
    src = sg.counts_in_unit("cps")
    ds_cps, _, _ = sg.downsample(6, 6, method="max", data=src)
    ds_cnt, _, _ = sg.downsample(6, 6, method="max")
    # постоянный делитель 2.0 -> прорежённая cps = прорежённые counts / 2
    assert np.allclose(ds_cps, ds_cnt / 2.0)


def test_downsample_data_shape_mismatch_raises():
    sg = _make_sg(ns=8, nc=8)
    with pytest.raises(ValueError):
        sg.downsample(4, 4, data=np.zeros((3, 3)))


# ---------- #41: кривая спектра на секущей плоскости 3D ----------

def test_spectrum_curve_is_cyan(app):
    # 2D-спектр среза — бирюзовый (совпадает с цветом плоскости Времени)
    p = SlicePanel()
    pen = pg.mkPen(p._spectrum_curve.opts["pen"])
    r, g, b, _ = pen.color().getRgb()
    assert (r, g, b) == (51, 217, 242)
    assert pen.width() == 2


def test_view3d_time_plane_profile_is_projection(app):
    # Задача #47: на плоскости Времени — проекция-сумма рельефа (не срез по одному индексу).
    # Одна плоскость видима -> окно суммирования = вся ось времени.
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=30), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.0, True)
    line = v._planes[("time", 0)]["line"]
    assert line.visible() is True
    assert line.pos.shape == (v._nc, 3)                # точка на каждый канал
    # Задача #52: проекция = Z-шкала(Σ sum-LOD counts) = спектр окна, нормированная в высоту
    s = apply_z_scale(v._z_counts_sum.sum(axis=0), v._z_mode, gain=v._gain,
                      gamma=v._gamma, clip=v._clip).astype(np.float64)
    s = s / s.max() * v._height_scale
    assert np.allclose(line.pos[:, 2], s)              # высота = нормированная проекция суммы


def test_view3d_time_projection_between_planes(app):
    # Задача #47: обе плоскости Времени видимы -> профиль = сумма рельефа в окне МЕЖДУ ними;
    # обе плоскости показывают одну и ту же проекцию.
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=24, nc=20), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.2, True)
    v.set_plane("time", 1, 0.8, True)
    i0, i1 = v._clip_windows()[0:2]
    s = apply_z_scale(v._z_counts_sum[i0:i1 + 1, :].sum(axis=0), v._z_mode, gain=v._gain,
                      gamma=v._gamma, clip=v._clip).astype(np.float64)
    s = s / s.max() * v._height_scale
    assert np.allclose(v._planes[("time", 0)]["line"].pos[:, 2], s)
    assert np.allclose(v._planes[("time", 1)]["line"].pos[:, 2], s)


def test_view3d_profile_always_on_top(app):
    # Профиль рисуется поверх непрозрачного рельефа: depth-тест выключен (иначе тонет).
    from awf.ui.view3d import _PROFILE_GL
    from OpenGL.GL import GL_DEPTH_TEST
    assert _PROFILE_GL[GL_DEPTH_TEST] is False
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=15, nc=25))
    line = v._planes[("energy", 1)]["line"]
    opts = getattr(line, "_GLGraphicsItem__glOpts", {})
    assert opts.get(GL_DEPTH_TEST) is False


def test_view3d_counts_plane_has_no_profile(app):
    # Горизонтальная плоскость уровня Отсчётов 1D-профиля не имеет — кривая скрыта.
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=15, nc=25))
    v.set_plane("counts", 0, 0.5, True)
    assert v._planes[("counts", 0)]["line"].visible() is False


def test_view3d_profile_depth_value_above_surface(app):
    # Задача #49: профиль имеет больший depthValue, чем поверхность -> рисуется ПОСЛЕ неё
    # (поверхность пересоздаётся каждый рендер и иначе перекрыла бы линию снаружи плоскостей).
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=15, nc=25))
    line = v._planes[("time", 0)]["line"]
    assert line.depthValue() == 10
    assert line.depthValue() > v._surface.depthValue()


def test_view3d_profile_matches_window_spectrum(app):
    # Задача #52: профиль на плоскости = спектр верхнего-правого окна (sg.sum_spectrum по окну).
    # Устойчивый пик (бо́льший ИНТЕГРАЛ) выше короткого транзиента-башни — как в окне; старый
    # max-метод (#50) сажал максимум на башню. Без прорежки каналов профиль поканально = sum_spectrum.
    ns, nc = 40, 100
    counts = np.random.RandomState(3).poisson(5, size=(ns, nc)).astype(np.int64)
    counts[:, 70] += 30                         # устойчивый пик: интеграл 30*40=1200
    counts[ns // 2:ns // 2 + 2, 20] += 400      # транзиент-башня: интеграл 800, но высота 405
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64)
    sg = Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                     real_time_s=np.full(ns, 1.0), live_time_s=np.full(ns, 1.0))
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.0, True)
    prof = v._planes[("time", 0)]["line"].pos[:, 2]
    win = np.asarray(sg.sum_spectrum(0, ns), dtype=np.float64)   # спектр окна (полное разрешение)
    assert int(np.argmax(prof)) == int(np.argmax(win)) == 70     # плоскость = окно (sum), не max
    assert prof[70] > prof[20]                                   # устойчивый пик выше транзиента


def _profile_cps_check(sg, counts, live):
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)   # nc<512, ns<400 -> без прорежки
    v.set_plane("time", 0, 0.0, True)
    prof = v._planes[("time", 0)]["line"].pos[:, 2]
    assert int(np.argmax(counts.sum(axis=0))) == 30                    # по counts выше пик B
    assert int(np.argmax((counts / live[:, None]).sum(axis=0))) == 10  # по cps выше пик A
    assert int(np.argmax(prof)) == 10                                  # профиль идёт за cps


def test_view3d_profile_is_cps_not_counts(app):
    # Задача #53: голубая линия = сумма CPS на срезе (counts/live_time), не сумма counts.
    # Дефолт-единицы рельефа теперь cps -> два пика с разным live_time меняют местами «кто выше».
    ns, nc = 20, 50
    counts = np.zeros((ns, nc), dtype=np.int64)
    live = np.empty(ns, dtype=np.float64)
    live[: ns // 2] = 0.5            # «быстрые» срезы -> высокий cps
    live[ns // 2:] = 4.0            # «медленные» срезы -> низкий cps
    counts[: ns // 2, 10] = 100     # пик A: counts-Σ=1000, cps-Σ=2000
    counts[ns // 2:, 30] = 200      # пик B: counts-Σ=2000, cps-Σ=500
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64)
    sg = Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                     real_time_s=live.copy(), live_time_s=live.copy())
    _profile_cps_check(sg, counts, live)


# ---------- #43: лог/лин шкала спектра ----------

def test_spectrum_log_toggle(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg())
    p.set_spectrum_log(True)
    assert p._spec_log is True
    assert p._spectrum_plot.getAxis("left").logMode is True
    assert p._spectrum_plot.getAxis("bottom").logMode is False
    p.set_spectrum_log(False)
    assert p._spectrum_plot.getAxis("left").logMode is False


# ---------- #42: маркеры сечений на нижнем графике ----------

def test_series_section_markers_time_only(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([10.0, 40.0], [None, None])    # две плоскости Времени -> 2 метки
    assert len(p._series_section_items) == 2
    assert all(it.angle == 90 for it in p._series_section_items)   # вертикали по оси времени


def test_series_section_markers_none_when_no_time(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([None, None], [5.0, 20.0])     # только энергии -> нет временных меток
    assert len(p._series_section_items) == 0


def test_series_section_markers_cleared_on_new_file(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([10.0, None], [None, None])
    assert len(p._series_section_items) == 1
    p.set_spectrogram(_make_sg(ns=20, nc=30))      # новый файл -> метки сброшены
    assert len(p._series_section_items) == 0


# ---------- #44: SlicePanel единицы ----------

def test_slice_spectrum_cps_scaled(app):
    sg = _make_sg(ns=10, nc=20, t_step=2.0)
    p = SlicePanel()
    p.set_spectrogram(sg)                            # интегральный спектр, lt_total = 20 с
    p.set_unit_mode("cps")
    y = p._spectrum_curve.getData()[1]
    assert np.allclose(y, sg.total_spectrum() / 20.0)
    assert "/с" in p._spectrum_plot.getAxis("left").labelText


def test_slice_series_cps_scaled(app):
    sg = _make_sg(ns=10, nc=20, t_step=2.0)
    p = SlicePanel()
    p.set_spectrogram(sg)
    p.set_unit_mode("cps")
    y = p._series_curve.getData()[1]
    assert np.allclose(y, sg.band_time_series(0, sg.n_channels) / 2.0)


def test_slice_unit_back_to_counts(app):
    sg = _make_sg(ns=10, nc=20)
    p = SlicePanel()
    p.set_spectrogram(sg)
    p.set_unit_mode("cps")
    p.set_unit_mode("counts")
    y = p._spectrum_curve.getData()[1]
    assert np.allclose(y, sg.total_spectrum().astype(np.float64))
    assert p._spectrum_plot.getAxis("left").labelText == "Отсчёты"


# ---------- #44: HeatmapPanel единицы ----------

def test_heatmap_cps_halves_small_map(app):
    sg = _make_sg(ns=20, nc=30, t_step=2.0)         # мельче cap -> без прорежения
    h = HeatmapPanel()
    h.set_spectrogram(sg)
    h.set_unit_mode("counts")                       # Задача #53: дефолт cps — берём counts-базу
    cnt_max = float(h._disp_counts.max())
    h.set_unit_mode("cps")
    assert h._unit == "cps"
    assert float(h._disp_counts.max()) == pytest.approx(cnt_max / 2.0, rel=1e-5)


# ---------- #44: view3d единицы ----------

def test_view3d_unit_mode_smoke(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=30), max_time=400, max_chan=512)
    v.set_unit_mode("cps")
    assert v._unit == "cps"
    v.set_plane("counts", 0, 0.5, True)
    _value, unit = v.plane_value("counts", 0.5)
    assert "отсч/с" in unit


# ---------- #44: глобальный веер из тулбара главного окна ----------

def test_mainwindow_unit_combo_fans_out(app):
    w = MainWindow()
    sg = _make_sg(ns=20, nc=30)
    w._view3d.set_spectrogram(sg)
    w._heatmap.set_spectrogram(sg)
    w._slices.set_spectrogram(sg)
    assert w._unit_combo.itemData(0) == "counts"
    assert w._unit_combo.itemData(1) == "cps"
    assert w._unit_combo.currentIndex() == 1        # Задача #53: дефолт — cps
    w._unit_combo.setCurrentIndex(0)                # -> counts веером на все панели
    assert w._view3d._unit == "counts"
    assert w._heatmap._unit == "counts"
    assert w._slices._unit == "counts"
    w._unit_combo.setCurrentIndex(1)                # -> cps веером обратно
    assert w._view3d._unit == "cps"
    assert w._heatmap._unit == "cps"
    assert w._slices._unit == "cps"
    w.close()


# ---------- #45/#46: контраст профиля и затенение рельефа ----------

def test_view3d_profile_matches_frame(app):
    # Задача #48: профиль красится цветом рамки/оси (time=бирюза, energy=пурпур), не белым.
    from awf.ui.view3d import _AXIS_RGB
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=15, nc=25))
    v.set_plane("time", 0, 0.3, True)
    col = np.asarray(v._planes[("time", 0)]["line"].color, dtype=float).ravel()
    assert np.allclose(col[:3], _AXIS_RGB["time"])
    v.set_plane("energy", 0, 0.4, True)
    cole = np.asarray(v._planes[("energy", 0)]["line"].color, dtype=float).ravel()
    assert np.allclose(cole[:3], _AXIS_RGB["energy"])


def test_surface_shading_none_when_zero():
    from awf.ui.view3d import _surface_shading
    z = np.random.RandomState(0).random((10, 12)).astype(np.float32)
    assert _surface_shading(z, 0.0) is None     # intensity 0 -> без затенения


def test_surface_shading_in_range():
    from awf.ui.view3d import _surface_shading, _SHADE_AMBIENT
    z = (np.random.RandomState(0).random((10, 12)) * 5).astype(np.float32)
    sh = _surface_shading(z, 1.0)
    assert sh.shape == z.shape
    assert sh.min() >= _SHADE_AMBIENT - 1e-6 and sh.max() <= 1.0 + 1e-6


def test_view3d_light_darkens_colors(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=30))
    c0 = v._colors_full.copy()
    v.set_light_intensity(0.8)
    c1 = v._colors_full
    assert not np.allclose(c0[..., :3], c1[..., :3])    # рельеф затенён
    assert np.allclose(c0[..., 3], c1[..., 3])           # alpha не трогаем
    assert (c1[..., :3] <= c0[..., :3] + 1e-6).all()     # затенение только затемняет


def test_view3d_light_zero_restores(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=15, nc=25))
    c0 = v._colors_full.copy()
    v.set_light_intensity(0.5)
    v.set_light_intensity(0.0)
    assert np.allclose(c0, v._colors_full)               # 0 == как без теней


def test_mainwindow_light_slider(app):
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg(ns=15, nc=25))
    assert w._light_slider.value() == 0                  # по умолчанию тени выключены
    w._adjust._global.setChecked(True)                   # Задача #60: дефолт выкл — включаем
    w._adjust.rows["light"].set_on(True)
    w._light_slider.setValue(60)
    assert w._view3d._light == pytest.approx(0.6)
    w.close()


# ---------- #51: кнопка сброса настроек отображения ----------

def _assert_display_defaults(w):
    assert w._z_combo.currentIndex() == 2
    assert w._cmap_combo.currentIndex() == 0
    assert w._unit_combo.currentIndex() == 1     # Задача #53: дефолт — cps
    assert w._axes_check.isChecked() is True
    assert w._hl_check.isChecked() is False
    assert w._gain_slider.value() == 100
    assert w._gamma_slider.value() == 100
    assert w._clip_slider.value() == 100
    assert w._smooth_slider.value() == 0
    assert w._light_slider.value() == 0
    assert w._view3d._unit == "cps"              # Задача #53: дефолт — cps
    assert w._view3d._light == pytest.approx(0.0)
    assert w._view3d._gain == pytest.approx(1.0)


def test_display_reset_restores_defaults(app):
    w = MainWindow()
    sg = _make_sg(ns=20, nc=30)
    w._view3d.set_spectrogram(sg)
    w._heatmap.set_spectrogram(sg)
    w._slices.set_spectrogram(sg)
    w._z_combo.setCurrentIndex(0)          # увести все контролы с умолчаний
    w._cmap_combo.setCurrentIndex(1)
    w._unit_combo.setCurrentIndex(0)       # Задача #53: дефолт cps -> уводим в counts
    w._axes_check.setChecked(False)
    w._hl_check.setChecked(True)
    w._gain_slider.setValue(250)
    w._gamma_slider.setValue(150)
    w._clip_slider.setValue(90)
    w._smooth_slider.setValue(7)
    w._light_slider.setValue(60)
    w._on_reset_display()
    _assert_display_defaults(w)             # дефолты доехали и до контролов, и до панелей
    w.close()
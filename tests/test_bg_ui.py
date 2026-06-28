"""UI-тесты фона/вычитания (Задача #96): наложение кривой фона на спектр среза (SlicePanel)
и вычет фона из всего водопада (MainWindow -> 3D/2D/срезы), аналитика — на исходных данных."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import SlicePanel
from awf.ui.main_window import MainWindow
from awf.ui.background_dialog import BackgroundDialog


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=12, nc=20, t_step=2.0):
    counts = np.random.RandomState(7).poisson(40, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])               # E(ch) = ch кэВ
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def _empty(curve) -> bool:
    x, _ = curve.getData()
    return x is None or len(x) == 0


# ---------- SlicePanel: наложение кривой фона ----------

def test_slice_bg_hidden_until_overlay_enabled(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=10))
    p.set_background(np.ones(10))
    assert _empty(p._bg_curve)                         # фон задан, наложение выкл -> пусто
    p.set_background_overlay(True)
    x, _ = p._bg_curve.getData()
    assert x is not None and len(x) == 10              # появилась кривая по всем каналам


def test_slice_bg_unit_scaling(app):
    # live_time на срез = 2 с -> в counts кривая фона = bg_cps * 2, в cps = bg_cps
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=3, nc=8))
    p.show_time_slice(0)                                # окно среза, lt_total = 2 c
    p.set_background(np.ones(8))
    p.set_background_overlay(True)
    _, y_cps = p._bg_curve.getData()                   # дефолт — cps
    assert np.allclose(y_cps, 1.0)
    p.set_unit_mode("counts")
    _, y_counts = p._bg_curve.getData()
    assert np.allclose(y_counts, 2.0)                  # bg_cps * живое_время_окна (2 c)


def test_slice_bg_cleared_on_none(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    p.set_background(np.ones(8))
    p.set_background_overlay(True)
    assert not _empty(p._bg_curve)
    p.set_background(None)                              # снять фон -> кривая пуста даже при overlay
    assert _empty(p._bg_curve)


def test_slice_bg_size_mismatch_not_drawn(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    p.set_background(np.ones(5))                        # длина != числу каналов
    p.set_background_overlay(True)
    assert _empty(p._bg_curve)                          # несогласованный фон не рисуется


# ---------- MainWindow: вычет фона из всего водопада ----------

def test_mainwindow_subtract_reduces_waterfall(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=12, nc=20))
    orig = float(w._sg.counts.sum())
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._on_bg_subtract_toggled(True)                    # вычет включён
    assert float(w._slices._sg.counts.sum()) < orig    # 3D/2D/срезы — на вычтенных данных
    assert float(w._heatmap._sg.counts.sum()) < orig
    assert (w._slices._sg.counts >= 0).all()           # отрицательные клипнуты к 0
    w._on_bg_subtract_toggled(False)                   # выключение -> исходный водопад
    assert float(w._slices._sg.counts.sum()) == pytest.approx(orig)
    w.close()


def test_mainwindow_analytics_stays_on_original(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=10, nc=16))
    orig = float(w._sg.counts.sum())
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._on_bg_subtract_toggled(True)
    assert float(w._sg.counts.sum()) == orig           # _sg (источник аналитики) не меняется
    w.close()


def test_mainwindow_reset_background_on_reload(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=10, nc=16))
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._act_bg_overlay.setEnabled(True)
    w._act_bg_subtract.setEnabled(True)
    w._on_bg_subtract_toggled(True)
    w._on_loaded(_make_sg(ns=8, nc=16))                # новый файл -> фон сброшен
    assert w._bg_cps is None
    assert w._bg_subtract is False and w._bg_overlay is False
    assert not w._act_bg_overlay.isEnabled()
    assert not w._act_bg_subtract.isEnabled()
    w.close()


# ---------- BackgroundDialog: сбор выбора источника ----------

def test_bg_dialog_range_result(app):
    dlg = BackgroundDialog(20, time_offsets=np.arange(20.0))
    dlg._lo.setValue(3); dlg._hi.setValue(10)
    dlg._on_accept()                                   # дефолт — радиокнопка диапазона
    assert dlg.result_spec() == ("range", 3, 10)


def test_bg_dialog_file_result(app):
    dlg = BackgroundDialog(20)
    dlg._path = "C:/tmp/bg.aswf"                        # имитируем выбор файла из «Обзор…»
    dlg._rb_file.setChecked(True)
    dlg._on_accept()
    assert dlg.result_spec() == ("file", "C:/tmp/bg.aswf")


def test_bg_dialog_default_full_range(app):
    dlg = BackgroundDialog(15)
    assert dlg._lo.value() == 0 and dlg._hi.value() == 15   # по умолчанию весь диапазон
    assert dlg.result_spec() is None                        # до accept — None


# ---------- #99: на лог-шкале нулевой фон ВЭ-хвоста маскируется (нет «частокола») ----------

def test_slice_bg_log_masks_nonpositive(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    bg = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])  # нули = пустой ВЭ-хвост фона
    p.set_background(bg)
    p.set_background_overlay(True)
    p.set_spectrum_log(True)                                 # лог Y -> нули фона в nan (#99)
    _, y = p._bg_curve.getData()
    y = np.asarray(y, dtype=np.float64)
    assert np.isnan(y[1]) and np.isnan(y[3]) and np.isnan(y[4])   # разрывы там, где фон == 0
    assert np.isfinite(y[0]) and np.isfinite(y[2]) and np.isfinite(y[5])
    p.set_spectrum_log(False)                                # линейная -> маскирования нет
    _, y2 = p._bg_curve.getData()
    assert np.isfinite(np.asarray(y2, dtype=np.float64)).all()


# ---------- #100: сброс зума графиков среза и времени ----------

def test_slice_has_reset_zoom_button(app):
    p = SlicePanel()
    assert hasattr(p, "_reset_zoom_btn")
    assert p._reset_zoom_btn.text() == "Сброс зума"


def test_slice_reset_zoom_restores_full_view(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=10, nc=16, t_step=2.0))    # энергии 0..15, время 0..18
    sp_vb = p._spectrum_plot.getViewBox()
    sr_vb = p._series_plot.getViewBox()
    sp_vb.setXRange(5.0, 6.0, padding=0)                     # зум обоих графиков в узкое окно
    sr_vb.setXRange(3.0, 4.0, padding=0)
    p.reset_zoom()
    app.processEvents()
    (sx0, sx1), _ = sp_vb.viewRange()
    (tx0, tx1), _ = sr_vb.viewRange()
    assert sx1 - sx0 > 1.5 and tx1 - tx0 > 1.5              # вид шире узкого зума 1.0


# ---------- #98: «фоновая простыня» на 3D-водопаде ----------

def test_view3d_bg_sheet_toggle(app):
    from awf.ui.view3d import Waterfall3DView
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=12, nc=20))
    v.set_background_sheet(np.ones(20))
    assert v._bg_sheet is None                          # фон задан, наложение выкл -> простыни нет
    v.set_background_sheet_visible(True)
    assert v._bg_sheet is not None                      # включили наложение -> простыня построена
    v.set_background_sheet_visible(False)
    assert v._bg_sheet is None                          # выключили -> простыня снята


# ---------- #101: нижняя граница оси Y графика спектра зафиксирована ----------

def test_slice_spectrum_y_floor_linear(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=6, nc=12))
    ymin, _ = p._spectrum_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == 0.0                                  # лин.: 0 прибит к низу (счёт ≥ 0)


def test_slice_spectrum_y_floor_log(app):
    p = SlicePanel()
    sg = _make_sg(ns=6, nc=12)
    p.set_spectrogram(sg)
    p.set_spectrum_log(True)
    _, s, lt = p._raw_spec
    disp = p._spec_to_unit(s, lt)
    expected = float(np.log10(disp[disp > 0].min()))
    ymin, _ = p._spectrum_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == pytest.approx(expected)              # лог: пол = log10(min>0)


# ---------- #102: окно «Цветовая палитра» с превью-градиентами ----------

def test_palette_dialog_lists_all_and_selects(app):
    from awf.ui.palette_dialog import PaletteDialog
    from awf.ui.colormaps import COLORMAPS
    dlg = PaletteDialog("insight")
    assert len(dlg._rows) == len(COLORMAPS)             # строка на каждую палитру (15)
    assert dlg._rows["insight"].property("selected") is True
    got = []
    dlg.selected.connect(got.append)
    dlg._on_pick("turbo")                               # выбор палитры кликом
    assert dlg.selected_key() == "turbo" and got == ["turbo"]
    assert dlg._rows["turbo"].property("selected") is True
    assert dlg._rows["insight"].property("selected") is False
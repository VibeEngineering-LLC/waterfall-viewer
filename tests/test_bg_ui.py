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
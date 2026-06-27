"""Тесты панели регулировок-«рукояток» (Задача #55): QSlider-совместимый API Knob,
bypass-семантика KnobRow/AdjustPanel и проводка в MainWindow (выкл → стандартный вид)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.knobs import Knob, KnobRow, AdjustPanel
from awf.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=20, nc=30, t_step=2.0):
    counts = np.random.RandomState(2).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- Knob: QSlider-совместимый API ----------

def test_knob_qslider_api(app):
    k = Knob(20, 500, 100)
    assert (k.minimum(), k.maximum(), k.value()) == (20, 500, 100)
    seen = []
    k.valueChanged.connect(seen.append)
    k.setValue(250)
    assert k.value() == 250 and seen == [250]
    k.setValue(9999)                       # кламп к максимуму
    assert k.value() == 500
    k.setValue(500)                        # без изменения — сигнал не повторяется
    assert seen == [250, 500]
    k.setRange(0, 15)                      # сужение диапазона переклампит значение
    assert k.value() == 15


def test_knob_is_horizontal_fader(app):
    """Задача #58: регулятор — горизонтальный движок (ширина > высоты), одной колонкой."""
    k = Knob(0, 100, 50)
    assert k.sizeHint().width() > k.sizeHint().height()


# ---------- KnobRow: per-row bypass ----------

def test_knobrow_bypass_preserves_position(app):
    row = KnobRow("gain", "Усиление", 20, 500, 100, lambda v: f"{v}")
    row.set_on(True)                             # Задача #60: ряд по умолчанию выкл — включаем
    row.setValue(250)
    assert row.effective_value() == 250          # включён → значение ручки
    row.set_on(False)
    assert row.is_on() is False
    assert row.effective_value() == 100          # выключен → дефолт (bypass)
    assert row.value() == 250                    # позиция ручки сохранена
    row.set_on(True)
    assert row.effective_value() == 250          # включение восстанавливает значение
    row.reset()
    assert row.value() == 100 and row.effective_value() == 100


# ---------- AdjustPanel: глобальный bypass и сброс ----------

_DEFAULTS = {"gain": 100, "gamma": 100, "clip": 100, "smooth": 0, "light": 0, "tbin": 100}


def test_panel_defaults_all_off(app):
    """Задача #60: по умолчанию все ряды — ВЫКЛ (эталон-скриншот).
    Задача #91: общего выключателя больше нет — bypass по каждому ряду отдельно."""
    p = AdjustPanel()
    assert not hasattr(p, "_global")             # #91: мастер-выключатель убран
    for r in p.rows.values():
        assert r.is_on() is False
        assert r._chk.text() == "выкл"
        assert r.knob.isEnabled() is False
    assert p.values() == _DEFAULTS               # всё выкл → нейтральные дефолты (bypass)


def test_adjustpanel_per_row_bypass(app):
    # Задача #91: общий выключатель убран — bypass только по каждому ряду отдельно.
    p = AdjustPanel()
    assert p.values() == _DEFAULTS               # старт — всё выкл → дефолты (bypass, #60)
    assert all(not r.is_on() for r in p.rows.values())   # #60: все ряды выкл
    p.rows["gain"].set_on(True)                  # включаем два ряда — без всякого мастер-гейта
    p.rows["smooth"].set_on(True)
    p.rows["gain"].setValue(250)
    p.rows["smooth"].setValue(7)
    assert p.values()["gain"] == 250 and p.values()["smooth"] == 7
    p.rows["gain"].set_on(False)                 # один ряд выкл → его дефолт, второй держит значение
    assert p.values()["gain"] == 100 and p.values()["smooth"] == 7
    assert p.rows["gain"].value() == 250         # позиция ручки сохранена
    p.rows["gain"].set_on(True)                  # назад — снова 250
    assert p.values()["gain"] == 250 and p.values()["smooth"] == 7


def test_adjustpanel_reset_all(app):
    p = AdjustPanel()
    p.rows["gain"].set_on(True)                   # #91: включаем ряды напрямую (мастера нет)
    p.rows["gain"].setValue(300)
    p.rows["light"].set_on(True)
    p.rows["light"].setValue(40)
    p.reset_all()
    assert all(not r.is_on() for r in p.rows.values())   # сброс → все ряды выкл
    assert all(not r.knob.isEnabled() for r in p.rows.values())  # #91: ручки погашены
    assert p.values() == _DEFAULTS


# ---------- MainWindow: панель в доке + bypass возвращает стандартный вид ----------

def test_mainwindow_adjust_dock_exists(app):
    w = MainWindow()
    names = {d.objectName() for d in w.findChildren(QtWidgets.QDockWidget)}
    assert "dock_adjust" in names
    assert isinstance(w._adjust, AdjustPanel)
    w.close()


def test_mainwindow_bypass_reverts_gain(app):
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg())
    w._adjust.rows["gain"].set_on(True)          # #91: ряд включаем напрямую (мастера нет)
    w._adjust.rows["gain"].setValue(250)         # усиление 2.5×
    assert w._view3d._gain == pytest.approx(2.5)
    w._adjust.rows["gain"].set_on(False)         # выкл ряда → стандартный вид (gain 1.0)
    assert w._view3d._gain == pytest.approx(1.0)
    assert w._adjust.rows["gain"].value() == 250  # позиция ручки сохранена
    w._adjust.rows["gain"].set_on(True)          # назад — снова 2.5×
    assert w._view3d._gain == pytest.approx(2.5)
    w.close()


def test_mainwindow_rows_off_revert_all(app):
    # Задача #91: выключение рядов (по отдельности) возвращает стандартный вид —
    # общего выключателя больше нет, каждый ряд сам себе bypass.
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg())
    w._adjust.rows["gain"].set_on(True)
    w._adjust.rows["light"].set_on(True)
    w._adjust.rows["gain"].setValue(250)
    w._adjust.rows["light"].setValue(60)
    assert w._view3d._gain == pytest.approx(2.5)
    assert w._view3d._light == pytest.approx(0.6)
    w._adjust.rows["gain"].set_on(False)         # ряды выкл → всё к дефолтам
    w._adjust.rows["light"].set_on(False)
    assert w._view3d._gain == pytest.approx(1.0)
    assert w._view3d._light == pytest.approx(0.0)
    w.close()


def test_mainwindow_tbin_changes_time_bins(app):
    """Задача #56: ручка «Окно t» меняет число временны́х бинов (max_time) 3D-водопада;
    bypass ряда → стандартная ширина (max_time=400), позиция ручки сохраняется."""
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg(ns=120, nc=30))   # ns>бинов — LOD реально режет
    assert w._view3d._max_time == 400                    # дефолт — стандартная ширина
    w._adjust.rows["tbin"].set_on(True)                  # #91: ряд включаем напрямую
    w._adjust.rows["tbin"].setValue(200)                 # ширина ×2 → меньше бинов («сжатие»)
    assert w._view3d._max_time == 200
    w._adjust.rows["tbin"].setValue(50)                  # ширина ×0.5 → больше бинов («растяжение»)
    assert w._view3d._max_time == 800
    w._adjust.rows["tbin"].set_on(False)                 # bypass ряда → стандартный max_time
    assert w._view3d._max_time == 400
    assert w._adjust.rows["tbin"].value() == 50          # позиция ручки сохранена
    w.close()


def test_mainwindow_tbin_persists_across_load(app):
    """Задача #56: ширина выборки переживает загрузку нового файла — аргументный
    set_spectrogram(sg) не сбрасывает max_time (None-дефолт сохраняет текущее)."""
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg(ns=120))
    w._adjust.rows["tbin"].set_on(True)                  # #91: ряд включаем напрямую
    w._adjust.rows["tbin"].setValue(200)                 # max_time=200
    assert w._view3d._max_time == 200
    w._view3d.set_spectrogram(_make_sg(ns=140))          # новый файл, аргументный вызов как в _on_loaded
    assert w._view3d._max_time == 200                    # ширина выборки сохранилась
    w.close()

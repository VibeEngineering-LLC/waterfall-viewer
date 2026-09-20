"""Задача #UI-243: фактические дата/время на шкалах времени (2D-карта, нижний график, 3D)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import HeatmapPanel, SlicePanel
from awf.ui.view3d import Waterfall3DView
from awf.ui.timefmt import parse_t0, clock_label, date_label


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(t0_iso="2026-06-25T02:47:43Z", ns=30, nc=50, t_step=60.0):
    """Спектрограмма с шагом по времени t_step секунд; t0_iso=None — файл без метки старта."""
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=Calibration(coeffs=[0.0, 1.0]),
                       time_offsets_s=t, real_time_s=np.full(ns, t_step),
                       live_time_s=np.full(ns, t_step), t0_iso=t0_iso)


def test_timefmt_basics():
    """timefmt: разбор ISO-метки, часы от смещения, дата для подписи оси."""
    t0 = parse_t0("2026-06-25T02:47:43Z")
    assert t0 is not None
    assert clock_label(t0, 0) == "02:47:43"
    assert clock_label(t0, 3725) == "03:49:48"
    assert clock_label(t0, 3725.4, True) == "03:49:48.400"
    assert date_label(t0) == "25.06.2026"
    assert parse_t0(None) is None
    assert parse_t0("") is None
    assert parse_t0("не время") is None
    assert clock_label(None, 5) == ""
    assert date_label(None) == ""


def test_heatmap_axis_switches_to_clock(app):
    """2D-карта: в абсолютном режиме тики Y — ЧЧ:ММ:СС, дата уходит в подпись оси."""
    hm = HeatmapPanel()
    hm.set_spectrogram(_make_sg())
    # относительный режим
    assert hm._time_axis.tickStrings([0, 10, 20], 1, 1) == ["0", "600", "1200"]
    assert "Время, с" in hm._plot.getAxis("left").labelText
    # абсолютный режим
    hm.set_absolute_time(True)
    assert hm._time_axis.tickStrings([0, 10, 20], 1, 1) == ["02:47:43", "02:57:43", "03:07:43"]
    assert "25.06.2026" in hm._plot.getAxis("left").labelText
    # возврат к относительному
    hm.set_absolute_time(False)
    assert hm._time_axis.tickStrings([0, 10, 20], 1, 1) == ["0", "600", "1200"]


def test_series_axis_switches_to_clock(app):
    """Нижний график: X-ось в абсолютном режиме печатает фактическое время."""
    sp = SlicePanel()
    sp.set_spectrogram(_make_sg())
    sp.set_absolute_time(True)
    assert sp._series_time_axis.tickStrings([0.0, 600.0, 1200.0], 1, 100) == ["02:47:43", "02:57:43", "03:07:43"]
    assert "25.06.2026" in sp._series_plot.getAxis("bottom").labelText


def test_absolute_mode_inert_without_t0(app):
    """Без t0 в файле абсолютный режим не включается — шкала остаётся относительной."""
    hm = HeatmapPanel()
    hm.set_spectrogram(_make_sg(t0_iso=None))
    hm.set_absolute_time(True)
    assert hm._time_axis.tickStrings([0, 10], 1, 1) == ["0", "600"]
    label = hm._plot.getAxis("left").labelText
    assert "Время, с" in label
    assert "20" not in label


def test_view3d_axis_labels_switch_to_clock(app):
    """3D: подписи делений оси времени переключаются на фактическое время."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=40, nc=60))
    texts = [getattr(i, "text", "") for i in v._axis_items]
    assert any(t.endswith(" с") for t in texts)
    v.set_absolute_time(True)
    texts2 = [getattr(i, "text", "") for i in v._axis_items]
    assert any(t.count(":") == 2 for t in texts2)
    assert any(t.endswith(" кэВ") for t in texts2)

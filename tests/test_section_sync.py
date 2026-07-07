import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView
from awf.ui.panels import HeatmapPanel, SlicePanel
from awf.ui.main_window import MainWindow, SETTINGS_ORG, SETTINGS_APP


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=40, t_step=2.0):
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- #38/#39: active_plane_values ----------

def test_active_plane_values_reports_visible_only(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=40), max_time=400, max_chan=512)
    v.set_plane("time", 0, 0.5, True)
    v.set_plane("energy", 1, 0.25, True)
    state = v.active_plane_values()
    assert state["time"][0] is not None and state["time"][1] is None
    assert state["energy"][1] is not None and state["energy"][0] is None
    assert 0.0 <= state["time"][0] <= 58.0      # реальные секунды
    assert 0.0 <= state["energy"][1] <= 39.0    # реальные кэВ
    # скрытие плоскости -> снова None
    v.set_plane("time", 0, 0.5, False)
    assert v.active_plane_values()["time"][0] is None


# ---------- #38: SlicePanel.sync_sections ----------

def test_slice_sync_single_time_plane_shows_slice(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([10.0, None], [None, None])   # одна плоскость времени -> спектр среза
    assert "Срез времени" in p._header.text()


def test_slice_sync_both_time_planes_show_roi(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([10.0, 40.0], [None, None])   # обе плоскости времени -> ROI окна времени
    assert "Выборка" in p._header.text()


def test_slice_sync_both_energy_planes_set_window(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=30, nc=40))
    p.sync_sections([None, None], [5.0, 20.0])    # обе плоскости энергии -> энергоокно
    assert p._ewin_active == (5.0, 20.0)


def test_slice_sync_noop_without_spectrogram(app):
    p = SlicePanel()
    p.sync_sections([10.0, 40.0], [5.0, 20.0])    # нет данных -> без исключения, без эффекта
    assert p._sg is None


# ---------- #39: HeatmapPanel.set_section_markers ----------

def test_heatmap_markers_count_matches_visible(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=30, nc=40))
    h.set_section_markers([10.0, None], [5.0, 20.0])   # 1 время + 2 энергии = 3 линии
    assert len(h._section_items) == 3
    h.set_section_markers([None, None], [None, None])  # всё скрыто -> очистка
    assert len(h._section_items) == 0


def test_heatmap_marker_angles(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=30, nc=40))
    h.set_section_markers([10.0, None], [5.0, None])
    angles = sorted(it.angle for it in h._section_items)
    assert angles == [0, 90]   # время = горизонталь (0), энергия = вертикаль (90)


def test_heatmap_markers_cleared_on_new_file(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=30, nc=40))
    h.set_section_markers([10.0, None], [5.0, None])
    assert len(h._section_items) == 2
    h.set_spectrogram(_make_sg(ns=20, nc=30))          # новый файл -> маркеры сброшены
    assert len(h._section_items) == 0


# ---------- #40: запоминание раскладки (QSettings) ----------

def test_layout_persistence_roundtrip(app, tmp_path):
    ini = str(tmp_path / "layout.ini")
    w = MainWindow()
    w._settings = QtCore.QSettings(ini, QtCore.QSettings.IniFormat)  # изолируем от реестра
    w.resize(900, 640)
    w.close()                                          # closeEvent -> сохранить раскладку
    s2 = QtCore.QSettings(ini, QtCore.QSettings.IniFormat)
    geo = s2.value("geometry")
    state = s2.value("windowState")
    assert geo is not None and len(geo) > 0
    assert state is not None and len(state) > 0
    # новое окно восстанавливает сохранённую геометрию без ошибок
    w2 = MainWindow()
    assert w2.restoreGeometry(geo) is True


def test_settings_names_exported():
    assert SETTINGS_ORG and SETTINGS_APP   # константы доступны для main()/тестов
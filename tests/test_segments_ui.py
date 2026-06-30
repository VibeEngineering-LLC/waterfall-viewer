"""UI-тесты Задача #131: проводка панели «Сегментация по времени» в MainWindow.

Проверяется, что MainWindow создаёт _segments_panel / _segments_dock / _act_segments,
что действие меню запускает сегментацию реальной записи, дерево заполняется ≥2 сегментами,
слабые временно-разнесённые источники всплывают каждый в своём сегменте (K-40 раньше Cs-137),
статус обновляется, пункт есть в меню «Инструменты», а пустой пересчёт (без записи) безопасен.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.main_window import MainWindow
from awf.ui import i18n   # язык интерфейса может быть RU или EN (восстанавливается из QSettings)


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _gauss(nch, center, height, fwhm):
    """Детерминированная гауссиана высотой height (отсчётов) на сетке каналов [0, nch)."""
    x = np.arange(nch, dtype=float)
    sigma = fwhm / 2.3548200450309493
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _two_regime_counts(ns_a, ns_b, nch, peak_a, peak_b, *, bg=2.0, h_a=40.0, h_b=40.0, fwhm=24.0):
    """Срезы [0,ns_a) с пиком в peak_a, [ns_a, ns_a+ns_b) с пиком в peak_b."""
    rows = []
    base = np.full(nch, bg)
    for _ in range(ns_a):
        rows.append(np.rint(base + _gauss(nch, peak_a, h_a, fwhm)).astype(np.int64))
    for _ in range(ns_b):
        rows.append(np.rint(base + _gauss(nch, peak_b, h_b, fwhm)).astype(np.int64))
    return np.vstack(rows)


def _make_sg(counts, *, slice_live=10.0):
    """Собрать Spectrogram из 2D counts [ns, nch] с калибровкой energy=channel."""
    counts = np.asarray(counts, dtype=np.int64)
    ns, nch = counts.shape
    to = np.arange(ns, dtype=float) * slice_live
    lt = np.full(ns, slice_live, dtype=float)
    return Spectrogram(counts=counts, calibration=Calibration(coeffs=[0.0, 1.0]),
                       time_offsets_s=to, real_time_s=lt.copy(), live_time_s=lt.copy())


def _win_with_record(app):
    """MainWindow с загруженной 2-фазной записью (K-40 ранняя фаза, Cs-137 поздняя)."""
    w = MainWindow()
    counts = _two_regime_counts(40, 40, 1600, peak_a=1460, peak_b=662)
    w._on_loaded(_make_sg(counts))
    return w


def test_segments_widgets_exist(app):
    w = MainWindow()
    assert hasattr(w, '_segments_panel')
    assert hasattr(w, '_segments_dock')
    assert hasattr(w, '_act_segments')
    assert w._segments_dock.objectName() == "dock_segments"


def test_segments_panel_empty_recompute_safe(app):
    w = MainWindow()
    w._on_segment_recompute(2.0)
    assert w._segments_panel._tree.topLevelItemCount() == 0
    assert "—" in w._segments_panel._status.text()


def test_segments_action_builds_tree(app):
    w = _win_with_record(app)
    w._on_segments_action()
    assert w._segments_panel._tree.topLevelItemCount() >= 2
    # offscreen: isVisible() зависит от показанного top-level окна; show() гарантирует isHidden()==False
    assert not w._segments_dock.isHidden()


def test_segments_status_updated(app):
    w = _win_with_record(app)
    w._on_segments_action()
    txt = w._segments_panel._status.text()
    assert txt.startswith(i18n.tr("Сегментов"))   # язык-независимо (RU «Сегментов», EN «Segments»)
    num = int(txt.split(":")[1].strip())
    assert num >= 2


def test_segments_separate_sources_k40_before_cs137(app):
    w = _win_with_record(app)
    w._on_segments_action()
    tree = w._segments_panel._tree
    k_idx = []
    cs_idx = []
    for k in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(k)
        for j in range(top.childCount()):
            child = top.child(j)
            if child.text(0) == "K-40":
                k_idx.append(k)
            elif child.text(0) == "Cs-137":
                cs_idx.append(k)
    assert k_idx
    assert cs_idx
    assert min(cs_idx) > max(k_idx)


def test_segments_in_tools_menu(app):
    w = MainWindow()
    menu = w._menus["tools"]
    labels = [a.text() for a in menu.actions()]
    assert i18n.tr("Сегментация по времени") in labels   # язык-независимо (RU/EN)

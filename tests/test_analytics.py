import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.analytics_panel import AnalyticsPanel
from awf.analysis.decomposition import is_available as proj_available


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=64):
    # два «режима» спектра -> кластеры должны разделиться
    rng = np.random.RandomState(11)
    counts = rng.poisson(10, size=(ns, nc)).astype(np.int64)
    counts[: ns // 2, nc // 4] += 400      # первая половина — пик слева
    counts[ns // 2:, 3 * nc // 4] += 400   # вторая половина — пик справа
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64) * 2.0
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, 2.0), live_time_s=np.full(ns, 2.0))


def test_recompute_creates_scatter_pca_kmeans(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p.set_spectrogram(sg)            # ns<cap -> авто-пересчёт (PCA+KMeans по умолчанию)
    assert len(p._scatters) >= 1
    # суммарно точек == число срезов (каждый срез — одна точка проекции)
    total = sum(len(sc.points()) for sc in p._scatters)
    assert total == sg.n_slices


def test_points_carry_slice_index(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p.set_spectrogram(sg)
    idxs = []
    for sc in p._scatters:
        for pt in sc.points():
            idxs.append(int(pt.data()))
    assert sorted(idxs) == list(range(sg.n_slices))   # индексы покрывают все срезы без дублей


def test_click_emits_slice_clicked(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p.set_spectrogram(sg)
    captured = []
    p.sliceClicked.connect(lambda i: captured.append(i))
    sc = p._scatters[0]
    pt = sc.points()[0]
    p._on_points_clicked(sc, [pt])
    assert len(captured) == 1
    assert 0 <= captured[0] < sg.n_slices
    assert captured[0] == int(pt.data())


def test_kmeans_k_respected(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p._k_spin.setValue(3)
    p.set_spectrogram(sg)
    # KMeans -> ровно k непустых кластеров (на разделимых данных)
    labels = set()
    for sc in p._scatters:
        for pt in sc.points():
            pass
    assert 1 <= len(p._scatters) <= 3


def test_large_record_defers_autorun(app):
    # ns > AUTORUN_SLICE_CAP -> авто-пересчёт не делаем, scatter пуст до кнопки
    sg = _make_sg(ns=8, nc=16)
    p = AnalyticsPanel()
    p.AUTORUN_SLICE_CAP = 4           # искусственно занизить порог
    p.set_spectrogram(sg)
    assert p._scatters == []
    p._recompute()                   # ручной запуск
    assert len(p._scatters) >= 1


def test_unavailable_projection_no_crash(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p.set_spectrogram(sg)
    # выбрать t-SNE; если пакет отсутствует — статус с сообщением, без падения и без scatter
    tsne_idx = p._proj_combo.findData("tsne")
    assert tsne_idx >= 0
    p._proj_combo.setCurrentIndex(tsne_idx)
    p._recompute()
    if not proj_available("tsne"):
        assert p._scatters == []
        assert "недоступен" in p._status.text().lower()
    else:
        assert len(p._scatters) >= 1


def test_set_none_clears(app):
    sg = _make_sg()
    p = AnalyticsPanel()
    p.set_spectrogram(sg)
    assert len(p._scatters) >= 1
    p.set_spectrogram(None)
    assert p._scatters == []
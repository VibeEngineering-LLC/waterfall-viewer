import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import SlicePanel
from awf.analysis.peakmap import DEFAULT_WINDOWS


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=24, nc=1500, t_step=2.0):
    rng = np.random.RandomState(7)
    counts = rng.poisson(20, size=(ns, nc)).astype(np.int64)
    counts[:, 662] += 500          # линия Cs-137 (E=ch при калибровке [0,1])
    counts[:, 1461] += 300         # линия K-40
    cal = Calibration(coeffs=[0.0, 1.0])   # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_point_spectrum_matches_slice(app):
    # «спектр в точке» (Задача 19.1): show_time_slice рисует ровно energy_spectrum(i) при smooth=0
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s.show_time_slice(5)
    x, y = s._spectrum_curve.getData()
    assert np.allclose(y, sg.energy_spectrum(5).astype(np.float64))
    assert np.allclose(x, sg.energies())


def test_energy_window_profile_matches_band(app):
    # профиль по времени в энергоокне (19.2) == energy_band_time_series; ось X == времена
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s.show_energy_window(646.7, 676.7)            # окно Cs-137
    x, y = s._ewin_curve.getData()
    expected = sg.energy_band_time_series(646.7, 676.7).astype(np.float64)
    assert np.allclose(y, expected)
    assert np.allclose(x, sg.time_offsets_s)
    assert y.size == sg.n_slices
    assert s._ewin_active == (646.7, 676.7)


def test_energy_window_default_on_load(app):
    # после загрузки активное окно выставлено и жёлтая кривая заполнена (длина = срезы)
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    assert s._ewin_active is not None
    _, y = s._ewin_curve.getData()
    assert y is not None and y.size == sg.n_slices
    lo, hi = s._ewin_active
    assert hi > lo


def test_energy_window_reversed_args_sorted(app):
    # перепутанный порядок границ нормализуется (sorted) — результат как у прямого порядка
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s.show_energy_window(676.7, 646.7)
    _, y = s._ewin_curve.getData()
    expected = sg.energy_band_time_series(646.7, 676.7).astype(np.float64)
    assert np.allclose(y, expected)
    assert s._ewin_active == (646.7, 676.7)


def test_preset_combobox_sets_window(app):
    # выбор пресета Cs-137 в комбобоксе выставляет границы спинбоксов и перерисовывает профиль
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s._ewin_preset.setCurrentIndex(1)             # 0 = «вручную», 1 = Cs-137 662
    w = DEFAULT_WINDOWS[0]
    assert abs(s._ewin_lo.value() - w.e_lo) < 1.0
    assert abs(s._ewin_hi.value() - w.e_hi) < 1.0
    lo, hi = s._ewin_active
    expected = sg.energy_band_time_series(lo, hi).astype(np.float64)
    _, y = s._ewin_curve.getData()
    assert np.allclose(y, expected)


def test_manual_spin_edit_resets_preset(app):
    # ручная правка границ сбрасывает пресет в «вручную» (idx 0) и обновляет профиль
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s._ewin_preset.setCurrentIndex(2)             # K-40
    s._ewin_lo.setValue(600.0); s._ewin_hi.setValue(620.0)
    s._on_ewin_spin()
    assert s._ewin_preset.currentIndex() == 0     # сброшен в «вручную»
    assert s._ewin_active == (600.0, 620.0)


def test_preset_ignored_before_load(app):
    # пресет до загрузки данных не падает и ничего не строит
    s = SlicePanel()
    s._ewin_preset.setCurrentIndex(1)
    assert s._ewin_active is None


def test_roi_band_curve_independent_of_window(app):
    # жёлтая кривая (энергоокно) и магента (полоса ROI) — раздельные кривые
    sg = _make_sg()
    s = SlicePanel()
    s.set_spectrogram(sg)
    s.show_roi(0, sg.n_slices, 0, sg.n_channels)
    s.show_energy_window(646.7, 676.7)
    _, y_roi = s._series_curve.getData()
    _, y_win = s._ewin_curve.getData()
    assert not np.allclose(y_roi, y_win)          # разные данные на разных кривых
    assert np.allclose(y_roi, sg.band_time_series(0, sg.n_channels).astype(np.float64))
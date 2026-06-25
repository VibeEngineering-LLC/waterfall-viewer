import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis.peakmap import (
    EnergyWindow, DEFAULT_WINDOWS, WindowSeries, window_series, peak_map,
)


def _gauss(nc, center, sigma, area):
    ch = np.arange(nc, dtype=np.float64)
    g = np.exp(-0.5 * ((ch - center) / sigma) ** 2)
    return g / g.sum() * area


def _sg_with_peak(ns=30, nc=720, base=20, center=662, sigma=4.0,
                  area=5000.0, present=(10, 21), t_step=1.0):
    """Плоский континуум base; гауссов пик в срезах present[0]:present[1]."""
    counts = np.full((ns, nc), base, dtype=np.int64)
    bump = np.round(_gauss(nc, center, sigma, area)).astype(np.int64)
    counts[present[0]:present[1], :] += bump
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch кэВ
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_default_windows_registry():
    assert len(DEFAULT_WINDOWS) >= 5
    names = [w.name for w in DEFAULT_WINDOWS]
    assert any("Cs-137" in n for n in names)
    assert any("K-40" in n for n in names)


def test_energy_window_bounds():
    w = EnergyWindow("test", 100.0, 10.0)
    assert w.e_lo == 90.0 and w.e_hi == 110.0


def test_gross_matches_energy_band_series():
    sg = _sg_with_peak()
    w = EnergyWindow("Cs", 662.0, 15.0)
    ws = window_series(sg, w)
    manual = sg.energy_band_time_series(w.e_lo, w.e_hi).astype(np.float64)
    assert np.array_equal(ws.gross, manual)


def test_net_zero_on_flat_continuum():
    # без пика: плоский континуум, нетто ровно 0 (боковые полосы оценивают базу точно)
    sg = _sg_with_peak(area=0.0, present=(0, 0))
    ws = window_series(sg, EnergyWindow("Cs", 662.0, 15.0))
    assert np.allclose(ws.net, 0.0, atol=1e-6)


def test_net_detects_peak_in_time():
    sg = _sg_with_peak(area=5000.0, present=(10, 21))
    ws = window_series(sg, EnergyWindow("Cs", 662.0, 15.0))
    assert isinstance(ws, WindowSeries)
    # нетто высокий в срезах с пиком, ~0 вне
    present_net = ws.net[10:21]
    absent_net = np.concatenate([ws.net[:10], ws.net[21:]])
    assert present_net.min() > 0.8 * 5000.0
    assert np.abs(absent_net).max() < 0.05 * 5000.0
    # пик локализуется по времени внутри окна присутствия
    assert 10 <= int(np.argmax(ws.net)) < 21


def test_peak_map_returns_all_windows():
    sg = _sg_with_peak()
    res = peak_map(sg)
    assert len(res) == len(DEFAULT_WINDOWS)
    assert all(isinstance(r, WindowSeries) for r in res)


def test_peak_map_custom_windows():
    sg = _sg_with_peak()
    custom = [EnergyWindow("A", 662.0, 15.0), EnergyWindow("B", 300.0, 10.0)]
    res = peak_map(sg, custom)
    assert [r.window.name for r in res] == ["A", "B"]


def test_window_near_edge_no_crash():
    sg = _sg_with_peak(nc=720)
    # окно у правого края спектра — боковая полоса справа усечена, без падения
    ws = window_series(sg, EnergyWindow("edge", 715.0, 8.0))
    assert ws.gross.shape == (sg.n_slices,)
    assert np.all(np.isfinite(ws.net))
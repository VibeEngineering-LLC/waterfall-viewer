"""Задача #114: модель разрешения FWHM(E) + per-channel ширина matched-фильтра.

Ядро: фиксированная скалярная ширина фильтра (8 каналов в проде) согласована
только с УЗКИМИ пиками; широкий сцинтилляционный фотопик (FWHM растёт по энергии)
этим узким −G''-фильтром интегрируется частично → значимость рушится → пик
ПРОПУСКАЕТСЯ. Массив ширины из FWHM(E)-модели использует локальную ширину в каждом
канале и находит ОБА пика. Скалярный путь обязан остаться бит-в-бит.
"""
import numpy as np
import pytest
from awf.analysis.peaks import (
    find_peaks, FwhmModel, default_fwhm_model, fwhm_model_keV,
    estimate_fwhm_model, fwhm_channels_from_model,
)

N = 2000
KEV_PER_CH = 0.43
ENERGIES = 3.0 + KEV_PER_CH * np.arange(N, dtype=np.float64)
NARROW_CH = 300
WIDE_CH = 1500
NARROW_FWHM_CH = 10.0
WIDE_FWHM_CH = 90.0

def _gauss(center_ch, height, fwhm_ch):
    sigma = fwhm_ch / 2.3548
    ch = np.arange(N, dtype=np.float64)
    return height * np.exp(-((ch - center_ch)**2) / (2 * sigma**2))

def _make_two_width_spectrum(seed=0):
    rng = np.random.default_rng(seed)
    counts = 80.0 + np.zeros(N, dtype=np.float64)
    counts += _gauss(NARROW_CH, 900.0, NARROW_FWHM_CH)
    counts += _gauss(WIDE_CH, 220.0, WIDE_FWHM_CH)
    return rng.poisson(np.clip(counts, 0.0, None)).astype(np.float64)

def _model_for_two_peaks():
    e_n = ENERGIES[NARROW_CH]
    e_w = ENERGIES[WIDE_CH]
    fw_n_keV = NARROW_FWHM_CH * KEV_PER_CH
    fw_w_keV = WIDE_FWHM_CH * KEV_PER_CH
    A = np.array([[1.0, e_n], [1.0, e_w]])
    B = np.array([fw_n_keV**2, fw_w_keV**2])
    a, b = np.linalg.solve(A, B)
    return FwhmModel(a=float(a), b=float(b), c=0.0, source="auto")

def _near(peaks, ch, tol):
    return [p for p in peaks if abs(p.channel - ch) <= tol]

def test_scalar_width_misses_wide_peak():
    counts = _make_two_width_spectrum(0)
    peaks = find_peaks(counts, 8.0, sigma_threshold=3.0, energies=ENERGIES)
    assert _near(peaks, NARROW_CH, 30)
    assert not _near(peaks, WIDE_CH, 60)

def test_array_width_finds_both_peaks_above_3sigma():
    counts = _make_two_width_spectrum(0)
    warr = fwhm_channels_from_model(_model_for_two_peaks(), ENERGIES)
    peaks = find_peaks(counts, warr, sigma_threshold=3.0, energies=ENERGIES)
    narrow = _near(peaks, NARROW_CH, 30)
    wide = _near(peaks, WIDE_CH, 60)
    assert narrow
    assert wide
    assert narrow[0].significance > 3.0
    assert wide[0].significance > 3.0

def test_array_width_has_local_values_at_anchors():
    warr = fwhm_channels_from_model(_model_for_two_peaks(), ENERGIES)
    assert warr.shape == (N,)
    assert warr[NARROW_CH] == pytest.approx(NARROW_FWHM_CH, abs=0.5)
    assert warr[WIDE_CH] == pytest.approx(WIDE_FWHM_CH, abs=0.5)
    assert warr[WIDE_CH] > warr[NARROW_CH]

def test_scalar_path_bit_exact_vs_callable_and_array():
    counts = _make_two_width_spectrum(0)
    scalar = find_peaks(counts, 8.0, sigma_threshold=3.0, energies=ENERGIES)
    callab = find_peaks(counts, (lambda ch: 8.0), sigma_threshold=3.0, energies=ENERGIES)
    const_arr = np.full(N, 8.0, float)
    arr = find_peaks(counts, const_arr, sigma_threshold=3.0, energies=ENERGIES)
    assert [(round(p.channel, 9), round(p.significance, 9)) for p in scalar] == \
           [(round(p.channel, 9), round(p.significance, 9)) for p in callab]
    assert [(round(p.channel, 9), round(p.significance, 9)) for p in scalar] == \
           [(round(p.channel, 9), round(p.significance, 9)) for p in arr]

def test_array_wrong_length_raises():
    counts = _make_two_width_spectrum(0)
    with pytest.raises(ValueError):
        find_peaks(counts, np.full(N - 5, 8.0), sigma_threshold=3.0)

def test_default_model_is_physical_and_monotonic():
    m = default_fwhm_model()
    assert m.source == "default"
    assert float(m(662.0)) == pytest.approx(0.07 * 662.0, rel=1e-6)
    Es = np.linspace(100, 2500, 60)
    w = np.asarray(m(Es))
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    assert np.all(np.diff(w) >= -1e-9)
    assert float(m(1460.0)) / 1460.0 < float(m(662.0)) / 662.0

def test_explicit_coeffs_match_model_and_function():
    a, b, c = 5.0, 3.0, 1e-4
    m = FwhmModel(a=a, b=b, c=c)
    Es = np.linspace(50, 3000, 40)
    assert np.allclose(np.asarray(m(Es)), fwhm_model_keV(Es, a, b, c))
    ch_model = fwhm_channels_from_model(m, ENERGIES)
    ch_tuple = fwhm_channels_from_model((a, b, c), ENERGIES)
    ch_call = fwhm_channels_from_model(lambda E: m(E), ENERGIES)
    assert np.allclose(ch_model, ch_tuple)
    assert np.allclose(ch_model, ch_call)

def test_fwhm_channels_uses_local_dispersion():
    m = default_fwhm_model()
    warr = fwhm_channels_from_model(m, ENERGIES)
    i662 = int(np.searchsorted(ENERGIES, 662.0))
    expected = float(m(662.0)) / KEV_PER_CH
    assert warr[i662] == pytest.approx(expected, rel=0.02)

def test_bad_input_falls_back_to_floor():
    warr = fwhm_channels_from_model("garbage", ENERGIES, floor=1.0)
    assert warr.shape == (N,)
    assert np.allclose(warr, 1.0)

def test_estimate_recovers_growing_model_on_clean_anchors():
    counts = _make_two_width_spectrum(0)
    e_n = ENERGIES[NARROW_CH]
    e_w = ENERGIES[WIDE_CH]
    model = estimate_fwhm_model(counts, ENERGIES, anchor_energies=(e_n, e_w), search_keV=30.0, min_anchors=2)
    assert model.source in ("auto", "default")
    Es = np.linspace(ENERGIES[10], ENERGIES[-10], 50)
    w = np.asarray(model(Es))
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    assert np.all(np.diff(w) >= -1e-9)
    if model.source == "auto":
        assert float(model(e_w)) > float(model(e_n))

def test_estimate_falls_back_when_anchors_missing():
    flat = 80.0 + np.zeros(N, float)
    model = estimate_fwhm_model(flat, ENERGIES, anchor_energies=(300.0, 600.0), min_anchors=2)
    assert model.source == "default"

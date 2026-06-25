from __future__ import annotations
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis import (
    FoundPeak, PeakArea, MdaResult, LineMatch, IdentResult,
    spectrum_from_selection,
)

def _make_sg(n_slices: int = 4, n_channels: int = 5) -> Spectrogram:
    counts = np.arange(n_slices * n_channels, dtype=np.int64).reshape(n_slices, n_channels)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch
    t = np.arange(n_slices, dtype=np.float64) * 10.0
    rt = np.full(n_slices, 10.0)
    lt = np.full(n_slices, 10.0)
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t, real_time_s=rt, live_time_s=lt)

def test_public_api_importable():
    assert isinstance(FoundPeak, type)
    assert isinstance(PeakArea, type)
    assert isinstance(MdaResult, type)
    assert isinstance(LineMatch, type)
    assert isinstance(IdentResult, type)
    assert callable(spectrum_from_selection)

def test_found_peak_fields():
    p = FoundPeak(channel=10.0, energy=661.7, height=500.0, fwhm_channels=3.0, significance=8.0, area_estimate=1250.0)
    assert p.channel == 10.0
    assert p.energy == pytest.approx(661.7)
    assert p.significance == 8.0

def test_peak_area_fields():
    a = PeakArea(net=900.0, d_net=40.0, gross=1100.0, baseline=200.0, roi_lo=8, roi_hi=14)
    assert a.net == 900.0
    assert a.gross - a.baseline == pytest.approx(a.net)
    assert a.roi_hi > a.roi_lo

def test_mda_result_fields():
    m = MdaResult(l_c=12.0, l_d=27.0, a_mda=3.5)
    assert m.l_c == 12.0 and m.l_d == 27.0 and m.a_mda == pytest.approx(3.5)

def test_ident_result_defaults():
    r = IdentResult(nuclide="K-40", confidence=0.9)
    assert r.matches == ()
    assert r.category is None

    lm = LineMatch(nuclide="K-40", line_energy=1460.8, peak_energy=1461.0, delta_keV=0.2,
                   intensity_pct=10.66)
    r2 = IdentResult(nuclide="K-40", confidence=0.9, matches=(lm,), category="natural")
    assert r2.matches[0].nuclide == "K-40"
    assert r2.category == "natural"

def test_types_are_frozen():
    m = MdaResult(l_c=1.0, l_d=2.0, a_mda=3.0)
    with pytest.raises(FrozenInstanceError):
        m.l_c = 5.0

def test_spectrum_from_selection_matches_sum():
    sg = _make_sg()
    windows = [(None, None), (0, 2), (1, 4), (2, 3)]
    for lo, hi in windows:
        got = spectrum_from_selection(sg, lo, hi)
        assert got.dtype == np.float64
        assert np.array_equal(got, sg.sum_spectrum(lo, hi).astype(np.float64))

def test_spectrum_from_selection_full_equals_total():
    sg = _make_sg()
    assert np.array_equal(spectrum_from_selection(sg), sg.total_spectrum().astype(np.float64))

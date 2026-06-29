import numpy as np
import pytest
from awf.analysis.peaks import find_peaks
from awf.analysis.types import FoundPeak

N = 512
FWHM = 8.0
CENTERS = (100, 256, 400)
HEIGHTS = (400.0, 1200.0, 2500.0)   # строго возрастают
SPIKE_CH = 330
SPIKE_H = 600.0


def _gauss(n, center, height, fwhm):
    sigma = fwhm / 2.355
    ch = np.arange(n, dtype=np.float64)
    return height * np.exp(-((ch - center) ** 2) / (2.0 * sigma ** 2))


def _make_spectrum(with_spike=True):
    """Наклонный фон + 3 гауссианы (+ опционально 1-канальный спайк)."""
    ch = np.arange(N, dtype=np.float64)
    counts = 100.0 - 0.10 * ch          # линейный наклон 100 -> ~49, всюду > 0
    for c, h in zip(CENTERS, HEIGHTS):
        counts = counts + _gauss(N, c, h, FWHM)
    if with_spike:
        counts[SPIKE_CH] += SPIKE_H     # узкий 1-канальный выброс
    return counts


def test_finds_exactly_three_gaussians():
    counts = _make_spectrum(with_spike=True)
    peaks = find_peaks(counts, FWHM, sigma_threshold=3.0)
    assert len(peaks) == 3
    found = sorted(p.channel for p in peaks)
    for c, fc in zip(CENTERS, found):
        assert abs(fc - c) <= 2
    # спайк отсеян: ни один пик не рядом с SPIKE_CH
    assert all(abs(p.channel - SPIKE_CH) > 3 for p in peaks)


def test_significance_monotonic_with_height():
    counts = _make_spectrum(with_spike=False)
    peaks = sorted(find_peaks(counts, FWHM, sigma_threshold=3.0), key=lambda p: p.channel)
    assert len(peaks) == 3
    sig = [p.significance for p in peaks]
    assert sig[0] < sig[1] < sig[2]
    h = [p.height for p in peaks]
    assert h[0] < h[1] < h[2]


def test_spike_filter_contrast():
    counts = _make_spectrum(with_spike=True)
    on = find_peaks(counts, FWHM, sigma_threshold=3.0, spike_min_fwhm_frac=0.3)
    off = find_peaks(counts, FWHM, sigma_threshold=3.0, spike_min_fwhm_frac=0.0)
    assert len(off) == len(on) + 1
    assert any(abs(p.channel - SPIKE_CH) <= 1 for p in off)
    assert all(abs(p.channel - SPIKE_CH) > 1 for p in on)


def test_energy_filled_from_energies_array():
    counts = _make_spectrum(with_spike=False)
    energies = 3.0 * np.arange(N, dtype=np.float64)   # 3 кэВ/канал
    peaks = find_peaks(counts, FWHM, sigma_threshold=3.0, energies=energies)
    assert len(peaks) == 3
    for p in peaks:
        assert p.energy == pytest.approx(3.0 * int(p.channel))


def test_energy_defaults_to_channel_without_calibration():
    counts = _make_spectrum(with_spike=False)
    peaks = find_peaks(counts, FWHM, sigma_threshold=3.0)
    assert all(p.energy == float(p.channel) for p in peaks)


def test_short_spectrum_returns_empty():
    assert find_peaks(np.ones(30, dtype=np.float64), FWHM) == []
    assert find_peaks(np.zeros(0, dtype=np.float64), FWHM) == []


def test_returns_foundpeak_instances():
    counts = _make_spectrum(with_spike=False)
    peaks = find_peaks(counts, FWHM, sigma_threshold=3.0)
    assert peaks and all(isinstance(p, FoundPeak) for p in peaks)
    for p in peaks:
        assert p.height > 0 and p.fwhm_channels > 0 and p.area_estimate > 0


def test_callable_fwhm_matches_scalar():
    counts = _make_spectrum(with_spike=False)
    scalar = find_peaks(counts, FWHM, sigma_threshold=3.0)
    callab = find_peaks(counts, (lambda ch: FWHM), sigma_threshold=3.0)
    assert [p.channel for p in scalar] == [p.channel for p in callab]

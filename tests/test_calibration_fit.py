"""Тесты #215 — фит калибровки по парам (channel, true_E)."""
from __future__ import annotations

import numpy as np
import pytest

from awf.analysis.calibration_fit import (
    PRESETS,
    apply_polynomial,
    compute_rmse,
    fit_calibration,
    format_coeffs,
)


def _fake_calibration(coeffs):
    """E(ch) прямо по PRYMOMU polyval (совместимо с awf.model.spectrogram.Calibration)."""
    import numpy.polynomial.polynomial as P
    def e(ch):
        return P.polyval(np.asarray(ch, dtype=np.float64), np.asarray(coeffs, dtype=np.float64))
    return e


def test_fit_recovers_known_polynomial():
    # синтетическая калибровка E = 5 + 0.35*ch + 1.2e-5*ch^2
    true_coeffs = [5.0, 0.35, 1.2e-5]
    E = _fake_calibration(true_coeffs)
    channels = [100.0, 500.0, 1200.0, 2500.0, 4000.0, 6500.0]
    pairs = [(ch, float(E(ch))) for ch in channels]

    got = fit_calibration(pairs, deg=2)

    assert len(got) == 3
    np.testing.assert_allclose(got, true_coeffs, atol=1e-6, rtol=1e-6)
    assert compute_rmse(got, pairs) < 1e-3


def test_fit_th232_pairs_smoke():
    # реалистичный сценарий: 4 пика Th-232, начальная калибровка примерно линейная
    # моделируем "истинную" калибровку quadratic, находим channels обратно
    true_coeffs = [2.0, 0.32, 8.0e-6]
    E = _fake_calibration(true_coeffs)
    energies_kev = [238.63, 583.19, 911.20, 2614.51]
    # находим channels численно (bisection)
    def find_ch(e_target):
        lo, hi = 0.0, 8191.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if E(mid) < e_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    pairs = [(find_ch(e), e) for e in energies_kev]

    got = fit_calibration(pairs, deg=2)
    rmse = compute_rmse(got, pairs)
    assert rmse < 0.5  # kev


def test_fit_degree_clipped_to_n_minus_1():
    # 3 pary, deg=4 -> reznemsya na 2
    pairs = [(0.0, 0.0), (100.0, 30.0), (1000.0, 300.0)]
    got = fit_calibration(pairs, deg=4)
    # deg=2 -> 3 koefficienta
    assert len(got) == 3


def test_fit_min_pairs():
    # 2 pary, deg=1 -> tochno lineynaya
    pairs = [(0.0, 5.0), (1000.0, 305.0)]
    got = fit_calibration(pairs, deg=1)
    assert len(got) == 2
    np.testing.assert_allclose(got, [5.0, 0.3], atol=1e-9)


def test_fit_too_few_pairs_raises():
    with pytest.raises(ValueError):
        fit_calibration([(0.0, 0.0)], deg=1)


def test_fit_bad_deg_raises():
    pairs = [(0.0, 0.0), (100.0, 30.0)]
    with pytest.raises(ValueError):
        fit_calibration(pairs, deg=0)


def test_presets_have_expected_isotopes():
    assert "Th-232 (цепочка)" in PRESETS
    assert "Cs-137" in PRESETS
    assert "Co-60" in PRESETS
    assert "K-40" in PRESETS
    assert "Am-241" in PRESETS

    # cs-137 - odna liniya 661.66
    assert PRESETS["Cs-137"] == [661.66]
    # co-60 - dve linii
    assert len(PRESETS["Co-60"]) == 2
    assert 1173.0 < PRESETS["Co-60"][0] < 1174.0
    assert 1332.0 < PRESETS["Co-60"][1] < 1333.0


def test_format_coeffs_readable():
    s = format_coeffs([2.0, 0.32, 8.0e-6])
    assert "c0=" in s
    assert "c1=" in s
    assert "c2=" in s


def test_apply_polynomial_matches_calibration_convention():
    # sverim s numpy.polynomial forward order
    import numpy.polynomial.polynomial as P
    coeffs = [1.0, 0.5, 1.0e-4]
    chs = np.array([0.0, 100.0, 1000.0, 5000.0])
    expected = P.polyval(chs, np.asarray(coeffs))
    got = apply_polynomial(coeffs, chs)
    np.testing.assert_allclose(got, expected)
"""
Tests for awf.model.dose (Task #104) -- dose rate from a gamma waterfall.

Synthetic only: a tiny Spectrogram is built directly, NO real file / serial.
Mirrors the style of tests/test_rcspg_android_text.py but self-contained.
"""
from __future__ import annotations
import numpy as np
import pytest

from awf.model.spectrogram import Spectrogram, Calibration
from awf.model import dose


def _make_sg(counts, *, coeffs=(0.0, 1.0), live_time_s=None):
    """Build a Spectrogram with E_i = poly(coeffs) per channel.

    With coeffs=(0,1) the energy of channel i is exactly i (keV), so hand
    computation of sum(E*cps) is trivial.
    """
    counts = np.asarray(counts, dtype=np.uint32)
    n_slices, n_channels = counts.shape
    if live_time_s is None:
        live_time_s = np.ones(n_slices, dtype=np.float64)
    live_time_s = np.asarray(live_time_s, dtype=np.float64)
    t_off = np.arange(n_slices, dtype=np.float64)
    return Spectrogram(
        counts=counts,
        calibration=Calibration(coeffs=np.asarray(coeffs, dtype=np.float64)),
        time_offsets_s=t_off,
        real_time_s=live_time_s.copy(),
        live_time_s=live_time_s.copy(),
    )


def test_energy_deposition_rate_manual():
    # E_i = i; counts 2x4; live_time [1,2] -> cps = counts/live_time.
    counts = [[1, 2, 3, 4],
              [10, 0, 0, 8]]
    sg = _make_sg(counts, live_time_s=[1.0, 2.0])
    # drop_last=1 -> drop channel 3 (E=3). Slice0 cps=[1,2,3], E=[0,1,2]:
    #   P0 = 1*0 + 2*1 + 3*2 = 8 keV/s
    # Slice1 cps=[5,0,0], E=[0,1,2]: P1 = 5*0 = 0 keV/s
    P = dose.energy_deposition_rate(sg, drop_last=1)
    np.testing.assert_allclose(P, [8.0, 0.0])


def test_energy_deposition_rate_no_drop():
    counts = [[1, 2, 3, 4]]
    sg = _make_sg(counts, live_time_s=[1.0])
    # drop_last=0 keeps channel 3 (E=3): P = 1*0+2*1+3*2+4*3 = 20
    P = dose.energy_deposition_rate(sg, drop_last=0)
    np.testing.assert_allclose(P, [20.0])


def test_drop_last_actually_drops_last_channel():
    counts = [[0, 0, 0, 1000]]   # all weight in the last (overflow) channel
    sg = _make_sg(counts, live_time_s=[1.0])
    # With drop_last=1 the only contributing channel is removed -> P=0.
    np.testing.assert_allclose(dose.energy_deposition_rate(sg, drop_last=1), [0.0])
    # Without dropping, E=3 channel contributes 1000*3 = 3000.
    np.testing.assert_allclose(dose.energy_deposition_rate(sg, drop_last=0), [3000.0])


def test_dose_rate_series_is_kcal_times_P():
    counts = [[1, 2, 3, 4]]
    sg = _make_sg(counts, live_time_s=[1.0])
    P = dose.energy_deposition_rate(sg, drop_last=1)   # = 1*0+2*1+3*2 = 8
    k = 2.0e-3
    D = dose.dose_rate_series(sg, unit="mSv/h", k_cal=k, drop_last=1)
    np.testing.assert_allclose(D, k * P)


def test_unit_conversion_uSv_is_1000x_mSv():
    counts = [[0, 5, 7, 0]]
    sg = _make_sg(counts, live_time_s=[1.0])
    k = 1.5e-4
    d_msvh = dose.dose_rate_series(sg, unit="mSv/h", k_cal=k)
    d_usvh = dose.dose_rate_series(sg, unit="uSv/h", k_cal=k)
    np.testing.assert_allclose(d_usvh, d_msvh * 1000.0)


def test_unknown_unit_raises():
    sg = _make_sg([[1, 1]], live_time_s=[1.0])
    with pytest.raises(ValueError):
        dose.dose_rate_series(sg, unit="rem/h")


def test_cps_divides_by_live_time():
    # Same counts, different live_time -> cps halved -> P halved.
    counts = [[0, 4, 6, 0],
              [0, 4, 6, 0]]
    sg = _make_sg(counts, live_time_s=[1.0, 2.0])
    P = dose.energy_deposition_rate(sg, drop_last=1)
    # E=[0,1,2]; per-count contribution = 4*1+6*2 = 16. Slice0 lt=1 -> 16; slice1 lt=2 -> 8.
    np.testing.assert_allclose(P, [16.0, 8.0])


def test_dead_slice_live_time_zero_gives_zero():
    counts = [[1, 2, 3, 4],
              [9, 9, 9, 9]]
    sg = _make_sg(counts, live_time_s=[1.0, 0.0])   # slice1 is "dead"
    P = dose.energy_deposition_rate(sg, drop_last=1)
    assert P[1] == 0.0
    D = dose.dose_rate_series(sg, drop_last=1)
    assert D[1] == 0.0


def test_negative_live_time_also_dead():
    sg = _make_sg([[5, 5, 5, 5]], live_time_s=[-3.0])
    assert dose.energy_deposition_rate(sg, drop_last=1)[0] == 0.0


def test_output_length_equals_n_slices():
    counts = np.arange(5 * 4, dtype=np.uint32).reshape(5, 4)
    sg = _make_sg(counts)
    assert dose.energy_deposition_rate(sg).shape == (5,)
    assert dose.dose_rate_series(sg).shape == (5,)


def test_default_constant_is_positive_and_small():
    # k_cal = 75 / 2.242e8 ~ 3.35e-7 (mSv/h)/(keV/s)
    assert 3.0e-7 < dose.DOSE_CAL_RC103 < 3.7e-7


def test_calibrate_constant_roundtrip():
    k = dose.calibrate_constant(2.242e8, 75.0)
    assert k == pytest.approx(75.0 / 2.242e8, rel=1e-12)
    # applying it back to the anchor P reproduces the anchor dose
    assert k * 2.242e8 == pytest.approx(75.0, rel=1e-12)


def test_calibrate_constant_rejects_nonpositive_P():
    with pytest.raises(ValueError):
        dose.calibrate_constant(0.0, 75.0)


def test_absorbed_dose_cross_check_order_of_magnitude():
    # 1 keV = 1.602176634e-16 J. For 1 g crystal and P=1e8 keV/s:
    #   Gy/s = 1e8 * 1.602e-16 / 1e-3 = 1.602e-5 ; mSv/h = *1000*3600
    counts = [[0, 0, 0, 0]]
    # build a slice whose P is known: put all counts in channel with E=100
    cc = np.zeros((1, 200), dtype=np.uint32)
    cc[0, 100] = 1_000_000          # cps=1e6 at E=100 -> P=1e8 keV/s
    sg = _make_sg(cc, live_time_s=[1.0])
    P = dose.energy_deposition_rate(sg, drop_last=1)
    assert P[0] == pytest.approx(1e8, rel=1e-9)
    d = dose.absorbed_dose_rate_in_crystal(sg, crystal_mass_g=1.0, unit="mSv/h", drop_last=1)
    expected = 1e8 * dose.KEV_IN_JOULE / 1e-3 * 1000.0 * 3600.0
    np.testing.assert_allclose(d, [expected])


def test_absorbed_dose_rejects_bad_mass():
    sg = _make_sg([[1, 1, 1, 1]], live_time_s=[1.0])
    with pytest.raises(ValueError):
        dose.absorbed_dose_rate_in_crystal(sg, crystal_mass_g=0.0)

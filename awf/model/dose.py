"""
Dose-rate estimation from a gamma waterfall spectrogram (Task #104).

Pure-numpy, no Qt. Works on top of awf.model.spectrogram.Spectrogram.

METHOD (energy-deposition proxy):
    For each time slice k we form the *measured energy-deposition rate*
        P[k] = sum_i  cps[k, i] * E_i        [keV / s]
    where
        cps[k, i] = counts[k, i] / live_time_s[k]   (sg.counts_in_unit('cps'))
        E_i       = sg.energies()                    (file calibration, keV)
    The last ADC channel is dropped by default (drop_last=1) because on this
    instrument it accumulates overflow garbage (see Spectrogram.trimmed_channels
    and remark IV-R5).

    The dose rate is taken proportional to that deposition rate:
        D[k] = k_cal * P[k]
    with a single linear calibration constant k_cal in (mSv/h) / (keV/s).

PROVENANCE / HONESTY (this is metrology -- be explicit, anti-hallucination):
    * P[k] is MEASURED: it comes only from counts, live_time and the file's own
      energy calibration. No tuning.
    * k_cal is EMPIRICALLY FITTED to match the RadiaCode-103 app readout for one
      reference spectrogram (operator requirement "like in the app"). It is NOT
      derived from first principles (no crystal mass, no detector efficiency, no
      build-up / attenuation model).
    * SATURATION CAVEAT: at the peak the count rate reaches ~1.9e5 1/s, where the
      detector is in heavy dead-time / pile-up and the spectrum is distorted, so
      the true count rate is UNDER-estimated. The app peak "75+ mSv/h" is a LOWER
      bound; absolute accuracy at the peak is limited. The model is linear between
      the two app anchors (peak ~75 mSv/h and low region ~2 mSv/h); their ratio
      (~37x) is consistent with a single linear constant, which justifies the
      linear "dose ~ sum(E*cps)" model in this range.
"""

from __future__ import annotations

import numpy as np

# 1 keV expressed in joule (CODATA), for the absorbed-dose cross-check only.
KEV_IN_JOULE = 1.602_176_634e-16


# --- empirical dose calibration --------------------------------------------

# DOSE_CAL_RC103 -- linear dose calibration constant, units (mSv/h) / (keV/s).
#
#   D[k] [mSv/h] = DOSE_CAL_RC103 * P[k] [keV/s]
#
# HOW IT WAS OBTAINED (provenance, Task #104):
#   * MEASURED on the reference RadiaCode-103 spectrogram (Android .rcspg,
#     1236 slices x 1024 channels, calibration E = 3.94 + 2.37*ch + 3.8e-4*ch^2):
#       peak energy-deposition rate  P_peak = 2.242e8 keV/s  (slice 632, t~796 s,
#       total_cps ~1.92e5);  median P ~1.18e7 keV/s.
#     Reproduce with scripts/dose_proxy_probe.py <file.rcspg>.
#   * The RadiaCode-103 APP reports for the SAME spectrogram:
#       peak dose rate ~= 75+ mSv/h  (LOWER bound -- detector saturated at the peak)
#       low region    ~=  2  mSv/h.
#   * k_cal is FITTED to the PEAK anchor (operator's choice):
#       k_cal = 75 / P_peak = 75 / 2.242e8 = 3.346e-7  (mSv/h)/(keV/s).
#   * CONSISTENCY of the two anchors (this passes): peak/low ~= 75/2 = 37.5x;
#     a single linear constant reproduces both, hence the linear model.
#
# WHAT IS MEASURED vs FITTED vs ASSUMED:
#   * P[k]=sum(E*cps)   -- MEASURED   (counts, live_time, file calibration).
#   * k_cal             -- FITTED     (matched to the app, not first-principles).
#   * linearity & scale -- ASSUMED    (single constant between the two anchors;
#                                      peak is a lower bound due to saturation).
DOSE_CAL_RC103 = 75.0 / 2.242e8   # = 3.34522...e-7 (mSv/h)/(keV/s)

# Unit conversion factors FROM the native mSv/h.
_UNIT_FROM_MSVH = {
    "mSv/h": 1.0,
    "uSv/h": 1000.0,
}


def energy_deposition_rate(sg, *, drop_last: int = 1) -> np.ndarray:
    """Measured energy-deposition rate per time slice, P[k] in keV/s.

        P[k] = sum_i cps[k, i] * E_i

    with cps = sg.counts_in_unit('cps') (counts[k]/live_time_s[k], dead slices
    with live_time<=0 -> 0) and E_i = sg.energies() (keV).

    drop_last (default 1) removes the last channel(s) before summing: the final
    ADC channel holds overflow garbage on this instrument (remark IV-R5).
    Returns float64 array of length sg.n_slices.
    """
    d = int(drop_last)
    cps = sg.counts_in_unit("cps")          # (n_slices, n_channels), float64
    energies = sg.energies()                # (n_channels,), keV
    if d > 0:
        if sg.n_channels - d < 1:
            raise ValueError("energy_deposition_rate: drop_last leaves no channels")
        cps = cps[:, :-d]
        energies = energies[:-d]
    # sum over channels: keV/s
    return cps.astype(np.float64, copy=False) @ energies.astype(np.float64, copy=False)


def dose_rate_series(sg, *, unit: str = "mSv/h",
                     k_cal: float = DOSE_CAL_RC103,
                     drop_last: int = 1) -> np.ndarray:
    """Dose-rate series per time slice, D[k] = k_cal * P[k], in `unit`.

    unit in {'mSv/h', 'uSv/h'}. k_cal is in (mSv/h)/(keV/s); the native result
    is mSv/h and is then converted to the requested unit. See DOSE_CAL_RC103 for
    the calibration provenance and the saturation caveat. Dead slices -> 0.
    """
    if unit not in _UNIT_FROM_MSVH:
        raise ValueError(f"dose_rate_series: unknown unit {unit!r}; "
                         f"expected one of {sorted(_UNIT_FROM_MSVH)}")
    p = energy_deposition_rate(sg, drop_last=drop_last)   # keV/s
    d_msvh = float(k_cal) * p                             # mSv/h
    return d_msvh * _UNIT_FROM_MSVH[unit]


def calibrate_constant(p_anchor_keVps: float, dose_anchor_mSvh: float) -> float:
    """Return k_cal (mSv/h)/(keV/s) so that dose_anchor = k_cal * p_anchor.

    Helper to re-fit the linear constant from any (P, dose) reference pair, e.g.
    from a different instrument or anchor. p_anchor_keVps must be > 0.
    """
    p = float(p_anchor_keVps)
    if p <= 0.0:
        raise ValueError("calibrate_constant: p_anchor_keVps must be > 0")
    return float(dose_anchor_mSvh) / p


def absorbed_dose_rate_in_crystal(sg, *, crystal_mass_g, unit: str = "mSv/h",
                                  drop_last: int = 1) -> np.ndarray:
    """Order-of-magnitude PHYSICAL cross-check (NOT the calibrated product).

    Treats the deposited energy as if fully absorbed in a CsI(Tl) crystal of mass
    `crystal_mass_g` grams and converts to absorbed dose rate:

        dose[k] [Gy/s]   = P[k] [keV/s] * KEV_IN_JOULE [J/keV] / mass [kg]
        -> Sv/h (Q=1 for gamma) -> mSv/h or uSv/h.

    This DELIBERATELY ignores detector efficiency, escape, build-up and the
    difference between energy absorbed in the small crystal and tissue kerma in
    the field; it only checks that the magnitude is plausibly in the tens of
    mSv/h at the peak (consistent with the app's "75+"). The CALIBRATED product
    is dose_rate_series(); this routine is a sanity sweep only.

    `crystal_mass_g` must be supplied by the caller -- the RC-103 crystal mass is
    NOT hard-coded here (it is not in the source file); pass a literature/datasheet
    value and treat the result as an order-of-magnitude reference. CsI(Tl) density
    is 4.51 g/cm^3 if a volume is known instead of a mass.
    """
    if unit not in _UNIT_FROM_MSVH:
        raise ValueError(f"absorbed_dose_rate_in_crystal: unknown unit {unit!r}")
    mass_kg = float(crystal_mass_g) / 1000.0
    if mass_kg <= 0.0:
        raise ValueError("absorbed_dose_rate_in_crystal: crystal_mass_g must be > 0")
    p = energy_deposition_rate(sg, drop_last=drop_last)   # keV/s
    gy_per_s = p * KEV_IN_JOULE / mass_kg                 # J/kg/s = Gy/s
    msvh = gy_per_s * 1000.0 * 3600.0                     # Gy/s -> mSv/h (Q=1)
    return msvh * _UNIT_FROM_MSVH[unit]

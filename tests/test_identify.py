import pytest
from awf.io.nuclide_lib import Nuclide, GammaLine
from awf.analysis.types import FoundPeak, IdentResult, LineMatch
from awf.analysis.identify import (
    lookup_by_energy, identify_peaks, default_fwhm_keV, get_prior,
    RARE_ISOTOPE_PRIOR,
)


def _fp(energy, area=1000.0, sig=30.0):
    return FoundPeak(channel=energy, energy=energy, height=area / 10.0,
                     fwhm_channels=3.0, significance=sig, area_estimate=area)


def _nuc(name, lines, category=None):
    gl = tuple(GammaLine(energy=e, intensity=i) for (e, i) in lines)
    return Nuclide(name=name, lines=gl, category=category)


def _fwhm6(E):
    # узкая константная модель FWHM, чтобы окно матчинга было предсказуемым
    return 6.0


def test_default_fwhm_model():
    assert default_fwhm_keV(661.66) == pytest.approx(0.07 * 661.66, rel=1e-6)
    assert default_fwhm_keV(1460.0) > default_fwhm_keV(661.66)
    assert default_fwhm_keV(0.0) > 0.0


def test_lookup_by_energy_window_and_sort():
    lib = [
        _nuc("Cs-137", [(661.66, 85.1)]),
        _nuc("Bi-214", [(609.31, 46.0)]),
        _nuc("K-40", [(1460.82, 10.66)]),
    ]
    hits = lookup_by_energy(660.0, 5.0, lib)
    assert [h.nuclide for h in hits] == ["Cs-137"]
    assert hits[0].delta_keV == pytest.approx(1.66, abs=1e-6)
    hits2 = lookup_by_energy(661.0, 60.0, lib)
    assert [h.nuclide for h in hits2] == ["Cs-137", "Bi-214"]


def test_lookup_by_energy_min_intensity():
    lib = [_nuc("X", [(100.0, 0.5), (100.2, 40.0)])]
    hits = lookup_by_energy(100.0, 5.0, lib, min_intensity_pct=1.0)
    assert len(hits) == 1
    assert hits[0].intensity_pct == 40.0


def test_identify_basic_ranking():
    peaks = [_fp(661.66, 1000.0), _fp(1173.2, 900.0), _fp(1332.5, 900.0)]
    lib = [
        _nuc("K-40", [(1460.82, 10.66)]),          # нет пика — не обнаружен
        _nuc("Cs-137", [(661.66, 85.1)]),
        _nuc("Co-60", [(1173.2, 99.85), (1332.5, 99.98)]),
    ]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    names = {r.nuclide for r in res}
    assert names == {"Cs-137", "Co-60"}
    confs = [r.confidence for r in res]
    assert confs == sorted(confs, reverse=True)
    by = {r.nuclide: r for r in res}
    assert by["Co-60"].confidence == pytest.approx(1.0)
    assert by["Cs-137"].confidence == pytest.approx(1.0)
    assert len(by["Co-60"].matches) == 2


def test_detection_gate_characteristic_absent():
    peaks = [_fp(1000.0, 500.0)]
    lib = [_nuc("Cs-137", [(661.66, 85.1)])]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    assert res == []


def test_proportionality_rejects_false():
    peaks = [_fp(661.66, 1000.0), _fp(1332.5, 800.0)]
    lib = [
        _nuc("Cs-137", [(661.66, 85.1)]),
        _nuc("Fake", [(661.5, 90.0), (1332.0, 10.0)]),
    ]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    names = {r.nuclide for r in res}
    assert "Cs-137" in names
    assert "Fake" not in names


def test_proportionality_accepts_consistent():
    peaks = [_fp(1173.2, 1000.0), _fp(1332.5, 980.0)]
    lib = [_nuc("Co-60", [(1173.2, 99.85), (1332.5, 99.98)])]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    assert len(res) == 1
    assert res[0].nuclide == "Co-60"
    assert res[0].confidence == pytest.approx(1.0)


def test_category_populated():
    peaks = [_fp(661.66, 1000.0)]
    lib = [_nuc("Cs-137", [(661.66, 85.1)], category="fission")]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    assert res[0].category == "fission"
    lib2 = [_nuc("K-40", [(1460.82, 10.66)])]
    res2 = identify_peaks([_fp(1460.82, 500.0)], lib2, fwhm_model=_fwhm6)
    assert res2[0].category == "natural"


def test_min_confidence_filter():
    peaks = [_fp(1332.5, 900.0)]
    lib = [_nuc("Co-60", [(1173.2, 99.85), (1332.5, 99.98)])]
    res_all = identify_peaks(peaks, lib, fwhm_model=_fwhm6)
    assert len(res_all) == 1
    assert res_all[0].confidence == pytest.approx(99.98 / (99.85 + 99.98))
    res_filtered = identify_peaks(peaks, lib, fwhm_model=_fwhm6, min_confidence=0.6)
    assert res_filtered == []


def test_apply_priors():
    peaks = [_fp(1173.2, 1000.0), _fp(1332.5, 1000.0)]
    lib = [_nuc("Co-60", [(1173.2, 99.85), (1332.5, 99.98)])]
    res = identify_peaks(peaks, lib, fwhm_model=_fwhm6, apply_priors=True)
    assert res[0].confidence == pytest.approx(0.5, abs=1e-9)
    assert get_prior("Co-60") == 0.5
    assert get_prior("UnknownNuclide") == 1.0
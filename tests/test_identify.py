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


def test_energy_quality_tolerates_calibration_drift():
    # #159: файл оператора сдвинут по шкале примерно на −1.4 % у 1461 кэВ
    # (пик 1440.6 vs K-40 1460.8). Квадратичный штраф качества энергии:
    # Δ = 2/3 окна → качество ≈ 1−(Δ/win)² ≈ 0.55, а не 0.33 (линейный).
    lib = [_nuc("K-40", [(1460.82, 10.66)])]
    res = identify_peaks([_fp(1440.6, 500.0)], lib, fwhm_model=lambda E: 60.0)
    assert len(res) == 1
    d, win = 1460.82 - 1440.6, 30.0
    assert res[0].confidence == pytest.approx(1.0 - (d / win) ** 2, rel=1e-6)


def test_th232_high_energy_lines_enabled_in_shipped_library():
    # #159: равновесные линии цепочки Th-232 в диапазоне 460..2614 кэВ должны
    # быть рабочими (used=True), иначе матчинг/маркеры теряют зону 911..2614.
    from awf.io.nuclide_lib import default_library
    th = next(n for n in default_library() if n.name == "Th-232")
    enabled = {round(ln.energy, 1) for ln in th.lines if ln.used}
    for e in (463.0, 727.3, 969.0, 1588.2, 1620.5, 1630.6, 2614.5):
        assert e in enabled, f"линия {e} кэВ выключена (used=False)"


def test_real_file_peak_set_k40_and_th232():
    # #159: энергии пиков реального файла оператора (сдвинутая шкала −0.2..−1.4 %).
    # K-40 должен войти в кандидаты с запасом над порогом 0.30 и выше La-138;
    # Th-232 должен зацепить высокоэнергетические линии 969 и 1588.
    from awf.io.nuclide_lib import default_library
    from awf.io.nuclide_families import collapse_families
    peak_es = [237.2, 335.0, 460.8, 501.9, 580.6, 660.2, 713.9,
               902.5, 963.8, 1440.6, 1573.0, 2607.2]
    peaks = [_fp(e, 500.0) for e in peak_es]
    lib = collapse_families(default_library())
    fwhm = lambda E: 0.062 * (661.66 * E) ** 0.5   # ≈ авто-модель файла
    res = identify_peaks(peaks, lib, fwhm_model=fwhm, apply_priors=True)
    by = {r.nuclide: r for r in res}
    assert "K-40" in by and by["K-40"].confidence >= 0.45
    assert by["K-40"].confidence > by.get("La-138", by["K-40"]).confidence \
        or "La-138" not in by
    th_lines = {round(m.line_energy, 1) for m in by["Th-232"].matches}
    assert {969.0, 1588.2} <= th_lines
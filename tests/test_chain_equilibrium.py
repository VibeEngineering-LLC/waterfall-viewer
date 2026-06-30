"""Тесты цепочечного равновесия (Задача #133)."""
from __future__ import annotations

import math

from awf.analysis.types import FoundPeak, LineMatch, IdentResult
from awf.analysis.identify import relative_efficiency
from awf.analysis.chain_equilibrium import (
    RA226_GROUP_A, RA226_GROUP_B, RA226_GROUP_C, TH232_GROUP,
    ChainGroupResult, ChainResult,
    check_ra226_chain, check_th232_chain,
    analyze_chains, identify_with_chains,
)

_BRANCHING = {"Tl-208": 0.36}


def _area_for(nuclide: float, energy: float, intensity: float, activity: float) -> float:
    """Площадь пика для равной активности (точно пропорциональна I*eff*branching)."""
    br = _BRANCHING.get(nuclide, 1.0)
    return activity * intensity * relative_efficiency(energy) * br


def _build(lines, *, activity=1.0, area_scale=None):
    """Построить (ident_results, found_peaks) из списка (nuclide, energy, intensity).

    area_scale: опц. dict {nuclide: factor} — домножить площадь нуклида (для имитации утечки Rn:
    группа B масштабируется <1). По умолчанию все 1.0.
    Возвращает (list[IdentResult], list[FoundPeak]). Один IdentResult на нуклид со всеми его
    LineMatch; на каждую линию — один FoundPeak с тем же energy и area_estimate.
    """
    area_scale = area_scale or {}
    by_nuc = {}
    peaks = []
    for (nuc, energy, intensity) in lines:
        area = _area_for(nuc, energy, intensity, activity) * area_scale.get(nuc, 1.0)
        peaks.append(FoundPeak(
            channel=0, energy=float(energy), height=float(area),
            fwhm_channels=3.0, significance=30.0, area_estimate=float(area),
        ))
        m = LineMatch(nuclide=nuc, line_energy=float(energy), peak_energy=float(energy),
                      delta_keV=0.0, intensity_pct=float(intensity))
        by_nuc.setdefault(nuc, []).append(m)
    results = [
        IdentResult(nuclide=nuc, confidence=0.9, matches=tuple(ms), category="natural")
        for nuc, ms in by_nuc.items()
    ]
    return results, peaks


# Th-232: Ac-228 911(26%),969(16%); Pb-212 238(44%); Bi-212 727(7%); Tl-208 583(85%),2614(99%).
TH232_LINES = [
    ("Ac-228", 911.16, 26.0), ("Ac-228", 968.97, 16.0),
    ("Pb-212", 238.63, 44.0), ("Bi-212", 727.33, 7.0),
    ("Tl-208", 583.19, 85.0), ("Tl-208", 2614.51, 99.0),
]

# Ra-226 равновесие: Ra-226 186(3.6%); Pb-214 295(18%),352(36%); Bi-214 609(45%),1120(15%),1764(15%).
RA226_LINES = [
    ("Ra-226", 186.21, 3.6),
    ("Pb-214", 295.22, 18.0), ("Pb-214", 351.93, 36.0),
    ("Bi-214", 609.31, 45.0), ("Bi-214", 1120.29, 15.0), ("Bi-214", 1764.49, 15.0),
]


def test_th232_chain_consistent():
    results, peaks = _build(TH232_LINES)
    chains = analyze_chains(results, peaks)
    th = [c for c in chains if c.chain_name == "Th-232"]
    assert len(th) == 1
    c = th[0]
    assert c.chain_consistent is True
    assert "Th232_daughters" in c.group_results
    g = c.group_results["Th232_daughters"]
    assert g.n_lines == 6
    assert g.passed is True
    assert g.n_pairs_passed == g.n_pairs_total  # равная активность с ветвлением → все пары точны
    assert c.rn_retention_ratio is None          # для Th-232 удержание Rn неприменимо
    assert set(c.member_nuclides) == {"Ac-228", "Pb-212", "Bi-212", "Tl-208"}


def test_th232_branching_matters():
    # площади построены с branching, затем Tl-208 искусственно поднят в (1/0.36) → как «полная активность»;
    # модель ждёт Tl с ветвлением 0.36 → рассинхрон ≈2.78×. С узким допуском 2.0 пары с Tl проваливаются.
    results, peaks = _build(TH232_LINES, area_scale={"Tl-208": 1.0 / 0.36})
    chains = analyze_chains(results, peaks, ratio_tolerance=2.0)
    c = [x for x in chains if x.chain_name == "Th-232"][0]
    g = c.group_results["Th232_daughters"]
    # пары с участием Tl-208 (отношение ≈2.78 > допуск 2.0) проваливаются → не все пары прошли
    assert g.n_pairs_passed < g.n_pairs_total


def test_ra226_equilibrium_retention_near_one():
    results, peaks = _build(RA226_LINES)  # равная активность всех групп
    chains = analyze_chains(results, peaks)
    ra = [c for c in chains if c.chain_name == "Ra-226"][0]
    assert ra.rn_retention_ratio is not None
    assert math.isclose(ra.rn_retention_ratio, 1.0, rel_tol=0.05)  # эфф-нормировка → ≈1.0
    # группа B (Pb-214+Bi-214) внутренне пропорциональна
    gb = ra.group_results["Rn222_daughters"]
    assert gb.n_lines == 5
    assert gb.passed is True
    assert ra.chain_consistent is True


def test_ra226_radon_escape_low_retention():
    results, peaks = _build(RA226_LINES, area_scale={"Pb-214": 0.3, "Bi-214": 0.3})
    chains = analyze_chains(results, peaks)
    ra = [c for c in chains if c.chain_name == "Ra-226"][0]
    assert ra.rn_retention_ratio is not None
    assert math.isclose(ra.rn_retention_ratio, 0.3, rel_tol=0.1)  # B/A ≈ 0.3 → утечка Rn
    gb = ra.group_results["Rn222_daughters"]
    assert gb.passed is True  # группа B внутренне всё ещё пропорциональна


def test_pb210_informational_only():
    lines = RA226_LINES + [("Pb-210", 46.54, 4.0)]
    results, peaks = _build(lines)
    chains = analyze_chains(results, peaks)
    ra = [c for c in chains if c.chain_name == "Ra-226"][0]
    assert "Pb210_long_lived" in ra.group_results
    gc = ra.group_results["Pb210_long_lived"]
    assert gc.passed is True            # информационная — всегда passed
    assert "Pb-210" in ra.member_nuclides
    assert ra.chain_consistent is True  # Pb-210 не ломает консистентность


def test_no_chain_when_unrelated():
    lines = [("Cs-137", 661.66, 85.0), ("K-40", 1460.82, 11.0)]
    results, peaks = _build(lines)
    chains = analyze_chains(results, peaks)
    assert chains == []


def test_single_line_group_deferred():
    lines = [("Bi-214", 609.31, 45.0)]
    results, peaks = _build(lines)
    chains = analyze_chains(results, peaks)
    ra = [c for c in chains if c.chain_name == "Ra-226"]
    assert len(ra) == 1
    gb = ra[0].group_results["Rn222_daughters"]
    assert gb.n_lines == 1
    assert gb.passed is True            # <2 линий → отложено, не FAIL
    assert ra[0].rn_retention_ratio is None  # нет Ra-226 (группа A пуста) → удержание не считается


def test_identify_with_chains_smoke():
    from awf.analysis import find_peaks  # noqa: F401  (проверка, что пакет импортируется)
    # пустой вход: identify_peaks итерирует library напрямую (for nuc in library) → пустой список.
    res, chains = identify_with_chains([], [])
    assert isinstance(res, list)
    assert isinstance(chains, list)
    assert res == []
    assert chains == []

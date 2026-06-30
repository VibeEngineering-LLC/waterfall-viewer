"""Цепочечное (decay-chain) равновесие для природных рядов (Задача #133).

Порт из SpectraVibe gamma-spectrum-analysis/scripts/gamma/identification/chain_equilibrium.py
(docstring строки 1–56, RA_226_CHAIN_GROUPS строки 70–92, TH_232_CHAIN_GROUPS строки 94–106,
ChainEquilibriumResult строки 109–121, check_ra226_chain_equilibrium строки 124–237).

Переписан под структуры вьюера (IdentResult/LineMatch/FoundPeak), а НЕ скопирован: оригинал
зависит от gamma.identification.proportionality и несёт поля library_E_keV/library_I_pct/
significance_currie/peak_area, которых во вьюере нет. Кривая эффективности переиспользуется из
awf.analysis.identify.relative_efficiency (не дублируется второй раз).

Группы заданы ЧЛЕНСТВОМ ПО ИМЕНИ нуклида (идентификация уже присвоила имена), опорные энергии —
только для оценки удержания Rn (A/B). Каждое физическое утверждение — со ссылкой на строку
оригинала (анти-галлюцинация).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from awf.analysis.types import FoundPeak, IdentResult, LineMatch
from awf.analysis.identify import relative_efficiency


# Группы цепочки Ra-226 — членство по имени нуклида (оригинал RA_226_CHAIN_GROUPS строки 70–92).
RA226_GROUP_A = ("Ra-226",)            # intrinsic 186.21 кэВ — НЕ страдает от утечки Rn (строки 33–35,72–74)
RA226_GROUP_B = ("Pb-214", "Bi-214")   # короткоживущие дочки Rn-222 (строки 36–40,80–86); утечка Rn гасит группу равномерно
RA226_GROUP_C = ("Pb-210",)            # 46.5 кэВ источник экранировки, НЕ для валидации образца (строки 41–43,87–91)

# Группа цепочки Th-232 — одна группа в секулярном равновесии (оригинал TH_232_CHAIN_GROUPS строки 94–106).
TH232_GROUP = ("Ac-228", "Pb-212", "Bi-212", "Tl-208")

# Опорные линии для оценки удержания Rn (отношение групп B/A): оригинал строки 202 (Ra 186.21), 210 (Bi-214 609.31).
RA226_REF_E_KEV = 186.21
BI214_REF_E_KEV = 609.31
_REF_TOL_KEV = 4.0

# Ветвление Bi-212 → Tl-208 = 36% (оригинал строка 104). Активность Tl-208 = 0.36 активности
# цепочки Th-232 → при межнуклидной пропорции ВНУТРИ группы ожидаемую интенсивность линий
# Tl-208 умножаем на этот фактор. Прочие члены — 1.0 (полная активность цепочки).
_CHAIN_BRANCHING = {"Tl-208": 0.36}

# Допуск эфф-пропорциональности по умолчанию. Линии цепочки разнесены широко по энергии
# (186 vs 2614 кэВ) → берём «широкий» допуск 5.0 (оригинал proportionality.py
# WIDE_RATIO_TOLERANCE_FACTOR=5.0 строка 89). Доля прошедших пар для PASS — 0.6 (строка 267).
DEFAULT_RATIO_TOLERANCE = 5.0
DEFAULT_MIN_FRACTION = 0.6


@dataclass(frozen=True)
class _ChainLine:
    """Одна совпавшая линия нуклида-члена цепочки (амплитуда = площадь пика)."""
    nuclide: str
    energy: float       # библиотечная энергия линии, кэВ
    intensity: float    # библиотечная интенсивность линии, %
    area: float         # площадь сопоставленного найденного пика (амплитуда-прокси)


@dataclass(frozen=True)
class ChainGroupResult:
    """Результат проверки пропорциональности ВНУТРИ одной группы цепочки."""
    group: str                      # имя группы (напр. "Rn222_daughters")
    nuclides: Tuple[str, ...]       # реально найденные нуклиды-члены этой группы
    n_lines: int                    # число надёжных линий в проверке
    n_pairs_passed: int
    n_pairs_total: int
    passed: bool                    # доля прошедших пар >= min_fraction; <2 линий → True (отложено)
    reason: str


@dataclass(frozen=True)
class ChainResult:
    """Результат анализа одной распадной цепочки."""
    chain_name: str                          # "Ra-226" / "Th-232"
    group_results: Dict[str, ChainGroupResult]
    member_nuclides: Tuple[str, ...]         # найденные нуклиды-члены цепочки (по убыванию числа линий, затем по имени)
    rn_retention_ratio: Optional[float]      # B/A (только Ra-226): 1.0=равновесие, <1=утечка Rn, >1=избыток дочек; None если нечем считать
    chain_consistent: bool                   # каждая заполненная группа (>=2 линий, кроме C) пропорциональна внутри
    notes: str


def _match_area(peak_energy: float, found_peaks: Sequence[FoundPeak]) -> float:
    """Площадь найденного пика, ближайшего по энергии к peak_energy. 0.0 если пиков нет.

    LineMatch.peak_energy в identify_peaks выставлен равным FoundPeak.energy, поэтому
    обычно совпадение точное; берём ближайший на случай численных расхождений.
    """
    best_area = 0.0
    best_d = None
    for pk in found_peaks:
        d = abs(pk.energy - peak_energy)
        if best_d is None or d < best_d:
            best_d = d
            best_area = pk.area_estimate
    return float(best_area)


def _collect_chain_lines(
    ident_results: Sequence[IdentResult],
    found_peaks: Sequence[FoundPeak],
) -> Dict[str, List[_ChainLine]]:
    """Собрать совпавшие линии по нуклидам: {nuclide: [_ChainLine, ...]}.

    Для каждого IdentResult и каждого его LineMatch берём библиотечную энергию/интенсивность
    линии и площадь сопоставленного пика (_match_area по peak_energy). Линии с intensity<=0
    или area<=0 пропускаются (для пропорциональности бесполезны).
    """
    by_nuclide: Dict[str, List[_ChainLine]] = {}
    for res in ident_results:
        for m in res.matches:
            area = _match_area(m.peak_energy, found_peaks)
            if m.intensity_pct <= 0 or area <= 0:
                continue
            cl = _ChainLine(nuclide=m.nuclide, energy=m.line_energy,
                            intensity=m.intensity_pct, area=area)
            by_nuclide.setdefault(m.nuclide, []).append(cl)
    return by_nuclide


def _group_proportionality(
    group_name: str,
    lines: Sequence[_ChainLine],
    ratio_tolerance: float,
    min_fraction: float,
) -> ChainGroupResult:
    """Доля пар линий с эфф-пропорциональными площадями (с поправкой на ветвление).

    Для пары (i, j):
        eff_i = relative_efficiency(E_i), eff_j = relative_efficiency(E_j)
        br_i  = _CHAIN_BRANCHING.get(nuclide_i, 1.0), br_j аналогично
        expected = (I_i * eff_i * br_i) / (I_j * eff_j * br_j)
        observed = area_i / area_j
        пара проходит, если 1/ratio_tolerance <= observed/expected <= ratio_tolerance.
    Линии с intensity<=0 или area<=0 уже отфильтрованы в _collect_chain_lines, но повторно
    защититься. Найденные нуклиды группы = отсортированный set имён.
    При <2 линий: passed=True, reason "<2 линий — отложено", пар 0.
    """
    reliable = [ln for ln in lines if ln.intensity > 0 and ln.area > 0]
    nuclides = tuple(sorted({ln.nuclide for ln in reliable}))
    n_lines = len(reliable)
    if n_lines < 2:
        return ChainGroupResult(
            group=group_name, nuclides=nuclides, n_lines=n_lines,
            n_pairs_passed=0, n_pairs_total=0, passed=True,
            reason=f"{n_lines} линий (<2) — проверка отложена",
        )
    n_pass = 0
    n_total = 0
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            li = reliable[i]
            lj = reliable[j]
            eff_i = relative_efficiency(li.energy)
            eff_j = relative_efficiency(lj.energy)
            br_i = _CHAIN_BRANCHING.get(li.nuclide, 1.0)
            br_j = _CHAIN_BRANCHING.get(lj.nuclide, 1.0)
            denom = lj.intensity * eff_j * br_j
            if denom <= 0:
                continue
            expected = (li.intensity * eff_i * br_i) / denom
            if expected <= 0:
                continue
            if lj.area <= 0:
                continue
            observed = li.area / lj.area
            ratio = observed / expected
            n_total += 1
            if (1.0 / ratio_tolerance) <= ratio <= ratio_tolerance:
                n_pass += 1
    if n_total == 0:
        return ChainGroupResult(
            group=group_name, nuclides=nuclides, n_lines=n_lines,
            n_pairs_passed=0, n_pairs_total=0, passed=True,
            reason="нет пригодных пар — проверка отложена",
        )
    frac = n_pass / n_total
    passed = frac >= min_fraction
    return ChainGroupResult(
        group=group_name, nuclides=nuclides, n_lines=n_lines,
        n_pairs_passed=n_pass, n_pairs_total=n_total, passed=passed,
        reason=(f"{n_pass}/{n_total} пар эфф-пропорциональны (допуск {ratio_tolerance:.0f}×); "
                f"{'PASS' if passed else 'FAIL'} (нужно >={min_fraction:.0%})"),
    )


def _nearest_norm_amplitude(
    lines: Sequence[_ChainLine], ref_energy: float, tol: float
) -> Optional[float]:
    """area/(I*eff) линии, ближайшей по энергии к ref_energy в пределах ±tol; иначе None.

    Нормировка по эффективности → «активность-эквивалент» (в отличие от оригинала, который
    делил только на I — оригинал строки 204/212). Эфф-нормировка даёт B/A≈1.0 при равновесии,
    что соответствует документированной семантике удержания Rn.
    """
    best = None
    best_d = tol
    for ln in lines:
        d = abs(ln.energy - ref_energy)
        if d <= best_d and ln.intensity > 0 and ln.area > 0:
            eff = relative_efficiency(ln.energy)
            if eff > 0:
                best = ln.area / (ln.intensity * eff)
                best_d = d
    return best


def check_ra226_chain(
    by_nuclide: Dict[str, List[_ChainLine]],
    *,
    ratio_tolerance: float = DEFAULT_RATIO_TOLERANCE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> ChainResult:
    """Проверить цепочку Ra-226 с учётом нарушения равновесия по Rn (оригинал строки 124–237).

    Группа A (Ra-226 intrinsic) — обычно 1 линия (186), проверка откладывается. Группа B
    (Pb-214+Bi-214) — основная проверка пропорциональности (короткие T½ ↔ взаимное равновесие).
    Группа C (Pb-210) — информационная, НЕ валидирует цепочку. Удержание Rn = B/A по опорным
    линиям Ra-226 186 и Bi-214 609 (эфф-нормированным): 1.0=равновесие, <1=утечка Rn.
    chain_consistent = все заполненные группы (>=2 линий, кроме C) пропорциональны внутри.
    """
    a_lines = [ln for n in RA226_GROUP_A for ln in by_nuclide.get(n, [])]
    b_lines = [ln for n in RA226_GROUP_B for ln in by_nuclide.get(n, [])]
    c_lines = [ln for n in RA226_GROUP_C for ln in by_nuclide.get(n, [])]
    group_results = {}
    group_results["Ra-226_intrinsic"] = _group_proportionality("Ra-226_intrinsic", a_lines, ratio_tolerance, min_fraction)
    group_results["Rn222_daughters"] = _group_proportionality("Rn222_daughters", b_lines, ratio_tolerance, min_fraction)
    if c_lines:
        group_results["Pb210_long_lived"] = ChainGroupResult(group="Pb210_long_lived",
            nuclides=("Pb-210",), n_lines=len(c_lines), n_pairs_passed=0, n_pairs_total=0, passed=True,
            reason="Pb-210 обнаружен, но НЕ используется для валидации цепочки (источник — свинец защиты детектора)")
    rn = None
    if a_lines and b_lines:
        ra = _nearest_norm_amplitude(a_lines, RA226_REF_E_KEV, _REF_TOL_KEV)
        bi = _nearest_norm_amplitude(b_lines, BI214_REF_E_KEV, _REF_TOL_KEV)
        if ra is not None and bi is not None and ra > 0:
            rn = bi / ra  # B/A: <1 = утечка Rn
    member_nuclides = tuple(sorted(
        {n for n in (RA226_GROUP_A + RA226_GROUP_B + RA226_GROUP_C) if by_nuclide.get(n, [])},
        key=lambda x: (-len(by_nuclide.get(x, [])), x)
    ))
    chain_consistent = all(r.passed for key, r in group_results.items()
                           if key != "Pb210_long_lived" and r.n_lines >= 2)
    if not any(r.n_lines >= 2 for key, r in group_results.items() if key != "Pb210_long_lived"):
        chain_consistent = True
    notes = "Цепочка Ra-226 валидируется пропорциональностью ВНУТРИ групп; Pb-210 не используется (свинец защиты); B/A = удержание Rn-222."
    return ChainResult(chain_name="Ra-226", group_results=group_results, member_nuclides=member_nuclides,
                       rn_retention_ratio=rn, chain_consistent=chain_consistent, notes=notes)


def check_th232_chain(
    by_nuclide: Dict[str, List[_ChainLine]],
    *,
    ratio_tolerance: float = DEFAULT_RATIO_TOLERANCE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> ChainResult:
    """Проверить цепочку Th-232 (оригинал TH_232_CHAIN_GROUPS строки 94–106).

    Одна группа в секулярном равновесии; Tl-208 учитывается с ветвлением 0.36 (строка 104).
    Газообразного промежуточного звена нет → удержание Rn неприменимо (None).
    """
    g_lines = [ln for n in TH232_GROUP for ln in by_nuclide.get(n, [])]
    gr = _group_proportionality("Th232_daughters", g_lines, ratio_tolerance, min_fraction)
    group_results = {"Th232_daughters": gr}
    member_nuclides = tuple(sorted(
        [n for n in TH232_GROUP if by_nuclide.get(n, [])],
        key=lambda x: (-len(by_nuclide.get(x, [])), x)
    ))
    chain_consistent = gr.passed if gr.n_lines >= 2 else True
    notes = "Цепочка Th-232: секулярное равновесие; Tl-208 с ветвлением Bi-212→Tl-208 36%."
    return ChainResult(chain_name="Th-232", group_results=group_results, member_nuclides=member_nuclides,
                       rn_retention_ratio=None, chain_consistent=chain_consistent, notes=notes)


def analyze_chains(
    ident_results: Sequence[IdentResult],
    found_peaks: Sequence[FoundPeak],
    *,
    ratio_tolerance: float = DEFAULT_RATIO_TOLERANCE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> List[ChainResult]:
    """Собрать распадные цепочки из результатов идентификации (Задача #133).

    Возвращает список ChainResult ТОЛЬКО для цепочек, у которых найден хотя бы один
    нуклид-член (member_nuclides непуст). Порядок: Ra-226, затем Th-232 (если присутствуют).
    """
    by_nuclide = _collect_chain_lines(ident_results, found_peaks)
    out: List[ChainResult] = []
    ra = check_ra226_chain(by_nuclide, ratio_tolerance=ratio_tolerance, min_fraction=min_fraction)
    if ra.member_nuclides:
        out.append(ra)
    th = check_th232_chain(by_nuclide, ratio_tolerance=ratio_tolerance, min_fraction=min_fraction)
    if th.member_nuclides:
        out.append(th)
    return out


def identify_with_chains(
    found_peaks,
    library,
    fwhm_model=None,
    *,
    ratio_tolerance: float = DEFAULT_RATIO_TOLERANCE,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    **identify_kwargs,
) -> Tuple[List[IdentResult], List[ChainResult]]:
    """identify_peaks(...) + analyze_chains(...). Возвращает (результаты_ид, цепочки).

    Импорт identify_peaks локальный (внутри функции), чтобы избежать циклического импорта
    на уровне модуля (identify уже импортирован для relative_efficiency — но identify_peaks
    тянем лениво для чистоты).
    """
    from awf.analysis.identify import identify_peaks
    results = identify_peaks(found_peaks, library, fwhm_model, **identify_kwargs)
    chains = analyze_chains(results, found_peaks,
                            ratio_tolerance=ratio_tolerance, min_fraction=min_fraction)
    return results, chains


__all__ = [
    "RA226_GROUP_A", "RA226_GROUP_B", "RA226_GROUP_C", "TH232_GROUP",
    "ChainGroupResult", "ChainResult",
    "check_ra226_chain", "check_th232_chain",
    "analyze_chains", "identify_with_chains",
]

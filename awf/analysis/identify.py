"""Идентификация нуклидов по найденным пикам (Задача 11, ТЗ-A.10; Задача #130).

Упрощённый Lsrm-подход (Algorithmic Foundations §6) с доработкой #130 — устранение
систематически НЕВЕРНОЙ идентификации (на ториевой пробе движок выдавал Be-7/Am-243/
Bi-207/Nb-95/F-18 вместо Th-232/Tl-208/Th-228/Ac-228/Pb-212 + слабого урана и K-40):

  1. Для каждого нуклида-кандидата берём characteristic line — линию с максимальной
     интенсивностью.
  2. ДОПУСК (#130). Прежний жёсткий гейт «найден пик characteristic line» отвергал
     реальные цепочечные нуклиды, у которых самая интенсивная линия — недетектируемый
     низкоэнергетический рентген (Ac-228 15.8 кэВ, U-238 15.8, Th-234 16.2, Bi-212
     12.1), хотя сильные гамма-линии найдены. Теперь кандидат допускается, если найден
     пик characteristic line ИЛИ совпало не менее 2 различных пиков.
  3. Сопоставление пик-линия — жадно по убыванию интенсивности линии; один найденный
     пик принадлежит максимум одной (сильнейшей) линии (#130: дедуп, иначе две близкие
     линии цепляются за один пик и порождают ложную «непропорциональную» пару).
  4. ПРОПОРЦИОНАЛЬНОСТЬ с учётом эффективности (#130). Площадь пика пропорц. I*eps(E),
     а не просто I: низкоэнергетические линии регистрируются эффективнее. Без поправки
     на eps(E) реальные цепочки (Th-232: 238 кэВ против 2614 кэВ) отбраковывались как
     «нарушение пропорций». Проверяются только НАДЁЖНЫЕ линии (не менее 10 % характери-
     стической интенсивности и пик не менее 2 % характеристической площади) — слабые
     рентген/шумовые совпадения исключаются. Если доля прошедших пар надёжных линий
     ниже min_prop_fraction и пары есть — кандидат отвергается.
  5. КАЧЕСТВО ЭНЕРГИИ (#130). Множитель eq = sum(I*max(0,1-|Δ|/окно))/sum(I) по
     совпавшим линиям демотирует кандидатов, чьи линии «зацепились» у самого края окна
     (источник ложных опознаний — Be-7 477.6 кэВ на хвосте). Точное совпадение (Δ=0)
     даёт eq=1.0.
  6. confidence из [0..1] = coverage * prop_factor * eq. При apply_priors=True
     домножается на plausibility(nuclide) — априор по имени (RARE_ISOTOPE_PRIOR) или
     категории (CATEGORY_PRIOR): природные цепочки встречаются всегда, осколочные/
     активационные/медицинские — редко.

Окно матчинга: tol(E) = tol_factor * FWHM(E). FWHM(E) берётся из переданной модели
fwhm_model(E) или из грубой сцинтилляционной модели default_fwhm_keV.

Модуль Qt-free; зависит только от stdlib + awf.io.nuclide_lib +
awf.io.nuclide_categories + awf.analysis.types.

Источник методики: Lsrm Algorithmic Foundations 2022 §6, §14.2, §14.3; кривая
эффективности eps(E) — типовая (форма ~Гамма-1С, разрешена оператором как типовая
2026-06-25); rare-isotope и category priors — практическая калибровка.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from awf.io.nuclide_lib import GammaLine, Nuclide
from awf.io.nuclide_categories import category_of
from awf.analysis.types import FoundPeak, IdentResult, LineMatch


# Приоритеты редких изотопов: множитель, понижающий уверенность для нуклидов
# с низкой априорной вероятностью встречи. Применяется только при apply_priors=True.
RARE_ISOTOPE_PRIOR = {
    "Zn-65": 0.05,
    "Cs-134": 0.3,
    "Co-58": 0.1,
    "Mn-54": 0.1,
    "Co-60": 0.5,
    "Eu-152": 0.5,
    "Eu-154": 0.2,
    "Ba-133": 0.2,
    "Na-22": 0.1,
    "Am-241": 0.3,
    "Be-7": 0.5,
    "Sb-125": 0.2,
    "Ru-106": 0.2,
    "Ag-110m": 0.2,
    "I-131": 0.2,
    "I-133": 0.1,
    "Co-57": 0.2,
    "In-111": 0.1,
    # Задача #130: пер-именные приоритеты для распространённых техногенных/дочерних,
    # которые НЕ следует давить как редкую «экзотику» (перекрывают CATEGORY_PRIOR ниже).
    "Cs-137": 0.5,    # очень распространён (фоллаут/медицина/поверки) — умеренный приоритет
    "Ba-137m": 0.15,  # дочерний Cs-137, та же линия 661.66 кэВ — дубль, демотировать
}
DEFAULT_PRIOR = 1.0


def get_prior(nuclide: str) -> float:
    """Априорный множитель уверенности для нуклида (1.0 — обычный)."""
    return RARE_ISOTOPE_PRIOR.get(nuclide, DEFAULT_PRIOR)


# Задача #130: типовая ОТНОСИТЕЛЬНАЯ эффективность регистрации в фотопике eps(E).
# Форма ~Гамма-1С (сцинтилляционный детектор): рост от низких энергий за счёт ухода
# из-под порога, максимум ~60 кэВ, затем степенной спад. Узлы (E_кэВ, eps_отн);
# интерполяция — линейная в log-log, плоская экстраполяция за краями. Абсолютная
# нормировка не важна — в пропорциональности входят только ОТНОШЕНИЯ eps(E_i)/eps(E_j).
_EFF_E_ANCHORS = (
    20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 150.0, 186.0, 238.0, 300.0,
    351.0, 511.0, 583.0, 662.0, 911.0, 1173.0, 1332.0, 1461.0, 2000.0, 2614.0,
)
_EFF_F_ANCHORS = (
    0.30, 0.55, 0.85, 1.00, 0.97, 0.88, 0.66, 0.54, 0.45, 0.36,
    0.31, 0.205, 0.175, 0.152, 0.107, 0.082, 0.072, 0.065, 0.045, 0.032,
)


def relative_efficiency(E_keV: float) -> float:
    """Типовая относительная эффективность фотопика eps(E) (Задача #130).

    Логарифмически-линейная интерполяция по узлам (_EFF_E_ANCHORS, _EFF_F_ANCHORS).
    За пределами диапазона узлов — плоская экстраполяция краевыми значениями.
    Возвращает положительное число (относительные единицы).
    """
    E = float(E_keV)
    xs = _EFF_E_ANCHORS
    ys = _EFF_F_ANCHORS
    if E <= xs[0]:
        return ys[0]
    if E >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= E:
            lo = mid
        else:
            hi = mid
    lx0 = math.log(xs[lo])
    lx1 = math.log(xs[hi])
    ly0 = math.log(ys[lo])
    ly1 = math.log(ys[hi])
    t = (math.log(E) - lx0) / (lx1 - lx0)
    return math.exp(ly0 + t * (ly1 - ly0))


# Задача #130: априоры правдоподобия по КАТЕГОРИИ. Природные ряды (U/Th/K-40)
# присутствуют почти в любой пробе → 1.0; осколочные/активационные/медицинские/
# космогенные — редки, демотируются. Перекрываются пер-именными RARE_ISOTOPE_PRIOR.
CATEGORY_PRIOR = {
    "natural": 1.0,
    "cosmogenic": 0.3,
    "technogenic": 0.4,
    "medical": 0.2,
    "fission": 0.25,
}


def plausibility(nuclide: Nuclide) -> float:
    """Априорный множитель правдоподобия нуклида (Задача #130).

    Пер-именной приоритет из RARE_ISOTOPE_PRIOR имеет приоритет; иначе — априор по
    категории (CATEGORY_PRIOR), категория берётся из nuclide.category или category_of.
    """
    if nuclide.name in RARE_ISOTOPE_PRIOR:
        return RARE_ISOTOPE_PRIOR[nuclide.name]
    cat = nuclide.category if nuclide.category is not None else category_of(nuclide.name)
    return CATEGORY_PRIOR.get(cat, DEFAULT_PRIOR)


def default_fwhm_keV(E_keV: float, *, resolution_at_662: float = 0.07) -> float:
    """Грубая модель FWHM(E) сцинтиллятора.

    Относительное разрешение R(E) пропорц. 1/sqrt(E), поэтому
    FWHM(E) = E*R(E) = resolution_at_662 * sqrt(661.66 * E). По умолчанию 7 %
    при 661.66 кэВ (типично для CsI(Tl) AtomSpectra).
    """
    E = max(float(E_keV), 1.0)
    return float(resolution_at_662) * math.sqrt(661.66 * E)


def _window_keV(E_keV: float,
                fwhm_model: Optional[Callable[[float], float]],
                tol_factor: float) -> float:
    """Полуширина окна матчинга на энергии E: tol_factor * FWHM(E)."""
    fwhm = fwhm_model(E_keV) if fwhm_model is not None else default_fwhm_keV(E_keV)
    return max(float(tol_factor) * float(fwhm), 1e-6)


def lookup_by_energy(E_keV: float,
                     tol_keV: float,
                     library: Sequence[Nuclide],
                     *,
                     min_intensity_pct: float = 0.0) -> List[LineMatch]:
    """Все библиотечные линии в окне ±tol_keV от E_keV, по возрастанию |Δ|.

    Подзадача 11.1. Возвращает LineMatch с peak_energy=E_keV и
    delta_keV = line.energy - E_keV. Линии с intensity < min_intensity_pct
    и не used пропускаются.
    """
    hits: List[LineMatch] = []
    for nuc in library:
        for line in nuc.lines:
            if not line.used:
                continue
            if line.intensity < min_intensity_pct:
                continue
            delta = line.energy - E_keV
            if abs(delta) <= tol_keV:
                hits.append(LineMatch(
                    nuclide=nuc.name,
                    line_energy=line.energy,
                    peak_energy=E_keV,
                    delta_keV=delta,
                    intensity_pct=line.intensity,
                ))
    hits.sort(key=lambda h: abs(h.delta_keV))
    return hits


def _nearest_peak(peaks: Sequence[FoundPeak],
                  E_keV: float,
                  window_keV: float) -> Optional[FoundPeak]:
    """Ближайший по энергии пик в пределах ±window_keV, либо None."""
    best: Optional[FoundPeak] = None
    best_d = window_keV
    for pk in peaks:
        d = abs(pk.energy - E_keV)
        if d <= best_d:
            best = pk
            best_d = d
    return best


# Задача #130: пороги «надёжной» линии для эфф-пропорциональности. Линия учитывается
# в проверке пропорций, только если её интенсивность не ниже доли _PROP_REL_FRAC от
# характеристической И площадь её пика не ниже доли _PROP_REL_AREA от характеристической
# площади. Так слабые рентген/шумовые совпадения не отравляют тест пропорций.
_PROP_REL_FRAC = 0.10
_PROP_REL_AREA = 0.02


def _proportionality_eff(matched: Sequence[Tuple[GammaLine, FoundPeak, float]],
                         ratio_tolerance: float,
                         *,
                         char_intensity: float,
                         char_peak_area: float) -> Tuple[float, int]:
    """Доля пар НАДЁЖНЫХ линий с эфф-пропорциональными площадями и число пар (#130).

    Для пары надёжных линий (i, j):
      expected = (I_i * eps(E_i)) / (I_j * eps(E_j)),  observed = area_i / area_j.
      Пара проходит, если 1/ratio_tolerance <= observed/expected <= ratio_tolerance.
    Надёжная линия: intensity >= _PROP_REL_FRAC*char_intensity и
    area >= _PROP_REL_AREA*char_peak_area. При числе пар 0 возвращает (1.0, 0).
    """
    reliable = []
    for (ln, pk, _w) in matched:
        if ln.intensity <= 0 or pk.area_estimate <= 0:
            continue
        if char_intensity > 0 and ln.intensity < _PROP_REL_FRAC * char_intensity:
            continue
        if char_peak_area > 0 and pk.area_estimate < _PROP_REL_AREA * char_peak_area:
            continue
        reliable.append((ln, pk))
    n_pass = 0
    n_total = 0
    for i in range(len(reliable)):
        for j in range(i + 1, len(reliable)):
            ln_i, pk_i = reliable[i]
            ln_j, pk_j = reliable[j]
            eff_i = relative_efficiency(ln_i.energy)
            eff_j = relative_efficiency(ln_j.energy)
            denom = ln_j.intensity * eff_j
            if denom <= 0:
                continue
            expected = (ln_i.intensity * eff_i) / denom
            if expected <= 0:
                continue
            observed = pk_i.area_estimate / pk_j.area_estimate
            ratio = observed / expected
            n_total += 1
            if (1.0 / ratio_tolerance) <= ratio <= ratio_tolerance:
                n_pass += 1
    if n_total == 0:
        return 1.0, 0
    return n_pass / n_total, n_total


def _energy_quality(matched: Sequence[Tuple[GammaLine, FoundPeak, float]]) -> float:
    """Множитель качества энергии eq из [0..1] по совпавшим линиям (#130).

    eq = sum(I * max(0, 1 - |Δ|/окно)) / sum(I), Δ = pk.energy - ln.energy. Точные
    совпадения (Δ=0) дают 1.0; зацепы у края окна тянут eq вниз. При нулевой сумме
    интенсивностей возвращает 0.0.
    """
    num = 0.0
    den = 0.0
    for (ln, pk, win) in matched:
        if ln.intensity <= 0:
            continue
        w = win if win > 0 else 1e-6
        closeness = 1.0 - abs(pk.energy - ln.energy) / w
        if closeness < 0.0:
            closeness = 0.0
        num += ln.intensity * closeness
        den += ln.intensity
    if den <= 0:
        return 0.0
    return num / den


def identify_peaks(found_peaks: Sequence[FoundPeak],
                   library: Sequence[Nuclide],
                   fwhm_model: Optional[Callable[[float], float]] = None,
                   *,
                   tol_factor: float = 0.5,
                   ratio_tolerance: float = 3.0,
                   min_prop_fraction: float = 0.5,
                   min_confidence: float = 0.0,
                   apply_priors: bool = False) -> List[IdentResult]:
    """Идентифицировать нуклиды по найденным пикам (Задача 11; доработка #130).

    Для каждого нуклида из library допускает кандидата (characteristic line найдена
    ИЛИ совпало не менее 2 пиков), жадно сопоставляет линии пикам (дедуп), проверяет
    эфф-пропорциональность площадей надёжных линий и считает confidence из [0..1] =
    coverage * prop_factor * energy_quality. Возвращает обнаруженные кандидаты по
    убыванию confidence.

    Args:
        found_peaks: найденные фотопики (FoundPeak с energy/area_estimate).
        library: список Nuclide.
        fwhm_model: callable(E_keV)->FWHM(E) в кэВ; None -> default_fwhm_keV.
        tol_factor: полуширина окна матчинга = tol_factor * FWHM(E).
        ratio_tolerance: допуск эфф-пропорциональности площадей (~3.0).
        min_prop_fraction: если доля прошедших пар надёжных линий ниже этого и пары
            есть — кандидат отвергается (пропорции активно нарушены).
        min_confidence: отбросить кандидатов с confidence ниже порога.
        apply_priors: домножать confidence на plausibility(nuclide) (#130).

    Returns:
        list[IdentResult], по убыванию confidence.
    """
    peaks = list(found_peaks)
    results: List[IdentResult] = []

    for nuc in library:
        lines = [ln for ln in nuc.lines if ln.used and ln.intensity > 0]
        if not lines:
            continue

        characteristic = max(lines, key=lambda ln: ln.intensity)

        # #130: жадный дедуп «пик-линия» по убыванию интенсивности линии; один найденный
        # пик принадлежит максимум одной (сильнейшей) линии. Иначе две близкие линии
        # (напр. Ac-228 964.8 и 969.0) цепляются за один пик и дают ложную пару.
        claimed = set()
        matched: List[Tuple[GammaLine, FoundPeak, float]] = []
        for ln in sorted(lines, key=lambda l: l.intensity, reverse=True):
            win = _window_keV(ln.energy, fwhm_model, tol_factor)
            best = None
            best_d = win
            for pk in peaks:
                if id(pk) in claimed:
                    continue
                d = abs(pk.energy - ln.energy)
                if d <= best_d:
                    best = pk
                    best_d = d
            if best is not None:
                claimed.add(id(best))
                matched.append((ln, best, win))
        if not matched:
            continue

        # #130: допуск кандидата. Прежний гейт «characteristic line найдена» рубил
        # цепочечные нуклиды, у которых самая интенсивная линия — недетектируемый
        # низкоэнергетический рентген. Допускаем, если найдена characteristic ИЛИ
        # совпало не менее 2 различных пиков.
        char_found = any(ln is characteristic for (ln, _pk, _w) in matched)
        if not char_found and len(matched) < 2:
            continue

        if char_found:
            char_peak_area = next(pk.area_estimate
                                  for (ln, pk, _w) in matched if ln is characteristic)
        else:
            char_peak_area = max(pk.area_estimate for (_ln, pk, _w) in matched)

        total_I = sum(ln.intensity for ln in lines)
        matched_I = sum(ln.intensity for (ln, _pk, _w) in matched)
        coverage = matched_I / total_I if total_I > 0 else 0.0

        prop_fraction, n_pairs = _proportionality_eff(
            matched, ratio_tolerance,
            char_intensity=characteristic.intensity,
            char_peak_area=char_peak_area)
        if n_pairs > 0 and prop_fraction < min_prop_fraction:
            continue  # площади надёжных линий противоречат эфф-пропорциям — отбраковка
        prop_factor = prop_fraction if n_pairs > 0 else 1.0

        eq_factor = _energy_quality(matched)

        confidence = coverage * prop_factor * eq_factor
        if apply_priors:
            confidence *= plausibility(nuc)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < min_confidence:
            continue

        matches = tuple(
            LineMatch(
                nuclide=nuc.name,
                line_energy=ln.energy,
                peak_energy=pk.energy,
                delta_keV=pk.energy - ln.energy,
                intensity_pct=ln.intensity,
            )
            for (ln, pk, _w) in matched)
        category = nuc.category if nuc.category is not None else category_of(nuc.name)
        results.append(IdentResult(
            nuclide=nuc.name,
            confidence=confidence,
            matches=matches,
            category=category,
        ))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


__all__ = [
    "RARE_ISOTOPE_PRIOR",
    "DEFAULT_PRIOR",
    "get_prior",
    "relative_efficiency",
    "CATEGORY_PRIOR",
    "plausibility",
    "default_fwhm_keV",
    "lookup_by_energy",
    "identify_peaks",
]

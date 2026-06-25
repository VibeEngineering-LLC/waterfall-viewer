"""Идентификация нуклидов по найденным пикам (Задача 11, ТЗ-A.10).

Упрощённый Lsrm-подход (Algorithmic Foundations §6):
  1. Для каждого нуклида-кандидата берём characteristic line — линию с
     максимальной интенсивностью (без модели эффективности/MDA это разумное
     приближение к «линии с минимальным MDA»).
  2. Если в окне ±tol(E) characteristic line нет найденного пика — кандидат
     отвергается (не обнаружен).
  3. Иначе ищем пики для всех линий кандидата, собираем совпадения.
  4. Пропорциональность: соотношение площадей совпавших пиков должно
     соответствовать табличным интенсивностям в пределах ratio_tolerance.
     Если большинство пар нарушают пропорции — кандидат отвергается.
  5. Индекс уверенности confidence из [0..1] = coverage * prop_factor, где
     coverage — доля суммарной интенсивности линий, объяснённая найденными
     пиками. (В отличие от безразмерного log-CI Lsrm §14.3, для UI нужен
     ограниченный [0..1] индекс.)

Окно матчинга: tol(E) = tol_factor * FWHM(E). FWHM(E) берётся из переданной
модели fwhm_model(E) или из грубой сцинтилляционной модели default_fwhm_keV.

Модуль Qt-free; зависит только от stdlib + awf.io.nuclide_lib +
awf.io.nuclide_categories + awf.analysis.types.

Источник методики: Lsrm Algorithmic Foundations 2022 §6, §14.2, §14.3;
rare-isotope priors — практическая калибровка (порт proportionality.py).
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
}
DEFAULT_PRIOR = 1.0


def get_prior(nuclide: str) -> float:
    """Априорный множитель уверенности для нуклида (1.0 — обычный)."""
    return RARE_ISOTOPE_PRIOR.get(nuclide, DEFAULT_PRIOR)


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


def _proportionality(matched_pairs: Sequence[Tuple[GammaLine, FoundPeak]],
                     ratio_tolerance: float) -> Tuple[float, int]:
    """Доля пар линий с пропорциональными площадями и число проверенных пар.

    Для каждой пары используемых линий (area>0, intensity>0):
      expected = I_i / I_j,  observed = area_i / area_j.
      Пара проходит, если 1/ratio_tolerance <= observed/expected <= ratio_tolerance.
    При n_pairs == 0 возвращает (1.0, 0) — проверять нечем, не штрафуем.
    """
    usable = [(ln, pk) for (ln, pk) in matched_pairs
              if pk.area_estimate > 0 and ln.intensity > 0]
    n_pass = 0
    n_total = 0
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            ln_i, pk_i = usable[i]
            ln_j, pk_j = usable[j]
            expected = ln_i.intensity / ln_j.intensity
            observed = pk_i.area_estimate / pk_j.area_estimate
            if expected <= 0:
                continue
            ratio = observed / expected
            n_total += 1
            if (1.0 / ratio_tolerance) <= ratio <= ratio_tolerance:
                n_pass += 1
    if n_total == 0:
        return 1.0, 0
    return n_pass / n_total, n_total


def identify_peaks(found_peaks: Sequence[FoundPeak],
                   library: Sequence[Nuclide],
                   fwhm_model: Optional[Callable[[float], float]] = None,
                   *,
                   tol_factor: float = 0.5,
                   ratio_tolerance: float = 3.0,
                   min_prop_fraction: float = 0.5,
                   min_confidence: float = 0.0,
                   apply_priors: bool = False) -> List[IdentResult]:
    """Идентифицировать нуклиды по найденным пикам (Задача 11).

    Для каждого нуклида из library проверяет наличие characteristic line,
    собирает совпавшие линии, проверяет пропорциональность площадей и
    считает confidence из [0..1]. Возвращает обнаруженные кандидаты,
    отсортированные по убыванию confidence.

    Args:
        found_peaks: найденные фотопики (FoundPeak с energy/area_estimate).
        library: список Nuclide.
        fwhm_model: callable(E_keV)->FWHM(E) в кэВ; None -> default_fwhm_keV.
        tol_factor: полуширина окна матчинга = tol_factor * FWHM(E).
        ratio_tolerance: допуск пропорциональности площадей (~3.0).
        min_prop_fraction: если доля прошедших пар < этого и пары есть —
            кандидат отвергается (пропорции активно нарушены).
        min_confidence: отбросить кандидатов с confidence ниже порога.
        apply_priors: домножать confidence на rare-isotope prior.

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
        char_window = _window_keV(characteristic.energy, fwhm_model, tol_factor)
        if _nearest_peak(peaks, characteristic.energy, char_window) is None:
            continue  # characteristic line не найдена — кандидат не обнаружен

        matches: List[LineMatch] = []
        matched_pairs: List[Tuple[GammaLine, FoundPeak]] = []
        for ln in lines:
            win = _window_keV(ln.energy, fwhm_model, tol_factor)
            pk = _nearest_peak(peaks, ln.energy, win)
            if pk is not None:
                matches.append(LineMatch(
                    nuclide=nuc.name,
                    line_energy=ln.energy,
                    peak_energy=pk.energy,
                    delta_keV=pk.energy - ln.energy,
                    intensity_pct=ln.intensity,
                ))
                matched_pairs.append((ln, pk))

        total_I = sum(ln.intensity for ln in lines)
        matched_I = sum(ln.intensity for (ln, _) in matched_pairs)
        coverage = matched_I / total_I if total_I > 0 else 0.0

        prop_fraction, n_pairs = _proportionality(matched_pairs, ratio_tolerance)
        if n_pairs > 0 and prop_fraction < min_prop_fraction:
            continue  # пропорции площадей противоречат — отбраковка
        prop_factor = prop_fraction if n_pairs > 0 else 1.0

        confidence = coverage * prop_factor
        if apply_priors:
            confidence *= get_prior(nuc.name)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < min_confidence:
            continue

        category = nuc.category if nuc.category is not None else category_of(nuc.name)
        results.append(IdentResult(
            nuclide=nuc.name,
            confidence=confidence,
            matches=tuple(matches),
            category=category,
        ))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


__all__ = [
    "RARE_ISOTOPE_PRIOR",
    "DEFAULT_PRIOR",
    "get_prior",
    "default_fwhm_keV",
    "lookup_by_energy",
    "identify_peaks",
]
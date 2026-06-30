"""Авто-сегментация спектрограммы по времени + посегментная идентификация (Задача #131).

Запись часто временно-структурирована: источники сменяются (напр. ториевый электрод →
урановое стекло → фон). В интегральном по времени спектре длинный/яркий сегмент
доминирует и забивает короткие — слабый источник (уран, K-40) не идентифицируется.
Модуль делит ось времени на участки квазистационарного спектрального состава
(Пуассоновский change-point по макро-полосам энергии) и прогоняет идентификацию
по суммарному спектру каждого сегмента отдельно.

Qt-free; numpy + awf.analysis.peaks/identify/types + awf.model.spectrogram.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from awf.model.spectrogram import Spectrogram
from awf.analysis.peaks import (
    auto_calibrate_fwhm_model,
    fwhm_channels_from_model,
    find_peaks,
)
from awf.analysis.identify import identify_peaks
from awf.analysis.types import FoundPeak, IdentResult
from awf.io.nuclide_lib import Nuclide


@dataclass(frozen=True)
class TimeSegment:
    """Один временной сегмент записи [t_lo, t_hi) (индексы срезов)."""
    t_lo: int               # первый срез, включительно
    t_hi: int               # последний срез + 1, исключительно
    t_start_s: float        # time_offset первого среза, с
    t_end_s: float          # time_offset последнего среза сегмента, с
    live_time_s: float      # суммарное живое время сегмента, с
    total_counts: int       # суммарные отсчёты сегмента

    @property
    def n_slices(self) -> int:
        return self.t_hi - self.t_lo


@dataclass(frozen=True)
class SegmentIdent:
    """Результат идентификации по одному сегменту."""
    segment: TimeSegment
    peaks: Tuple[FoundPeak, ...] = field(default_factory=tuple)
    idents: Tuple[IdentResult, ...] = field(default_factory=tuple)


def _macro_bins(counts: np.ndarray, n_bands: int) -> np.ndarray:
    """Свернуть [ns, nch] -> [ns, K] суммированием каналов в K смежных РАВНЫХ по числу
    каналов полос. Возвращает float64. K фактический = min(n_bands, nch)."""
    ns, nch = counts.shape
    K = max(1, min(int(n_bands), nch))
    edges = np.linspace(0, nch, K + 1).astype(np.int64)
    result = np.empty((ns, K), dtype=np.float64)
    for k in range(K):
        result[:, k] = counts[:, edges[k]:edges[k+1]].sum(axis=1)
    return result


def _seg_score(Pcum, Ecum, a: int, b: int) -> float:
    S = Pcum[b] - Pcum[a]            # [K]
    T = float(Ecum[b] - Ecum[a])
    if T <= 0.0:
        return 0.0
    pos = S > 0.0
    if not np.any(pos):
        return 0.0
    return float(np.sum(S[pos] * np.log(S[pos] / T)))


def _best_split(Pcum, Ecum, a: int, b: int, min_slices: int) -> Tuple[int, float]:
    """Вернуть (s*, gain*) — точку разбиения с макс. приростом F. Если допустимых точек
    нет — вернуть (-1, 0.0). Допустимы s, при которых обе половины имеют >= min_slices
    срезов: a + min_slices <= s <= b - min_slices."""
    
    def _row_F(S, T):
        # S: [m,K], T: [m]; вернуть F по строкам [m], с правилом 0·ln0=0 и T<=0 -> 0
        T = np.asarray(T, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = S / T[:, None]
            term = np.where(S > 0.0, S * np.log(ratio), 0.0)
        Frow = term.sum(axis=1)
        Frow[T <= 0.0] = 0.0
        return Frow

    lo = a + min_slices
    hi = b - min_slices
    if lo > hi:
        return (-1, 0.0)

    s = np.arange(lo, hi + 1)  # точки разбиения, левая половина [a,s), правая [s,b)
    Sl = Pcum[s] - Pcum[a]      # [m,K]
    Tl = Ecum[s] - Ecum[a]      # [m]
    Sr = Pcum[b] - Pcum[s]      # [m,K]
    Tr = Ecum[b] - Ecum[s]      # [m]

    Fp = _seg_score(Pcum, Ecum, a, b)
    gains = _row_F(Sl, Tl) + _row_F(Sr, Tr) - Fp
    j = int(np.argmax(gains))
    return (int(s[j]), float(gains[j]))


def segment_by_time(sg: Spectrogram, *,
                    n_bands: int = 32,
                    min_slices: int = 3,
                    max_segments: int = 12,
                    penalty: Optional[float] = None,
                    pen_factor: float = 2.0) -> List[TimeSegment]:
    """Авто-сегментация оси времени по Пуассоновскому change-point.

    Делит [0, n_slices) на участки квазистационарного спектрального состава методом
    жадного нисходящего (best-first) дробления по макро-полосам энергии. Точка
    разбиения принимается, пока прирост Пуассоновского лог-правдоподобия превышает
    штраф penalty и число сегментов < max_segments, а обе половины имеют >= min_slices.

    penalty=None -> авто-BIC: penalty = pen_factor * K * ln(max(2, n_slices)),
    где K — фактическое число макро-полос. Больше pen_factor -> меньше сегментов.

    Возвращает список TimeSegment, упорядоченный по времени (по t_lo).
    Деградация: n_slices < 2*min_slices ИЛИ один сегмент не делится -> один сегмент на всю запись.
    """
    ns = sg.n_slices
    expo = np.asarray(sg.live_time_s, float)
    if expo.sum() <= 0:
        expo = np.asarray(sg.real_time_s, float)
    if expo.sum() <= 0:
        expo = np.ones(ns)

    def _mk(a: int, b: int) -> TimeSegment:
        to = np.asarray(sg.time_offsets_s, float)
        t_start = float(to[a])
        t_end = float(to[b-1])
        lt = float(expo[a:b].sum())
        tc = int(np.asarray(sg.counts[a:b]).sum(dtype=np.int64))
        return TimeSegment(a, b, t_start, t_end, lt, tc)

    if ns < 2 * min_slices:
        return [_mk(0, ns)]

    M = _macro_bins(sg.counts, n_bands)
    K = M.shape[1]
    Pcum = np.vstack([np.zeros((1, K)), np.cumsum(M, axis=0)])
    Ecum = np.concatenate([[0.0], np.cumsum(expo)])

    pen = penalty if penalty is not None else pen_factor * K * math.log(max(2, ns))

    # best-first дробление; стартовый кандидат — лучший split всей записи (Задача #131:
    # без этого вызова s_star=-1 и цикл прерывается сразу, отдавая один сегмент)
    s0, g0 = _best_split(Pcum, Ecum, 0, ns, min_slices)
    candidates = [(0, ns, s0, g0)]  # (a, b, s_star, gain_star)
    while len(candidates) < max_segments:
        # найти лучший кандидат
        best_idx = -1
        best_gain = -1.0
        for i, (_, _, _, gain) in enumerate(candidates):
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or candidates[best_idx][2] == -1 or best_gain <= pen:
            break

        a, b, s_star, _ = candidates.pop(best_idx)
        new_a, new_b = (a, s_star), (s_star, b)

        # проверить, что обе половины можно делить
        if s_star - a >= min_slices and b - s_star >= min_slices:
            for new_a, new_b in [new_a, new_b]:
                s_star_new, gain_new = _best_split(Pcum, Ecum, new_a, new_b, min_slices)
                candidates.append((new_a, new_b, s_star_new, gain_new))

    # собрать финальные сегменты
    boundaries = sorted(set([c[0] for c in candidates] + [c[1] for c in candidates]))
    segments = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i+1]
        segments.append(_mk(a, b))

    return segments


def identify_segments(sg: Spectrogram,
                      library: Sequence[Nuclide],
                      segments: Sequence[TimeSegment],
                      *,
                      fwhm_model: Optional[Callable[[float], float]] = None,
                      peak_sigma: float = 3.0,
                      max_energy_keV: float = 3000.0,
                      tol_factor: float = 0.5,
                      min_confidence: float = 0.30,
                      apply_priors: bool = True) -> List[SegmentIdent]:
    """Идентифицировать нуклиды по суммарному спектру КАЖДОГО сегмента отдельно.

    Переиспользует существующий движок: find_peaks (#107/#114) + identify_peaks (#130).
    Модель разрешения FWHM(E) строится ОДИН раз по интегральному спектру (разрешение от
    времени не зависит) и общая для всех сегментов; fwhm_model можно передать готовую.
    Порог энергии max_energy_keV отсекает пики выше предела отображения (#119).
    """
    energies = np.asarray(sg.energies(), float)
    model = fwhm_model if fwhm_model is not None else auto_calibrate_fwhm_model(
        np.asarray(sg.total_spectrum(), float), energies)
    widths = fwhm_channels_from_model(model, energies)

    result = []
    for seg in segments:
        spec = np.asarray(sg.sum_spectrum(seg.t_lo, seg.t_hi), dtype=np.float64)
        pk = find_peaks(spec, widths, sigma_threshold=peak_sigma, energies=energies)
        pk = [p for p in pk if float(p.energy) <= max_energy_keV]
        idents = identify_peaks(pk, library, fwhm_model=model, tol_factor=tol_factor,
            min_confidence=min_confidence, apply_priors=apply_priors)
        result.append(SegmentIdent(seg, tuple(pk), tuple(idents)))

    return result

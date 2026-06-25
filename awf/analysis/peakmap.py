"""Карта отдельных пиков по времени (Задача 21, ТЗ-B.7 — временной вариант).

Для набора энергетических окон интереса строит временной ряд интенсивности: gross
(полный счёт в окне) и net (нетто после вычитания континуума, оценённого по боковым
полосам — трапеция Ковелла). Окна редактируемы. Qt-free, только numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from awf.model.spectrogram import Spectrogram


@dataclass(frozen=True)
class EnergyWindow:
    """Энергетическое окно интереса (центр ± полуширина, кэВ)."""
    name: str
    center: float
    half_width: float
    nuclide: str = ""

    @property
    def e_lo(self) -> float:
        return self.center - self.half_width

    @property
    def e_hi(self) -> float:
        return self.center + self.half_width


# Редактируемый реестр по умолчанию: ключевые гамма-линии (кэВ).
DEFAULT_WINDOWS = (
    EnergyWindow("Cs-137 662", 661.7, 15.0, "Cs-137"),
    EnergyWindow("K-40 1461", 1460.8, 25.0, "K-40"),
    EnergyWindow("U 609", 609.3, 15.0, "Bi-214"),
    EnergyWindow("U 1764", 1764.5, 25.0, "Bi-214"),
    EnergyWindow("Th 2614", 2614.5, 30.0, "Tl-208"),
)


@dataclass(frozen=True)
class WindowSeries:
    """Временные ряды одного энергоокна."""
    window: EnergyWindow
    t: np.ndarray          # центры времени, с (len = n_slices)
    gross: np.ndarray      # полный счёт в окне по срезам
    baseline: np.ndarray   # интеграл континуума под окном по срезам
    net: np.ndarray        # gross - baseline (нетто), >= не гарантируется


def _window_channels(sg: Spectrogram, w: EnergyWindow):
    """Каналы границ окна [ch_lo, ch_hi) для энергий окна."""
    nch = sg.n_channels
    ch_lo = int(sg.calibration.channel_of_energy(w.e_lo, nch))
    ch_hi = int(sg.calibration.channel_of_energy(w.e_hi, nch)) + 1
    ch_lo = max(0, min(ch_lo, nch - 1))
    ch_hi = max(ch_lo + 1, min(ch_hi, nch))
    return ch_lo, ch_hi


def window_series(sg: Spectrogram, w: EnergyWindow, *, side_channels: int = 3) -> WindowSeries:
    """Gross/baseline/net временные ряды для одного окна.

    Континуум оценивается как трапеция между средним счётом на канал в левой и правой
    боковых полосах шириной ``side_channels`` (примыкают к окну). Полосы обрезаются по
    границам спектра; при отсутствии полосы её среднее берётся равным имеющейся.
    """
    nch = sg.n_channels
    ch_lo, ch_hi = _window_channels(sg, w)
    nwin = ch_hi - ch_lo
    counts = sg.counts.astype(np.float64, copy=False)

    gross = counts[:, ch_lo:ch_hi].sum(axis=1)

    s = max(1, int(side_channels))
    l0, l1 = max(0, ch_lo - s), ch_lo
    r0, r1 = ch_hi, min(nch, ch_hi + s)
    left = counts[:, l0:l1].mean(axis=1) if l1 > l0 else None
    right = counts[:, r0:r1].mean(axis=1) if r1 > r0 else None
    if left is None and right is None:
        per_ch = np.zeros(sg.n_slices, dtype=np.float64)
    elif left is None:
        per_ch = right
    elif right is None:
        per_ch = left
    else:
        per_ch = 0.5 * (left + right)
    baseline = per_ch * float(nwin)

    net = gross - baseline
    return WindowSeries(window=w, t=sg.time_offsets_s, gross=gross,
                        baseline=baseline, net=net)


def peak_map(sg: Spectrogram, windows=None, *, side_channels: int = 3):
    """Список WindowSeries по набору окон (по умолчанию — DEFAULT_WINDOWS)."""
    ws = tuple(DEFAULT_WINDOWS if windows is None else windows)
    return [window_series(sg, w, side_channels=side_channels) for w in ws]
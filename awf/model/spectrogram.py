"""
Модуль для работы с waterfall-спектрограммами гамма-спектрометра.
Содержит классы Calibration и Spectrogram для численных операций над данными.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional


@dataclass(frozen=True)
class Calibration:
    coeffs: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "coeffs", np.asarray(self.coeffs, dtype=np.float64).ravel())
        if self.coeffs.size == 0:
            raise ValueError("Calibration: пустые коэффициенты")

    def energy_of_channel(self, ch) -> np.ndarray | float:
        import numpy.polynomial.polynomial as P
        ch_arr = np.asarray(ch, dtype=np.float64)
        result = P.polyval(ch_arr, self.coeffs)
        if np.ndim(ch_arr) == 0:
            return float(result)
        return result

    def energies(self, n_channels: int) -> np.ndarray:
        if n_channels <= 0:
            raise ValueError("n_channels должен быть положительным")
        ch = np.arange(n_channels, dtype=np.float64)
        return self.energy_of_channel(ch)

    def channel_of_energy(self, energy, n_channels: int) -> np.ndarray | float:
        grid = self.energies(n_channels)
        mono = bool(np.all(np.diff(grid) >= 0))
        if not mono:
            order = np.argsort(grid)
            grid_sorted = grid[order]
        else:
            order = None
            grid_sorted = grid

        e = np.asarray(energy, dtype=np.float64)
        idx = np.searchsorted(grid_sorted, e)
        idx = np.clip(idx, 0, n_channels - 1).astype(np.int64)
        if order is not None:
            idx = order[idx]

        if np.ndim(e) == 0:
            return int(idx)
        return idx

    @classmethod
    def from_coeff_string(cls, text: str) -> "Calibration":
        if not text.strip():
            raise ValueError("Calibration.from_coeff_string: пустая строка")
        vals = [float(x) for x in text.split()]
        return cls(coeffs=np.asarray(vals, dtype=np.float64))


class Spectrogram:
    def __init__(self, *,
                 counts: np.ndarray,
                 calibration: Calibration,
                 time_offsets_s: np.ndarray,
                 real_time_s: np.ndarray,
                 live_time_s: np.ndarray,
                 t0_iso: str | None = None,
                 source_path: str | None = None):
        if counts.ndim != 2:
            raise ValueError("counts должен быть двумерным массивом")
        n_slices, n_channels = counts.shape
        if n_slices < 1 or n_channels < 1:
            raise ValueError("n_slices и n_channels должны быть положительными")

        time_offsets_s = np.asarray(time_offsets_s, dtype=np.float64)
        real_time_s = np.asarray(real_time_s, dtype=np.float64)
        live_time_s = np.asarray(live_time_s, dtype=np.float64)

        if not (time_offsets_s.ndim == 1 and
                real_time_s.ndim == 1 and
                live_time_s.ndim == 1 and
                time_offsets_s.size == n_slices and
                real_time_s.size == n_slices and
                live_time_s.size == n_slices):
            raise ValueError("Временные массивы должны быть одномерными и иметь размерность n_slices")

        self.counts = counts
        self.calibration = calibration
        self.time_offsets_s = time_offsets_s
        self.real_time_s = real_time_s
        self.live_time_s = live_time_s
        self.t0_iso = t0_iso
        self.source_path = source_path

    @property
    def n_slices(self) -> int:
        return self.counts.shape[0]

    @property
    def n_channels(self) -> int:
        return self.counts.shape[1]

    def energies(self) -> np.ndarray:
        return self.calibration.energies(self.n_channels)

    def energy_spectrum(self, i: int) -> np.ndarray:
        if not (-self.n_slices <= i < self.n_slices):
            raise IndexError("Индекс времени вне диапазона")
        i = i % self.n_slices
        return self.counts[i]

    def channel_time_series(self, j: int) -> np.ndarray:
        if not (0 <= j < self.n_channels):
            raise IndexError("Индекс канала вне диапазона")
        return self.counts[:, j]

    def band_time_series(self, ch_lo: int, ch_hi: int) -> np.ndarray:
        lo, hi = sorted((ch_lo, ch_hi))
        lo = max(0, lo)
        hi = min(self.n_channels, hi)
        if hi <= lo:
            raise ValueError("Неверный диапазон каналов")
        return self.counts[:, lo:hi].sum(axis=1, dtype=np.int64)

    def energy_band_time_series(self, e_lo: float, e_hi: float) -> np.ndarray:
        lo_e, hi_e = sorted((e_lo, e_hi))
        ch_lo = int(self.calibration.channel_of_energy(lo_e, self.n_channels))
        ch_hi = int(self.calibration.channel_of_energy(hi_e, self.n_channels)) + 1
        return self.band_time_series(ch_lo, ch_hi)

    def sum_spectrum(self, t_lo: int | None = None, t_hi: int | None = None) -> np.ndarray:
        lo = 0 if t_lo is None else max(0, t_lo)
        hi = self.n_slices if t_hi is None else min(self.n_slices, t_hi)
        if hi <= lo:
            raise ValueError("Неверный диапазон времени")
        return self.counts[lo:hi, :].sum(axis=0, dtype=np.int64)

    def total_spectrum(self) -> np.ndarray:
        return self.sum_spectrum(None, None)

    def trimmed_channels(self, drop_last: int = 1) -> "Spectrogram":
        """Вернуть новую спектрограмму без последних drop_last каналов (Замечание IV-R5:
        последний канал АЦП содержит мусор — переполнение). Калибровка и временные оси не
        меняются (энергии каналов 0..n-1-drop_last те же). drop_last<=0 — вернуть self."""
        d = int(drop_last)
        if d <= 0:
            return self
        if self.n_channels - d < 1:
            raise ValueError("trimmed_channels: после обрезки не остаётся каналов")
        return Spectrogram(
            counts=self.counts[:, :-d].copy(),
            calibration=self.calibration,
            time_offsets_s=self.time_offsets_s,
            real_time_s=self.real_time_s,
            live_time_s=self.live_time_s,
            t0_iso=self.t0_iso,
            source_path=self.source_path,
        )

    def roi_sum(self, t_lo: int, t_hi: int, ch_lo: int, ch_hi: int) -> int:
        lo, hi = sorted((t_lo, t_hi))
        ch_lo, ch_hi = sorted((ch_lo, ch_hi))
        lo = max(0, lo)
        hi = min(self.n_slices, hi)
        ch_lo = max(0, ch_lo)
        ch_hi = min(self.n_channels, ch_hi)
        if hi <= lo or ch_hi <= ch_lo:
            raise ValueError("Неверный диапазон выборки")
        return int(self.counts[lo:hi, ch_lo:ch_hi].sum(dtype=np.int64))

    def downsample(self, max_time: int, max_chan: int, method: str = "max") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ns, ncyh = self.n_slices, self.n_channels
        nt = min(int(max_time), ns)
        nc = min(int(max_chan), ncyh)
        nt = max(1, nt)
        nc = max(1, nc)

        t_edges = np.unique(np.linspace(0, ns, nt + 1).astype(np.int64))
        ch_edges = np.unique(np.linspace(0, ncyh, nc + 1).astype(np.int64))

        t_starts = t_edges[:-1]
        ch_starts = ch_edges[:-1]

        data = self.counts.astype(np.float64, copy=False)

        if method == "max":
            red = np.maximum
        elif method == "sum":
            red = np.add
        else:
            raise ValueError("Метод должен быть 'max' или 'sum'")

        step1 = red.reduceat(data, t_starts, axis=0)
        counts_ds = red.reduceat(step1, ch_starts, axis=1)

        t_off = self.time_offsets_s
        en = self.energies()

        t_centers = np.empty(len(t_starts), dtype=np.float64)
        ch_centers = np.empty(len(ch_starts), dtype=np.float64)

        for k in range(len(t_starts)):
            a = int(t_edges[k])
            b = int(t_edges[k + 1])
            if b > a:
                t_centers[k] = float(t_off[a:b].mean())
            else:
                t_centers[k] = float(t_off[min(a, ns - 1)])

        for k in range(len(ch_starts)):
            a = int(ch_edges[k])
            b = int(ch_edges[k + 1])
            if b > a:
                ch_centers[k] = float(en[a:b].mean())
            else:
                ch_centers[k] = float(en[min(a, ncyh - 1)])

        return counts_ds, t_centers, ch_centers

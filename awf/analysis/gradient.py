"""Градиентный анализ по оси времени (Задача 22, ТЗ-B.8 — временной вариант).

Производная d(счёт)/d(время) локализует ФРОНТЫ: появление/исчезновение источника,
границу зоны при движении детектора. Работает с полным счётом среза либо со счётом
в энергоокне. Qt-free, только numpy. Опциональное скользящее усреднение перед
дифференцированием подавляет пуассоновский шум (производная его усиливает).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from awf.model.spectrogram import Spectrogram


def moving_average(y, radius: int) -> np.ndarray:
    """Box-сглаживание 1D-ряда шириной 2*radius+1 (edge-паддинг краёв).

    radius<=0 или длина<2 — возврат входа без изменений (float64).
    """
    r = int(radius)
    a = np.asarray(y, dtype=np.float64).ravel()
    if r <= 0 or a.size < 2:
        return a
    r = min(r, a.size)
    k = 2 * r + 1
    pad = np.pad(a, (r, r), mode="edge")
    csum = np.concatenate([[0.0], np.cumsum(pad)])
    return (csum[k:] - csum[:-k]) / float(k)


@dataclass(frozen=True)
class GradientResult:
    """Результат градиентного анализа временного ряда."""
    t: np.ndarray          # центры времени, с (len = n)
    counts: np.ndarray     # сглаженный временной ряд счёта (len = n)
    gradient: np.ndarray   # d(counts)/d(t), отсч/с (len = n)
    front_index: int       # индекс максимума |gradient| — локализованный фронт
    front_time: float      # время фронта, с


def total_counts_series(sg: Spectrogram) -> np.ndarray:
    """Полный счёт каждого временного среза (сумма по всем каналам), float64."""
    return sg.counts.sum(axis=1, dtype=np.int64).astype(np.float64)


def time_gradient(counts_series, t, *, smooth_radius: int = 0) -> GradientResult:
    """Градиент d(счёт)/d(время) одномерного временного ряда.

    ``counts_series`` — счёт по срезам; ``t`` — времена срезов (с). Если ``t`` строго
    возрастает — используется как ось дифференцирования (физ. единицы отсч/с); иначе
    (вырожденные/неупорядоченные времена) — единичный шаг по индексу. ``smooth_radius>0``
    включает скользящее среднее перед дифференцированием. Фронт = argmax|grad|.
    """
    y = np.asarray(counts_series, dtype=np.float64).ravel()
    tt = np.asarray(t, dtype=np.float64).ravel()
    if y.size != tt.size:
        raise ValueError("time_gradient: длины counts_series и t не совпадают")
    if y.size < 2:
        raise ValueError("time_gradient: нужно >= 2 срезов")
    ys = moving_average(y, smooth_radius)
    if np.all(np.diff(tt) > 0):
        grad = np.gradient(ys, tt)
    else:
        grad = np.gradient(ys)
    fi = int(np.argmax(np.abs(grad)))
    return GradientResult(t=tt, counts=ys, gradient=grad,
                          front_index=fi, front_time=float(tt[fi]))


def band_gradient(sg: Spectrogram, e_lo: float, e_hi: float,
                  *, smooth_radius: int = 0) -> GradientResult:
    """Градиент временного ряда счёта в энергоокне [e_lo, e_hi] кэВ."""
    series = sg.energy_band_time_series(e_lo, e_hi).astype(np.float64)
    return time_gradient(series, sg.time_offsets_s, smooth_radius=smooth_radius)
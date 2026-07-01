"""
Фон и вычитание фона для waterfall-спектрограмм (Задача #96).

Фон хранится как поканальная скорость счёта (cps на канал), выровненная по энергетической
оси целевой спектрограммы. Два источника:
  * диапазон времени текущего измерения — средняя скорость по выбранным срезам;
  * отдельный файл — суммарная скорость; при иной калибровке интерполируется по спектральной
    плотности (cps/кэВ) и пересчитывается на ширину каналов целевой спектрограммы.

Вычитание идёт на уровне отсчётов: counts[t,ch] - bg_cps[ch]*live_time[t]. Остаток ЗНАКОВЫЙ
(без клипа к 0, Задача #134) — только так интеграл по времени сокращается точно: когда фон =
самому спектру, Σ_t (counts − bg_cps·lt) = total_counts − bg_cps·total_lt = 0 поканально.
Отрицательные ячейки гасятся к нулю уже на этапе ОТОБРАЖЕНИЯ (zscale._base_transform →
np.maximum(a,0)), поэтому 3D/2D-виды выглядят как прежде, а суммарный (интегральный) спектр
среза корректно уходит в ноль.
"""
from __future__ import annotations

import numpy as np

from awf.model.spectrogram import Spectrogram


def background_from_range(sg, t_lo: int, t_hi: int) -> np.ndarray:
    """Поканальный фон (cps) как средняя скорость по срезам [t_lo:t_hi] текущей спектрограммы:
    bg[ch] = sum(counts[t_lo:t_hi, ch]) / sum(live_time[t_lo:t_hi]). Калибровка та же — без
    интерполяции. Длина результата = sg.n_channels."""
    lo = max(0, int(t_lo))
    hi = min(sg.n_slices, int(t_hi))
    if hi <= lo:
        raise ValueError("background_from_range: пустой диапазон времени")
    gross = sg.counts[lo:hi, :].sum(axis=0, dtype=np.float64)
    lt = float(np.asarray(sg.live_time_s, dtype=np.float64)[lo:hi].sum())
    if lt <= 0.0:
        raise ValueError("background_from_range: нулевое живое время окна")
    return gross / lt


def _channel_widths(energies) -> np.ndarray:
    """Ширина каналов по энергии (кэВ/канал) через градиент монотонной шкалы энергий."""
    e = np.asarray(energies, dtype=np.float64)
    if e.size < 2:
        return np.ones_like(e)
    return np.abs(np.gradient(e))


def background_from_spectrogram(bg_sg, target_sg) -> np.ndarray:
    """Поканальный фон (cps) из отдельной спектрограммы, выровненный по энергии target_sg.
    Считаем спектральную плотность скорости (cps/кэВ) фон-файла, интерполируем по энергии на
    каналы цели и умножаем на ширину каналов цели. При совпадающей калибровке сводится к
    поканальному cps. Каналы вне диапазона фон-файла -> 0 (без экстраполяции). Длина = цель."""
    e_bg = np.asarray(bg_sg.energies(), dtype=np.float64)
    e_t = np.asarray(target_sg.energies(), dtype=np.float64)
    lt_bg = float(np.asarray(bg_sg.live_time_s, dtype=np.float64).sum())
    if lt_bg <= 0.0:
        raise ValueError("background_from_spectrogram: нулевое живое время фон-файла")
    bg_cps = bg_sg.counts.sum(axis=0, dtype=np.float64) / lt_bg     # cps на канал фон-файла
    dens_bg = bg_cps / _channel_widths(e_bg)                        # cps/кэВ
    xp, fp = e_bg, dens_bg
    if xp.size >= 2 and xp[0] > xp[-1]:                             # шкала по убыванию -> развернуть
        xp, fp = xp[::-1], fp[::-1]
    dens_t = np.interp(e_t, xp, fp, left=0.0, right=0.0)
    return dens_t * _channel_widths(e_t)


def subtract_background(sg, bg_cps) -> Spectrogram:
    """Новая спектрограмма с вычтенным фоном: counts[t,ch] - bg_cps[ch]*live_time[t] (Задача #96).
    Остаток ЗНАКОВЫЙ, без клипа к 0 (Задача #134): поканальный клип асимметрично разрушал
    сокращение и оставлял ложный ~18 % положительный остаток при фон=спектр (должно быть 0).
    Без клипа Σ_t остаток = total_counts − bg_cps·total_lt = 0 поканально при фон=весь файл.
    Калибровка/временные оси сохраняются; отрицательные ячейки гасятся к 0 на отображении
    (zscale._base_transform), поэтому 3D/2D-виды не меняются, а интегральный спектр уходит в 0."""
    bg = np.asarray(bg_cps, dtype=np.float64).ravel()
    if bg.size != sg.n_channels:
        raise ValueError("subtract_background: длина bg_cps != числу каналов")
    lt = np.asarray(sg.live_time_s, dtype=np.float64)
    sub = sg.counts.astype(np.float64) - bg[None, :] * lt[:, None]
    return Spectrogram(counts=sub, calibration=sg.calibration,
                       time_offsets_s=sg.time_offsets_s, real_time_s=sg.real_time_s,
                       live_time_s=sg.live_time_s, t0_iso=sg.t0_iso, source_path=sg.source_path)
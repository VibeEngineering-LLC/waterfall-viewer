"""Задача #215 — фит калибровки E(channel) по парам (канал ↔ истинная энергия).

Полином степени deg (1..4): E(ch) = c0 + c1*ch + c2*ch^2 + ...
Коэффициенты возвращаются в ПРЯМОМ порядке (c0..cN), совместимо с
`awf.model.spectrogram.Calibration` (использует `numpy.polynomial.polynomial.polyval`)
и с записью `hdr["calibration"]` в ASWF-заголовке.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# Пресеты — типичные фотопики известных источников. Значения — кэВ, лабораторные линии.
# Не заполняются автоматически, используются как справочник в UI (список кнопок).
PRESETS: dict[str, list[float]] = {
    "Th-232 (цепочка)": [
        238.63,   # Pb-212
        338.32,   # Ac-228
        583.19,   # Tl-208 (двойной эскейп 2614)
        727.33,   # Bi-212
        911.20,   # Ac-228
        969.00,   # Ac-228
        1588.20,  # Ac-228
        2614.51,  # Tl-208 (жёсткая линия, отличный якорь)
    ],
    "U-238 / Ra-226 (цепочка)": [
        186.21,   # Ra-226
        295.22,   # Pb-214
        351.93,   # Pb-214
        609.31,   # Bi-214
        1120.29,  # Bi-214
        1764.49,  # Bi-214
        2204.21,  # Bi-214
    ],
    "K-40": [1460.83],
    "Cs-137": [661.66],
    "Co-60": [1173.23, 1332.49],
    "Am-241": [59.54],
    "Na-22": [511.00, 1274.54],
    "Ba-133": [80.998, 302.85, 356.02, 383.85],
    "Eu-152": [121.78, 344.28, 778.90, 964.06, 1112.08, 1408.01],
}


def _to_arrays(pairs: Iterable[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Список пар [(channel, true_E_keV), ...] -> (np.ndarray channels, np.ndarray energies)."""
    arr = np.asarray(list(pairs), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("fit_calibration: ожидается список пар (channel, true_E), shape (N, 2)")
    if arr.shape[0] < 2:
        raise ValueError("fit_calibration: нужно ≥2 пар для фита")
    return arr[:, 0], arr[:, 1]


def fit_calibration(pairs: Iterable[Sequence[float]], deg: int = 2) -> list[float]:
    """Фит полиномиальной калибровки E(ch) = Σ c_i * ch^i по парам (канал, истинная_E).

    Параметры:
      pairs — итерируемое пар (channel_float, true_E_keV_float).
      deg   — степень полинома (1..4). deg > N-1 автоматически ограничивается N-1.

    Возвращает: список [c0, c1, ..., c_deg] (прямой порядок, как в ASWF header).
    Бросает ValueError, если пар < 2 или (после клипа) deg < 1.
    """
    chs, true_es = _to_arrays(pairs)
    max_deg = int(chs.size) - 1
    d = int(deg)
    if d < 1:
        raise ValueError(f"fit_calibration: deg={deg} < 1")
    if d > max_deg:
        d = max_deg
    # numpy.polyfit возвращает коэффициенты в ОБРАТНОМ порядке (высшая степень первой);
    # reverse — прямой порядок [c0, c1, ..., c_d], который использует Calibration.
    reverse = np.polyfit(chs, true_es, d)
    forward = reverse[::-1].astype(np.float64)
    return forward.tolist()


def apply_polynomial(coeffs: Sequence[float], channels: np.ndarray) -> np.ndarray:
    """E(ch) по коэффициентам в прямом порядке [c0..cN] и массиву каналов."""
    import numpy.polynomial.polynomial as P
    return P.polyval(np.asarray(channels, dtype=np.float64), np.asarray(coeffs, dtype=np.float64))


def compute_rmse(coeffs: Sequence[float], pairs: Iterable[Sequence[float]]) -> float:
    """RMSE фита (кэВ) — sqrt(mean((E_fit(ch) - true_E)^2))."""
    chs, true_es = _to_arrays(pairs)
    predicted = apply_polynomial(coeffs, chs)
    return float(np.sqrt(np.mean((predicted - true_es) ** 2)))


def format_coeffs(coeffs: Sequence[float]) -> str:
    """Строка вида «c0=0.123, c1=1.234, c2=5.6e-05» для отображения в диалоге."""
    parts = []
    for i, c in enumerate(coeffs):
        v = float(c)
        parts.append(f"c{i}={v:.6g}")
    return ", ".join(parts)
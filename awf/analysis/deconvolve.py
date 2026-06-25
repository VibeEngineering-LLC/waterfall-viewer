"""Деконволюция мультиплетов (Задача 23).

Разделение перекрывающихся фотопиков при фиксированных позициях и ширинах (из калибровки
энергии и модели FWHM): линейная задача с неотрицательными площадями A_k>=0 и континуумом
(constant/linear). Решатель — scipy.optimize.lsq_linear (trf, bounds); fallback —
numpy.linalg.lstsq с клиппингом отрицательных площадей. Возвращает площади, их погрешности,
континуум, модельную кривую и chi^2/dof. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeconvolutionResult:
    areas: np.ndarray        # площади компонент (counts), >= 0
    d_areas: np.ndarray      # погрешности площадей (1 сигма, прибл.)
    continuum: np.ndarray    # коэффициенты континуума (const[, slope])
    fit: np.ndarray          # модельная кривая (len = len(y))
    chi2: float              # пуассон-взвешенный chi^2
    dof: int                 # число степеней свободы
    chi2_dof: float          # chi^2 / dof
    centers: np.ndarray      # позиции компонент (каналы), эхо входа
    sigmas: np.ndarray       # сигмы компонент (каналы), эхо входа
    method: str              # "lsq_linear" | "lstsq"


def _gauss_col(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """Нормированный на единичную площадь гауссиан (sum_x ≈ 1) → параметр = площадь."""
    s = max(float(s), 1e-6)
    return np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def _design(x: np.ndarray, centers, sigmas, continuum: str):
    """Матрица плана: столбцы пиков + столбцы континуума. Возврат (M, n_peaks)."""
    cols = [_gauss_col(x, c, s) for c, s in zip(centers, sigmas)]
    npk = len(cols)
    if continuum == "linear":
        cols.append(np.ones_like(x))
        cols.append(x - x.mean())
    elif continuum == "constant":
        cols.append(np.ones_like(x))
    elif continuum == "none":
        pass
    else:
        raise ValueError("continuum: ожидается 'linear' | 'constant' | 'none'")
    M = np.column_stack(cols) if cols else np.empty((x.size, 0))
    return M, npk


def _solve_bounded(M, y, npk, use_scipy: bool):
    """Решение с A_k>=0 (пики) и свободным континуумом. Возврат (params, method)."""
    nparam = M.shape[1]
    lo = np.full(nparam, -np.inf)
    hi = np.full(nparam, np.inf)
    lo[:npk] = 0.0
    if use_scipy:
        try:
            from scipy.optimize import lsq_linear
            res = lsq_linear(M, y, bounds=(lo, hi), method="trf")
            return np.asarray(res.x, dtype=np.float64), "lsq_linear"
        except Exception:
            pass
    p, *_ = np.linalg.lstsq(M, y, rcond=None)
    p = np.asarray(p, dtype=np.float64)
    p[:npk] = np.clip(p[:npk], 0.0, None)   # неотрицательность площадей
    return p, "lstsq"


def deconvolve_multiplet(counts_1d, centers, sigmas, *, x=None,
                         continuum: str = "linear",
                         use_scipy: bool = True) -> DeconvolutionResult:
    """Разделить мультиплет с фикс. ``centers``/``sigmas`` (каналы) над ``counts_1d``.

    ``x`` — ось каналов (по умолчанию arange). ``continuum`` — форма подложки. ``use_scipy=False``
    форсирует numpy-fallback. Площади компонент неотрицательны; погрешности — из ковариации
    нормальных уравнений (приближённо). Возвращает DeconvolutionResult.
    """
    y = np.asarray(counts_1d, dtype=np.float64).ravel()
    centers = np.asarray(centers, dtype=np.float64).ravel()
    sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
    if centers.size != sigmas.size:
        raise ValueError("deconvolve_multiplet: centers и sigmas разной длины")
    if centers.size == 0:
        raise ValueError("deconvolve_multiplet: нужен хотя бы один компонент")
    if x is None:
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("deconvolve_multiplet: x и counts_1d разной длины")

    M, npk = _design(x, centers, sigmas, continuum)
    params, method = _solve_bounded(M, y, npk, use_scipy)

    fit = M @ params
    resid = y - fit
    dof = max(1, y.size - M.shape[1])
    rss = float(resid @ resid)
    var = rss / dof
    try:
        cov = var * np.linalg.pinv(M.T @ M)
        d_all = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        d_all = np.full(M.shape[1], np.nan)

    w = 1.0 / np.maximum(fit, 1.0)             # пуассоновские веса по модели
    chi2 = float((resid ** 2 * w).sum())

    return DeconvolutionResult(
        areas=params[:npk], d_areas=d_all[:npk],
        continuum=params[npk:], fit=fit,
        chi2=chi2, dof=dof, chi2_dof=chi2 / dof,
        centers=centers, sigmas=sigmas, method=method,
    )
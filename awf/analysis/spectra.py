"""Спектральные хелперы пакета анализа: подготовка 1D-спектров из Spectrogram."""

from __future__ import annotations

import numpy as np

from awf.model.spectrogram import Spectrogram


def spectrum_from_selection(sg: Spectrogram,
                            t_lo: int | None = None,
                            t_hi: int | None = None) -> np.ndarray:
    """Суммарный спектр временного окна [t_lo, t_hi) как float64.

    Тонкая обёртка над ``Spectrogram.sum_spectrum``: численные методы анализа
    (поиск пиков, континуум, площади) работают с float, а ``sum_spectrum``
    возвращает int64. Значения поканально идентичны исходной целочисленной сумме.
    """
    return sg.sum_spectrum(t_lo, t_hi).astype(np.float64)

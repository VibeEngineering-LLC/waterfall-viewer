"""Пакет численного анализа waterfall-спектрограмм (Qt-free, тестируется без графики).

Контракт входов: спектр — 1D float ndarray (длина = число каналов);
ось энергий — Calibration.energies(n) (кэВ, ascending); FWHM — float (каналы)
или Callable[[float], float]. Методы возвращают структуры из awf.analysis.types.
Реализации методов добавляются по задачам backlog (peaks/continuum/area/mda/...).
"""
from __future__ import annotations

from awf.analysis.types import (
    FoundPeak, PeakArea, MdaResult, LineMatch, IdentResult,
)
from awf.analysis.spectra import spectrum_from_selection
from awf.analysis.identify import (
    identify_peaks, lookup_by_energy, default_fwhm_keV, get_prior,
)

__all__ = [
    "FoundPeak", "PeakArea", "MdaResult", "LineMatch", "IdentResult",
    "spectrum_from_selection",
    "identify_peaks", "lookup_by_energy", "default_fwhm_keV", "get_prior",
]

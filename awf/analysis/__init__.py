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
from awf.analysis.peaks import (
    find_peaks, peak_time_mask, snip_baseline,
    find_transient_peaks,   # Задача #113: транзиентные (время-локализованные) пики
    # Задача #114: модель разрешения FWHM(E) и per-channel ширина matched-фильтра
    FwhmModel, default_fwhm_model, fwhm_model_keV,
    estimate_fwhm_model, fwhm_channels_from_model,
)
from awf.analysis.identify import (
    identify_peaks, lookup_by_energy, default_fwhm_keV, get_prior,
)
# Группа V — аналитика по матрице время×энергия (Задачи 21–25)
from awf.analysis.gradient import (
    GradientResult, moving_average, total_counts_series, time_gradient, band_gradient,
)
from awf.analysis.peakmap import (
    EnergyWindow, DEFAULT_WINDOWS, WindowSeries, window_series, peak_map,
)
from awf.analysis.decomposition import (
    ProjectionResult, feature_matrix, project, pca,
    is_available as projection_available,
)
from awf.analysis.deconvolve import DeconvolutionResult, deconvolve_multiplet
from awf.analysis.cluster import (
    ClusterResult, kmeans, cluster, segments,
    is_available as cluster_available,
)
# Задача #131: авто-сегментация по времени + посегментная идентификация
from awf.analysis.segment import (
    TimeSegment, SegmentIdent, segment_by_time, identify_segments,
)
# Задача #133: цепочечное (decay-chain) равновесие Ra-226 / Th-232
from awf.analysis.chain_equilibrium import (
    ChainGroupResult, ChainResult,
    check_ra226_chain, check_th232_chain, analyze_chains, identify_with_chains,
)

__all__ = [
    "FoundPeak", "PeakArea", "MdaResult", "LineMatch", "IdentResult",
    "spectrum_from_selection",
    "find_peaks", "peak_time_mask", "snip_baseline", "find_transient_peaks",
    # Задача #114
    "FwhmModel", "default_fwhm_model", "fwhm_model_keV",
    "estimate_fwhm_model", "fwhm_channels_from_model",
    "identify_peaks", "lookup_by_energy", "default_fwhm_keV", "get_prior",
    # Группа V
    "GradientResult", "moving_average", "total_counts_series", "time_gradient", "band_gradient",
    "EnergyWindow", "DEFAULT_WINDOWS", "WindowSeries", "window_series", "peak_map",
    "ProjectionResult", "feature_matrix", "project", "pca", "projection_available",
    "DeconvolutionResult", "deconvolve_multiplet",
    "ClusterResult", "kmeans", "cluster", "segments", "cluster_available",
    # Задача #131
    "TimeSegment", "SegmentIdent", "segment_by_time", "identify_segments",
    # Задача #133
    "ChainGroupResult", "ChainResult",
    "check_ra226_chain", "check_th232_chain", "analyze_chains", "identify_with_chains",
]
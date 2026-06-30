"""Тесты авто-сегментации по времени и посегментной идентификации (Задача #131)."""
import math

import numpy as np
import pytest

from awf.model.spectrogram import Spectrogram, Calibration
from awf.io.nuclide_lib import default_library
from awf.analysis.segment import (
    TimeSegment,
    SegmentIdent,
    segment_by_time,
    identify_segments,
    _macro_bins,
    _seg_score,
)


def _make_sg(counts, *, slice_live=10.0):
    """Собрать Spectrogram из 2D-массива counts [ns, nch] с калибровкой energy=channel."""
    counts = np.asarray(counts, dtype=np.int64)
    ns, nch = counts.shape
    to = np.arange(ns, dtype=float) * slice_live
    lt = np.full(ns, slice_live, dtype=float)
    return Spectrogram(counts=counts, calibration=Calibration(coeffs=[0.0, 1.0]),
                       time_offsets_s=to, real_time_s=lt.copy(), live_time_s=lt.copy())


def _gauss(nch, center, height, fwhm):
    """Детерминированная гауссиана высотой height (отсчётов) на сетке каналов [0, nch)."""
    x = np.arange(nch, dtype=float)
    sigma = fwhm / 2.3548200450309493
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _two_regime_counts(ns_a, ns_b, nch, peak_a, peak_b, *, bg=2.0, h_a=30.0, h_b=30.0, fwhm=24.0):
    """Две временные фазы: срезы [0,ns_a) с пиком в peak_a, [ns_a, ns_a+ns_b) с пиком в peak_b.
    Пик каждой фазы отсутствует в другой → источники временно разнесены."""
    rows = []
    base = np.full(nch, bg)
    for _ in range(ns_a):
        rows.append(np.rint(base + _gauss(nch, peak_a, h_a, fwhm)).astype(np.int64))
    for _ in range(ns_b):
        rows.append(np.rint(base + _gauss(nch, peak_b, h_b, fwhm)).astype(np.int64))
    return np.vstack(rows)


def test_macro_bins_shape_and_sum():
    counts = np.arange(5 * 100).reshape(5, 100)
    M = _macro_bins(counts, 8)
    assert M.shape == (5, 8)
    assert M.dtype == np.float64
    assert np.allclose(M.sum(axis=1), counts.sum(axis=1))
    # При n_bands > nch фактическое K = nch
    M_large = _macro_bins(counts, 1000)
    assert M_large.shape == (5, 100)


def test_seg_score_poisson():
    M = np.array([[4.0, 0.0], [0.0, 6.0]])
    expo = np.array([1.0, 1.0])
    Pcum = np.vstack([np.zeros((1, 2)), np.cumsum(M, axis=0)])
    Ecum = np.concatenate([[0.0], np.cumsum(expo)])
    # Для всего [0, 2)
    S = [4, 6]
    T = 2
    F = 4 * math.log(2.0) + 6 * math.log(3.0)
    assert _seg_score(Pcum, Ecum, 0, 2) == pytest.approx(F)
    # При T <= 0 (a==b)
    assert _seg_score(Pcum, Ecum, 1, 1) == 0.0


def test_segment_by_time_two_phases():
    counts = _two_regime_counts(40, 40, 256, peak_a=60, peak_b=190)
    segs = segment_by_time(_make_sg(counts), min_slices=3)
    assert len(segs) >= 2
    assert any(abs(s.t_lo - 40) <= 4 for s in segs)


def test_segment_by_time_coverage_no_overlap():
    counts = _two_regime_counts(40, 40, 256, peak_a=60, peak_b=190)
    segs = segment_by_time(_make_sg(counts), min_slices=3)
    # Сортировка по t_lo
    segs.sort(key=lambda s: s.t_lo)
    assert segs[0].t_lo == 0
    assert segs[-1].t_hi == 80
    for i in range(len(segs) - 1):
        assert segs[i].t_hi == segs[i + 1].t_lo
    for s in segs:
        assert s.n_slices >= 3
        assert s.total_counts > 0
    total = sum(s.total_counts for s in segs)
    assert total == counts.sum()


def test_segment_by_time_large_penalty():
    counts = _two_regime_counts(40, 40, 256, peak_a=60, peak_b=190)
    segs = segment_by_time(_make_sg(counts), penalty=1e18)
    assert len(segs) == 1
    assert segs[0].t_lo == 0 and segs[0].t_hi == 80


def test_segment_by_time_short_record():
    short = _make_sg(_two_regime_counts(2, 2, 64, 20, 50))
    segs = segment_by_time(short, min_slices=3)
    assert len(segs) == 1
    assert segs[0].t_lo == 0 and segs[0].t_hi == 4


def test_segment_by_time_max_segments():
    blocks = [_two_regime_counts(20, 0, 256, c, c) for c in (20, 60, 100, 140, 180, 220)]
    counts = np.vstack(blocks)
    segs = segment_by_time(_make_sg(counts), min_slices=3, max_segments=4, pen_factor=0.0)
    assert len(segs) <= 4


def test_identify_segments_structure():
    counts = _two_regime_counts(40, 40, 1600, peak_a=1460, peak_b=662, h_a=40.0, h_b=40.0)
    sg = _make_sg(counts)
    segs = segment_by_time(sg, min_slices=3)
    sidents = identify_segments(sg, default_library(), segs)
    assert len(sidents) == len(segs)
    for k, si in enumerate(sidents):
        assert isinstance(si, SegmentIdent)
        assert isinstance(si.segment, TimeSegment)
        assert si.segment.t_lo == segs[k].t_lo  # соответствие сегментов по порядку
        assert isinstance(si.peaks, tuple)
        assert isinstance(si.idents, tuple)


def test_identify_segments_separate_sources():
    counts = _two_regime_counts(40, 40, 1600, peak_a=1460, peak_b=662, h_a=40.0, h_b=40.0)
    sg = _make_sg(counts)
    segs = segment_by_time(sg, min_slices=3)
    sidents = identify_segments(sg, default_library(), segs)
    cs_idx = [i for i, si in enumerate(sidents) if any(r.nuclide == "Cs-137" for r in si.idents)]
    k_idx = [i for i, si in enumerate(sidents) if any(r.nuclide == "K-40" for r in si.idents)]
    assert cs_idx, "Cs-137 не идентифицирован ни в одном сегменте"
    assert k_idx, "K-40 не идентифицирован ни в одном сегменте"
    assert min(cs_idx) > max(k_idx)


def test_identify_segments_with_fwhm_model():
    counts = _two_regime_counts(40, 40, 1600, peak_a=1460, peak_b=662, h_a=40.0, h_b=40.0)
    sg = _make_sg(counts)
    segs = segment_by_time(sg, min_slices=3)
    from awf.analysis.peaks import auto_calibrate_fwhm_model
    model = auto_calibrate_fwhm_model(np.asarray(sg.total_spectrum(), float),
                                       np.asarray(sg.energies(), float))
    out = identify_segments(sg, default_library(), segs, fwhm_model=model)
    assert len(out) == len(segs)

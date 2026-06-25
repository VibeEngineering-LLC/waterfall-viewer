import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis.gradient import (
    moving_average, total_counts_series, time_gradient, band_gradient, GradientResult,
)


def _step_sg(ns=40, nc=32, step_at=20, low=10, high=200, t_step=2.0):
    """Спектрограмма со ступенькой полного счёта на срезе step_at."""
    counts = np.full((ns, nc), low, dtype=np.int64)
    counts[step_at:, :] = high
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch кэВ
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_moving_average_identity_and_shape():
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    assert np.array_equal(moving_average(y, 0), y)        # radius<=0 — тождество
    sm = moving_average(y, 1)
    assert sm.shape == y.shape                            # форма сохраняется
    assert abs(sm.sum() - y.sum()) < y.sum() * 0.5        # среднее не уносит масштаб


def test_moving_average_smooths_spike():
    y = np.array([0.0, 0.0, 100.0, 0.0, 0.0])
    sm = moving_average(y, 1)
    assert sm[2] < 100.0                                  # пик размазан
    assert sm[1] > 0.0 and sm[3] > 0.0                    # энергия растеклась к соседям


def test_total_counts_series_matches_manual():
    sg = _step_sg()
    s = total_counts_series(sg)
    assert s.shape == (sg.n_slices,)
    assert np.array_equal(s, sg.counts.sum(axis=1).astype(np.float64))


def test_time_gradient_localizes_front():
    sg = _step_sg(ns=40, step_at=20)
    res = time_gradient(total_counts_series(sg), sg.time_offsets_s)
    assert isinstance(res, GradientResult)
    # фронт ступени локализуется на переходе (±1 срез от step_at)
    assert abs(res.front_index - 20) <= 1
    assert res.front_time == sg.time_offsets_s[res.front_index]
    # вне фронта градиент ~0 (счёт-плато)
    assert abs(res.gradient[5]) < abs(res.gradient[res.front_index])


def test_time_gradient_physical_units():
    # ступень high-low=190 за один шаг t_step=2 c -> пик градиента ~95 отсч/с
    sg = _step_sg(ns=30, step_at=15, low=10, high=200, t_step=2.0)
    res = time_gradient(total_counts_series(sg), sg.time_offsets_s)
    peak = np.abs(res.gradient).max()
    nc = sg.n_channels
    expected = (200 - 10) * nc / 2.0 / 2.0   # (Δсчёт_полный)/Δt, центральная разность
    assert peak == pytest.approx(expected, rel=0.2)


def test_time_gradient_errors():
    with pytest.raises(ValueError):
        time_gradient([1.0], [0.0])                        # < 2 срезов
    with pytest.raises(ValueError):
        time_gradient([1.0, 2.0, 3.0], [0.0, 1.0])         # рассинхрон длин


def test_time_gradient_nonmonotonic_time_falls_back():
    y = [0.0, 0.0, 10.0, 10.0]
    t = [0.0, 0.0, 0.0, 0.0]                               # вырожденное время
    res = time_gradient(y, t)                              # не делит на ноль
    assert np.all(np.isfinite(res.gradient))


def test_band_gradient_uses_energy_window():
    sg = _step_sg(ns=40, nc=32, step_at=20)
    res = band_gradient(sg, 5.0, 15.0)                     # окно каналов ~5..15
    assert abs(res.front_index - 20) <= 1                  # тот же фронт во времени
    assert np.all(np.isfinite(res.gradient))
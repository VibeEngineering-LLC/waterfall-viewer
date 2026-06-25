import numpy as np
import pytest
from awf.ui.zscale import (
    apply_z_scale, Z_MODES, DEFAULT_GAIN, DEFAULT_GAMMA, DEFAULT_CLIP,
)


def test_backward_compat_defaults():
    """При дефолтных параметрах apply_z_scale == базовая Z-шкала (short-circuit)."""
    x = np.array([[0.0, 1.0, 4.0], [9.0, 100.0, -5.0]], dtype=np.float64)
    assert np.allclose(apply_z_scale(x, "linear"), np.maximum(x, 0.0))
    assert np.allclose(apply_z_scale(x, "sqrt"), np.sqrt(np.maximum(x, 0.0)))
    assert np.allclose(apply_z_scale(x, "log"), np.log10(1.0 + np.maximum(x, 0.0)))


def test_returns_float32():
    out = apply_z_scale(np.arange(10.0), "linear", gamma=0.5)
    assert out.dtype == np.float32


def test_gamma_lifts_lows():
    """gamma<1 поднимает середину: 50 из [0,50,100] -> 0.5**0.5*100 ≈ 70.71."""
    x = np.array([0.0, 50.0, 100.0])
    out = apply_z_scale(x, "linear", gamma=0.5)
    assert np.isclose(out[0], 0.0, atol=1e-4)
    assert np.isclose(out[1], (0.5 ** 0.5) * 100.0, atol=1e-3)
    assert np.isclose(out[2], 100.0, atol=1e-4)


def test_gain_brightens_and_saturates():
    """gain=2 насыщает: середина 50 -> 100, верх остаётся 100."""
    x = np.array([0.0, 50.0, 100.0])
    out = apply_z_scale(x, "linear", gain=2.0)
    assert np.isclose(out[1], 100.0, atol=1e-4)
    assert np.isclose(out[2], 100.0, atol=1e-4)
    assert np.isclose(out[0], 0.0, atol=1e-4)


def test_clip_percentile_clamps_outlier():
    """Верхний перцентиль отсекает выброс: max выхода == pctl(t,90) < исходного максимума."""
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=np.float64)
    out = apply_z_scale(x, "linear", clip=(0.0, 90.0))
    p90 = np.percentile(np.maximum(x, 0.0), 90.0)
    assert out.max() < 1000.0
    assert np.isclose(out.max(), p90, rtol=1e-4)


def test_output_within_clip_range():
    rng = np.random.default_rng(1)
    x = rng.random(500) * 1000.0
    lo_p, hi_p = 5.0, 95.0
    out = apply_z_scale(x, "linear", clip=(lo_p, hi_p))
    lo_v = np.percentile(x, lo_p)
    hi_v = np.percentile(x, hi_p)
    assert out.min() >= lo_v - 1e-3
    assert out.max() <= hi_v + 1e-3


def test_monotonic_nondecreasing():
    """Конвейер монотонен по входу: порядок сортировки сохраняется."""
    rng = np.random.default_rng(2)
    x = rng.random(300) * 500.0
    out = apply_z_scale(x, "sqrt", gain=1.5, gamma=0.7, clip=(2.0, 98.0))
    order = np.argsort(x)
    s = out[order]
    assert np.all(np.diff(s) >= -1e-4)


def test_clip_swapped_order_ok():
    """clip с переставленными границами не падает (внутренняя перестановка)."""
    x = np.linspace(0, 100, 50)
    a = apply_z_scale(x, "linear", clip=(90.0, 10.0))
    b = apply_z_scale(x, "linear", clip=(10.0, 90.0))
    assert np.allclose(a, b)


def test_constant_array_no_crash():
    x = np.full(20, 7.0)
    out = apply_z_scale(x, "linear", clip=(10.0, 90.0), gain=2.0, gamma=0.5)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_empty_array():
    out = apply_z_scale(np.array([], dtype=np.float64), "log", gamma=0.5)
    assert out.size == 0


def test_defaults_exported():
    assert DEFAULT_GAIN == 1.0 and DEFAULT_GAMMA == 1.0 and DEFAULT_CLIP == (0.0, 100.0)
    assert len(Z_MODES) == 3
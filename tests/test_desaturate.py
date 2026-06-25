import numpy as np

from awf.ui.zscale import desaturate_rgba, DEFAULT_DESAT


def test_desat_default_is_zero():
    assert DEFAULT_DESAT == 0.0


def test_desat_zero_identity():
    c = np.array([[0.2, 0.4, 0.6, 1.0], [0.9, 0.1, 0.3, 0.5]], dtype=np.float32)
    out = desaturate_rgba(c, 0.0)
    assert np.allclose(out, c)


def test_desat_full_is_gray():
    c = np.array([[0.2, 0.4, 0.6, 1.0]], dtype=np.float32)
    out = desaturate_rgba(c, 1.0)
    # rgb равны между собой => серый
    assert np.allclose(out[0, 0], out[0, 1]) and np.allclose(out[0, 1], out[0, 2])
    lum = 0.299 * 0.2 + 0.587 * 0.4 + 0.114 * 0.6
    assert np.allclose(out[0, 0], lum, atol=1e-5)


def test_desat_alpha_preserved():
    c = np.array([[0.2, 0.4, 0.6, 0.33]], dtype=np.float32)
    out = desaturate_rgba(c, 0.7)
    assert np.allclose(out[0, 3], 0.33)


def test_desat_monotonic_reduces_spread():
    c = np.array([[0.1, 0.5, 0.9, 1.0]], dtype=np.float32)
    spread0 = float(c[0, :3].max() - c[0, :3].min())
    s_half = desaturate_rgba(c, 0.5)[0, :3]
    s_full = desaturate_rgba(c, 1.0)[0, :3]
    assert float(s_half.max() - s_half.min()) < spread0
    assert float(s_full.max() - s_full.min()) <= float(s_half.max() - s_half.min()) + 1e-6


def test_desat_clamps_amount():
    c = np.array([[0.2, 0.4, 0.6, 1.0]], dtype=np.float32)
    # amount>1 => серый (как 1.0); amount<0 => identity (как 0.0)
    assert np.allclose(desaturate_rgba(c, 5.0), desaturate_rgba(c, 1.0))
    assert np.allclose(desaturate_rgba(c, -3.0), c)
import numpy as np
import pytest

from awf.analysis.deconvolve import deconvolve_multiplet, DeconvolutionResult


def _gnorm(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def _multiplet(x, comps, const=50.0, slope=0.0):
    y = np.full_like(x, 0.0)
    for c, s, a in comps:
        y = y + a * _gnorm(x, c, s)
    y = y + const + slope * (x - x.mean())
    return y


def test_recovers_two_close_areas_noiseless():
    x = np.arange(61, dtype=np.float64)
    comps = [(20.0, 3.0, 8000.0), (28.0, 3.0, 4000.0)]
    y = _multiplet(x, comps, const=40.0)
    res = deconvolve_multiplet(y, [20.0, 28.0], [3.0, 3.0], x=x, continuum="linear")
    assert isinstance(res, DeconvolutionResult)
    assert res.areas[0] == pytest.approx(8000.0, rel=0.02)
    assert res.areas[1] == pytest.approx(4000.0, rel=0.02)


def test_areas_nonnegative_when_component_absent():
    x = np.arange(61, dtype=np.float64)
    y = _multiplet(x, [(20.0, 3.0, 8000.0)], const=30.0)   # второго пика нет
    res = deconvolve_multiplet(y, [20.0, 28.0], [3.0, 3.0], x=x)
    assert res.areas[1] >= 0.0
    assert res.areas[1] < 0.05 * 8000.0
    assert res.areas[0] == pytest.approx(8000.0, rel=0.03)


def test_chi2_small_for_exact_model():
    x = np.arange(61, dtype=np.float64)
    y = _multiplet(x, [(20.0, 3.0, 6000.0), (30.0, 4.0, 3000.0)], const=20.0, slope=0.5)
    res = deconvolve_multiplet(y, [20.0, 30.0], [3.0, 4.0], x=x)
    assert res.chi2_dof < 1.0          # модель точна -> остатки малы


def test_fallback_matches_scipy():
    x = np.arange(61, dtype=np.float64)
    comps = [(22.0, 3.0, 7000.0), (30.0, 3.0, 5000.0)]
    y = _multiplet(x, comps, const=35.0)
    sp = deconvolve_multiplet(y, [22.0, 30.0], [3.0, 3.0], x=x, use_scipy=True)
    np_ = deconvolve_multiplet(y, [22.0, 30.0], [3.0, 3.0], x=x, use_scipy=False)
    assert sp.method == "lsq_linear"
    assert np_.method == "lstsq"
    assert np.allclose(sp.areas, np_.areas, rtol=0.02, atol=1.0)


def test_continuum_modes_run():
    x = np.arange(41, dtype=np.float64)
    y = _multiplet(x, [(20.0, 3.0, 5000.0)], const=10.0)
    for mode, ncont in (("linear", 2), ("constant", 1), ("none", 0)):
        res = deconvolve_multiplet(y, [20.0], [3.0], x=x, continuum=mode)
        assert res.continuum.shape == (ncont,)
        assert res.fit.shape == x.shape


def test_with_poisson_noise_recovers():
    x = np.arange(81, dtype=np.float64)
    comps = [(30.0, 3.5, 9000.0), (40.0, 3.5, 6000.0)]
    y_true = _multiplet(x, comps, const=60.0)
    y = np.random.RandomState(3).poisson(np.maximum(y_true, 0.0)).astype(np.float64)
    res = deconvolve_multiplet(y, [30.0, 40.0], [3.5, 3.5], x=x)
    assert res.areas[0] == pytest.approx(9000.0, rel=0.1)
    assert res.areas[1] == pytest.approx(6000.0, rel=0.1)
    assert np.all(res.d_areas >= 0.0)


def test_input_validation():
    x = np.arange(20, dtype=np.float64)
    y = np.ones(20)
    with pytest.raises(ValueError):
        deconvolve_multiplet(y, [5.0, 10.0], [2.0], x=x)        # рассинхрон centers/sigmas
    with pytest.raises(ValueError):
        deconvolve_multiplet(y, [], [], x=x)                    # нет компонент
    with pytest.raises(ValueError):
        deconvolve_multiplet(y, [5.0], [2.0], x=np.arange(10))  # x и y разной длины
    with pytest.raises(ValueError):
        deconvolve_multiplet(y, [5.0], [2.0], continuum="weird")
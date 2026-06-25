import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis.decomposition import (
    METHODS, ProjectionResult, feature_matrix, is_available, pca, project,
)


def _two_cluster_X(n_each=15, nc=200, c0=40, c1=140, sigma=5.0, seed=0):
    rng = np.random.RandomState(seed)
    ch = np.arange(nc, dtype=np.float64)

    def spec(center):
        g = np.exp(-0.5 * ((ch - center) / sigma) ** 2)
        return g / g.max() * 100.0

    rows, labels = [], []
    for _ in range(n_each):
        rows.append(spec(c0) + rng.poisson(3, nc)); labels.append(0)
    for _ in range(n_each):
        rows.append(spec(c1) + rng.poisson(3, nc)); labels.append(1)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels)


def _sg(ns=10, nc=64, t_step=1.0, seed=1):
    counts = np.random.RandomState(seed).poisson(20, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_methods_and_availability():
    assert set(("pca", "tsne", "umap")).issubset(set(METHODS))
    assert is_available("pca") is True
    assert is_available("bogus") is False


def test_feature_matrix_shape_and_options():
    sg = _sg(ns=10, nc=64)
    X = feature_matrix(sg)
    assert X.shape == (10, 64)
    Xn = feature_matrix(sg, normalize=True)
    assert np.allclose(Xn.sum(axis=1), 1.0)        # L1-нормировка строк
    Xl = feature_matrix(sg, log=True)
    assert Xl.max() < X.max()                      # log сжимает диапазон


def test_pca_separates_two_clusters():
    X, labels = _two_cluster_X()
    res = pca(X, 2)
    assert isinstance(res, ProjectionResult)
    assert res.coords.shape == (X.shape[0], 2)
    a = res.coords[labels == 0, 0]
    b = res.coords[labels == 1, 0]
    gap = abs(a.mean() - b.mean())
    spread = max(a.std(), b.std())
    assert gap > 4.0 * spread                      # PC1 разделяет кластеры

def test_pca_explained_variance_descending():
    X, _ = _two_cluster_X()
    res = pca(X, 3)
    evr = res.explained_variance
    assert evr is not None and evr.shape == (3,)
    assert evr[0] >= evr[1] >= evr[2]              # доли дисперсии не возрастают
    assert 0.0 < evr.sum() <= 1.0 + 1e-9


def test_pca_clamps_components():
    X = np.random.RandomState(0).normal(size=(4, 7))
    res = pca(X, 10)                               # k > n,m -> усечётся
    assert res.coords.shape[1] <= min(4, 7)


def test_project_dispatch_pca_and_unknown():
    X, _ = _two_cluster_X()
    assert project(X, "pca").method == "pca"
    with pytest.raises(ValueError):
        project(X, "nope")


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_optional_methods_graceful(method):
    X, _ = _two_cluster_X(n_each=8, nc=60)
    if is_available(method):
        res = project(X, method, n_components=2)   # пакет есть — должен отработать
        assert res.coords.shape == (X.shape[0], 2)
    else:
        with pytest.raises(ImportError):           # пакета нет — понятная ошибка
            project(X, method, n_components=2)
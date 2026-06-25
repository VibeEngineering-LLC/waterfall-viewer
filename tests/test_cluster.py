import numpy as np
import pytest

from awf.analysis.cluster import (
    METHODS, ClusterResult, kmeans, cluster, segments, is_available,
)


def _blobs(per=12, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    X, y = [], []
    for li, c in enumerate(centers):
        X.append(rng.normal(c, 0.4, size=(per, 2)))
        y += [li] * per
    return np.vstack(X), np.asarray(y)


def _maps_bijectively(true_labels, pred_labels, k):
    """Каждый истинный кластер -> ровно одна (своя) предсказанная метка."""
    mapping = {}
    for tl in range(k):
        preds = set(pred_labels[true_labels == tl].tolist())
        if len(preds) != 1:
            return False
        mapping[tl] = preds.pop()
    return len(set(mapping.values())) == k


def test_methods_and_availability():
    assert set(("kmeans", "dbscan", "hdbscan")).issubset(set(METHODS))
    assert is_available("kmeans") is True
    assert is_available("bogus") is False


def test_kmeans_separates_blobs():
    X, y = _blobs(per=15, seed=1)
    res = kmeans(X, 3, random_state=0)
    assert isinstance(res, ClusterResult)
    assert res.labels.shape == (X.shape[0],)
    assert res.n_clusters == 3
    assert _maps_bijectively(y, res.labels, 3)
    assert res.centers.shape == (3, 2)
    assert res.inertia >= 0.0


def test_kmeans_deterministic():
    X, _ = _blobs(seed=2)
    r1 = kmeans(X, 3, random_state=7)
    r2 = kmeans(X, 3, random_state=7)
    assert np.array_equal(r1.labels, r2.labels)
    assert r1.inertia == pytest.approx(r2.inertia)


def test_kmeans_clamps_k_to_n():
    X = np.random.RandomState(0).normal(size=(3, 2))
    res = kmeans(X, 10)               # k > n -> усечётся до n
    assert res.labels.shape == (3,)
    assert res.n_clusters <= 3


def test_cluster_dispatch_and_unknown():
    X, _ = _blobs()
    assert cluster(X, "kmeans", 3).method == "kmeans"
    with pytest.raises(ValueError):
        cluster(X, "nope")


def test_segments_contiguous_runs():
    labels = [0, 0, 1, 1, 1, 2, 0]
    segs = segments(labels)
    assert segs == [(0, 2, 0), (2, 5, 1), (5, 6, 2), (6, 7, 0)]
    assert segments([]) == []


def test_segments_align_with_time():
    # синтетика: первая половина времени — кластер A, вторая — B
    X = np.vstack([np.zeros((10, 2)), np.full((10, 2), 8.0)])
    res = kmeans(X, 2, random_state=0)
    segs = segments(res.labels)
    assert len(segs) == 2                       # ровно два временных сегмента
    assert segs[0][1] == 10 and segs[1][0] == 10


@pytest.mark.parametrize("method", ["dbscan", "hdbscan"])
def test_optional_methods_graceful(method):
    X, _ = _blobs()
    if is_available(method):
        res = cluster(X, method)
        assert res.labels.shape == (X.shape[0],)
    else:
        with pytest.raises(ImportError):
            cluster(X, method)
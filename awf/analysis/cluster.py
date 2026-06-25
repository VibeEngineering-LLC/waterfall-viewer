"""Кластерный анализ временных срезов (Задача 25, ТЗ-B.9).

Разбиение срезов на группы разного изотопного состава по матрице признаков (спектры,
интегралы окон Задача 21, PCA-компоненты Задача 24). KMeans реализован на чистом numpy
(k-means++ init, фикс. random_state) — доступен всегда; DBSCAN/HDBSCAN опциональны через
scikit-learn. Метки переносятся на ось времени непрерывными сегментами. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

METHODS = ("kmeans", "dbscan", "hdbscan")


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray                 # метка кластера на срез (n_slices,), -1 = шум
    n_clusters: int                    # число непустых кластеров (без шума)
    method: str                        # "kmeans" | "dbscan" | "hdbscan"
    inertia: Optional[float] = None    # сумма квадратов до центров (KMeans)
    centers: Optional[np.ndarray] = None


def is_available(method: str) -> bool:
    """Доступен ли метод кластеризации в текущем окружении."""
    m = method.lower()
    if m == "kmeans":
        return True
    if m == "dbscan":
        try:
            import sklearn.cluster  # noqa: F401
            return True
        except Exception:
            return False
    if m == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN  # noqa: F401
            return True
        except Exception:
            try:
                import hdbscan  # noqa: F401
                return True
            except Exception:
                return False
    return False


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Квадраты расстояний (n, k) между точками X и центрами C."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def _init_pp(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """k-means++ инициализация центров."""
    n = X.shape[0]
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = _sqdist(X, np.asarray(centers)).min(axis=1)
        tot = d2.sum()
        probs = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    return np.asarray(centers, dtype=np.float64)


def kmeans(X, n_clusters: int, *, random_state: int = 0,
           n_init: int = 4, max_iter: int = 100) -> ClusterResult:
    """KMeans (Lloyd) на numpy с k-means++ инициализацией, фикс. random_state."""
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("kmeans: ожидается 2D матрица признаков")
    n = A.shape[0]
    k = max(1, min(int(n_clusters), n))
    rng = np.random.RandomState(random_state)
    best_labels, best_centers, best_inertia = None, None, np.inf
    for _ in range(max(1, n_init)):
        C = _init_pp(A, k, rng)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            d = _sqdist(A, C)
            new_labels = d.argmin(axis=1)
            newC = np.array([A[new_labels == j].mean(axis=0) if np.any(new_labels == j)
                             else C[j] for j in range(k)])
            if np.array_equal(new_labels, labels) and np.allclose(newC, C):
                C = newC
                labels = new_labels
                break
            C, labels = newC, new_labels
        inertia = float(_sqdist(A, C).min(axis=1).sum())
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels.copy(), C.copy(), inertia
    return ClusterResult(labels=best_labels, n_clusters=int(len(np.unique(best_labels))),
                         method="kmeans", inertia=best_inertia, centers=best_centers)


def _sklearn_cluster(X, method, **kw) -> ClusterResult:
    A = np.asarray(X, dtype=np.float64)
    if method == "dbscan":
        try:
            from sklearn.cluster import DBSCAN
        except Exception as e:
            raise ImportError("DBSCAN требует scikit-learn (pip install scikit-learn)") from e
        labels = DBSCAN(**kw).fit_predict(A)
    else:  # hdbscan
        try:
            from sklearn.cluster import HDBSCAN as _H
        except Exception:
            try:
                from hdbscan import HDBSCAN as _H
            except Exception as e:
                raise ImportError(
                    "HDBSCAN требует scikit-learn>=1.3 или пакет hdbscan"
                ) from e
        labels = _H(**kw).fit_predict(A)
    labels = np.asarray(labels, dtype=np.int64)
    nclu = int(len(set(labels.tolist()) - {-1}))
    return ClusterResult(labels=labels, n_clusters=nclu, method=method)


def cluster(X, method: str = "kmeans", n_clusters: int = 2,
            *, random_state: int = 0, **kw) -> ClusterResult:
    """Кластеризовать матрицу признаков. KMeans всегда; DBSCAN/HDBSCAN — при наличии sklearn."""
    m = method.lower()
    if m == "kmeans":
        return kmeans(X, n_clusters, random_state=random_state, **kw)
    if m in ("dbscan", "hdbscan"):
        return _sklearn_cluster(X, m, **kw)
    raise ValueError(f"cluster: неизвестный метод {method!r}; ожидается один из {METHODS}")


def segments(labels):
    """Непрерывные сегменты одинаковой метки по оси времени.

    Возврат списка (start_index, end_index_exclusive, label) — для раскраски временной оси.
    """
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    out, start = [], 0
    for i in range(1, lab.size):
        if lab[i] != lab[i - 1]:
            out.append((start, i, int(lab[i - 1])))
            start = i
    out.append((start, lab.size, int(lab[-1])))
    return out
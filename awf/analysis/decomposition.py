"""Понижение размерности по временным срезам (Задача 24, ТЗ-B.6).

Проекция спектров временных срезов в 2D для группировки/поиска аномалий. PCA —
на чистом numpy (SVD), доступна всегда. t-SNE/UMAP — опциональны (scikit-learn / umap-learn);
при отсутствии пакета метод выключается с понятным сообщением. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from awf.model.spectrogram import Spectrogram

METHODS = ("pca", "tsne", "umap")


@dataclass(frozen=True)
class ProjectionResult:
    """2D-проекция набора срезов."""
    coords: np.ndarray                      # (n_slices, n_components)
    method: str                             # "pca" | "tsne" | "umap"
    explained_variance: Optional[np.ndarray] = None  # доля дисперсии по компонентам (PCA)


def feature_matrix(sg: Spectrogram, *, normalize: bool = False,
                   log: bool = False) -> np.ndarray:
    """Матрица признаков: строка = спектр временного среза (n_slices, n_channels).

    ``log`` — log1p (сжатие динамического диапазона); ``normalize`` — L1-нормировка строки
    (форма спектра без зависимости от полной активности среза). log применяется до нормировки.
    """
    X = sg.counts.astype(np.float64, copy=True)
    if log:
        X = np.log1p(np.maximum(X, 0.0))
    if normalize:
        s = X.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        X = X / s
    return X


def is_available(method: str) -> bool:
    """Доступен ли метод проекции в текущем окружении."""
    m = method.lower()
    if m == "pca":
        return True
    if m == "tsne":
        try:
            import sklearn.manifold  # noqa: F401
            return True
        except Exception:
            return False
    if m == "umap":
        try:
            import umap  # noqa: F401
            return True
        except Exception:
            return False
    return False


def pca(X: np.ndarray, n_components: int = 2) -> ProjectionResult:
    """PCA через SVD центрированной матрицы. Возвращает счета и долю дисперсии."""
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("pca: ожидается 2D матрица признаков")
    n, m = A.shape
    k = max(1, min(int(n_components), n, m))
    mean = A.mean(axis=0, keepdims=True)
    Ac = A - mean
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    scores = U[:, :k] * S[:k]
    total = float((S ** 2).sum())
    evr = (S[:k] ** 2 / total) if total > 0 else np.zeros(k)
    return ProjectionResult(coords=scores, method="pca",
                            explained_variance=np.asarray(evr, dtype=np.float64))


def _tsne(X, n_components, random_state, **kw) -> ProjectionResult:
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        raise ImportError(
            "t-SNE требует scikit-learn (pip install scikit-learn)"
        ) from e
    A = np.asarray(X, dtype=np.float64)
    per = min(30, max(2, A.shape[0] - 1))
    model = TSNE(n_components=n_components, random_state=random_state,
                 perplexity=kw.pop("perplexity", per), init="pca")
    coords = model.fit_transform(A)
    return ProjectionResult(coords=np.asarray(coords, dtype=np.float64), method="tsne")


def _umap(X, n_components, random_state, **kw) -> ProjectionResult:
    try:
        import umap
    except Exception as e:
        raise ImportError(
            "UMAP требует umap-learn (pip install umap-learn)"
        ) from e
    A = np.asarray(X, dtype=np.float64)
    model = umap.UMAP(n_components=n_components, random_state=random_state, **kw)
    coords = model.fit_transform(A)
    return ProjectionResult(coords=np.asarray(coords, dtype=np.float64), method="umap")


def project(X, method: str = "pca", n_components: int = 2,
            *, random_state: int = 0, **kw) -> ProjectionResult:
    """Спроецировать матрицу признаков выбранным методом ("pca"|"tsne"|"umap").

    PCA доступна всегда; t-SNE/UMAP бросают ImportError с понятным сообщением, если
    соответствующий пакет не установлен.
    """
    m = method.lower()
    if m == "pca":
        return pca(X, n_components)
    if m == "tsne":
        return _tsne(X, n_components, random_state, **kw)
    if m == "umap":
        return _umap(X, n_components, random_state, **kw)
    raise ValueError(f"project: неизвестный метод {method!r}; ожидается один из {METHODS}")
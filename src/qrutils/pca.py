from __future__ import annotations
from typing import Any, cast
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]

def _center_and_scale(
    X: FloatArray, standardize: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Center (and optionally standardize) columns of X."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    Xc = X - mu
    if standardize:
        s = Xc.std(axis=0, ddof=1)
        s = np.where(s == 0, 1.0, s)
        Xc = Xc / s
    else:
        s = np.ones_like(mu)
    return cast(FloatArray, Xc), cast(FloatArray, mu), cast(FloatArray, s)

def pca_svd(
    X: FloatArray, k: int, *, standardize: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """PCA via SVD, returns components, explained variance, ratios, and scores."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not (1 <= k <= min(n, d)):
        raise ValueError(f"k must be in [1, {min(n, d)}], got {k}")
    Xc, _, _ = _center_and_scale(cast(FloatArray, X), standardize=standardize)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S**2) / (n - 1)
    total_var = ev.sum()
    idx = slice(0, k)
    components = Vt[idx, :]
    explained_var = ev[idx]
    explained_var_ratio = explained_var / total_var
    scores = Xc @ components.T
    return (
        cast(FloatArray, components),
        cast(FloatArray, explained_var),
        cast(FloatArray, explained_var_ratio),
        cast(FloatArray, scores),
    )





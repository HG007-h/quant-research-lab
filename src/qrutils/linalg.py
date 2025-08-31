from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.floating[Any]]
ArrayLikeF = npt.ArrayLike


def l2_norm(x: ArrayLikeF, axis: int | None = None) -> np.floating[Any] | FloatArray:
    """
    L2 norm. Returns a NumPy scalar if axis is None, else an array.
    """
    arr = np.asarray(x, dtype=float)
    out = np.linalg.norm(arr, ord=2, axis=axis)
    if axis is None:
        return cast(np.floating[Any], out)
    return cast(FloatArray, out)


def dot(a: ArrayLikeF, b: ArrayLikeF) -> np.floating[Any] | FloatArray:
    """
    Dot product. Returns a NumPy scalar if both inputs are 1-D, else an array.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.dot(a_arr, b_arr)
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return cast(np.floating[Any], out)
    return cast(FloatArray, out)


def rowwise_normalize(X: ArrayLikeF, eps: float = 1e-12) -> FloatArray:
    """
    Normalize rows of X to unit L2 norm with numerical stability.
    """
    X_arr = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    Xn = X_arr / np.maximum(norms, eps)
    return cast(FloatArray, Xn)


def cosine_similarity_matrix(X: ArrayLikeF) -> FloatArray:
    """
    Cosine similarity matrix for rows of X.
    """
    Xn = rowwise_normalize(X)
    S = Xn @ Xn.T
    return cast(FloatArray, S)


def pairwise_euclidean_distances(X: ArrayLikeF) -> FloatArray:
    """
    Pairwise Euclidean distances between rows of X (vectorized, numerically stable).
    Ensures exact zeros on the diagonal.
    """
    X_arr = np.asarray(X, dtype=float)
    xx: FloatArray = cast(FloatArray, np.sum(X_arr * X_arr, axis=1, keepdims=True))  # (n,1)
    D2 = xx + xx.T - 2.0 * (X_arr @ X_arr.T)  # squared distances
    np.maximum(D2, 0.0, out=D2)  # clip tiny negatives
    np.fill_diagonal(D2, 0.0)  # exact zeros on diagonal
    D = np.sqrt(D2, out=D2)
    return cast(FloatArray, D)


def pairwise_cosine_distances(X: ArrayLikeF) -> FloatArray:
    """
    Pairwise cosine distances = 1 - cosine_similarity.
    """
    sim = cosine_similarity_matrix(X)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return cast(FloatArray, D)

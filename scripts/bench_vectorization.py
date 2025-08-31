"""
Micro-benchmarks: loops vs vectorized NumPy for rolling mean and pairwise distances.
Run:
    python scripts/bench_vectorization.py
"""

from __future__ import annotations

import time

import numpy as np


def timeit(fn, *args, repeats: int = 5, **kwargs) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def rolling_mean_loop(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n - window + 1, dtype=float)
    for i in range(n - window + 1):
        out[i] = x[i : i + window].mean()
    return out


def rolling_mean_vec(x: np.ndarray, window: int) -> np.ndarray:
    # cumulative sum trick
    c = np.cumsum(np.r_[0.0, x])
    return (c[window:] - c[:-window]) / window


def pairwise_dist_loop(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            D[i, j] = np.sqrt(np.dot(diff, diff))
    return D


def pairwise_dist_vec(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=False)
    xx = np.sum(X * X, axis=1, keepdims=True)
    D2 = xx + xx.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, out=D2)


def main():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2_000_000)
    X = rng.standard_normal((3_000, 50))
    window = 50

    t_loop_rm = timeit(rolling_mean_loop, x, window)
    t_vec_rm = timeit(rolling_mean_vec, x, window)
    t_loop_pd = timeit(pairwise_dist_loop, X)
    t_vec_pd = timeit(pairwise_dist_vec, X)

    print(
        f"Rolling mean   - loop: {t_loop_rm:.3f}s | vectorized: {t_vec_rm:.3f}s | speedup ~{t_loop_rm/t_vec_rm:,.1f}x"
    )
    print(
        f"Pairwise dist  - loop: {t_loop_pd:.3f}s | vectorized: {t_vec_pd:.3f}s | speedup ~{t_loop_pd/t_vec_pd:,.1f}x"
    )


if __name__ == "__main__":
    main()

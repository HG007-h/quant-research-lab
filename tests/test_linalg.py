import numpy as np

from qrutils.linalg import (
    cosine_similarity_matrix,
    dot,
    l2_norm,
    pairwise_cosine_distances,
    pairwise_euclidean_distances,
)


def test_norm_and_dot():
    x = np.array([3.0, 4.0])
    assert np.isclose(l2_norm(x), 5.0)
    y = np.array([1.0, 2.0])
    assert np.isclose(dot(x, y), 11.0)


def test_cosine_similarity_matrix_basic():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = cosine_similarity_matrix(X)
    assert S.shape == (2, 2)
    assert np.isclose(S[0, 0], 1.0)
    assert np.isclose(S[1, 1], 1.0)
    assert np.isclose(S[0, 1], 0.0)


def test_pairwise_distances_match():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    D_euc = pairwise_euclidean_distances(X)
    # symmetry and zeros on diagonal
    assert np.allclose(D_euc, D_euc.T, atol=1e-10)
    assert np.allclose(np.diag(D_euc), 0.0, atol=1e-10)

    D_cos = pairwise_cosine_distances(X)
    assert np.all(D_cos >= -1e-10)  # small numerical noise tolerated
    assert np.allclose(np.diag(D_cos), 0.0, atol=1e-10)

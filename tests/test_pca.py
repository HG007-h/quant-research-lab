
import numpy as np
import numpy.testing as npt
from sklearn.decomposition import PCA as SKPCA

from qrutils.pca import pca_svd

def test_pca_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    k = 5
    comps, ev, evr, _ = pca_svd(X, k)
    sk = SKPCA(n_components=k).fit(X)
    npt.assert_allclose(ev, sk.explained_variance_, rtol=1e-5, atol=1e-8)
    npt.assert_allclose(evr, sk.explained_variance_ratio_, rtol=1e-5, atol=1e-8)

def test_reconstruction_error_decreases():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 8))
    comps2, _, _, _ = pca_svd(X, 2)
    comps3, _, _, _ = pca_svd(X, 3)
    Xc = X - X.mean(axis=0)
    rec2 = (Xc - (Xc @ comps2.T) @ comps2).var()
    rec3 = (Xc - (Xc @ comps3.T) @ comps3).var()
    assert rec3 <= rec2 + 1e-12

def test_pca_with_standardize_branch():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5))
    k = 3
    comps, ev, evr, scores = pca_svd(X, k, standardize=True)
    assert comps.shape == (k, X.shape[1])
    assert ev.shape == (k,)
    assert scores.shape == (X.shape[0], k)
    s = float(evr.sum())
    assert 0.0 < s < 1.0          # only top-k variance fraction
    assert np.all(evr >= 0)       # ratios are non-negative


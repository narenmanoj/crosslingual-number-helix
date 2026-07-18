"""Fit the number 'helix' (Kantamneni & Tegmark 2502.00873 recipe).

For integer n, the Fourier feature basis is:
    B(n) = [ n/n_max ,  cos(2*pi*n/T), sin(2*pi*n/T)  for T in periods ]
We PCA-reduce the residual-stream activations, then linearly regress the PCA
coordinates onto B(n). R^2 measures how much of the (reduced) activation variance
the helix explains. The helix SUBSPACE in model space is the image of the Fourier
features under the fitted map, which is what we compare across surface forms.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

DEFAULT_PERIODS = (2, 5, 10, 100)


def fourier_basis(numbers, periods=DEFAULT_PERIODS, include_linear=True, nmax=None) -> np.ndarray:
    """Fourier feature basis for integers.

    nmax: normalization for the linear term. MUST be passed (from the fit) when reconstructing on
    a different range than the fit, else the linear component is mis-scaled (e.g. fit 0-99 nmax=99
    vs reconstruct 0-9 nmax=9 -> 11x too large). Defaults to numbers.max() only for a fresh fit.

    The sin term at period 2 (and 1) is identically zero for integer inputs (sin(pi*n)=0), so it is
    dropped -- otherwise it is a dead column that makes the basis rank-deficient and injects an
    arbitrary direction into the orthonormalized helix subspace. cos at period 2 = (-1)^n is kept.
    """
    nums = np.asarray(numbers, dtype=float)
    if nmax is None:
        nmax = max(nums.max(), 1.0)
    feats = []
    if include_linear:
        feats.append(nums / nmax)
    for T in periods:
        feats.append(np.cos(2 * np.pi * nums / T))
        if T not in (1, 2):  # sin(2*pi*n/T) == 0 for all integer n when T divides 2
            feats.append(np.sin(2 * np.pi * nums / T))
    return np.stack(feats, axis=1)  # [n, d_fourier]


def fit_helix(H: np.ndarray, numbers, periods=DEFAULT_PERIODS, k_pca: int = 20) -> dict:
    """H: [n, d_model] activations (rows aligned to `numbers`)."""
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    nmax = max(float(np.asarray(numbers, dtype=float).max()), 1.0)
    k = min(k_pca, n - 1, H.shape[1])
    pca = PCA(n_components=k)
    Z = pca.fit_transform(H)  # [n, k]
    B = fourier_basis(numbers, periods, nmax=nmax)  # [n, d_fourier]

    # least-squares: Z ~ B  =>  W [d_fourier, k]
    W, *_ = np.linalg.lstsq(B, Z, rcond=None)
    Z_hat = B @ W
    ss_res = ((Z - Z_hat) ** 2).sum()
    ss_tot = ((Z - Z.mean(0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # helix directions back in model space: [d_fourier, d_model]
    helix_dirs_model = W @ pca.components_

    return {
        "r2": r2,
        "W": W,
        "pca": pca,
        "Z": Z,
        "helix_dirs_model": helix_dirs_model,
        "mean": H.mean(0),
        "periods": periods,
        "nmax": nmax,
    }


def shuffled_control_r2(H, numbers, seed=0, **kw) -> float:
    """Fit the helix against SHUFFLED number labels. Should collapse toward 0 if the
    structure is genuinely number-indexed rather than an artifact of the fit's capacity."""
    rng = np.random.default_rng(seed)
    shuffled = list(numbers)
    rng.shuffle(shuffled)
    return fit_helix(H, shuffled, **kw)["r2"]

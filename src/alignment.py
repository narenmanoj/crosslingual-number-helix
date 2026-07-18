"""Compare number geometry across surface forms.

Metrics (all in [0,1], rows paired by number):
  - subspace_alignment: principal angles between the two helix SUBSPACES. PRIMARY,
    transport-relevant metric -- 'same literal directions'. Read against random_subspace_floor.
  - orthogonal_procrustes_cv: held-out R^2 of the best ROTATION aligning the two helices.
    Necessary-not-sufficient ('same shape, maybe rotated').
  - linear_cka: representational-similarity sanity check; weak discriminator here.

Controls:
  - random_subspace_floor (what subspace_cos looks like for an unrelated subspace).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles


def orthonormal_basis(dirs: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """dirs: [r, d_model] rows span a subspace -> [d_model, rank] orthonormal columns.

    Rank-revealing SVD (not QR): drops near-degenerate directions so a rank-deficient `dirs`
    (e.g. a dead Fourier column) doesn't contribute an arbitrary numerical axis to the subspace."""
    U, S, _ = np.linalg.svd(dirs.T, full_matrices=False)   # U: [d_model, min(d_model, r)]
    if S.size == 0 or S[0] == 0:
        return U[:, :0]
    keep = S > tol * S[0]
    return U[:, keep]


def subspace_alignment(dirs_a: np.ndarray, dirs_b: np.ndarray) -> dict:
    Qa = orthonormal_basis(dirs_a)
    Qb = orthonormal_basis(dirs_b)
    angles = subspace_angles(Qa, Qb)
    cos = np.cos(angles)
    return {
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "max_cos": float(cos.max()),
        "principal_cosines": [float(c) for c in cos],
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro")
    ny = np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (nx * ny)) if nx > 0 and ny > 0 else 0.0


def random_subspace_floor(dirs_ref: np.ndarray, d_model: int, n_trials: int = 20, seed: int = 0) -> float:
    """Mean principal-angle cosine between the reference subspace and random subspaces of
    the same dimension. This is the 'no real alignment' baseline."""
    rng = np.random.default_rng(seed)
    r = dirs_ref.shape[0]
    Qref = orthonormal_basis(dirs_ref)
    vals = []
    for _ in range(n_trials):
        R = rng.standard_normal((r, d_model))
        Qr = orthonormal_basis(R)
        vals.append(float(np.cos(subspace_angles(Qref, Qr)).mean()))
    return float(np.mean(vals))


def orthogonal_procrustes_cv(X: np.ndarray, Y: np.ndarray, k: int = 12,
                             train_frac: float = 0.8, seed: int = 0) -> float:
    """Held-out R^2 of the best ROTATION aligning form X's geometry onto form Y's.

    Each form is reduced with its OWN PCA to k dims (so this is robust to the two forms
    occupying different directions in the model), then an orthogonal map is fit on a train
    split of numbers and scored on held-out numbers. High score = 'both encode the numbers
    as the same-shaped object, up to rotation'. This is NECESSARY-not-sufficient: it is high
    for essentially any two competent number encoders. The discriminating, transport-enabling
    metric is subspace_alignment (same literal directions). Use this as a sanity floor and to
    distinguish 'different directions but same shape' (transport via a learned map) from
    'no shared geometry / tokenization destroyed it' (this metric also drops).

    k is deliberately small (~helix dim): a k-dim orthogonal map has k(k-1)/2 free params, so a
    large k overfits the ~70-80 train numbers and gives NEGATIVE held-out R^2 (seen at k=30 on
    7B for byte-fragmented scripts). k=12 keeps the map well-determined; report robustness to k."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, float); Y = np.asarray(Y, float)
    kk = min(k, X.shape[0] - 1, X.shape[1], Y.shape[1])
    Xr = PCA(n_components=kk).fit_transform(X)
    Yr = PCA(n_components=kk).fit_transform(Y)  # each form in its OWN principal axes
    Xc = Xr - Xr.mean(0); Yc = Yr - Yr.mean(0)

    n = Xc.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]

    # orthogonal Procrustes: argmin_R ||Xc[tr] R - Yc[tr]||,  R = U V^T
    U, _, Vt = np.linalg.svd(Xc[tr].T @ Yc[tr], full_matrices=False)
    R = U @ Vt
    Yhat = Xc[te] @ R
    ss_res = ((Yc[te] - Yhat) ** 2).sum()
    ss_tot = ((Yc[te] - Yc[te].mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

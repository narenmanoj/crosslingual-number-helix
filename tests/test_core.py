"""Unit tests locking in the audit fixes (no model needed -- pure numpy on synthetic activations).

Run:  python -m pytest tests/ -q      (or)      python tests/test_core.py
Each test targets a specific audit item so a regression points straight at the cause.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helix import fit_helix, fourier_basis, heldout_r2, DEFAULT_PERIODS
from src.patching import (helix_reconstruct, helix_subspace_basis, make_patched_vector,
                          random_subspace_basis, covariance_matched_basis, subspace_energy)
from src.alignment import (subspace_alignment, orthogonal_procrustes_cv, canonical_map_cosines,
                           permutation_alignment_null, orthonormal_basis)

RNG = np.random.default_rng(0)
NUMS = list(range(100))


def _linear_code(d=64, scale=3.0, noise=0.0, seed=0):
    """Activations carrying an exact centered linear-in-n code along one direction."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    n = np.array(NUMS, float)
    return np.outer((n - n.mean()) / n.std(), u) * scale + rng.standard_normal(d) * noise, u


def _helix_code(d=64, noise=0.05, seed=0):
    """Activations with signal in ALL Fourier features: H = (B - B_mean) @ D + noise, for a known
    feature->model map D. Unlike a pure linear code, every helix direction is meaningful, so
    coordinate-identity and subspace-alignment tests exercise the real object."""
    rng = np.random.default_rng(seed)
    B = fourier_basis(NUMS, DEFAULT_PERIODS)
    Bc = B - B.mean(0, keepdims=True)
    D = rng.standard_normal((B.shape[1], d))
    return Bc @ D + rng.standard_normal((len(NUMS), d)) * noise, D


# ---- audit #3: centered Fourier basis ----
def test_centered_basis_recovers_linear_code():
    H, _ = _linear_code()
    fit = fit_helix(H, NUMS, k_pca=20)
    assert fit["r2"] > 0.999, f"R^2 should be ~1 on an exact linear code, got {fit['r2']}"
    assert "B_mean" in fit, "fit must store B_mean for consistent reconstruction"


def test_reconstruction_matches_activations():
    H, _ = _linear_code()
    fit = fit_helix(H, NUMS, k_pca=20)
    Hhat = helix_reconstruct(fit, NUMS)
    rel = np.linalg.norm(Hhat - H) / np.linalg.norm(H)
    assert rel < 1e-6, f"reconstruction should recover activations, rel-err={rel}"


def test_nmax_reuse_consistent_across_ranges():
    # reconstructing a NARROW range must reuse the fit's nmax (else the linear term is mis-scaled)
    H, _ = _linear_code()
    fit = fit_helix(H, NUMS, k_pca=20)
    narrow = helix_reconstruct(fit, list(range(10)))
    full = helix_reconstruct(fit, NUMS)[:10]
    assert np.allclose(narrow, full), "reconstruction on a sub-range must match the full-range slice"


# ---- audit #8: deterministic PCA ----
def test_fit_helix_deterministic():
    H, _ = _linear_code(noise=0.1)
    a, b = fit_helix(H, NUMS, k_pca=20), fit_helix(H, NUMS, k_pca=20)
    assert np.allclose(a["W"], b["W"]) and a["r2"] == b["r2"]


def test_covariance_matched_deterministic():
    H, _ = _linear_code(noise=0.5)
    assert np.allclose(covariance_matched_basis(H, 8, seed=1), covariance_matched_basis(H, 8, seed=1))


def test_procrustes_deterministic_and_leakage_free():
    H, _ = _linear_code(noise=0.3, seed=1)
    H2 = H @ orthonormal_basis(RNG.standard_normal((64, 64))).T[:64]  # rotate
    a = orthogonal_procrustes_cv(H, H, seed=0)
    b = orthogonal_procrustes_cv(H, H, seed=0)
    assert a == b, "same seed must give identical result (no randomized SVD / leakage RNG drift)"
    assert a > 0.9, "a form vs itself should align near-perfectly"


# ---- audit #7: held-out R^2 ----
def test_heldout_r2_high_for_real_structure_low_for_noise():
    H, _ = _linear_code(noise=0.05)
    ho, _ = heldout_r2(H, NUMS, k_pca=20)
    assert ho > 0.9, f"held-out R^2 should be high for a real linear code, got {ho}"
    noise = RNG.standard_normal((100, 64))
    ho_n, _ = heldout_r2(noise, NUMS, k_pca=20)
    assert ho_n < 0.5, f"held-out R^2 should be low for pure noise, got {ho_n}"


# ---- audit #1: coordinate-level identity vs span overlap ----
def test_canonical_cosines_high_for_same_code_and_detect_rotation():
    # two forms built from the SAME feature->model map D => coordinate cosines ~1
    H, D = _helix_code(noise=0.05, seed=2)
    fit_a = fit_helix(H, NUMS, k_pca=20)
    fit_b = fit_helix(_helix_code(noise=0.05, seed=2)[0] + RNG.standard_normal((100, 64)) * 0.02, NUMS, k_pca=20)
    coord = canonical_map_cosines(fit_a, fit_b)
    assert coord["mean_abs_cos"] > 0.8, f"same code => high coord identity, got {coord['mean_abs_cos']}"
    # a random-direction 'form' should NOT share coordinates (near 0)
    fit_r = fit_helix(RNG.standard_normal((100, 64)), NUMS, k_pca=20)
    assert canonical_map_cosines(fit_a, fit_r)["mean_abs_cos"] < 0.6


def test_permutation_null_below_real_alignment():
    # a real shared code aligns ABOVE the pipeline-matched permutation null
    H, D = _helix_code(noise=0.1, seed=3)
    Ha = H + RNG.standard_normal((100, 64)) * 0.02
    Hb = H + RNG.standard_normal((100, 64)) * 0.02
    real = subspace_alignment(fit_helix(Ha, NUMS)["helix_dirs_model"],
                              fit_helix(Hb, NUMS)["helix_dirs_model"])["mean_cos"]
    null = permutation_alignment_null(Ha, Hb, NUMS, n_perm=20)
    assert real > null["null_q95"], f"real alignment {real} should beat perm null q95 {null['null_q95']}"


# ---- patching primitives ----
def test_subspace_patch_preserves_orthocomplement():
    Q = random_subspace_basis(8, 64, seed=0)
    h, t = RNG.standard_normal(64), RNG.standard_normal(64)
    patched = make_patched_vector(h, t, Q=Q, mode="subspace")
    # component orthogonal to Q must be unchanged; component in Q must equal target's
    assert np.allclose((patched - Q @ (Q.T @ patched)), (h - Q @ (Q.T @ h)))
    assert np.allclose(Q @ (Q.T @ patched), Q @ (Q.T @ t))


def test_subspace_energy_zero_at_mean():
    Q = random_subspace_basis(8, 64, seed=0)
    m = RNG.standard_normal(64)
    assert subspace_energy(Q, m, m) < 1e-10


def test_fourier_basis_drops_degenerate_sin():
    B = fourier_basis(NUMS, periods=(2, 10))
    # linear + [cos2] (sin2 dropped) + [cos10, sin10] = 4 columns
    assert B.shape == (100, 4), f"expected 4 cols (sin@T=2 dropped), got {B.shape[1]}"


def test_alignment_identity_and_orthogonal():
    dirs = RNG.standard_normal((8, 64))
    same = subspace_alignment(dirs, dirs)
    assert same["mean_cos"] > 0.999
    # an orthogonal complement subspace should have low overlap
    Q = orthonormal_basis(dirs)                       # [64, 8]
    comp = np.linalg.svd(np.eye(64) - Q @ Q.T)[0][:, :8].T
    assert subspace_alignment(dirs, comp)["mean_cos"] < 0.1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)

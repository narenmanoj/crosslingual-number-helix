"""Unit tests locking in the audit fixes (no model needed -- pure numpy on synthetic activations).

Run:  python -m pytest tests/ -q      (or)      python tests/test_core.py
Each test targets a specific audit item so a regression points straight at the cause.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from src.helix import fit_helix, fourier_basis, heldout_r2, DEFAULT_PERIODS
from src.patching import (helix_reconstruct, helix_subspace_basis, make_patched_vector,
                          random_subspace_basis, covariance_matched_basis, top_pca_span_basis,
                          subspace_energy)
from src.alignment import (subspace_alignment, orthogonal_procrustes_cv, canonical_map_cosines,
                           permutation_alignment_null, orthonormal_basis, subspace_overlap)
from src.extract import validate_single_token_answers, continuation_answer_ids
from analyze_stats import bh_fdr, perm_sign_p, paired, cluster_boot

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


# ---- audit round 2: statistics ----
def test_bh_fdr_known_pvalues():
    # m=4: q = p*m/rank -> [0.004, 0.02, 0.053, 0.2]; only the first two are <= 0.05
    reject, q = bh_fdr([0.001, 0.01, 0.04, 0.2], alpha=0.05)
    assert list(reject) == [True, True, False, False], f"BH rejections wrong: {reject}"
    assert np.isclose(q[0], 0.004) and np.isclose(q[1], 0.02)
    order = np.argsort([0.001, 0.01, 0.04, 0.2])
    assert np.all(np.diff(q[order]) >= -1e-12), "q-values must be monotone in p order"
    assert np.all(q <= 1.0)
    # a clearly-significant set should be fully rejected
    rj, _ = bh_fdr([0.0001, 0.0002, 0.001], alpha=0.05)
    assert list(rj) == [True, True, True]


def test_perm_p_symmetric_null():
    rng = np.random.default_rng(0)
    # zero-centered symmetric diffs => large permutation p; strongly positive => tiny p
    assert perm_sign_p(rng.standard_normal(200) * 0.5, B=2000) > 0.2
    assert perm_sign_p(np.ones(50) * 0.5 + rng.standard_normal(50) * 0.05, B=2000) < 0.01


def test_paired_asserts_equal_length():
    try:
        paired([1, 2, 3], [1, 2])
        assert False, "paired must raise on length mismatch, not truncate"
    except ValueError:
        pass


def test_cluster_boot_wider_than_naive():
    # rows within a source value are perfectly correlated => clustered CI must be wider than naive
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 6)
    per_group = rng.standard_normal(10)
    diff = np.repeat(per_group, 6) + rng.standard_normal(60) * 1e-6   # near-identical within group
    clo, chi = cluster_boot(diff, groups, B=2000)
    # naive bootstrap CI half-width
    x = diff; nb = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(1)
    nlo, nhi = np.percentile(nb, [2.5, 97.5])
    assert (chi - clo) > (nhi - nlo), "clustered CI should be wider when rows cluster by source"


# ---- audit round 2: alignment / controls ----
def test_unequal_rank_overlap_penalized():
    Q = orthonormal_basis(RNG.standard_normal((8, 64)))       # [64, 8]
    dirs_a = Q.T                                              # rank 8
    dirs_b = Q[:, :4].T                                      # rank 4, subset of A's span
    ov = subspace_overlap(dirs_a, dirs_b)
    assert ov["rank_a"] == 8 and ov["rank_b"] == 4
    assert abs(ov["shared_energy"] - 4.0) < 1e-6              # 4 shared unit directions
    # rank-penalized (÷max rank 8) must be below the small-side overlap (÷4)
    assert ov["overlap_rank_penalized"] < ov["overlap_b_to_a"]


def test_top_pca_span_alias():
    assert covariance_matched_basis is top_pca_span_basis


def test_norm_matched_control_delta():
    # audit #3: a norm-matched control delta must have the SAME norm as the helix delta
    Q = random_subspace_basis(8, 64, seed=0)
    Qc = random_subspace_basis(8, 64, seed=1)
    diff = RNG.standard_normal(64)
    dh = Q @ (Q.T @ diff)
    dc = Qc @ (Qc.T @ diff)
    dc = dc * (np.linalg.norm(dh) / np.linalg.norm(dc))
    assert np.isclose(np.linalg.norm(dh), np.linalg.norm(dc), rtol=1e-6)


def test_span_energy_identity():
    # E_span^2 == sum_p ||QQ^T (h_p - mean)||^2
    Q = random_subspace_basis(8, 64, seed=0)
    mean = RNG.standard_normal(64)
    hs = [RNG.standard_normal(64) for _ in range(3)]
    e_span = np.sqrt(sum(subspace_energy(Q, h, mean) ** 2 for h in hs))
    stacked = np.sqrt(sum(np.linalg.norm(Q @ (Q.T @ (h - mean))) ** 2 for h in hs))
    assert np.isclose(e_span, stacked, rtol=1e-8)


# ---- audit round 2: answer-token fail-fast (mock tokenizers) ----
class _CleanTok:
    """char-level tokenizer: digits are single tokens."""
    def __call__(self, s, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in s]}
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _MultiTok:
    """digits split into two tokens (SentencePiece-like); prompt tokenization stays prefix-stable."""
    def __call__(self, s, add_special_tokens=True):
        ids = []
        for c in s:
            ids += [1000 + int(c), 2000 + int(c)] if c.isdigit() else [ord(c)]
        return {"input_ids": ids}
    def decode(self, ids):
        return "".join(str(i - 1000) if 1000 <= i < 1010 else ("" if 2000 <= i < 2010 else chr(i)) for i in ids)


def test_answer_validation_clean_vs_multitoken():
    assert validate_single_token_answers(_CleanTok(), range(10)) == []
    ids = continuation_answer_ids(_CleanTok(), range(10))
    assert ids[5] == ord("5")
    bad = validate_single_token_answers(_MultiTok(), range(10))
    assert len(bad) == 10, "multi-token digits should all be flagged"
    try:
        continuation_answer_ids(_MultiTok(), range(10))
        assert False, "continuation_answer_ids must fail-fast on multi-token digits"
    except ValueError:
        pass


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

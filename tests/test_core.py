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
from src.provenance import (require_schema, admits, stamp, git_metadata,
                            VALIDATED, LEGACY, EXPLORATORY, E_DELTA, E_ABSOLUTE)
from analyze_stats import (bh_fdr, paired, paired_by_key, cluster_boot, cluster_sign_p,
                           hier_boot, seed_stats)

RNG = np.random.default_rng(0)
NUMS = list(range(100))


def _linear_code(d=64, scale=3.0, noise=0.0, seed=0):
    """Activations carrying an exact centered linear-in-n code along one direction.
    Noise is PER-EXAMPLE (shape [n, d]); a 1-D noise vector would be a constant offset removed by
    centering, making 'noisy' tests effectively noiseless (audit r3 #9)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    n = np.array(NUMS, float)
    return np.outer((n - n.mean()) / n.std(), u) * scale + rng.standard_normal((len(NUMS), d)) * noise, u


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


def test_procrustes_deterministic_and_recovers_rotation():
    H, _ = _helix_code(noise=0.05, seed=1)
    Rq, _ = np.linalg.qr(RNG.standard_normal((64, 64)))       # a true orthogonal rotation of model space
    H_rot = H @ Rq
    a = orthogonal_procrustes_cv(H, H_rot, seed=0)
    b = orthogonal_procrustes_cv(H, H_rot, seed=0)
    assert a == b, "same seed must give identical result (no randomized SVD / leakage RNG drift)"
    # the whole point: Procrustes should RECOVER an orthogonal rotation between the two forms
    assert a > 0.9, f"Procrustes should recover the rotation (held-out R^2 high), got {a}"


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


def test_cluster_sign_p_null_and_signal():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 5)
    # zero-centered symmetric diffs => large p; strongly positive => small p
    assert cluster_sign_p(rng.standard_normal(100) * 0.5, g, B=2000) > 0.2
    assert cluster_sign_p(np.ones(100) * 0.5 + rng.standard_normal(100) * 0.05, g, B=2000) < 0.05


def test_cluster_sign_p_is_more_conservative_than_row_level():
    # audit r4 #11: with strong within-cluster correlation, flipping per cluster must NOT be more
    # significant than flipping per row (the row-level test overstates evidence).
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(6), 10)                       # 6 clusters x 10 identical-ish rows
    diff = np.repeat(rng.standard_normal(6) + 0.4, 10) + rng.standard_normal(60) * 1e-3
    p_cluster = cluster_sign_p(diff, g, B=4000)
    p_row = cluster_sign_p(diff, None, B=4000)            # groups=None => per-row flips
    assert p_cluster >= p_row, f"cluster p ({p_cluster}) must be >= row p ({p_row})"


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


def test_norm_matched_delta_uses_real_helpers():
    # audit r3 #9: exercise the ACTUAL helpers run_transport uses (not a reimplemented formula)
    from src.patching import subspace_delta, norm_match
    Q = random_subspace_basis(8, 64, seed=0)
    Qc = random_subspace_basis(8, 64, seed=1)
    diff = RNG.standard_normal(64)
    dh = subspace_delta(diff, Q)
    dc = norm_match(subspace_delta(diff, Qc), np.linalg.norm(dh))
    assert np.isclose(np.linalg.norm(dh), np.linalg.norm(dc), rtol=1e-6)
    assert np.isclose(np.linalg.norm(norm_match(np.zeros(64), 5.0)), 0.0)  # no-op on ~0


def test_norm_matched_ablation_equalizes_removed_energy():
    # audit r3 #3: the control ablation must remove the SAME energy as the helix ablation
    from src.patching import norm_matched_ablation
    Q = random_subspace_basis(8, 64, seed=0)
    Qc = random_subspace_basis(8, 64, seed=1)
    h, mean = RNG.standard_normal(64), RNG.standard_normal(64)
    helix_removed = np.linalg.norm(Q @ (Q.T @ (h - mean)))
    control_removed = np.linalg.norm(h - norm_matched_ablation(h, mean, Q_signal=Q, Q_control=Qc))
    assert np.isclose(helix_removed, control_removed, rtol=1e-6)


def test_axis_relabeling_span_vs_coordinates():
    # audit r3 #9: a true axis relabeling D_B = R D_A keeps the SPAN (principal angles ~1) but changes
    # per-feature COORDINATES (canonical cosines drop). This is the distinction the metrics must make.
    D_A = RNG.standard_normal((8, 64))
    Rq, _ = np.linalg.qr(RNG.standard_normal((8, 8)))         # orthogonal relabeling of the 8 feature axes
    D_B = Rq @ D_A
    assert subspace_alignment(D_A, D_B)["mean_cos"] > 0.999, "span (row space) is unchanged by R"
    coord = canonical_map_cosines({"helix_dirs_model": D_A}, {"helix_dirs_model": D_B})
    assert coord["mean_abs_cos"] < 0.9, "coordinate identity must DROP under axis relabeling"


# ---- audit round 4: schema enforcement, strict pairing, seed-level control stats ----
def _hdr(**kw):
    base = {"schema_version": "2.2", "experiment_type": "transport",
            "estimand": E_DELTA, "analysis_status": VALIDATED}
    base.update(kw)
    return base


def test_require_schema_rejects_each_dimension():
    ok = dict(expected_schema="2.2", expected_experiment="transport",
              allowed_estimands={E_DELTA}, allowed_statuses={VALIDATED})
    require_schema(_hdr(), **ok)                                    # admissible
    for bad in (_hdr(schema_version="2.1"),                         # stale schema
                _hdr(experiment_type="necessity"),                  # wrong experiment
                _hdr(estimand=E_ABSOLUTE),                          # unapproved estimand
                _hdr(analysis_status=EXPLORATORY),                  # unapproved status
                {}):                                                # unstamped legacy file
        try:
            require_schema(bad, **ok)
            assert False, f"require_schema must reject {bad}"
        except ValueError:
            pass
    # opt-in widening admits the legacy estimand/status
    assert admits(_hdr(estimand=E_ABSOLUTE, analysis_status=LEGACY),
                  expected_schema="2.2", expected_experiment="transport",
                  allowed_estimands={E_DELTA, E_ABSOLUTE}, allowed_statuses={VALIDATED, LEGACY})


def test_stamp_includes_provenance():
    s = stamp("2.2", "transport", estimand=E_DELTA)
    for k in ("schema_version", "experiment_type", "estimand", "analysis_status",
              "code_commit", "dirty_worktree"):
        assert k in s, f"stamp missing {k}"
    assert set(git_metadata()) == {"code_commit", "dirty_worktree"}


def test_paired_by_key_strict():
    ka = [(1, 2, 3), (4, 5, 6)]
    d, order = paired_by_key([10.0, 20.0], ka, [1.0, 2.0], ka)
    assert list(d) == [9.0, 18.0] and order == sorted(ka)
    # REORDERED second condition must still pair correctly by key (position would be wrong)
    d2, _ = paired_by_key([10.0, 20.0], ka, [2.0, 1.0], [(4, 5, 6), (1, 2, 3)])
    assert list(d2) == [9.0, 18.0]
    for bad in (([1.0, 2.0], [(1, 1, 1), (1, 1, 1)]),          # duplicate keys
                ([1.0, 2.0], [(9, 9, 9), (4, 5, 6)])):         # different case set
        try:
            paired_by_key([10.0, 20.0], ka, bad[0], bad[1])
            assert False, "paired_by_key must reject mismatched/duplicate keys"
        except ValueError:
            pass


def test_seed_stats_and_hier_boot():
    # signal beats every control seed -> P(beat)=1 and even the worst-control margin is positive
    M = np.full((20, 5), 0.8) + RNG.standard_normal((20, 5)) * 0.01
    s = seed_stats(M)
    assert s["p_beats_random_control"] == 1.0
    assert s["vs_worst_control"] > 0 and s["vs_strong_control_q90"] > 0
    lo, hi = hier_boot(M, None, B=1000)
    assert lo > 0, "hierarchical CI should exclude 0 when signal dominates every seed"
    # a signal that only beats the control MEAN has P(beat) well below 1
    M2 = np.tile(np.array([2.0, -1.0, 0.5, -0.5, 0.2]), (20, 1))
    assert seed_stats(M2)["p_beats_random_control"] < 0.8
    assert seed_stats(M2)["vs_worst_control"] < 0


def test_aggregate_uses_clean_contrasts():
    # audit r3 #8: cross-model H2 aggregation must consume the clean contrasts, not the confounded summary
    from aggregate_runs import axis_values
    vals, src = axis_values({"clean_contrasts": {"script (a)": 0.8, "notation (b)": 0.6, "language (c)": 0.3},
                             "axis_summary": {"language": {"subspace_cos": 0.9}}})
    assert src == "clean" and vals["language"] == 0.3
    vals2, src2 = axis_values({"axis_summary": {"script": {"subspace_cos": 0.9},
                                                "notation": {"subspace_cos": 0.7},
                                                "language": {"subspace_cos": 0.5}}})
    assert src2 == "confounded" and vals2["language"] == 0.5


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

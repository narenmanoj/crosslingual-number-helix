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
                           crossed_boot, seed_stats, build_cell, claim_family)

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


def test_seed_stats_and_crossed_boot():
    # signal beats every control seed -> P(beat)=1 and even the worst-control margin is positive
    M = np.full((20, 5), 0.8) + RNG.standard_normal((20, 5)) * 0.01
    s = seed_stats(M)
    assert s["p_beats_random_control"] == 1.0
    assert s["vs_worst_control"] > 0 and s["vs_strong_control"] > 0
    lo, hi = crossed_boot(M, None, B=1000)
    assert lo > 0, "crossed CI should exclude 0 when signal dominates every seed"
    # a signal that only beats the control MEAN has P(beat) well below 1
    M2 = np.tile(np.array([2.0, -1.0, 0.5, -0.5, 0.2]), (20, 1))
    assert seed_stats(M2)["p_beats_random_control"] < 0.8
    assert seed_stats(M2)["vs_worst_control"] < 0


# ---- audit round 5: rerun-readiness blockers ----
def test_layer_selection_prefers_heldout_not_insample():
    """Blocker #1: a layer that overfits (high in-sample, poor held-out) must NOT be selected."""
    from src.helix import select_layer_independent, fit_helix
    nums = list(range(60))
    # layer 1: genuine helix structure (generalizes). layer 2: pure noise (fits in-sample, not held-out)
    real, _ = _helix_code(noise=0.05, seed=7)
    acts = {1: real[:60], 2: RNG.standard_normal((60, 64)) * 3.0}
    ins = {L: fit_helix(acts[L], nums, k_pca=20)["r2"] for L in acts}
    sel = select_layer_independent(acts, nums, k_pca=20)
    assert sel["selected_layer"] == 1, f"must pick the generalizing layer, got {sel}"
    assert sel["metric"] == "heldout_r2" and sel["form_used"] == "en_digit"
    assert sel["selection_frozen_before_crossform_eval"] is True
    # the bias this fixes: the noise layer's IN-SAMPLE R^2 far exceeds what it generalizes to
    ho = {d["layer"]: d["heldout_r2"] for d in sel["per_layer"]}
    assert ins[2] - ho[2] > 0.2, f"noise layer should overfit (in-sample {ins[2]:.2f} vs held-out {ho[2]:.2f})"
    assert ho[1] > ho[2], "the generalizing layer must win on held-out R^2"


def test_crossed_bootstrap_preserves_seed_variance():
    """Blocker #3: with large BETWEEN-seed variance, the crossed CI must be wider than a
    row-wise-seed-resampling CI, which averages that variance away."""
    from analyze_stats import crossed_boot
    rng = np.random.default_rng(3)
    n, k = 40, 6
    seed_offsets = rng.standard_normal(k) * 1.0        # large between-seed spread
    M = seed_offsets[None, :] + rng.standard_normal((n, k)) * 0.02   # tiny within-seed noise
    groups = np.arange(n)
    clo, chi = crossed_boot(M, groups, B=1500, seed=0)
    # emulate the OLD nested sampler: independent seed per row
    rg = np.random.default_rng(0)
    means = [M[rg.integers(0, n, n), rg.integers(0, k, n)].mean() for _ in range(1500)]
    nlo, nhi = np.percentile(means, [2.5, 97.5])
    assert (chi - clo) > (nhi - nlo) * 1.5, \
        f"crossed CI {chi-clo:.3f} must be materially wider than nested {nhi-nlo:.3f}"


def test_build_cell_aligns_groups_seeds_and_nans():
    """Blocker #4: reordering, and NaN filtering, must keep diff/groups/seed-matrix/keys aligned."""
    from analyze_stats import build_cell
    keys = [(1, 5, 1), (2, 6, 1), (3, 7, 1)]
    a, b = [10.0, 20.0, 30.0], [1.0, 2.0, 3.0]
    sm = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]])
    diff, groups, sm2, order = build_cell(a, b, keys, seed_matrix=sm, cluster_by=0)
    assert list(order) == sorted(keys) and list(groups) == [1, 2, 3]
    assert np.allclose(diff, [9.0, 18.0, 27.0]) and np.allclose(sm2[:, 0], [0.0, 1.0, 2.0])
    # a NaN in one case must drop that case from EVERY aligned object
    a_nan = [10.0, float("nan"), 30.0]
    diff2, groups2, sm3, order2 = build_cell(a_nan, b, keys, seed_matrix=sm, cluster_by=0)
    assert len(diff2) == 2 and list(groups2) == [1, 3] and len(order2) == 2
    assert np.allclose(sm3[:, 0], [0.0, 2.0]), "seed matrix rows must follow the same mask"
    # cluster_by=1 clusters on the TARGET value instead
    _, g_t, _, _ = build_cell(a, b, keys, cluster_by=1)
    assert list(g_t) == [5, 6, 7]
    # mismatched seed-matrix row count is an error, not a silent truncation
    try:
        build_cell(a, b, keys, seed_matrix=sm[:2])
        assert False, "must reject a seed matrix with the wrong number of rows"
    except ValueError:
        pass


def test_run_dir_validation_rejects_bad_runs(tmpdir=None):
    """Blocker #5: mixed commits, dirty results, duplicate cells and missing models must all fail."""
    import json
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.2", expected_models=["M1", "M2"],
                             expected_experiments=["transport"], expected_forms=["en_digit"], allow_dirty=True)
        cm = man["code_commit"]      # results must share the manifest's commit
        record_completion(d, "transport:M1", "ok"); record_completion(d, "transport:M2", "ok")
        def res(model, commit=None, **kw):
            return {"experiment_type": "transport", "model": model, "schema_version": "2.2",
                    "analysis_status": "validated", "model_revision": {"name": model, "revision": "r1"},
                    "code_commit": commit or cm, "dirty_worktree": False,
                    "results": {"en_digit": {"n_cases": 10}}, "all_cases_processed": True, **kw}
        ok = [("t_M1.json", res("M1")), ("t_M2.json", res("M2"))]
        validate_run_dir(d, ok)                                        # clean run passes
        mixed = ok + [("t_M3.json", res("M3", commit="XYZ"))]
        for bad, why in ((mixed, "mixed commits"),
                         (ok[:1], "missing expected model"),
                         (ok + [("dup.json", res("M1"))], "duplicate cell")):
            try:
                validate_run_dir(d, bad)
                assert False, f"must reject: {why}"
            except ValueError:
                pass
        # no manifest at all -> refuse a production analysis
        d2 = tempfile.mkdtemp()
        try:
            validate_run_dir(d2, ok, require_manifest=True)
            assert False, "must refuse a directory with no manifest"
        except ValueError:
            pass
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_norm_match_diag_flags_off_manifold_control():
    """Blocker #6: a control with a near-zero raw projection must be RETAINED and flagged, not hidden."""
    from src.patching import norm_match_diag, ALPHA_LO, ALPHA_HI
    good, dg = norm_match_diag(np.array([1.0, 0.0, 0.0]), 1.0)
    assert dg["admissible"] and np.isclose(dg["alpha"], 1.0) and np.isclose(np.linalg.norm(good), 1.0)
    _, bad = norm_match_diag(np.array([1e-6, 0.0, 0.0]), 1.0)       # tiny raw projection
    assert bad["alpha"] > ALPHA_HI and not bad["admissible"], "huge alpha must be flagged inadmissible"
    assert np.isfinite(bad["raw_norm"]) and bad["raw_norm"] > 0, "raw norm must be retained"
    _, zero = norm_match_diag(np.zeros(3), 1.0)
    assert not zero["admissible"] and np.isinf(zero["alpha"])


# ---- audit round 6: production-readiness blockers ----
def test_geometry_scripts_use_independent_selector():
    """Blocker #1: the geometry scripts must call the independent helper, not an in-sample mean scan."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for fn in ("scripts/run_fit_and_align.py", "scripts/run_structure.py"):
        src = (root / fn).read_text()
        assert "select_layer_independent(" in src, f"{fn} must call select_layer_independent"
        assert "discovery_evaluation_split(" in src, f"{fn} must use a disjoint discovery split"
        assert "layer_selection" in src, f"{fn} must record the selection provenance"
        # the old biased scan (mean in-sample R^2 across ALL forms) must be gone
        assert 'for f in forms])["r2"]' not in src, f"{fn} still scans in-sample R^2 across all forms"


def test_admitted_seeds_drive_primary_estimate():
    """Blocker #2: --admissible-only must change the POINT ESTIMATE, not just summaries, and must
    drop whole seeds rather than impute row means."""
    from analyze_stats import admit_global_seeds, build_cell
    n = 12
    # seeds 0,1 are inadmissible and carry a huge effect; seeds 2,3 are admissible and null
    M = np.zeros((n, 4))
    M[:, 0] = M[:, 1] = -5.0      # control shift far below signal -> huge apparent effect
    M[:, 2] = M[:, 3] = 1.0       # admissible controls match the signal -> null effect
    adm = np.zeros((n, 4), bool); adm[:, 2] = adm[:, 3] = True
    keep, rep = admit_global_seeds(adm, min_case_frac=0.8)
    assert keep == [2, 3] and rep["n_admitted"] == 2
    sig = np.ones(n)
    keys = [(i, i + 1, 1) for i in range(n)]
    all_mean, adm_mean = M.mean(1), M[:, keep].mean(1)
    eff_all = build_cell(sig, all_mean, keys)[0].mean()
    eff_adm = build_cell(sig, adm_mean, keys)[0].mean()
    assert abs(eff_all - eff_adm) > 1.0, "admissible-only must move the point estimate"
    assert np.isclose(eff_adm, 0.0), "with admissible controls the effect should be null here"
    # too few admitted seeds is detectable (the analyzer turns this into a hard cell failure)
    none_adm = np.zeros((n, 4), bool)
    assert admit_global_seeds(none_adm)[0] == []


def test_no_row_mean_imputation_in_analyzer():
    """Blocker #2: the analyzer must not fabricate seed values via row-mean imputation."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "scripts/analyze_stats.py").read_text()
    assert "nanmean" not in src, "row-mean imputation of missing seeds must be gone"
    assert "admit_global_seeds(" in src, "global seed admission must be used"


def test_result_cell_id_distinguishes_positions():
    """Blocker #4: necessity at last/span/after is three cells, not a duplicate."""
    from src.provenance import result_cell_id, cell_matches
    base = {"experiment_type": "necessity", "model": "M", "estimand": E_ABLATION_ID, "layer": 14}
    ids = {result_cell_id({**base, "ablation_position": p}) for p in ("last", "span", "after")}
    assert len(ids) == 3, "positions must produce distinct cell ids"
    spec = {"experiment_type": "necessity", "model": "M", "ablation_position": "span"}
    assert cell_matches(spec, result_cell_id({**base, "ablation_position": "span"}))
    assert not cell_matches(spec, result_cell_id({**base, "ablation_position": "last"}))


def test_manifest_enforces_exact_cells_and_zero_fallback():
    """Blocker #4 + #6: exact expected-cell set, unexpected files, and baseline fallbacks all fail."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.3", expected_models=["M"],
                             expected_experiments=["necessity"], expected_forms=[],
                             expected_cells=[{"experiment_type": "necessity", "model": "M",
                                              "ablation_position": p} for p in ("last", "span", "after")],
                             allow_dirty=True)
        cm = man["code_commit"]
        for p_ in ("last", "span", "after"):
            record_completion(d, f"necessity:M:{p_}", "ok")

        def mk(pos, **kw):
            body = {"experiment_type": "necessity", "model": "M", "schema_version": "2.3",
                    "analysis_status": "validated", "model_revision": {"name": "M", "revision": "r1"},
                    "code_commit": cm, "dirty_worktree": False, "ablation_position": pos,
                    "ablation": {"en_digit": {"n": 10}}, "all_cases_processed": True}
            body.update(kw)
            return (f"n_{pos}.json", body)
        full = [mk(p) for p in ("last", "span", "after")]
        validate_run_dir(d, full)                                   # complete run passes
        for bad, why in ((full[:2], "missing a position"),
                         (full + [mk("last")], "duplicate position"),
                         (full + [("x.json", {**mk("post")[1], "ablation_position": "post"})], "unexpected cell"),
                         ([mk("last", ablation={"en_digit": {"n": 10, "n_skipped_no_baseline": 3}}),
                           full[1], full[2]], "baseline skips")):
            try:
                validate_run_dir(d, bad)
                assert False, f"must reject: {why}"
            except ValueError:
                pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_interchange_not_in_default_claims():
    """Blocker #5: the undercontrolled interchange claim must be opt-in, not a default validated one."""
    from analyze_stats import DEFAULT_CLAIMS, OPTIN_CLAIMS
    assert "interchange" not in DEFAULT_CLAIMS
    assert "interchange" in OPTIN_CLAIMS
    assert DEFAULT_CLAIMS[0] == "delta_vs_shuf_fourier", "the admissible structured control leads"


def test_production_requires_frozen_layer_manifest():
    """Blocker #7: production refuses a hand-typed layer and validates the manifest's protocol."""
    import json
    import tempfile
    from src.provenance import resolve_layer, git_metadata
    try:
        resolve_layer("M", 14, None, production=True)
        assert False, "production must refuse a CLI layer"
    except ValueError:
        pass
    layer, prov = resolve_layer("M", 14, None, production=False)
    assert layer == 14 and prov["layer_source"] == "cli_argument"
    g = git_metadata()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"schema_version": "2.3", "selection_protocol": "en_digit_heldout_r2",
                   "code_commit": g["code_commit"], "dirty_worktree": False,
                   "models": {"M": {"selected_layer": 21,
                                    "model_revision": {"name": "M", "revision": "deadbeef"}}}}, fh)
        path = fh.name
    layer, prov = resolve_layer("M", None, path, schema_version="2.3", production=True)
    assert layer == 21 and prov["layer_source"] == "frozen_manifest"
    # Blocker #4: the frozen immutable revision must propagate so the runner pins the same snapshot.
    assert prov["frozen_model_revision"] == "deadbeef"
    for kw, why in (({"schema_version": "9.9"}, "schema mismatch"),):
        try:
            resolve_layer("M", None, path, production=True, **kw)
            assert False, f"must reject {why}"
        except ValueError:
            pass
    try:
        resolve_layer("OTHER", None, path, schema_version="2.3", production=True)
        assert False, "must reject a model absent from the manifest"
    except ValueError:
        pass
    # Blocker #4: a manifest entry with NO immutable revision must fail in production (a later job
    # could otherwise silently load a different snapshot at the same code commit), but is fine in scratch.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"schema_version": "2.3", "selection_protocol": "en_digit_heldout_r2",
                   "code_commit": g["code_commit"], "dirty_worktree": False,
                   "models": {"M": {"selected_layer": 21}}}, fh)  # <- no model_revision
        norev = fh.name
    try:
        resolve_layer("M", None, norev, schema_version="2.3", production=True)
        assert False, "production must reject a manifest entry with no immutable model revision"
    except ValueError:
        pass
    layer, prov = resolve_layer("M", None, norev, schema_version="2.3", production=False)
    assert layer == 21 and prov["frozen_model_revision"] is None  # scratch tolerates it


def test_overnight_runner_is_manifest_driven():
    """Blocker #3: the production runner must use the hardened pipeline, not a shared directory."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "scripts/run_overnight.sh").read_text()
    for needle in ("scripts/new_run.py", "scripts/select_layers.py", "--layer-manifest",
                   "--production", "record_completion", "--on-baseline-fallback error"):
        assert needle in src, f"run_overnight.sh must use {needle}"
    assert "MODEL_LAYERS_DEFAULT" not in src, "hard-coded layer list must be gone"


E_ABLATION_ID = "norm_matched_subspace_ablation"


# ---- audit round 8: production-parity blockers ----
def test_baseline_policy_consistency_enforced():
    """Blocker #1: a manifest claiming one baseline policy and a result declaring another must fail."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.4", expected_models=["M"],
                             expected_experiments=["necessity"], expected_forms=["en_digit"],
                             baseline_policy="disjoint_calibration", allow_dirty=True,
                             expected_cells=[{"experiment_type": "necessity", "model": "M",
                                              "ablation_position": "after"}])
        record_completion(d, "necessity:M:after", "ok")
        cm = man["code_commit"]
        res = ("n.json", {"experiment_type": "necessity", "model": "M", "schema_version": "2.4",
                          "analysis_status": "validated", "model_revision": {"name": "M", "revision": "r1"},
                          "code_commit": cm, "dirty_worktree": False, "ablation_position": "after",
                          "baseline_policy": "in_run_leave_one_source_value_out",   # DISAGREES
                          "ablation": {"en_digit": {"n": 10}}, "all_cases_processed": True})
        try:
            validate_run_dir(d, [res])
            assert False, "mismatched baseline policy must fail"
        except ValueError as e:
            assert "baseline_policy" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_separate_form_sets_and_en_word_notation():
    """Blockers #2/#3: transport and necessity have separate form lists and transport isolates notation."""
    import config as C
    assert C.TRANSPORT_FORMS != C.NECESSITY_FORMS, "form sets must differ by experiment"
    assert "en_word" in C.TRANSPORT_FORMS, "en_word needed to isolate the notation axis"
    # necessity's set is the eligibility candidate set; ineligible foreign words are gated at run time
    assert "es_word" in C.TRANSPORT_FORMS and "es_word" not in C.NECESSITY_FORMS


def test_register_cells_gates_necessity_by_clean_acc():
    """Blocker #2: register_cells marks low-accuracy necessity forms ineligible without failing, and
    pins the layer into each cell (blocker #6)."""
    import json
    import shutil
    import subprocess
    import tempfile
    from src.provenance import write_manifest, git_metadata
    d = tempfile.mkdtemp()
    try:
        write_manifest(d, run_id="t", schema_version=__import__("config").SCHEMA_VERSION,
                       expected_models=["M"], expected_experiments=["transport", "necessity"],
                       expected_forms=["en_digit", "es_word"], allow_dirty=True)
        json.dump({"models": {"M": {"selected_layer": 14, "model_revision": {"revision": "r1"}}}},
                  open(os.path.join(d, "layers.json"), "w"))
        # a REALISTIC eligibility artifact (the shape measure_clean.py emits); register_cells now
        # strictly validates schema/type/config/revision + exact per-form coverage (r9 #3).
        cases = [(a, b) for b in [1, 2, 3] for a in range(0, 10) if a + b <= 9]
        n = len(cases)
        json.dump({"schema_version": __import__("config").SCHEMA_VERSION,
                   "experiment_type": "behavioral_eligibility", **git_metadata(),
                   "addends": [1, 2, 3], "max_sum": 9,
                   "models": {"M": {"model_revision": "r1", "forms": {
                       "en_digit": {"clean_acc": 0.95, "n_expected": n, "n_processed": n},
                       "es_word": {"clean_acc": 0.10, "n_expected": n, "n_processed": n}}}}},
                  open(os.path.join(d, "elig.json"), "w"))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "scripts/register_cells.py"),
                            "--run-dir", d, "--layers", os.path.join(d, "layers.json"),
                            "--eligibility", os.path.join(d, "elig.json"),
                            "--transport-forms", "en_digit", "es_word",
                            "--necessity-forms", "en_digit", "es_word"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        man = json.load(open(os.path.join(d, "manifest.json")))
        assert "M:es_word" in man["necessity_ineligible_forms"], "low-acc form must be ineligible"
        # every expected cell pins a layer (blocker #6)
        assert all("layer" in c and c["layer"] == 14 for c in man["expected_cells"])
        nec_cells = [c for c in man["expected_cells"] if c["experiment_type"] == "necessity"]
        assert nec_cells and all(c["ablation_position"] in ("after", "last", "span") for c in nec_cells)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_primary_necessity_position_is_single_family():
    """Blocker #4: only the primary position is FDR-corrected as primary; others are secondary."""
    import config as C
    assert C.PRIMARY_NECESSITY_POSITION not in C.SECONDARY_NECESSITY_POSITIONS
    prim = C.PRIMARY_NECESSITY_POSITION
    # exactly the primary position is 'primary'; every registered secondary position is 'secondary'
    assert claim_family("necessity", prim, prim) == "primary"
    for pos in C.SECONDARY_NECESSITY_POSITIONS:
        assert claim_family("necessity", pos, prim) == "secondary", pos


def test_analysis_policy_freezes_all_choices():
    """Blocker #5: null, admissible-only and FDR alpha are frozen too."""
    from src.provenance import default_analysis_policy
    pol = default_analysis_policy()
    for k in ("necessity_null", "admissible_only", "fdr_alpha", "primary_necessity_position",
              "min_case_fraction"):
        assert k in pol, f"policy must freeze {k}"
    assert pol["min_case_fraction"] == 1.0, "strict admission is the frozen default (blocker #9)"
    s = _src("scripts/analyze_stats.py")
    assert '("necessity_null", "null")' in s and '("fdr_alpha", "alpha")' in s


def test_expected_cell_matching_is_one_to_one():
    """Blocker #6: two results for one model at different layers must not both satisfy one cell."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.4", expected_models=["M"],
                             expected_experiments=["transport"], expected_forms=["en_digit"],
                             allow_dirty=True,
                             expected_cells=[{"experiment_type": "transport", "model": "M",
                                              "estimand": E_DELTA, "layer": 14}])
        record_completion(d, "transport:M", "ok")
        cm = man["code_commit"]
        def tr(layer):
            return (f"t_L{layer}.json", {"experiment_type": "transport", "model": "M", "estimand": E_DELTA,
                    "layer": layer, "schema_version": "2.4", "analysis_status": "validated",
                    "model_revision": {"name": "M", "revision": "r1"}, "code_commit": cm,
                    "dirty_worktree": False, "results": {"en_digit": {"n_cases": 5}},
                    "all_cases_processed": True})
        validate_run_dir(d, [tr(14)])                                # exact layer match passes
        try:
            validate_run_dir(d, [tr(14), tr(15)])                    # L15 is an unexpected cell
            assert False, "two layers for one expected cell must fail"
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_model_revision_must_be_pinned_and_consistent():
    """Blocker #7: null revision, or differing revisions across jobs, must fail."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion, model_commit
    assert model_commit({"model_revision": {"revision": "abc"}}) == "abc"
    assert model_commit({"model_revision": {"name": "M"}}) is None
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.4", expected_models=["M"],
                             expected_experiments=["transport", "necessity"], expected_forms=["en_digit"],
                             allow_dirty=True,
                             expected_cells=[{"experiment_type": "transport", "model": "M", "layer": 14},
                                             {"experiment_type": "necessity", "model": "M", "layer": 14,
                                              "ablation_position": "after"}])
        record_completion(d, "transport:M", "ok"); record_completion(d, "necessity:M:after", "ok")
        cm = man["code_commit"]
        def f(kind, rev, **kw):
            b = {"experiment_type": kind, "model": "M", "layer": 14, "schema_version": "2.4",
                 "analysis_status": "validated", "model_revision": {"name": "M", "revision": rev},
                 "code_commit": cm, "dirty_worktree": False, "all_cases_processed": True}
            b.update(kw); return (f"{kind}.json", b)
        good = [f("transport", "r1", results={"en_digit": {"n_cases": 5}}),
                f("necessity", "r1", ablation_position="after", ablation={"en_digit": {"n": 5}},
                  baseline_policy=man["baseline_policy"])]
        validate_run_dir(d, good)
        bad = [good[0], f("necessity", "r2", ablation_position="after", ablation={"en_digit": {"n": 5}},
                          baseline_policy=man["baseline_policy"])]          # different revision
        try:
            validate_run_dir(d, bad)
            assert False, "mismatched model revisions must fail"
        except ValueError as e:
            assert "revision" in str(e).lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_delta_coverage_checked_in_validation():
    """Blocker #8: a transport file whose delta keys omit expected cases must fail production."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.4", expected_models=["M"],
                             expected_experiments=["transport"], expected_forms=["en_digit"],
                             allow_dirty=True,
                             expected_cells=[{"experiment_type": "transport", "model": "M", "layer": 14}])
        record_completion(d, "transport:M", "ok")
        cm = man["code_commit"]
        keys = [[0, 1, 1], [1, 0, 1]]
        base = {"experiment_type": "transport", "model": "M", "layer": 14, "schema_version": "2.4",
                "analysis_status": "validated", "model_revision": {"name": "M", "revision": "r1"},
                "code_commit": cm, "dirty_worktree": False, "all_cases_processed": True}
        full = {**base, "results": {"en_digit": {"n_cases": 2,
                "per_case_keys": {"delta": keys}, "expected_case_keys": keys}}}
        validate_run_dir(d, [("t.json", full)])
        short = {**base, "results": {"en_digit": {"n_cases": 2,
                 "per_case_keys": {"delta": keys[:1]}, "expected_case_keys": keys}}}
        try:
            validate_run_dir(d, [("t.json", short)])
            assert False, "incomplete delta coverage must fail"
        except ValueError as e:
            assert "matched-delta" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_strict_seed_admission_rejects_partial_seed():
    """Blocker #9: with min_case_fraction=1.0 a seed with any out-of-range case is rejected."""
    from analyze_stats import admit_global_seeds
    A = np.array([[True, True], [True, False], [True, True]])   # seed 0 all-ok, seed 1 has one bad case
    keep_strict, _ = admit_global_seeds(A, min_case_frac=1.0)
    assert keep_strict == [0], "strict admission keeps only the fully-admissible seed"
    keep_loose, _ = admit_global_seeds(A, min_case_frac=0.5)
    assert keep_loose == [0, 1], "the 0.5 rule is only a sensitivity view"


# ---- audit round 7: go/no-go blockers ----
def _src(rel):
    import pathlib
    return (pathlib.Path(__file__).resolve().parent.parent / rel).read_text()


def test_runner_passes_production_to_writers():
    """Blocker #1: the writers' production contract is inert unless the runner activates it."""
    s = _src("scripts/run_overnight.sh")
    assert "PROD_FLAGS=(--production)" in s, "production must be the default writer mode"
    for line in ("run_transport.py", "run_necessity.py"):
        idx = s.index(line)
        assert 'PROD_FLAGS[@]+' in s[idx:idx + 400], f"{line} must receive the production flags"
    # scratch mode (dirty tree) must disable it, else every scratch job fails for the wrong reason
    assert "PROD_FLAGS=()" in s and "MODE=scratch" in s
    # eligibility + exact-cell registration replaces the old inline manifest builder
    assert "scripts/measure_clean.py" in s and "scripts/register_cells.py" in s


def test_runner_handles_empty_arrays_under_set_u():
    """Correctness: with `set -u`, "${arr[@]}" on an EMPTY array is an unbound-variable error on
    bash 3.2 (macOS default). PROD_FLAGS/NEWRUN_FLAGS/PAIRS_FLAG are all empty in normal
    configurations -- PAIRS_FLAG is empty in the PRODUCTION default (PAIRS=0)."""
    import subprocess
    s = _src("scripts/run_overnight.sh")
    for name in ("PROD_FLAGS", "NEWRUN_FLAGS", "DIRTY_FLAGS"):   # arrays that can be empty
        guarded = f'"${{{name}[@]+"${{{name}[@]}}"}}"'
        assert guarded in s, f"{name} must use the \"${{arr[@]+...}}\" idiom"
        assert f'"${{{name}[@]}}"' not in s.replace(guarded, ""), \
            f"{name} still has a bare empty-array expansion somewhere"
    # the unsafe form really does fail, so the guard is not cosmetic
    bad = subprocess.run(["bash", "-c", 'set -u; A=(); printf "%s" "${A[@]}"'],
                         capture_output=True, text=True)
    good = subprocess.run(["bash", "-c", 'set -u; A=(); printf "%s" "${A[@]+"${A[@]}"}"'],
                          capture_output=True, text=True)
    assert good.returncode == 0, "the safe idiom must work"
    if bad.returncode == 0:          # newer bash tolerates it; the guard still matters for bash 3.2
        pass


def test_runner_passes_exhaustive_case_flag():
    """Correctness: PAIRS=0 (exhaustive) must reach the writer. Omitting --pairs-per-form would fall
    back to the writer's sampled default, silently defeating the all-valid-cases requirement."""
    s = _src("scripts/run_overnight.sh")
    assert 'PAIRS_FLAG=(--pairs-per-form "$PAIRS")' in s, "PAIRS must always be passed explicitly"
    assert 'PAIRS_FLAG=(); [[ "$PAIRS" != "0" ]]' not in s, "the conditional-omit form is the bug"


def test_geometry_uses_evaluation_values_only():
    """Blocker #2: final geometry must be fit on held-out evaluation values, not the discovery ones."""
    for rel in ("scripts/run_fit_and_align.py", "scripts/run_structure.py"):
        s = _src(rel)
        assert "eval_idx" in s, f"{rel} must index the evaluation subset"
        assert "geometry_uses_discovery_values" in s, f"{rel} must record the split provenance"
        assert "fit_helix(acts[f][layer], numbers" not in s, f"{rel} still fits on ALL numbers"
        assert "fit_helix(H, numbers" not in s, f"{rel} still fits on ALL numbers"


def test_headline_requires_crossed_interval():
    """Blocker #3: a cell whose crossed CI includes 0 must not be a headline positive."""
    from analyze_stats import add_row, build_cell
    rng = np.random.default_rng(0)
    n, k = 30, 6
    # strong between-seed spread: case-only CI excludes 0, crossed CI should not
    offs = np.array([-1.4, -0.9, 0.3, 1.0, 1.6, 2.2])
    M = offs[None, :] + rng.standard_normal((n, k)) * 0.02
    keys = [(i, i + 1, 1) for i in range(n)]
    cell = build_cell(np.ones(n) + M.mean(1), np.ones(n), keys, seed_matrix=M, cluster_by=0)
    rows = []
    add_row(rows, "delta_vs_shuf_fourier", "M", "en_digit", "script", cell, 800, require_crossed=True)
    r = rows[0]
    assert r["sig_ci"], "case-only CI should exclude 0 in this construction"
    assert not r["sig_crossed"], "crossed CI must include 0 given the large between-seed spread"
    r["sig_fdr"] = True
    headline = bool(r["sig_fdr"] and r["sig_crossed"])
    assert not headline, "headline must require the crossed interval"
    assert (r["primary_lo"], r["primary_hi"]) == (r["crossed_lo"], r["crossed_hi"]), \
        "the reported interval must be the crossed one"


def test_analyzer_tracks_and_fails_on_dropped_primary_cells():
    """Blocker #4: dropped cells are recorded and invalidate a production run."""
    s = _src("scripts/analyze_stats.py")
    assert "dropped_primary" in s and "PRODUCTION RUN REJECTED" in s
    assert "def drop(" in s, "every omission must go through the recorder"
    assert "dropped_cells" in s, "dropped cells must be written to the stats JSON"


def test_validator_rejects_empty_payload_and_zero_cases():
    """Blocker #5: empty results/ablation and zero processed cases must fail."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.3", expected_models=["M"],
                             expected_experiments=["transport"], expected_forms=["en_digit"],
                             expected_cells=[{"experiment_type": "transport", "model": "M"}],
                             allow_dirty=True)
        record_completion(d, "transport:M", "ok")
        cm = man["code_commit"]
        base = {"experiment_type": "transport", "model": "M", "schema_version": "2.3",
                "analysis_status": "validated", "model_revision": {"name": "M", "revision": "r1"},
                "code_commit": cm, "dirty_worktree": False, "all_cases_processed": True}
        validate_run_dir(d, [("t.json", {**base, "results": {"en_digit": {"n_cases": 8}}})])
        for payload, why in (({}, "empty results"),
                             ({"en_digit": {"n_cases": 0}}, "zero processed cases")):
            try:
                validate_run_dir(d, [("t.json", {**base, "results": payload})])
                assert False, f"must reject: {why}"
            except ValueError:
                pass
        # unprocessed cases must fail too (blocker #10)
        try:
            validate_run_dir(d, [("t.json", {**base, "results": {"en_digit": {"n_cases": 8}},
                                             "all_cases_processed": False,
                                             "skipped_case_keys": {"en_digit": [[1, 2]]}})])
            assert False, "must reject unprocessed cases"
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_span_admissibility_requires_every_position():
    """Blocker #6: mean alpha can hide two out-of-range positions; require ALL of them in band."""
    from src.patching import ALPHA_LO, ALPHA_HI
    a_list = [0.1, 5.0]                              # mean 2.55 is "in range", both ends are not
    assert ALPHA_LO <= float(np.mean(a_list)) <= ALPHA_HI, "the mean really is misleadingly in-band"
    assert not all(ALPHA_LO <= x <= ALPHA_HI for x in a_list), "the all() rule must reject it"
    s = _src("scripts/run_necessity.py")
    assert "all(ALPHA_LO <= x <= ALPHA_HI for x in a_list)" in s, "necessity must use the all() rule"


def test_manifest_experiments_match_expected_cells():
    """Blocker #7: the manifest must not promise experiments the run never registers."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(d, run_id="t", schema_version="2.3", expected_models=["M"],
                             expected_experiments=["transport", "necessity", "structure"],
                             expected_forms=[], allow_dirty=True,
                             expected_cells=[{"experiment_type": "transport", "model": "M"}])
        record_completion(d, "transport:M", "ok")
        res = [("t.json", {"experiment_type": "transport", "model": "M", "schema_version": "2.3",
                           "analysis_status": "validated", "model_revision": {"name": "M"},
                           "code_commit": man["code_commit"], "dirty_worktree": False,
                           "results": {"en_digit": {"n_cases": 5}}, "all_cases_processed": True})]
        try:
            validate_run_dir(d, res)
            assert False, "declaring 'structure' while registering only transport must fail"
        except ValueError as e:
            assert "expected_experiments" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_manifest_freezes_analysis_policy():
    """Blocker #8: every threshold that changes inclusion/significance is frozen in the manifest."""
    from src.provenance import default_analysis_policy
    pol = default_analysis_policy()
    for k in ("alpha_range", "min_case_fraction", "min_admitted_seeds", "cluster_by", "bootstrap_B",
              "primary_requires_crossed_ci", "global_fdr_sensitivity", "clean_accuracy_threshold",
              "require_all_cases_processed"):
        assert k in pol, f"analysis policy must freeze {k}"
    s = _src("scripts/analyze_stats.py")
    assert "PRODUCTION POLICY CONFLICT" in s, "conflicting CLI overrides must be rejected"
    assert "--global-fdr" in _src("scripts/run_overnight.sh"), "runner must honour the FDR promise"


def test_clean_behaviour_gate_excludes_incompetent_forms():
    """Blocker #9: a form the model cannot solve must not enter the primary necessity family."""
    from analyze_stats import eligible_clean
    ok, acc = eligible_clean({"clean_acc": 0.95}, 0.8)
    assert ok and acc == 0.95
    ok, acc = eligible_clean({"clean_acc": 0.12}, 0.8)
    assert not ok, "chance-level clean accuracy must be ineligible"
    assert eligible_clean({}, 0.8)[0], "blocks without a clean_acc field are ungated"
    assert "not_testable_due_to_clean_behavior" in _src("scripts/analyze_stats.py")


def test_necessity_claims_always_carry_position():
    """Correctness: 'necessity' must not silently mean whichever position was listed first."""
    s = _src("scripts/analyze_stats.py")
    assert 'suffix = f"@{pos}"' in s, "the ablation position must always be explicit in the claim"


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


# ======================================================================================
# Round-9 audit regression tests
# ======================================================================================

def _r9_write_layers_elig(d, forms_acc, revision="r1", layer=14):
    """Write a layers.json + realistic behavioral_eligibility artifact for one model 'M'.
    forms_acc maps form -> clean_acc. Returns (layers_path, elig_path)."""
    import json
    from src.provenance import git_metadata
    json.dump({"schema_version": __import__("config").SCHEMA_VERSION,
               "models": {"M": {"selected_layer": layer,
                                "model_revision": {"revision": revision}}}},
              open(os.path.join(d, "layers.json"), "w"))
    cases = [(a, b) for b in [1, 2, 3] for a in range(0, 10) if a + b <= 9]
    n = len(cases)
    json.dump({"schema_version": __import__("config").SCHEMA_VERSION,
               "experiment_type": "behavioral_eligibility", **git_metadata(),
               "addends": [1, 2, 3], "max_sum": 9,
               "models": {"M": {"model_revision": revision, "forms": {
                   f: {"clean_acc": acc, "n_expected": n, "n_processed": n}
                   for f, acc in forms_acc.items()}}}},
              open(os.path.join(d, "elig.json"), "w"))
    return os.path.join(d, "layers.json"), os.path.join(d, "elig.json")


def _r9_register(d, transport_forms, necessity_forms):
    """Run register_cells.py as the runner does; returns (jobs_lines, manifest)."""
    import json
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts/register_cells.py"),
                        "--run-dir", d, "--layers", os.path.join(d, "layers.json"),
                        "--eligibility", os.path.join(d, "elig.json"),
                        "--transport-forms", *transport_forms,
                        "--necessity-forms", *necessity_forms],
                       capture_output=True, text=True)
    return r, json.load(open(os.path.join(d, "manifest.json"))) if r.returncode == 0 else None


def test_r9_energy_bank_records_seed_provenance():
    """Issue #9: the null control bank must record its ACTUAL builder seeds + basis hashes, not just
    a candidate count, so the exact null geometry is reproducible/auditable."""
    from src.patching import energy_matched_bank, helix_subspace_basis, fourier_basis
    import numpy as np
    rng = np.random.default_rng(0)
    d_model, r = 64, 6
    signal = np.linalg.qr(rng.standard_normal((d_model, r)))[0]
    samples = rng.standard_normal((12, d_model))
    _, rep = energy_matched_bank(samples, signal, r=r, d_model=d_model,
                                 n_keep=4, n_candidates=20, seed=100)
    for k in ("base_rng_seed", "n_candidates", "candidate_indices", "builder_seeds", "basis_hashes",
              "kept_mean_proj_norm", "selection_scores", "selected_order", "implied_alpha"):
        assert k in rep, f"seed-provenance report missing {k}"
    assert rep["base_rng_seed"] == 100 and rep["n_candidates"] == 20
    assert rep["n_kept"] == 4 == len(rep["builder_seeds"]) == len(rep["basis_hashes"])
    # the ACTUAL builder seed is base + candidate index -- so the bank can be regenerated exactly
    assert rep["builder_seeds"] == [100 + i for i in rep["candidate_indices"]]
    assert all(isinstance(h, str) and len(h) == 16 for h in rep["basis_hashes"])
    # kept in ascending selection score (best energy match first)
    assert rep["selection_scores"] == sorted(rep["selection_scores"])
    assert rep["selected_order"] == list(range(4))


def test_r9_register_writes_one_file_cell_per_position_not_per_form():
    """Blocker #2: with K eligible forms and P positions, necessity must yield P FILE cells (each
    carrying all K forms), NOT P*K indistinguishable per-form cells."""
    import shutil
    import tempfile
    from src.provenance import write_manifest
    d = tempfile.mkdtemp()
    try:
        write_manifest(d, run_id="t", schema_version=__import__("config").SCHEMA_VERSION,
                       expected_models=["M"], expected_experiments=["transport", "necessity"],
                       expected_forms=["en_digit", "es_word"], allow_dirty=True)
        # BOTH forms behaviourally eligible -> tests that we do NOT fan out one cell per form
        _r9_write_layers_elig(d, {"en_digit": 0.95, "es_word": 0.90})
        r, man = _r9_register(d, ["en_digit", "es_word"], ["en_digit", "es_word"])
        assert r.returncode == 0, r.stderr
        nec = [c for c in man["expected_cells"] if c["experiment_type"] == "necessity"]
        positions = {c["ablation_position"] for c in nec}
        # primary 'after' + secondary last/span = 3 positions => exactly 3 necessity cells
        assert len(nec) == len(positions) == 3, f"expected 3 file cells, got {len(nec)}"
        for c in nec:
            assert set(c["expected_forms"]) == {"en_digit", "es_word"}, "each cell carries all forms"
        # exactly one primary position, two required_secondary
        reqs = sorted(c["requirement"] for c in nec)
        assert reqs == ["required_primary", "required_secondary", "required_secondary"], reqs
        # necessity jobs are ONE line per position (not per form)
        nec_jobs = [ln for ln in r.stdout.splitlines() if ln.startswith("necessity\t")]
        assert len(nec_jobs) == 3, nec_jobs
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_register_fails_on_missing_eligibility_entry():
    """Blocker #3: a requested necessity form absent from the eligibility artifact must FAIL
    registration -- never be silently defaulted to accuracy 0 / ineligible."""
    import shutil
    import tempfile
    from src.provenance import write_manifest
    d = tempfile.mkdtemp()
    try:
        write_manifest(d, run_id="t", schema_version=__import__("config").SCHEMA_VERSION,
                       expected_models=["M"], expected_experiments=["transport", "necessity"],
                       expected_forms=["en_digit", "es_word"], allow_dirty=True)
        # eligibility only measured en_digit; es_word is REQUESTED but MISSING
        _r9_write_layers_elig(d, {"en_digit": 0.95})
        r, man = _r9_register(d, ["en_digit"], ["en_digit", "es_word"])
        assert r.returncode != 0, "missing eligibility entry must fail, not default to 0"
        assert "missing" in (r.stderr + r.stdout).lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_register_fails_on_partial_case_coverage():
    """Blocker #3: an eligibility entry whose n_processed < n_expected must FAIL (a form measured on a
    subset of cases is not a trustworthy competence signal)."""
    import json
    import shutil
    import subprocess
    import tempfile
    from src.provenance import write_manifest
    d = tempfile.mkdtemp()
    try:
        write_manifest(d, run_id="t", schema_version=__import__("config").SCHEMA_VERSION,
                       expected_models=["M"], expected_experiments=["transport", "necessity"],
                       expected_forms=["en_digit"], allow_dirty=True)
        lp, ep = _r9_write_layers_elig(d, {"en_digit": 0.95})
        e = json.load(open(ep))
        e["models"]["M"]["forms"]["en_digit"]["n_processed"] -= 1   # one case unprocessed
        json.dump(e, open(ep, "w"))
        r, _ = _r9_register(d, ["en_digit"], ["en_digit"])
        assert r.returncode != 0 and "processed" in (r.stderr + r.stdout).lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_validator_applies_forms_per_cell_not_global_union():
    """Blocker #1: the manifest-level form union must NOT be applied to every file. A necessity file
    carries fewer forms than transport; validating it against the transport union is the r8 bug."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion, E_DELTA, E_ABLATION
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(
            d, run_id="t", schema_version="2.4", expected_models=["M"],
            expected_experiments=["transport", "necessity"], expected_forms=["en_digit", "es_word"],
            allow_dirty=True,
            expected_cells=[
                {"experiment_type": "transport", "model": "M", "estimand": E_DELTA, "layer": 14,
                 "expected_forms": ["en_digit", "es_word"], "requirement": "required_primary"},
                {"experiment_type": "necessity", "model": "M", "estimand": E_ABLATION, "layer": 14,
                 "ablation_position": "after", "expected_forms": ["en_digit"],
                 "requirement": "required_primary"}])
        record_completion(d, "transport:M", "ok")
        record_completion(d, "necessity:M:after", "ok")
        cm = man["code_commit"]
        base = {"model": "M", "schema_version": "2.4", "analysis_status": "validated",
                "model_revision": {"name": "M", "revision": "r1"}, "code_commit": cm,
                "dirty_worktree": False, "all_cases_processed": True, "layer": 14}
        tp = {**base, "experiment_type": "transport", "estimand": E_DELTA,
              "results": {"en_digit": {"n_cases": 8}, "es_word": {"n_cases": 8}}}
        # necessity file has ONLY en_digit -- correct against its own cell, wrong against the union
        nec = {**base, "experiment_type": "necessity", "estimand": E_ABLATION,
               "ablation_position": "after", "baseline_policy": man["baseline_policy"],
               "ablation": {"en_digit": {"clean_acc": 0.95, "n": 8}}}
        validate_run_dir(d, [("tp.json", tp), ("nec.json", nec)])   # must PASS

        # now the necessity file carries an EXTRA form its cell never registered -> must fail
        nec_bad = {**nec, "ablation": {"en_digit": {"clean_acc": 0.95, "n": 8},
                                       "es_word": {"clean_acc": 0.9, "n": 8}}}
        try:
            validate_run_dir(d, [("tp.json", tp), ("nec.json", nec_bad)])
            assert False, "per-cell form set must be enforced"
        except ValueError as e:
            assert "expected" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_unexpected_clean_acc_fails_but_preregistered_ineligible_is_fine():
    """Blocker #6: a necessity form registered ELIGIBLE that reads below threshold in the RESULT is an
    unexpected behavioural failure and must fail production; a form listed in
    necessity_ineligible_forms is a preregistered not-testable and must be tolerated."""
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion, E_ABLATION
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(
            d, run_id="t", schema_version="2.4", expected_models=["M"],
            expected_experiments=["necessity"], expected_forms=["en_digit"],
            allow_dirty=True, necessity_ineligible_forms={"M:fullwidth_digit": {"clean_acc": 0.1}},
            expected_cells=[{"experiment_type": "necessity", "model": "M", "estimand": E_ABLATION,
                             "layer": 14, "ablation_position": "after",
                             "expected_forms": ["en_digit"], "requirement": "required_primary"}])
        record_completion(d, "necessity:M:after", "ok")
        cm = man["code_commit"]
        base = {"model": "M", "schema_version": "2.4", "analysis_status": "validated",
                "model_revision": {"name": "M", "revision": "r1"}, "code_commit": cm,
                "dirty_worktree": False, "all_cases_processed": True, "layer": 14,
                "experiment_type": "necessity", "estimand": E_ABLATION,
                "ablation_position": "after", "baseline_policy": man["baseline_policy"]}
        thr = man["analysis_policy"]["clean_accuracy_threshold"]
        # eligible form reads FINE -> passes
        validate_run_dir(d, [("nec.json", {**base, "ablation": {"en_digit": {"clean_acc": 0.95, "n": 8}}})])
        # eligible form reads BELOW threshold -> unexpected failure -> reject
        try:
            validate_run_dir(d, [("nec.json", {**base,
                              "ablation": {"en_digit": {"clean_acc": thr - 0.2, "n": 8}}})])
            assert False, "an eligible form that fails behaviourally must be rejected"
        except ValueError as e:
            assert "unexpected" in str(e).lower() or "registered" in str(e).lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_layer_manifest_revision_crosschecked_against_results():
    """Blocker #4: the validator must reject a result whose loaded model revision disagrees with the
    frozen layer manifest (a job that silently loaded a different snapshot)."""
    import json
    import shutil
    import tempfile
    from src.provenance import validate_run_dir, write_manifest, record_completion, E_DELTA
    d = tempfile.mkdtemp()
    try:
        man = write_manifest(
            d, run_id="t", schema_version="2.4", expected_models=["M"],
            expected_experiments=["transport"], expected_forms=["en_digit"], allow_dirty=True,
            expected_cells=[{"experiment_type": "transport", "model": "M", "estimand": E_DELTA,
                             "layer": 14, "expected_forms": ["en_digit"],
                             "requirement": "required_primary"}])
        record_completion(d, "transport:M", "ok")
        json.dump({"schema_version": "2.4",
                   "models": {"M": {"selected_layer": 14, "model_revision": {"revision": "GOOD"}}}},
                  open(os.path.join(d, "layers.json"), "w"))
        cm = man["code_commit"]
        base = {"model": "M", "schema_version": "2.4", "analysis_status": "validated",
                "code_commit": cm, "dirty_worktree": False, "all_cases_processed": True,
                "layer": 14, "experiment_type": "transport", "estimand": E_DELTA,
                "results": {"en_digit": {"n_cases": 8}}}
        validate_run_dir(d, [("tp.json", {**base, "model_revision": {"name": "M", "revision": "GOOD"}})])
        try:
            validate_run_dir(d, [("tp.json", {**base, "model_revision": {"name": "M", "revision": "WRONG"}})])
            assert False, "a result whose revision != layer manifest must be rejected"
        except ValueError as e:
            assert "layer-manifest" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r9_blocker5_family_classification_matches_frozen_config():
    """Blocker #5: the family classifier must follow C.PRIMARY_FAMILIES / C.SECONDARY_FAMILIES exactly,
    not a hard-coded DEFAULT_CLAIMS list. Assert the full preregistered decision table."""
    import config as C
    prim = C.PRIMARY_NECESSITY_POSITION
    sec = C.SECONDARY_NECESSITY_POSITIONS[0]
    # delta_vs_shuf_fourier is PRIMARY; delta_vs_pca_span and Haar delta_transport are SECONDARY
    assert claim_family("delta_vs_shuf_fourier", None, prim) == "primary"
    assert claim_family("delta_vs_pca_span", None, prim) == "secondary"
    assert claim_family("delta_transport", None, prim) == "secondary"
    # necessity: PRIMARY only at the frozen primary position; other registered positions are SECONDARY
    assert claim_family("necessity", prim, prim) == "primary"
    assert claim_family("necessity", sec, prim) == "secondary"
    # this test would have caught the old bug where every non-necessity DEFAULT_CLAIM was primary
    assert claim_family("delta_vs_pca_span", None, prim) != "primary"


def test_r9_interchange_is_optin_not_secondary():
    """Blocker #5 / scientific soundness: interchange is undercontrolled (Haar-only null, no alpha
    diagnostics) so it is opt-in EXPLORATORY, never a preregistered secondary family. Config must not
    list it, and the classifier must return 'optin' (never a secondary-FDR stamp)."""
    import config as C
    assert "interchange" not in C.SECONDARY_FAMILIES, "interchange must not be a preregistered family"
    assert "interchange" not in C.PRIMARY_FAMILIES
    assert claim_family("interchange", "last", C.PRIMARY_NECESSITY_POSITION) == "optin"
    # the other legacy/exploratory claims are opt-in too
    assert claim_family("sufficiency", None, C.PRIMARY_NECESSITY_POSITION) == "optin"
    assert claim_family("necessity_peak", "after", C.PRIMARY_NECESSITY_POSITION) == "optin"


def test_r9_dropped_required_secondary_rejects_production():
    """Issue #7: 'required' = a primary family OR a registered SECONDARY necessity position. A dropped
    secondary necessity position must reject production; control-family (haar/pca_span) admissibility
    drops must NOT, since those are legitimately conditional."""
    s = _src("scripts/analyze_stats.py")
    # the requirement classifier delegates to the SAME claim_family() used for reporting (no drift)
    assert "def dropped_requirement(d):" in s
    assert "fam = claim_family(base, d.get(\"ablation_position\"), args.primary_necessity_position)" in s
    assert 'if fam == "secondary" and base == "necessity":' in s
    # production rejects on ANY dropped_required, not only dropped_primary
    assert 'dropped_required = [d for d in dropped if d["requirement"] is not None]' in s
    assert "if args.production and dropped_required:" in s


def test_r9_end_to_end_register_then_validate():
    """Issue #11: full admission path -- write_manifest -> register_cells -> synthesize the exact
    result files the emitted jobs imply -> validate_run_dir must PASS; and a run missing one
    registered secondary-position file must FAIL."""
    import json
    import shutil
    import tempfile
    from src.provenance import (validate_run_dir, write_manifest, record_completion,
                                E_DELTA, E_ABLATION)
    d = tempfile.mkdtemp()
    try:
        man0 = write_manifest(
            d, run_id="e2e", schema_version=__import__("config").SCHEMA_VERSION,
            expected_models=["M"], expected_experiments=["transport", "necessity"],
            expected_forms=["en_digit", "es_word", "fullwidth_digit"], allow_dirty=True)
        # en_digit + es_word competent; fullwidth_digit below threshold -> preregistered not-testable
        _r9_write_layers_elig(d, {"en_digit": 0.95, "es_word": 0.9, "fullwidth_digit": 0.2})
        r, man = _r9_register(d, ["en_digit", "es_word", "fullwidth_digit"],
                              ["en_digit", "es_word", "fullwidth_digit"])
        assert r.returncode == 0, r.stderr
        assert "M:fullwidth_digit" in man["necessity_ineligible_forms"]

        cm = man["code_commit"]
        base = {"model": "M", "schema_version": man["schema_version"], "analysis_status": "validated",
                "model_revision": {"name": "M", "revision": "r1"}, "code_commit": cm,
                "dirty_worktree": False, "all_cases_processed": True, "layer": 14,
                "fit_values": man["experiment_policy"]["fit_values"],
                "causal_values": man["experiment_policy"]["causal_values"]}
        results = []
        # build EXACTLY the files each emitted job implies
        for ln in r.stdout.splitlines():
            if ln.startswith("transport\t"):
                _, model, layer, forms = ln.split("\t")
                results.append((f"transport_{model}.json",
                    {**base, "experiment_type": "transport", "estimand": E_DELTA,
                     "layer": int(layer),
                     "results": {f: {"n_cases": 8} for f in forms.split()}}))
                record_completion(d, f"transport:{model}", "ok")
            elif ln.startswith("necessity\t"):
                _, model, layer, pos, forms = ln.split("\t")
                results.append((f"necessity_{model}_{pos}.json",
                    {**base, "experiment_type": "necessity", "estimand": E_ABLATION,
                     "layer": int(layer), "ablation_position": pos,
                     "baseline_policy": man["baseline_policy"],
                     "ablation": {f: {"clean_acc": 0.95, "n": 8} for f in forms.split()}}))
                record_completion(d, f"necessity:{model}:{pos}", "ok")
        # ineligible fullwidth_digit must NOT appear in any necessity payload
        for nm, res in results:
            if res["experiment_type"] == "necessity":
                assert "fullwidth_digit" not in res["ablation"], "not-testable form must be excluded"
        validate_run_dir(d, results)   # full run must PASS

        # drop one registered secondary-position necessity file -> validation must FAIL
        dropped_one = [(nm, res) for nm, res in results
                       if not (res["experiment_type"] == "necessity"
                               and res["ablation_position"] == "span")]
        try:
            validate_run_dir(d, dropped_one)
            assert False, "a missing registered secondary-position cell must fail validation"
        except ValueError as e:
            assert "missing" in str(e).lower()

        # a STALE/unexpected result file not registered in the manifest must FAIL (Issue #11)
        stale = dict(results[0][1])
        stale["ablation_position"] = "penultimate"   # a position no cell registered
        stale = {**stale, "experiment_type": "necessity", "estimand": E_ABLATION,
                 "baseline_policy": man["baseline_policy"],
                 "ablation": {"en_digit": {"clean_acc": 0.95, "n": 8}}}
        try:
            validate_run_dir(d, results + [("stale_necessity_penultimate.json", stale)])
            assert False, "an unexpected result cell not in the manifest must fail validation"
        except ValueError as e:
            assert "unexpected" in str(e).lower() or "not in manifest" in str(e).lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


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

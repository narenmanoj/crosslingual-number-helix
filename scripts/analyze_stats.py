#!/usr/bin/env python
"""Fail-fast, estimand-aware statistics over the per-case data in experiments/*.json.

The analysis boundary is where validated, legacy, and exploratory results used to blur together.
This script now REFUSES files that do not declare the expected schema / experiment_type / estimand /
analysis_status (audit r4 #1), and only admits legacy or exploratory estimands behind explicit flags.

Default claim family (all `analysis_status: validated`):
  - delta_vs_shuf_fourier  : delta vs a shuffled-pipeline control   [PRIMARY -- admissible, alpha~1]
  - delta_vs_pca_span      : delta vs a top-PCA-span control        [admissible, alpha~2]
  - delta_transport        : delta vs a norm-matched Haar control   [usually DROPPED: a random 8-d
    subspace in ~1500-d needs alpha~8-10 to norm-match, so it is an extrapolation, not a control.
    --admissible-only (default) removes it; --all-controls restores it as a sensitivity view.]
  - interchange            : delta interchange  −  norm-matched control
  - necessity              : structured-null acc  −  helix-ablate acc (norm-matched, @ ablation layer)

Opt-in only:
  --include-legacy-absolute-patching -> adds `sufficiency` (legacy subspace−random absolute patching)
  --include-exploratory-sweeps       -> adds `necessity_peak` (confounded layer-sweep vulnerability)
Opt-in claims are excluded from the default FDR family, headline counts, and figures.

Inference: paired differences are matched strictly BY CASE KEY (never by position). CIs are cluster
bootstraps over source value; the permutation test flips signs at the CLUSTER level so the p-value and
the CI assume the same independent unit; BH-FDR is applied to that clustered p.

Usage:  python scripts/analyze_stats.py
        python scripts/analyze_stats.py --include-exploratory-sweeps --b 20000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.patching import ALPHA_LO, ALPHA_HI
from src.provenance import (require_schema, validate_run_dir, VALIDATED, LEGACY, EXPLORATORY,
                            E_DELTA, E_ABSOLUTE, E_ABLATION, E_LAYER_VULN)

AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}
DEFAULT_CLAIMS = ["delta_vs_shuf_fourier", "delta_vs_pca_span", "delta_transport", "necessity"]
# BLOCKER 5: interchange uses only Haar controls, no energy-matched bank and no alpha diagnostics, so
# its null design is weaker than delta transport's. Delta transport already provides the sufficiency
# test, so interchange is demoted to opt-in exploratory rather than shipped as a validated claim.
OPTIN_CLAIMS = {"sufficiency": "legacy absolute patching", "necessity_peak": "exploratory layer sweep",
                "interchange": "undercontrolled null (Haar only, no alpha diagnostics)"}
CLAIM_ORDER = ["delta_vs_shuf_fourier", "delta_vs_pca_span", "delta_transport",
               "necessity", "interchange", "necessity_peak", "sufficiency"]


# ----------------------------- inference primitives -----------------------------

def cluster_boot(diff, groups, B, seed=0, alpha=0.05):
    """Cluster bootstrap over group ids (rows sharing a source value are not independent)."""
    diff = np.asarray(diff, float)
    if groups is None:
        idx = np.arange(len(diff))
        rng = np.random.default_rng(seed)
        means = diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(1)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return (float("nan"), float("nan"))
        by_g = {g: np.where(groups == g)[0] for g in uniq}
        rng = np.random.default_rng(seed)
        means = np.empty(B)
        for j in range(B):
            pick = uniq[rng.integers(0, len(uniq), size=len(uniq))]
            means[j] = diff[np.concatenate([by_g[g] for g in pick])].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cluster_sign_p(diff, groups, B=10000, seed=0):
    """Two-sided sign-flip permutation p with signs flipped PER CLUSTER (audit r4 #11), so the test
    and the clustered CI assume the same independent unit. Add-one estimator."""
    diff = np.asarray(diff, float)
    n = len(diff)
    if n < 2:
        return float("nan")
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    obs = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(B, len(uniq)))     # one sign per cluster
    null = np.abs((signs[:, inv] * diff).mean(1))
    return float((1 + int(np.sum(null >= obs))) / (B + 1))


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg: (reject, q-values) aligned to input order."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    q = np.full_like(p, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return reject, q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    qr = p[order] * m / np.arange(1, m + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    q[order] = np.clip(qr, 0, 1)
    reject[order] = q[order] <= alpha
    return reject, q


def crossed_boot(seed_diff, groups, B, seed=0, alpha=0.05):
    """CROSSED bootstrap over case clusters x GLOBAL control seeds (audit r5 blocker #3).

    A control seed is one global subspace reused across every case, so the dependence is *crossed*,
    not nested. Sampling an independent seed per row (the old hierarchical version) invents an
    experiment where each case had its own control basis, which averages seed variance away and makes
    the CI too narrow. Here we resample case clusters AND resample the seed set, applying the sampled
    seeds to all sampled rows -- so between-seed variance survives."""
    M = np.asarray(seed_diff, float)                     # [n_cases, n_seeds]
    n, k = M.shape
    rng = np.random.default_rng(seed)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    means = np.empty(B)
    for j in range(B):
        rows = np.concatenate([by_g[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        seeds = rng.choice(k, size=k, replace=True)      # ONE seed set applied to ALL sampled rows
        means[j] = M[np.ix_(rows, seeds)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def seed_stats(seed_diff):
    """Control-distribution summaries: how the signal fares against the WHOLE control set, not just
    its mean (audit r4 #4). seed_diff[i][j] is oriented so positive supports the claim."""
    M = np.asarray(seed_diff, float)
    # NB: diff = signal - control, so the 10th percentile of diffs is the comparison against a
    # STRONG (90th-percentile) control. Named for what it measures, not the percentile index.
    return {"p_beats_random_control": float(np.mean(M > 0)),          # P(signal beats a random control draw)
            "vs_mean_control": float(M.mean(1).mean()),
            "vs_strong_control": float(np.mean(np.percentile(M, 10, axis=1))),
            "vs_worst_control": float(np.mean(M.min(axis=1)))}


def paired_by_key(values_a, keys_a, values_b, keys_b):
    """Strict paired differences matched BY CASE KEY (audit r4 #10). Equal length is not enough:
    filtering/sorting/duplication can silently mispair equal-length arrays."""
    ka = [tuple(k) for k in keys_a]
    kb = [tuple(k) for k in keys_b]
    if len(set(ka)) != len(ka):
        raise ValueError("duplicate case keys in condition A")
    if len(set(kb)) != len(kb):
        raise ValueError("duplicate case keys in condition B")
    if set(ka) != set(kb):
        raise ValueError(f"conditions contain different cases ({len(set(ka) ^ set(kb))} mismatched)")
    a, b = dict(zip(ka, values_a)), dict(zip(kb, values_b))
    ordered = sorted(a)
    return np.array([a[k] - b[k] for k in ordered], float), ordered


def paired(a, b):
    """Positional fallback when no keys exist; asserts equal length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError(f"paired arrays misaligned: len {len(a)} != {len(b)}")
    return a - b


# ----------------------------- row construction -----------------------------

def admit_global_seeds(admissible_matrix, min_case_frac=0.8, min_seeds=1):
    """Decide admissibility at the GLOBAL CONTROL-SEED level (audit r6 blocker #2).

    One seed == one global subspace reused across every case, so admissibility is a property of the
    SEED, not of individual cells. A seed is admitted when its norm-match scale sits inside the
    predefined band for at least `min_case_frac` of cases. Returns (kept_seed_indices, report).

    This replaces per-cell masking + row-mean imputation, which manufactured a rectangular matrix,
    shrank control-seed variance, and let cases with one surviving seed masquerade as fully sampled."""
    A = np.asarray(admissible_matrix, bool)          # [n_cases, n_seeds]
    if A.ndim != 2 or A.size == 0:
        return [], {"n_seeds": 0, "n_admitted": 0, "per_seed_case_frac": []}
    frac = A.mean(axis=0)                            # fraction of cases admissible, per seed
    keep = [j for j in range(A.shape[1]) if frac[j] >= min_case_frac]
    return keep, {"n_seeds": int(A.shape[1]), "n_admitted": len(keep),
                  "min_case_frac": min_case_frac,
                  "per_seed_case_frac": [float(x) for x in frac]}


def build_cell(values_a, values_b, keys, seed_matrix=None, cluster_by=0):
    """Assemble one analysis cell with EVERYTHING aligned by case key (audit r5 blocker #4).

    Pairs strictly by key (rejecting reordering / duplicates / differing case sets), then derives the
    cluster labels and the control seed-matrix rows FROM THE SAME sorted key order, and finally
    applies ONE validity mask to diff + groups + seed_matrix + keys together -- so NaN filtering can
    never silently misalign the pieces.

    cluster_by indexes the case key: 0=source value, 1=target value, 2=addend.
    Returns (diff, groups, seed_matrix, keys) or None if too few usable cases."""
    keys = [tuple(k) for k in keys]
    diff, order = paired_by_key(values_a, keys, values_b, keys)
    pos = {k: i for i, k in enumerate(keys)}
    if any(len(k) <= cluster_by for k in order):
        raise ValueError(f"--cluster-by {cluster_by} exceeds the case-key length "
                         f"{min(len(k) for k in order)}; pick a valid key index "
                         "(0=source, 1=target/addend depending on experiment)")
    groups = np.array([k[cluster_by] for k in order])
    sm = None
    if seed_matrix is not None:
        sm_full = np.asarray(seed_matrix, float)
        if len(sm_full) != len(keys):
            raise ValueError(f"seed matrix has {len(sm_full)} rows but there are {len(keys)} cases")
        sm = sm_full[[pos[k] for k in order]]           # reorder rows to the paired key order
    valid = np.isfinite(diff)
    if sm is not None:
        valid &= np.isfinite(sm).all(axis=1)
    diff, groups = diff[valid], groups[valid]
    order = [k for k, keep in zip(order, valid) if keep]
    if sm is not None:
        sm = sm[valid]
    return (diff, groups, sm, order) if len(diff) >= 2 else None


def add_row(rows, claim, tag, form, axis, cell, B, status=VALIDATED, extra=None,
            require_crossed=True):
    """cell = (diff, groups, seed_matrix|None, keys) from build_cell().

    HEADLINE GATE (audit r7 blocker #3): for any comparison against a random/shuffled control BANK,
    which global control subspaces happened to be drawn is part of the scientific uncertainty. So a
    cell counts as a positive result only if BOTH the case-clustered interval and the CROSSED
    (case x global-seed) interval exclude zero. `sig_ci` alone is a conditional-on-this-bank
    diagnostic and is never the headline."""
    diff, groups, seed_diff, keys = cell
    lo, hi = cluster_boot(diff, groups, B)
    row = {"claim": claim, "model": tag, "form": form, "axis": axis, "status": status,
           "effect": float(diff.mean()), "lo": lo, "hi": hi, "n": int(len(diff)),
           "n_clusters": int(len(np.unique(groups))),
           "perm_p": cluster_sign_p(diff, groups, B),
           "clustered": True,
           "sig_ci": bool(lo > 0 or hi < 0)}
    if seed_diff is not None:
        row.update(seed_stats(seed_diff))
        clo, chi = crossed_boot(seed_diff, groups, B)
        row["crossed_lo"], row["crossed_hi"] = clo, chi
        row["sig_crossed"] = bool(clo > 0 or chi < 0)
        # primary interval shown in tables/figures is the crossed one when it exists
        row["primary_lo"], row["primary_hi"] = (clo, chi) if require_crossed else (lo, hi)
        row["sig_primary_interval"] = row["sig_crossed"] if require_crossed else row["sig_ci"]
    else:
        row["primary_lo"], row["primary_hi"] = lo, hi
        row["sig_crossed"] = None
        row["sig_primary_interval"] = row["sig_ci"]
    if extra:
        row.update(extra)
    rows.append(row)


def axis_of(form):
    if form == "en_word":
        return "notation"
    return "language" if form.endswith("_word") else "script"


def want(tag, models):
    return models is None or any(m in tag for m in models)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--b", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--null", default="shuf_fourier", choices=["shuf_fourier", "cov_matched", "random"])
    p.add_argument("--schema", default=C.SCHEMA_VERSION, help="required schema_version (fail-fast)")
    p.add_argument("--include-legacy-absolute-patching", dest="legacy", action="store_true", default=False)
    p.add_argument("--include-exploratory-sweeps", dest="exploratory", action="store_true", default=False)
    p.add_argument("--strict", dest="strict", action="store_true", default=True,
                   help="raise on an inadmissible file (default); --no-strict skips it with a warning")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.add_argument("--production", action="store_true", default=False,
                   help="validate the whole run directory against its manifest: one commit, no dirty "
                        "results, no duplicate cells, no missing expected models (audit r5 #5)")
    p.add_argument("--admissible-only", dest="admissible_only", action="store_true", default=True,
                   help="PRIMARY: keep only control seeds inside the predefined alpha band (audit r5 #6)")
    p.add_argument("--all-controls", dest="admissible_only", action="store_false",
                   help="sensitivity analysis: include controls that needed extreme norm matching")
    p.add_argument("--cluster-by", type=int, default=0, choices=[0, 1, 2],
                   help="case-key index to cluster on: 0=source value (default), 1=target, 2=addend "
                        "-- rerun with each for the dependence sensitivity analysis (audit r5 #8)")
    p.add_argument("--positions", nargs="*", default=["span", "last", "after"],
                   help="necessity ablation positions to include, read from JSON metadata (audit r5 #10)")
    p.add_argument("--include-interchange", dest="interchange", action="store_true", default=False,
                   help="admit the UNDERCONTROLLED delta-interchange claim (Haar-only null; audit r6 #5)")
    p.add_argument("--min-admitted-seeds", type=int, default=3,
                   help="fail a cell if fewer control seeds survive global admissibility (audit r6 #2)")
    p.add_argument("--min-case-frac", type=float, default=0.8,
                   help="a seed is admitted when alpha is in-band for >= this fraction of cases")
    p.add_argument("--clean-acc-threshold", type=float, default=0.8,
                   help="necessity eligibility: forms below this clean accuracy are not testable (r7 #9)")
    p.add_argument("--require-crossed", dest="require_crossed", action="store_true", default=True,
                   help="headline positives require BOTH the clustered and CROSSED intervals (r7 #3)")
    p.add_argument("--no-require-crossed", dest="require_crossed", action="store_false")
    p.add_argument("--global-fdr", action="store_true", default=False,
                   help="also report a single BH correction across ALL primary cells (audit r5 #9)")
    return p.parse_args()


def load_admissible(path, args, *, experiment, estimands, statuses):
    """Read a result file only if it passes the schema/estimand/status gate."""
    d = json.load(open(path))
    try:
        require_schema(d, expected_schema=args.schema, expected_experiment=experiment,
                       allowed_estimands=estimands, allowed_statuses=statuses,
                       source=os.path.basename(path))
    except ValueError as e:
        if args.strict:
            raise SystemExit(f"\nREFUSED: {e}\n\nRegenerate this result, or re-run with the matching "
                             f"--include-* flag / --no-strict to skip it.\n")
        print(f"  skip (inadmissible): {e}")
        return None
    return d


def eligible_clean(block, threshold):
    """BLOCKER 9: a necessity result is meaningless when the model cannot do the clean task in that
    form -- a drop from chance to chance proves nothing. Returns (ok, clean_acc)."""
    acc = block.get("clean_acc")
    if acc is None:
        return True, None                      # transport-style blocks carry no gate
    return (float(acc) >= threshold), float(acc)


def main():
    args = parse_args()
    rows, skipped, loaded, dropped = [], [], [], []

    # BLOCKER 8: in production the manifest FREEZES the analysis policy. CLI values that disagree are
    # rejected rather than silently winning, so a report cannot be re-derived with friendlier settings.
    policy = None
    mpath = os.path.join(args.out_dir, "manifest.json")
    if args.production and os.path.exists(mpath):
        policy = (json.load(open(mpath)) or {}).get("analysis_policy")
    if policy:
        conflicts = []
        for key, attr in (("min_admitted_seeds", "min_admitted_seeds"),
                          ("min_case_fraction", "min_case_frac"),
                          ("cluster_by", "cluster_by"),
                          ("bootstrap_B", "b"),
                          ("clean_accuracy_threshold", "clean_acc_threshold"),
                          ("primary_requires_crossed_ci", "require_crossed")):
            frozen, got = policy.get(key), getattr(args, attr)   # NB: not `want` -- that shadows want()
            if frozen is None:
                continue
            explicit = any(a.startswith("--" + attr.replace("_", "-")) for a in sys.argv[1:])
            if explicit and frozen != got:
                conflicts.append(f"--{attr.replace('_', '-')}={got} conflicts with frozen {key}={frozen}")
            setattr(args, attr, frozen)
        if list(policy.get("alpha_range", [])) not in ([], [ALPHA_LO, ALPHA_HI]):
            conflicts.append(f"frozen alpha_range {policy['alpha_range']} != code ({ALPHA_LO}, {ALPHA_HI})")
        if conflicts:
            raise SystemExit("\nPRODUCTION POLICY CONFLICT:\n  " + "\n  ".join(conflicts) + "\n")
        if policy.get("global_fdr_sensitivity"):
            args.global_fdr = True
        print(f"analysis policy (frozen by manifest): min_admitted_seeds={args.min_admitted_seeds}, "
              f"cluster_by={args.cluster_by}, B={args.b}, clean_acc>={args.clean_acc_threshold}, "
              f"crossed_required={args.require_crossed}")

    def drop(claim, tag, form, reason, **kw):
        """BLOCKER 4: every omitted cell is recorded with a reason so production can fail on it
        instead of quietly shrinking the significance table."""
        dropped.append({"claim": claim, "model": tag, "form": form, "reason": reason, **kw})
        print(f"  DROP {claim} {tag}:{form} -- {reason}")

    # ---------------- transport: delta (primary) + optional legacy absolute ----------------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "transport_*_L*.json"))):
        tag = os.path.basename(f)[len("transport_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = load_admissible(f, args, experiment="transport",
                            estimands={E_DELTA} | ({E_ABSOLUTE} if args.legacy else set()),
                            statuses={VALIDATED} | ({LEGACY} if args.legacy else set()))
        if d is None:
            skipped.append(f); continue
        loaded.append((os.path.basename(f), d))
        for form, R in d.get("results", {}).items():
            pc, ax = R.get("per_case_shift", {}), R.get("axis", axis_of(form))
            keys = R.get("per_case_keys", {})
            dk = keys.get("delta", [])
            by_seed = R.get("delta_control_by_seed", {})
            adm_all = R.get("control_diagnostics", {})
            # PRIMARY: delta vs each norm-matched control family
            for fam, mode, claim in (("haar", "delta_rand", "delta_transport"),
                                     ("pca_span", "delta_pca_span", "delta_vs_pca_span"),
                                     ("shuf_fourier", "delta_shuf_fourier", "delta_vs_shuf_fourier")):
                if not ("delta" in pc and mode in pc and len(pc["delta"]) and dk):
                    continue
                sm, seed_report = by_seed.get(fam), None
                control_mean = np.asarray(pc[mode], float)   # stored mean over ALL seeds
                if sm is not None:
                    M = np.asarray(sm, float)                                    # [cases, seeds]
                    if args.admissible_only and adm_all.get(fam, {}).get("admissible"):
                        keep, seed_report = admit_global_seeds(adm_all[fam]["admissible"],
                                                               min_case_frac=args.min_case_frac)
                        if len(keep) < args.min_admitted_seeds:
                            drop(claim, tag, form,
                                 f"only {len(keep)} admitted control seed(s) < required "
                                 f"{args.min_admitted_seeds} (alpha outside {ALPHA_LO}-{ALPHA_HI})",
                                 n_admitted=len(keep))
                            continue
                        M = M[:, keep]                       # NO imputation: drop whole seeds only
                    # BLOCKER 2: the PRIMARY estimate/CI/p must come from the ADMITTED controls
                    control_mean = M.mean(axis=1)
                    sig = np.asarray(pc["delta"], float)
                    sd = sig[:, None] - M
                else:
                    sd = None
                cell = build_cell(pc["delta"], control_mean, dk, seed_matrix=sd, cluster_by=args.cluster_by)
                if cell is None:
                    drop(claim, tag, form, "fewer than 2 usable cases after keyed pairing")
                elif True:
                    add_row(rows, claim, tag, form, ax, cell, args.b, require_crossed=args.require_crossed,
                            extra={"alpha_frac_out": R.get("delta_alpha", {}).get(fam, {}).get("frac_out_of_range"),
                                   "seed_admission": seed_report,
                                   "controls_used": "admitted_global_seeds" if seed_report else "all_seeds"})
            # LEGACY (opt-in only): absolute subspace vs absolute random
            if args.legacy and "subspace" in pc and "random" in pc and keys.get("modes"):
                cell = build_cell(pc["subspace"], pc["random"], keys["modes"], cluster_by=args.cluster_by)
                if cell:
                    add_row(rows, "sufficiency", tag, form, ax, cell, args.b, status=LEGACY)

    # ---------------- necessity: ablation + delta interchange ----------------
    # Loaded by METADATA, not filename (audit r5 #10): every necessity file is considered and its
    # ablation_position is read from the JSON, so `after`-position runs are no longer invisible.
    for f in sorted(glob.glob(os.path.join(args.out_dir, "necessity_*.json"))):
        tag = os.path.basename(f)[len("necessity_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = load_admissible(f, args, experiment="necessity", estimands={E_ABLATION}, statuses={VALIDATED})
        if d is None:
            skipped.append(f); continue
        pos = d.get("ablation_position", "?")
        if args.positions and pos not in args.positions:
            continue
        loaded.append((os.path.basename(f), d))
        suffix = f"@{pos}"      # ALWAYS explicit: no position is silently "the" necessity claim
        for form, A in d.get("ablation", {}).items():
            pc = A.get("per_case", {})
            if args.null not in pc.get("controls", {}) or not pc.get("keys"):
                drop("necessity" + suffix, tag, form, "missing per-case control/keys arrays")
                continue
            ok_clean, clean_acc = eligible_clean(A, args.clean_acc_threshold)
            if not ok_clean:
                drop("necessity" + suffix, tag, form,
                     f"not_testable_due_to_clean_behavior (clean_acc={clean_acc:.2f} < "
                     f"{args.clean_acc_threshold})", clean_acc=clean_acc, not_testable=True)
                continue
            sm = pc.get("controls_by_seed", {}).get(args.null)
            adm = A.get("control_diagnostics", {}).get(args.null, {}).get("admissible")
            helix = np.asarray(pc["helix"], float)
            null_mean, seed_report, sd = np.asarray(pc["controls"][args.null], float), None, None
            if sm is not None:
                M = np.asarray(sm, float)
                if args.admissible_only and adm:
                    keep, seed_report = admit_global_seeds(adm, min_case_frac=args.min_case_frac)
                    if len(keep) < args.min_admitted_seeds:
                        drop("necessity" + suffix, tag, form,
                             f"only {len(keep)} admitted control seed(s) < required "
                             f"{args.min_admitted_seeds}", n_admitted=len(keep))
                        continue
                    M = M[:, keep]                           # whole seeds only, no imputation
                null_mean = M.mean(axis=1)                   # PRIMARY uses admitted controls (blocker #2)
                sd = M - helix[:, None]
            # effect = null_acc - helix_acc  (positive => ablating the helix hurts MORE)
            cell = build_cell(null_mean, pc["helix"], pc["keys"],
                              seed_matrix=sd, cluster_by=args.cluster_by)
            if cell is None:
                drop("necessity" + suffix, tag, form, "fewer than 2 usable cases after keyed pairing")
            else:
                add_row(rows, "necessity" + suffix, tag, form, A.get("axis", axis_of(form)), cell, args.b,
                        require_crossed=args.require_crossed, extra={"ablation_position": pos, "seed_admission": seed_report, "clean_acc": clean_acc,
                               "controls_used": "admitted_global_seeds" if seed_report else "all_seeds",
                               "n_skipped_no_baseline": A.get("n_skipped_no_baseline")})
        for form, I in (d.get("interchange", {}) if args.interchange else {}).items():
            pc = I.get("per_case", {})
            if not pc.get("keys"):
                continue
            sm = I.get("control_by_seed")
            sig = np.asarray(pc["subspace"], float)
            sd = (sig[:, None] - np.asarray(sm, float)) if sm else None
            cell = build_cell(pc["subspace"], pc["matched_random"], pc["keys"],
                              seed_matrix=sd, cluster_by=args.cluster_by)
            if cell:
                add_row(rows, "interchange" + suffix, tag, form, I.get("axis", axis_of(form)), cell, args.b,
                        status=EXPLORATORY,
                        extra={"interchange_position": I.get("intervention_position")})

    # ---------------- exploratory sweeps (opt-in only) ----------------
    if args.exploratory:
        for f in sorted(glob.glob(os.path.join(args.out_dir, "ablation_sweep_*.json"))):
            tag = os.path.basename(f)[len("ablation_sweep_"):-len(".json")]
            if not want(tag, args.models):
                continue
            d = load_admissible(f, args, experiment="ablation_sweep", estimands={E_LAYER_VULN},
                                statuses={EXPLORATORY})
            if d is None:
                skipped.append(f); continue
            loaded.append((os.path.basename(f), d))
            for form, Cv in d.get("curves", {}).items():
                ps = Cv.get("heldout_peak_structured", {}).get(args.null)
                if ps and "per_case" in ps and ps.get("keys"):
                    pcv = np.asarray(ps["per_case"], float)
                    cell = build_cell(pcv, np.zeros_like(pcv), ps["keys"], cluster_by=args.cluster_by)
                    if cell:
                        add_row(rows, "necessity_peak", tag, form, axis_of(form), cell, args.b,
                                status=EXPLORATORY)

    # ---------------- whole-run-directory validation for production reports (audit r5 #5) ----------
    if args.production:
        try:
            man = validate_run_dir(args.out_dir, loaded, require_manifest=True)
            print(f"\nRUN VALIDATED: {man.get('run_id')} @ {str(man.get('code_commit'))[:7]} "
                  f"({len(loaded)} result files, one commit, no duplicates, no missing models)")
        except ValueError as e:
            raise SystemExit(f"\nPRODUCTION RUN REJECTED: {e}\n")

    if not rows:
        msg = (f"No admissible per-case data found (schema {args.schema}). "
               "Re-run the experiments or pass --include-* flags.")
        if args.production:
            raise SystemExit(f"\nPRODUCTION RUN REJECTED: {msg}\n")
        print(msg)
        return

    # ---------------- FDR over the DEFAULT family only (opt-in claims excluded) ----------------
    # PREREGISTERED FAMILIES: each claim is its own BH family (they test different control families /
    # different interventions). --global-fdr additionally reports one correction across all primary
    # cells as a sensitivity analysis, since per-family correction is the more permissive choice (r5 #9).
    default_rows = [r for r in rows if r["claim"].split("@")[0] in DEFAULT_CLAIMS]
    for claim in {r["claim"] for r in default_rows}:
        cr = [r for r in default_rows if r["claim"] == claim]
        reject, q = bh_fdr([r["perm_p"] for r in cr], alpha=args.alpha)
        for r, rj, qq in zip(cr, reject, q):
            r["q"], r["sig_fdr"] = float(qq), bool(rj)
    if args.global_fdr and default_rows:
        rej_g, q_g = bh_fdr([r["perm_p"] for r in default_rows], alpha=args.alpha)
        for r, rj, qq in zip(default_rows, rej_g, q_g):
            r["q_global"], r["sig_fdr_global"] = float(qq), bool(rj)
    for r in rows:                       # opt-in claims are reported but never FDR-corrected here
        r.setdefault("q", float("nan"))
        r.setdefault("sig_fdr", False)
    # HEADLINE gate: FDR-significant AND the crossed (case x global-seed) interval excludes zero.
    for r in rows:
        crossed_ok = r["sig_crossed"] if r.get("sig_crossed") is not None else r["sig_ci"]
        r["headline"] = bool(r["sig_fdr"] and (crossed_ok if args.require_crossed else r["sig_ci"]))

    # ---------------- table ----------------
    def flag(b):
        return "✓" if b else "·"
    rows.sort(key=lambda r: (CLAIM_ORDER.index(r["claim"]) if r["claim"] in CLAIM_ORDER else 99,
                             r["model"], r["form"]))
    print("\n" + "=" * 134)
    print(f"PAIRED TESTS  (schema {args.schema}; B={args.b}; necessity null={args.null}; FDR alpha={args.alpha})")
    print("  CI shown = CROSSED (cases x global control seeds) | perm p = cluster sign flip | q = BH-FDR")
    print("  HEAD = headline positive: FDR-significant AND the crossed interval excludes 0")
    print("-" * 134)
    print(f"  {'claim':<21}{'model':<21}{'form':<18}{'axis':<9}{'effect':>8}{'95% CI':>17}"
          f"{'perm_p':>8}{'q':>7}{'P(beat)':>9}{'xCI':>5}{'FDR':>5}{'HEAD':>6}{'n':>5}")
    for r in rows:
        ci = f"[{r['primary_lo']:.2f}, {r['primary_hi']:.2f}]"
        pb = f"{r['p_beats_random_control']:.2f}" if "p_beats_random_control" in r else "  -"
        hh = flag(r["sig_crossed"]) if "sig_crossed" in r else "-"
        print(f"  {r['claim']:<21}{r['model'][:20]:<21}{r['form']:<18}{r['axis']:<9}{r['effect']:>8.3f}{ci:>17}"
              f"{r['perm_p']:>8.3f}{r['q']:>7.3f}{pb:>9}{hh:>5}{flag(r['sig_fdr']):>5}{flag(r['headline']):>6}{r['n']:>5}")
    print("=" * 134)
    optin = [r for r in rows if r["claim"] in OPTIN_CLAIMS]
    if optin:
        print(f"  NOTE: {len(optin)} row(s) are OPT-IN ({', '.join(sorted({r['claim'] for r in optin}))}) — "
              "excluded from the FDR family and from headline counts.")

    # ---------------- summaries ----------------
    print("\nSIGNIFICANCE SUMMARY (default validated family only; CI = clustered, FDR on cluster perm p):")
    for claim in DEFAULT_CLAIMS:
        cr = [r for r in default_rows if r["claim"].split("@")[0] == claim]
        if cr:
            print(f"  {claim:<22} xCI {sum(bool(r.get('sig_crossed')) for r in cr):>2}/{len(cr):<2}   "
                  f"FDR {sum(r['sig_fdr'] for r in cr):>2}/{len(cr):<2}   "
                  f"HEADLINE {sum(r['headline'] for r in cr):>2}/{len(cr):<2}")

    models = sorted({r["model"] for r in default_rows})
    present = [c for c in DEFAULT_CLAIMS if any(r["claim"].split("@")[0] == c for r in default_rows)]
    if models and present:
        print("\nPER-MODEL fraction of forms HEADLINE-positive (FDR and crossed CI):")
        print(f"  {'model':<24}" + "".join(f"{c[:14]:>16}" for c in present))
        for mdl in models:
            cells = "".join(
                (lambda cr: f"{(sum(r['headline'] for r in cr) / len(cr)):>16.2f}" if cr else f"{'-':>16}")(
                    [r for r in default_rows if r["model"] == mdl and r["claim"].split("@")[0] == c])
                for c in present)
            print(f"  {mdl[:23]:<24}{cells}")

    # ---------------- BLOCKER 4: dropped primary cells invalidate a production run -------------
    primary_prefixes = set(C.PRIMARY_FAMILIES)
    dropped_primary = [d for d in dropped if d["claim"].split("@")[0] in primary_prefixes]
    if dropped:
        print(f"\nDROPPED CELLS ({len(dropped)}; {len(dropped_primary)} in a PRIMARY family):")
        for d in dropped:
            mark = "PRIMARY" if d in dropped_primary else "       "
            print(f"  [{mark}] {d['claim']:<22}{d['model']:<20}{d['form']:<18}{d['reason']}")
    if args.production and dropped_primary:
        raise SystemExit(
            f"\nPRODUCTION RUN REJECTED: {len(dropped_primary)} primary analysis cell(s) were dropped "
            "-- a report cannot silently omit cells from its own denominator. Fix the underlying "
            "cause (controls / clean accuracy / case coverage) or preregister the omission.\n")

    # ---------------- json + forest (default family only) ----------------
    out = {"schema_version": args.schema, "bootstrap_B": args.b, "necessity_null": args.null,
           "alpha": args.alpha, "default_claims": DEFAULT_CLAIMS,
           "included_legacy": args.legacy, "included_exploratory": args.exploratory,
           "admissible_only": args.admissible_only, "cluster_by": args.cluster_by,
           "positions": args.positions, "production_validated": args.production,
           "analysis_policy": policy, "require_crossed": args.require_crossed,
           "clean_acc_threshold": args.clean_acc_threshold,
           "dropped_cells": dropped, "primary_families": C.PRIMARY_FAMILIES,
           "skipped_files": [os.path.basename(s) for s in skipped], "rows": rows}
    with open(os.path.join(args.out_dir, "stats_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    panels = [(c, c) for c in DEFAULT_CLAIMS if any(r["claim"].split("@")[0] == c for r in default_rows)]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 8))
        axes = np.atleast_1d(axes)
        for ax, (claim, xlabel) in zip(axes, panels):
            cr = sorted([r for r in default_rows if r["claim"].split("@")[0] == claim],
                        key=lambda r: (r["model"], r["form"]))
            y = np.arange(len(cr))[::-1]
            for yi, r in zip(y, cr):
                col = AXIS_COLORS.get(r["axis"], "#333")
                ax.plot([r["primary_lo"], r["primary_hi"]], [yi, yi], color=col, lw=1.5,
                        alpha=1 if r["headline"] else 0.4)
                if r["headline"]:
                    ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="o",
                               facecolors=col, edgecolors=col)
                elif r["sig_ci"]:
                    ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="o",
                               facecolors="none", edgecolors=col)
                else:
                    ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="x", color=col)
            ax.axvline(0, color="#999", lw=1, ls="--")
            ax.set_yticks(y); ax.set_yticklabels([f"{r['model']}:{r['form']}" for r in cr], fontsize=6)
            ax.set_xlabel(xlabel); ax.set_title(claim, fontsize=9)
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle("crossed (case x seed) intervals · filled = headline (FDR + crossed) · open = case-CI only · × = n.s.",
                 fontsize=9)
        fig.tight_layout()
        png = os.path.join(args.out_dir, "stats_forest.png")
        fig.savefig(png, dpi=130)
        print(f"\nSaved -> {os.path.join(args.out_dir, 'stats_summary.json')}\n         {png}\n")


if __name__ == "__main__":
    main()

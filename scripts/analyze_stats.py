#!/usr/bin/env python
"""Fail-fast, estimand-aware statistics over the per-case data in experiments/*.json.

The analysis boundary is where validated, legacy, and exploratory results used to blur together.
This script now REFUSES files that do not declare the expected schema / experiment_type / estimand /
analysis_status (audit r4 #1), and only admits legacy or exploratory estimands behind explicit flags.

Default claim family (all `analysis_status: validated`):
  - delta_transport        : matched-arithmetic delta  −  norm-matched Haar control   [PRIMARY sufficiency]
  - delta_vs_pca_span      : same signal vs a top-PCA-span control
  - delta_vs_shuf_fourier  : same signal vs a shuffled-pipeline control
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
from src.provenance import (require_schema, VALIDATED, LEGACY, EXPLORATORY,
                            E_DELTA, E_ABSOLUTE, E_ABLATION, E_LAYER_VULN)

AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}
DEFAULT_CLAIMS = ["delta_transport", "delta_vs_pca_span", "delta_vs_shuf_fourier", "interchange", "necessity"]
OPTIN_CLAIMS = {"sufficiency": "legacy absolute patching", "necessity_peak": "exploratory layer sweep"}
CLAIM_ORDER = ["delta_transport", "delta_vs_pca_span", "delta_vs_shuf_fourier", "interchange",
               "necessity", "necessity_peak", "sufficiency"]


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


def hier_boot(seed_diff, groups, B, seed=0, alpha=0.05):
    """Hierarchical bootstrap (audit r4 #4): resample CASES (clustered by group), then sample ONE
    control seed within each case, so control-direction uncertainty propagates into the interval."""
    M = np.asarray(seed_diff, float)                     # [n_cases, n_seeds]
    n, k = M.shape
    rng = np.random.default_rng(seed)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    means = np.empty(B)
    for j in range(B):
        rows = np.concatenate([by_g[g] for g in uniq[rng.integers(0, len(uniq), size=len(uniq))]])
        means[j] = M[rows, rng.integers(0, k, size=len(rows))].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def seed_stats(seed_diff):
    """Control-distribution summaries: how the signal fares against the WHOLE control set, not just
    its mean (audit r4 #4). seed_diff[i][j] is oriented so positive supports the claim."""
    M = np.asarray(seed_diff, float)
    return {"p_beats_random_control": float(np.mean(M > 0)),          # P(signal beats a random control draw)
            "vs_mean_control": float(M.mean(1).mean()),
            "vs_strong_control_q90": float(np.mean(np.percentile(M, 10, axis=1))),  # vs a strong control
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

def add_row(rows, claim, tag, form, axis, diff, B, groups=None, seed_diff=None, status=VALIDATED):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return
    est = float(diff.mean())
    lo, hi = cluster_boot(diff, groups, B)
    row = {"claim": claim, "model": tag, "form": form, "axis": axis, "status": status,
           "effect": est, "lo": lo, "hi": hi, "n": n,
           "perm_p": cluster_sign_p(diff, groups, B),
           "clustered": groups is not None,
           "sig_ci": bool(lo > 0 or hi < 0)}
    if seed_diff is not None and len(seed_diff) == n:
        row.update(seed_stats(seed_diff))
        hlo, hhi = hier_boot(seed_diff, groups, B)
        row["hier_lo"], row["hier_hi"] = hlo, hhi
        row["sig_hier"] = bool(hlo > 0 or hhi < 0)      # the most conservative interval
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


def main():
    args = parse_args()
    rows, skipped = [], []

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
        for form, R in d.get("results", {}).items():
            pc, ax = R.get("per_case_shift", {}), R.get("axis", axis_of(form))
            keys = R.get("per_case_keys", {})
            dk = [tuple(k) for k in keys.get("delta", [])]
            g_delta = [k[0] for k in dk] if dk else None
            by_seed = R.get("delta_control_by_seed", {})
            # PRIMARY: delta vs each norm-matched control family
            for fam, mode, claim in (("haar", "delta_rand", "delta_transport"),
                                     ("pca_span", "delta_pca_span", "delta_vs_pca_span"),
                                     ("shuf_fourier", "delta_shuf_fourier", "delta_vs_shuf_fourier")):
                if "delta" in pc and mode in pc and len(pc["delta"]):
                    sd = None
                    if by_seed.get(fam):
                        sig = np.asarray(pc["delta"], float)[:, None]
                        sd = sig - np.asarray(by_seed[fam], float)      # [cases, seeds], + supports claim
                    add_row(rows, claim, tag, form, ax, paired(pc["delta"], pc[mode]), args.b,
                            groups=g_delta, seed_diff=sd)
            # LEGACY (opt-in only): absolute subspace vs absolute random
            if args.legacy and "subspace" in pc and "random" in pc:
                mk = [tuple(k) for k in keys.get("modes", [])]
                gm = [k[0] for k in mk] if mk else None
                add_row(rows, "sufficiency", tag, form, ax, paired(pc["subspace"], pc["random"]),
                        args.b, groups=gm, status=LEGACY)

    # ---------------- necessity: ablation + delta interchange ----------------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "necessity_*_span.json"))
                    + glob.glob(os.path.join(args.out_dir, "necessity_*_last.json"))):
        tag = os.path.basename(f)[len("necessity_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = load_admissible(f, args, experiment="necessity", estimands={E_ABLATION}, statuses={VALIDATED})
        if d is None:
            skipped.append(f); continue
        for form, A in d.get("ablation", {}).items():
            pc = A.get("per_case", {})
            if args.null not in pc.get("controls", {}):
                continue
            keys = [tuple(k) for k in pc.get("keys", [])]
            g = [k[0] for k in keys] if keys else None
            # effect = null_acc - helix_acc  (positive => ablating the helix hurts MORE)
            diff = paired(pc["controls"][args.null], pc["helix"])
            sd = None
            if pc.get("controls_by_seed", {}).get(args.null):
                helix = np.asarray(pc["helix"], float)[:, None]
                sd = np.asarray(pc["controls_by_seed"][args.null], float) - helix
            add_row(rows, "necessity", tag, form, A.get("axis", axis_of(form)), diff, args.b,
                    groups=g, seed_diff=sd)
        for form, I in d.get("interchange", {}).items():
            pc = I.get("per_case", {})
            if not pc:
                continue
            keys = [tuple(k) for k in pc.get("keys", [])]
            g = [k[0] for k in keys] if keys else None
            sd = None
            if I.get("control_by_seed"):
                sig = np.asarray(pc["subspace"], float)[:, None]
                sd = sig - np.asarray(I["control_by_seed"], float)
            add_row(rows, "interchange", tag, form, I.get("axis", axis_of(form)),
                    paired(pc["subspace"], pc["matched_random"]), args.b, groups=g, seed_diff=sd)

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
            for form, Cv in d.get("curves", {}).items():
                ps = Cv.get("heldout_peak_structured", {}).get(args.null)
                if ps and "per_case" in ps:
                    g = [k[0] for k in ps["keys"]] if ps.get("keys") else None
                    add_row(rows, "necessity_peak", tag, form, axis_of(form),
                            np.array(ps["per_case"], float), args.b, groups=g, status=EXPLORATORY)

    if not rows:
        print("No admissible per-case data found. Re-run the experiments (schema "
              f"{args.schema}) or pass --include-* flags.")
        return

    # ---------------- FDR over the DEFAULT family only (opt-in claims excluded) ----------------
    default_rows = [r for r in rows if r["claim"] in DEFAULT_CLAIMS]
    for claim in {r["claim"] for r in default_rows}:
        cr = [r for r in default_rows if r["claim"] == claim]
        reject, q = bh_fdr([r["perm_p"] for r in cr], alpha=args.alpha)
        for r, rj, qq in zip(cr, reject, q):
            r["q"], r["sig_fdr"] = float(qq), bool(rj)
    for r in rows:                       # opt-in claims are reported but never FDR-corrected here
        r.setdefault("q", float("nan"))
        r.setdefault("sig_fdr", False)

    # ---------------- table ----------------
    def flag(b):
        return "✓" if b else "·"
    rows.sort(key=lambda r: (CLAIM_ORDER.index(r["claim"]) if r["claim"] in CLAIM_ORDER else 99,
                             r["model"], r["form"]))
    print("\n" + "=" * 134)
    print(f"PAIRED TESTS  (schema {args.schema}; B={args.b}; necessity null={args.null}; FDR alpha={args.alpha})")
    print("  CI = cluster bootstrap by source value | perm p = CLUSTER-level sign flip | q = BH-FDR (default family only)")
    print("  P(beat ctrl) = fraction of (case, control-seed) draws the signal beats | hier = hierarchical CI over cases x seeds")
    print("-" * 134)
    print(f"  {'claim':<21}{'model':<21}{'form':<18}{'axis':<9}{'effect':>8}{'95% CI':>17}"
          f"{'perm_p':>8}{'q':>7}{'P(beat)':>9}{'CI':>4}{'FDR':>5}{'hier':>6}{'n':>5}")
    for r in rows:
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        pb = f"{r['p_beats_random_control']:.2f}" if "p_beats_random_control" in r else "  -"
        hh = flag(r["sig_hier"]) if "sig_hier" in r else "-"
        print(f"  {r['claim']:<21}{r['model'][:20]:<21}{r['form']:<18}{r['axis']:<9}{r['effect']:>8.3f}{ci:>17}"
              f"{r['perm_p']:>8.3f}{r['q']:>7.3f}{pb:>9}{flag(r['sig_ci']):>4}{flag(r['sig_fdr']):>5}{hh:>6}{r['n']:>5}")
    print("=" * 134)
    optin = [r for r in rows if r["claim"] in OPTIN_CLAIMS]
    if optin:
        print(f"  NOTE: {len(optin)} row(s) are OPT-IN ({', '.join(sorted({r['claim'] for r in optin}))}) — "
              "excluded from the FDR family and from headline counts.")

    # ---------------- summaries ----------------
    print("\nSIGNIFICANCE SUMMARY (default validated family only; CI = clustered, FDR on cluster perm p):")
    for claim in DEFAULT_CLAIMS:
        cr = [r for r in default_rows if r["claim"] == claim]
        if cr:
            print(f"  {claim:<22} CI {sum(r['sig_ci'] for r in cr):>2}/{len(cr):<2}   "
                  f"FDR {sum(r['sig_fdr'] for r in cr):>2}/{len(cr):<2}")

    models = sorted({r["model"] for r in default_rows})
    present = [c for c in DEFAULT_CLAIMS if any(r["claim"] == c for r in default_rows)]
    if models and present:
        print("\nPER-MODEL fraction of forms significant (FDR):")
        print(f"  {'model':<24}" + "".join(f"{c[:14]:>16}" for c in present))
        for mdl in models:
            cells = "".join(
                (lambda cr: f"{(sum(r['sig_fdr'] for r in cr) / len(cr)):>16.2f}" if cr else f"{'-':>16}")(
                    [r for r in default_rows if r["model"] == mdl and r["claim"] == c])
                for c in present)
            print(f"  {mdl[:23]:<24}{cells}")

    # ---------------- json + forest (default family only) ----------------
    out = {"schema_version": args.schema, "bootstrap_B": args.b, "necessity_null": args.null,
           "alpha": args.alpha, "default_claims": DEFAULT_CLAIMS,
           "included_legacy": args.legacy, "included_exploratory": args.exploratory,
           "skipped_files": [os.path.basename(s) for s in skipped], "rows": rows}
    with open(os.path.join(args.out_dir, "stats_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    panels = [(c, c) for c in DEFAULT_CLAIMS if any(r["claim"] == c for r in default_rows)]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 8))
        axes = np.atleast_1d(axes)
        for ax, (claim, xlabel) in zip(axes, panels):
            cr = sorted([r for r in default_rows if r["claim"] == claim],
                        key=lambda r: (r["model"], r["form"]))
            y = np.arange(len(cr))[::-1]
            for yi, r in zip(y, cr):
                col = AXIS_COLORS.get(r["axis"], "#333")
                ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=1.5, alpha=1 if r["sig_ci"] else 0.4)
                if r["sig_ci"]:
                    ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="o",
                               facecolors=col if r["sig_fdr"] else "none", edgecolors=col)
                else:
                    ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="x", color=col)
            ax.axvline(0, color="#999", lw=1, ls="--")
            ax.set_yticks(y); ax.set_yticklabels([f"{r['model']}:{r['form']}" for r in cr], fontsize=6)
            ax.set_xlabel(xlabel); ax.set_title(claim, fontsize=9)
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle("validated family only · filled = FDR-significant · open = CI-only · × = n.s.", fontsize=9)
        fig.tight_layout()
        png = os.path.join(args.out_dir, "stats_forest.png")
        fig.savefig(png, dpi=130)
        print(f"\nSaved -> {os.path.join(args.out_dir, 'stats_summary.json')}\n         {png}\n")


if __name__ == "__main__":
    main()

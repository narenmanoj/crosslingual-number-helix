#!/usr/bin/env python
"""Bootstrap CIs + paired significance tests over the per-case data in experiments/*.json.

No model needed -- reads the per-case arrays run_transport / run_necessity / run_ablation_sweep save.
Per (model, form) it paired-tests, with BOTH a percentile bootstrap CI and a sign-flip PERMUTATION
test (a proper null-centered test; the bootstrap p alone is not -- audit #11), then applies a
Benjamini-Hochberg FDR correction ACROSS cells per claim (so "45 individually significant cells" is
not the unit) and reports a per-model aggregation.

Claims:
  - sufficiency     : subspace_shift - random_shift            (transport)
  - delta_transport : delta_shift - delta_rand_shift           (matched-arithmetic value transport, audit #2)
  - interchange     : subspace - norm-matched-random           (matched-source interchange)
  - necessity       : structured-null_acc - helix_acc @ share-layer
  - necessity_peak  : structured-null_acc - helix_acc @ necessity peak (strong null; from the sweep)

"sig" (CI) = 95% CI excludes 0. "FDR" = BH-adjusted q <= alpha across that claim's cells.

Usage:  python scripts/analyze_stats.py            (all models in experiments/)
        python scripts/analyze_stats.py --b 20000 --null cov_matched --models Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}
CLAIMS = ["sufficiency", "delta_transport", "interchange", "necessity_peak", "necessity"]


def boot(diff, B, seed=0, alpha=0.05):
    """Percentile bootstrap of the MEAN of a per-case paired-difference array.
    Returns (estimate, lo, hi, p_one_sided=P(resampled mean <= 0), n)."""
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), float("nan"), n)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(B, n))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi), float(np.mean(means <= 0)), n


def perm_sign_p(diff, B=10000, seed=0):
    """Two-sided paired sign-flip PERMUTATION p (audit #11): under H0 the distribution of paired
    differences is symmetric about 0, so each sign is exchangeable. p = P(|flipped mean| >= |obs|).
    A genuine null-centered test, unlike resampling the empirical (non-null) distribution."""
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    flipped = np.abs((rng.choice([-1.0, 1.0], size=(B, n)) * x).mean(1))
    return float(np.mean(flipped >= obs))


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR: returns (reject bool array, q-values), aligned to input order."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    q = np.full_like(p, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return reject, q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    qr = ranked * m / np.arange(1, m + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]      # enforce monotonicity
    q[order] = np.clip(qr, 0, 1)
    reject[order] = q[order] <= alpha
    return reject, q


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--models", nargs="*", default=None, help="restrict to these model tags (basename)")
    p.add_argument("--b", type=int, default=10000, help="bootstrap / permutation resamples")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--null", default="shuf_fourier", choices=["shuf_fourier", "cov_matched", "random"],
                   help="necessity control null to test helix against")
    return p.parse_args()


def want(tag, models):
    return models is None or any(m in tag for m in models)


def axis_of(form):
    if form == "en_word":
        return "notation"
    if form.endswith("_word"):
        return "language"
    return "script"


def add_row(rows, claim, tag, form, axis, diff, B):
    """Paired bootstrap CI + permutation p for one (claim, model, form) cell."""
    diff = np.asarray(diff, float)
    est, lo, hi, p, n = boot(diff, B)
    rows.append({"claim": claim, "model": tag, "form": form, "axis": axis,
                 "effect": est, "lo": lo, "hi": hi, "p": p, "perm_p": perm_sign_p(diff, B),
                 "n": n, "sig_ci": bool(lo > 0 or hi < 0)})


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = min(len(a), len(b))
    return a[:m] - b[:m]


def main():
    args = parse_args()
    rows = []

    # ---------- transport: sufficiency (subspace vs random) + delta transport (delta vs delta_rand) ----------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "transport_*_L*.json"))):
        tag = os.path.basename(f)[len("transport_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = json.load(open(f))
        for form, R in d.get("results", {}).items():
            pc, ax = R.get("per_case_shift"), R.get("axis", axis_of(form))
            if not pc:
                continue
            if "subspace" in pc and "random" in pc:
                add_row(rows, "sufficiency", tag, form, ax, paired(pc["subspace"], pc["random"]), args.b)
            if "delta" in pc and "delta_rand" in pc and len(pc["delta"]):
                add_row(rows, "delta_transport", tag, form, ax, paired(pc["delta"], pc["delta_rand"]), args.b)

    # ---------- necessity @ share-layer + matched-source interchange ----------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "necessity_*_span.json"))):
        tag = os.path.basename(f)[len("necessity_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = json.load(open(f))
        for form, A in d.get("ablation", {}).items():
            pc = A.get("per_case")
            if pc and args.null in pc.get("controls", {}):
                add_row(rows, "necessity", tag, form, A.get("axis", axis_of(form)),
                        paired(pc["controls"][args.null], pc["helix"]), args.b)
        for form, I in d.get("interchange", {}).items():
            pc = I.get("per_case")
            if pc:
                add_row(rows, "interchange", tag, form, I.get("axis", axis_of(form)),
                        paired(pc["subspace"], pc["matched_random"]), args.b)

    # ---------- necessity @ peak layer, vs the STRUCTURED null (strong claim; from the sweep) ----------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "ablation_sweep_*.json"))):
        tag = os.path.basename(f)[len("ablation_sweep_"):-len(".json")]
        if not want(tag, args.models):
            continue
        d = json.load(open(f))
        for form, C in d.get("curves", {}).items():
            ps = C.get("heldout_peak_structured", {}).get(args.null)
            if ps and "per_case" in ps:
                add_row(rows, "necessity_peak", tag, form, axis_of(form), np.array(ps["per_case"], float), args.b)

    if not rows:
        print("No per-case data found. Re-run run_transport/run_necessity (they save per_case).")
        return

    # ---------- BH-FDR across cells, PER CLAIM (the multiple-comparisons policy) ----------
    for claim in set(r["claim"] for r in rows):
        cr = [r for r in rows if r["claim"] == claim]
        reject, q = bh_fdr([r["p"] for r in cr], alpha=args.alpha)
        for r, rj, qq in zip(cr, reject, q):
            r["q"] = float(qq)
            r["sig_fdr"] = bool(rj)

    # ---------- table ----------
    def flag(b):
        return "✓" if b else "·"
    order = {c: i for i, c in enumerate(CLAIMS)}
    rows.sort(key=lambda r: (order.get(r["claim"], 9), r["model"], r["form"]))
    print("\n" + "=" * 128)
    print(f"PAIRED TESTS  (B={args.b}; necessity null = {args.null}; FDR alpha = {args.alpha})")
    print("  effect [95% CI] | boot p (P<=0, 1-sided) | perm p (sign-flip, 2-sided) | q (BH-FDR) | CI sig | FDR sig")
    print("-" * 128)
    print(f"  {'claim':<16}{'model':<24}{'form':<19}{'axis':<9}{'effect':>8}{'95% CI':>18}"
          f"{'boot_p':>8}{'perm_p':>8}{'q_fdr':>8}{'CI':>4}{'FDR':>5}{'n':>5}")
    for r in rows:
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"  {r['claim']:<16}{r['model'][:23]:<24}{r['form']:<19}{r['axis']:<9}{r['effect']:>8.3f}{ci:>18}"
              f"{r['p']:>8.3f}{r['perm_p']:>8.3f}{r['q']:>8.3f}{flag(r['sig_ci']):>4}{flag(r['sig_fdr']):>5}{r['n']:>5}")
    print("=" * 128)

    # ---------- overall + per-model aggregation ----------
    print("\nSIGNIFICANCE SUMMARY per claim (CI-based | FDR-corrected):")
    for claim in CLAIMS:
        cr = [r for r in rows if r["claim"] == claim]
        if cr:
            ci_n = sum(r["sig_ci"] for r in cr)
            fdr_n = sum(r["sig_fdr"] for r in cr)
            print(f"  {claim:<16} CI {ci_n:>2}/{len(cr):<2} ({100*ci_n/len(cr):3.0f}%)   "
                  f"FDR {fdr_n:>2}/{len(cr):<2} ({100*fdr_n/len(cr):3.0f}%)")

    models = sorted(set(r["model"] for r in rows))
    present = [c for c in CLAIMS if any(r["claim"] == c for r in rows)]
    print("\nPER-MODEL fraction of forms significant (FDR-corrected)   [aggregating over forms, not cells]:")
    print(f"  {'model':<26}" + "".join(f"{c[:12]:>13}" for c in present))
    for mdl in models:
        cells = "".join(
            (lambda cr: f"{(sum(r['sig_fdr'] for r in cr) / len(cr)):>13.2f}" if cr else f"{'-':>13}")(
                [r for r in rows if r["model"] == mdl and r["claim"] == c])
            for c in present)
        print(f"  {mdl[:25]:<26}{cells}")
    print("  (a cross-form dissociation -- e.g. sufficiency 1.00 but necessity low -- is the finding, not noise)")

    # ---------- json + forest ----------
    out = {"bootstrap_B": args.b, "necessity_null": args.null, "alpha": args.alpha, "rows": rows}
    with open(os.path.join(args.out_dir, "stats_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    panels = [("sufficiency", "subspace − random"),
              ("delta_transport", "delta − delta_rand  (matched-arith value)"),
              ("necessity_peak", f"{args.null} − helix @ peak"),
              ("necessity", f"{args.null} − helix @ share-layer")]
    panels = [p for p in panels if any(r["claim"] == p[0] for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 8))
    axes = np.atleast_1d(axes)
    for ax, (claim, xlabel) in zip(axes, panels):
        cr = [r for r in rows if r["claim"] == claim]
        cr.sort(key=lambda r: (r["model"], r["form"]))
        y = np.arange(len(cr))[::-1]
        for yi, r in zip(y, cr):
            col = AXIS_COLORS.get(r["axis"], "#333")
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=1.5, alpha=1 if r["sig_ci"] else 0.4)
            # filled circle = FDR-significant; open circle = CI-only; x = n.s.
            if r["sig_ci"]:
                ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="o",
                           facecolors=col if r.get("sig_fdr") else "none", edgecolors=col)
            else:
                ax.scatter([r["effect"]], [yi], s=30, zorder=3, marker="x", color=col)
        ax.axvline(0, color="#999", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([f"{r['model']}:{r['form']}" for r in cr], fontsize=6)
        ax.set_xlabel(xlabel); ax.set_title(claim)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("filled = FDR-significant · open circle = CI-only · × = n.s.", fontsize=9)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "stats_forest.png")
    fig.savefig(png, dpi=130)
    print(f"\nSaved -> {os.path.join(args.out_dir, 'stats_summary.json')}\n         {png}\n")


if __name__ == "__main__":
    main()

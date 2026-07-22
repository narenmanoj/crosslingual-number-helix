#!/usr/bin/env python
"""Bootstrap CIs + paired significance tests over the per-case data in experiments/*.json.

No model needed -- reads the per-case arrays that run_transport / run_necessity now save, and for
each (model, form) computes:
  - SUFFICIENCY: paired bootstrap of (subspace_shift - random_shift) per case -> effect [95% CI], p.
  - INTERCHANGE: paired bootstrap of (subspace - norm-matched-random) per case (matched-source).
  - NECESSITY:  paired bootstrap of (shuffled-Fourier-null acc - helix-ablate acc) per case -> the
    helix-SPECIFIC necessity effect [95% CI], p (vs the strongest structured null).

Outputs a printed table, experiments/stats_summary.json, and a forest plot (stats_forest.png).
One-sided p = P(bootstrap effect <= 0); "sig" = 95% CI excludes 0.

Usage:  python scripts/analyze_stats.py            (all models in experiments/)
        python scripts/analyze_stats.py --b 20000  --models Qwen2.5-7B Granite-4.0-h-tiny-base
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


def boot(diff, B, seed=0, alpha=0.05):
    """Percentile bootstrap of the MEAN of a per-case (paired-difference) array.
    Returns (estimate, lo, hi, p_one_sided) where p = P(resampled mean <= 0)."""
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), float("nan"), n)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(B, n))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi), float(np.mean(means <= 0)), n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--models", nargs="*", default=None, help="restrict to these model tags (basename)")
    p.add_argument("--b", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--null", default="shuf_fourier", choices=["shuf_fourier", "cov_matched", "random"],
                   help="necessity control null to test helix against")
    return p.parse_args()


def want(tag, models):
    return models is None or any(m in tag for m in models)


def main():
    args = parse_args()
    rows = []   # (claim, model, form, axis, est, lo, hi, p, n)

    # ---------- SUFFICIENCY (transport) + INTERCHANGE overlap: transport subspace vs random ----------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "transport_*_L*.json"))):
        tag = os.path.basename(f)[len("transport_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = json.load(open(f))
        for form, R in d.get("results", {}).items():
            pc = R.get("per_case_shift")
            if not pc or "subspace" not in pc or "random" not in pc:
                continue
            sub, rnd = np.array(pc["subspace"], float), np.array(pc["random"], float)
            m = min(len(sub), len(rnd))
            est, lo, hi, p, n = boot(sub[:m] - rnd[:m], args.b)
            rows.append(("sufficiency", tag, form, R.get("axis", "?"), est, lo, hi, p, n))

    # ---------- NECESSITY (whole-span ablation) + matched-source interchange ----------
    for f in sorted(glob.glob(os.path.join(args.out_dir, "necessity_*_span.json"))):
        tag = os.path.basename(f)[len("necessity_"):].rsplit("_L", 1)[0]
        if not want(tag, args.models):
            continue
        d = json.load(open(f))
        for form, A in d.get("ablation", {}).items():
            pc = A.get("per_case")
            if not pc or args.null not in pc.get("controls", {}):
                continue
            helix = np.array(pc["helix"], float)
            null = np.array(pc["controls"][args.null], float)
            m = min(len(helix), len(null))
            # effect = null_acc - helix_acc (positive => ablating the helix hurts MORE => helix-specific)
            est, lo, hi, p, n = boot(null[:m] - helix[:m], args.b)
            rows.append((f"necessity", tag, form, A.get("axis", "?"), est, lo, hi, p, n))
        for form, I in d.get("interchange", {}).items():
            pc = I.get("per_case")
            if not pc:
                continue
            sub, mr = np.array(pc["subspace"], float), np.array(pc["matched_random"], float)
            m = min(len(sub), len(mr))
            est, lo, hi, p, n = boot(sub[:m] - mr[:m], args.b)
            rows.append(("interchange", tag, form, I.get("axis", "?"), est, lo, hi, p, n))

    if not rows:
        print("No per-case data found. Re-run run_transport/run_necessity (they now save per_case).")
        return

    # ---------- report ----------
    def sig(lo, hi):
        return "***" if (lo > 0 or hi < 0) else "n.s."
    print("\n" + "=" * 104)
    print(f"BOOTSTRAP CIs + PAIRED TESTS  (B={args.b}; necessity null = {args.null}; effect [95% CI], one-sided p)")
    print("  sufficiency = subspace_shift - random_shift | necessity = null_acc - helix_acc | interchange = subspace - matched_random")
    print("-" * 104)
    print(f"  {'claim':<12}{'model':<26}{'form':<20}{'axis':<9}{'effect':>9}{'95% CI':>20}{'p':>9}{'sig':>6}{'n':>5}")
    for claim, tag, form, axis, est, lo, hi, p, n in rows:
        print(f"  {claim:<12}{tag[:25]:<26}{form:<20}{axis:<9}{est:>9.3f}"
              f"{('['+format(lo,'.2f')+', '+format(hi,'.2f')+']'):>20}{p:>9.3f}{sig(lo,hi):>6}{n:>5}")
    print("=" * 104)

    # summary: fraction of (model,form) that are significant, per claim
    print("\nSIGNIFICANCE SUMMARY (fraction of model x form with 95% CI excluding 0):")
    for claim in ["sufficiency", "necessity", "interchange"]:
        cr = [r for r in rows if r[0] == claim]
        if cr:
            frac = np.mean([(lo > 0 or hi < 0) for _, _, _, _, _, lo, hi, _, _ in cr])
            print(f"  {claim:<12} {frac*100:.0f}%  ({int(round(frac*len(cr)))}/{len(cr)})")

    # ---------- forest plot (sufficiency + necessity panels) ----------
    out = {"bootstrap_B": args.b, "necessity_null": args.null,
           "rows": [{"claim": c, "model": t, "form": fm, "axis": ax, "effect": e,
                     "ci_lo": lo, "ci_hi": hi, "p": p, "n": n, "sig": bool(lo > 0 or hi < 0)}
                    for (c, t, fm, ax, e, lo, hi, p, n) in rows]}
    with open(os.path.join(args.out_dir, "stats_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    panels = [("sufficiency", "subspace − random  (logit shift)"),
              ("necessity", f"{args.null} − helix  (accuracy drop)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 8))
    axes = np.atleast_1d(axes)
    for ax, (claim, xlabel) in zip(axes, panels):
        cr = [r for r in rows if r[0] == claim]
        cr.sort(key=lambda r: (r[1], r[2]))       # by model, then form
        labels = [f"{t}:{fm}" for _, t, fm, *_ in cr]
        y = np.arange(len(cr))[::-1]
        for yi, (_, _, _, ax_, e, lo, hi, p, n) in zip(y, cr):
            col = AXIS_COLORS.get(ax_, "#333")
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.5, alpha=0.5 if not (lo > 0 or hi < 0) else 1)
            ax.scatter([e], [yi], color=col, s=28, zorder=3,
                       marker="o" if (lo > 0 or hi < 0) else "x")
        ax.axvline(0, color="#999", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(xlabel); ax.set_title(f"{claim}  (o = 95% CI excludes 0)")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "stats_forest.png")
    fig.savefig(png, dpi=130)
    print(f"\nSaved -> {os.path.join(args.out_dir, 'stats_summary.json')}\n         {png}\n")


if __name__ == "__main__":
    main()

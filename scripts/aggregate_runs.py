#!/usr/bin/env python
"""Aggregate the cross-model H2 table from the AUTHORITATIVE clean contrasts (audit #8).

By default reads structure_*.json (run_structure.py), whose `clean_contrasts` use the correct
reference PER AXIS (script = en_digit vs digit-scripts, notation = en_digit vs en_word, language =
en_word vs foreign words). The `axis_summary` in align_*.json is the everything-vs-en_digit contrast,
which changes both notation and language and silently reintroduces the reference confound -- so the
cross-model H2 plot must NOT be built from it. If you point --glob at align_*.json, this script warns
loudly and labels the output confounded.

Usage:
    python scripts/aggregate_runs.py                              # clean contrasts (structure_*.json)
    python scripts/aggregate_runs.py --glob 'align_*.json'        # confounded (warned)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["script", "notation", "language"]
AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--glob", default="structure_*.json", help="authoritative clean-contrast source")
    return p.parse_args()


def axis_values(d):
    """Return {axis: subspace_cos} and a source tag. structure_*.json -> clean contrasts (authoritative);
    align_*.json -> the confounded everything-vs-en_digit axis_summary."""
    if "clean_contrasts" in d:                     # structure_*.json (clean, per-axis reference)
        cc = d["clean_contrasts"]

        def pick(ax):
            for k, v in cc.items():
                if k.startswith(ax):
                    return float(v)
            return float("nan")
        return {a: pick(a) for a in AXES}, "clean"
    ax = d.get("axis_summary", {})                 # align_*.json (confounded)
    return {a: ax.get(a, {}).get("subspace_cos", float("nan")) for a in AXES}, "confounded"


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.out_dir, args.glob)))
    if not files:
        print(f"No files matching {args.glob} in {args.out_dir}/. Run run_structure.py first "
              "(or pass --glob 'align_*.json' for the confounded per-form contrasts).")
        sys.exit(0)

    rows, sources = [], set()
    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        vals, src = axis_values(d)
        sources.add(src)
        rows.append({
            "model": d.get("model", "?").split("/")[-1],
            "pooling": d.get("pooling", "?"),
            "layer": d.get("layer", "?"),
            "floor": d.get("floor", d.get("random_subspace_floor", float("nan"))),
            "contrast": src,
            **vals,
        })
    if "confounded" in sources:
        print("\n  ⚠ WARNING: aggregating the CONFOUNDED everything-vs-en_digit contrast (align_*.json). "
              "For the H2 figure use structure_*.json (clean per-axis contrasts) -- see audit #8.\n")
    rows.sort(key=lambda r: (r["model"], str(r["pooling"]), str(r["layer"])))

    # --- table ---
    print("\n" + "=" * 92)
    print("CROSS-RUN SUMMARY — per-axis subspace_cos (H2: script >= notation >= language)")
    print("-" * 92)
    print(f"  {'model':<26}{'pool':<12}{'layer':>6}{'floor':>8}{'script':>10}{'notation':>10}{'language':>10}")
    for r in rows:
        print(f"  {r['model']:<26}{str(r['pooling']):<12}{str(r['layer']):>6}{r['floor']:>8.3f}"
              f"{r['script']:>10.3f}{r['notation']:>10.3f}{r['language']:>10.3f}")
    print("=" * 92)

    # --- csv ---
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "pooling", "layer", "floor", "contrast", *AXES])
        w.writeheader()
        w.writerows(rows)

    # --- grouped bar chart: one group per run, three bars (axes) ---
    labels = [f"{r['model']}\n{r['pooling']}·L{r['layer']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(rows)), 5))
    for i, axis in enumerate(AXES):
        ax.bar(x + (i - 1) * width, [r[axis] for r in rows], width,
               label=axis, color=AXIS_COLORS[axis])
    floor = np.nanmean([r["floor"] for r in rows])
    ax.axhline(floor, ls="--", color="#888", lw=1, label=f"random floor (~{floor:.02f})")
    ax.set_ylabel("subspace_cos (clean per-axis contrast)" if "confounded" not in sources
                  else "subspace_cos (CONFOUNDED vs en_digit)")
    ax.set_title("Cross-form number-helix sharing (H2)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "summary.png")
    fig.savefig(png, dpi=130)

    print(f"\n{len(rows)} run(s) aggregated.")
    print(f"Saved -> {csv_path}\n         {png}\n")


if __name__ == "__main__":
    main()

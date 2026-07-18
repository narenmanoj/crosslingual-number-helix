#!/usr/bin/env python
"""Aggregate experiments/align_*.json into one cross-model / cross-pooling comparison.

Reads every point-alignment result (produced by run_fit_and_align.py) and builds:
  - a printed table of per-axis subspace_cos for each (model, pooling, layer),
  - a CSV (experiments/summary.csv),
  - a grouped bar chart (experiments/summary.png) with the random-floor line.

Use this to assemble the H2 story across models once the cluster runs land. It intentionally
reads only align_*.json (point runs); layer-sweep JSONs are summarized by run_layer_sweep.py.

Usage:
    python scripts/aggregate_runs.py
    python scripts/aggregate_runs.py --out-dir experiments --glob 'align_*.json'
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
    p.add_argument("--glob", default="align_*.json")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.out_dir, args.glob)))
    if not files:
        print(f"No files matching {args.glob} in {args.out_dir}/. Run run_fit_and_align.py first.")
        sys.exit(0)

    rows = []
    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        axis = d.get("axis_summary", {})
        rows.append({
            "model": d.get("model", "?").split("/")[-1],
            "pooling": d.get("pooling", "?"),
            "layer": d.get("layer", "?"),
            "floor": d.get("random_subspace_floor", float("nan")),
            "r2_ref": d.get("r2", {}).get(d.get("reference", ""), float("nan")),
            **{ax: axis.get(ax, {}).get("subspace_cos", float("nan")) for ax in AXES},
        })
    rows.sort(key=lambda r: (r["model"], str(r["pooling"]), str(r["layer"])))

    # --- table ---
    print("\n" + "=" * 92)
    print("CROSS-RUN SUMMARY — per-axis subspace_cos (H2: script >= notation >= language)")
    print("-" * 92)
    print(f"  {'model':<20}{'pool':<12}{'layer':>6}{'floor':>8}{'script':>10}{'notation':>10}{'language':>10}")
    for r in rows:
        print(f"  {r['model']:<20}{str(r['pooling']):<12}{str(r['layer']):>6}{r['floor']:>8.3f}"
              f"{r['script']:>10.3f}{r['notation']:>10.3f}{r['language']:>10.3f}")
    print("=" * 92)

    # --- csv ---
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "pooling", "layer", "floor", "r2_ref", *AXES])
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
    ax.set_ylabel("subspace_cos (vs en_digit)")
    ax.set_title("Cross-form number-helix sharing by run")
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

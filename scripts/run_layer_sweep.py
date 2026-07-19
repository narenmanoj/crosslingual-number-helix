#!/usr/bin/env python
"""Layer sweep: where in the network is the number helix SHARED across forms?

Extracts activations once (all layers come free from output_hidden_states), then fits the
helix and computes cross-form alignment at EVERY layer. Produces subspace_cos-vs-layer curves
per axis (script/notation/language) plus the helix-R^2 profile, so you can see the layer band
where cross-form sharing peaks -- which is where you'd site the step-3 causal transport.

Usage:
    python scripts/run_layer_sweep.py --model Qwen/Qwen2.5-7B --pooling mean
    python scripts/run_layer_sweep.py --forms en_digit devanagari_digit es_word
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import load_model, extract_form_activations, model_revision
from src.helix import fit_helix
from src.alignment import (
    subspace_alignment, linear_cka, random_subspace_floor, orthogonal_procrustes_cv,
)

AXES = ["script", "notation", "language"]
AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=C.FORMS or D.DEFAULT_FORMS)
    p.add_argument("--reference", default=None)
    p.add_argument("--pooling", default=C.POOLING, choices=["last", "mean", "prompt_last"])
    p.add_argument("--max-num", type=int, default=max(C.NUMBERS))
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    numbers = list(range(0, args.max_num + 1))
    forms = list(args.forms)
    ref = args.reference or forms[0]
    if ref not in forms:
        forms = [ref] + forms

    print(f"\nModel: {args.model} | pooling: {args.pooling}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"Device: {device} | d_model: {d_model} | layers: {n_layers}\n")

    # --- extract once; every layer is already in hidden_states ---
    acts_by_form = {}
    for f in forms:
        print(f"Extracting: {f} ({D.FORMS[f].axis})")
        prompts = D.build_prompts(f, numbers)
        acts_by_form[f] = extract_form_activations(model, tok, device, prompts, pooling=args.pooling)

    layers = list(range(0, n_layers + 1))  # 0 = embeddings
    # floor depends only on subspace dim + d_model, so compute it once
    ref_mid = fit_helix(acts_by_form[ref][n_layers // 2], numbers, k_pca=args.k_pca)
    floor = random_subspace_floor(ref_mid["helix_dirs_model"], d_model)

    per_layer = []
    for layer in layers:
        fits = {f: fit_helix(acts_by_form[f][layer], numbers, k_pca=args.k_pca) for f in forms}
        ref_dirs = fits[ref]["helix_dirs_model"]
        ref_H = acts_by_form[ref][layer]
        form_rows = []
        for f in forms:
            if f == ref:
                continue
            sc = subspace_alignment(ref_dirs, fits[f]["helix_dirs_model"])["mean_cos"]
            proc = orthogonal_procrustes_cv(ref_H, acts_by_form[f][layer])
            cka = linear_cka(ref_H, acts_by_form[f][layer])
            form_rows.append((f, D.FORMS[f].axis, sc, proc, cka))
        axis_summary = {}
        for ax in AXES:
            v = [(sc, proc, cka) for _, a, sc, proc, cka in form_rows if a == ax]
            if v:
                arr = np.array(v)
                axis_summary[ax] = {"subspace_cos": float(arr[:, 0].mean()),
                                    "procrustes_cv": float(arr[:, 1].mean()),
                                    "linear_cka": float(arr[:, 2].mean())}
        per_layer.append({
            "layer": layer,
            "r2_mean": float(np.mean([fits[f]["r2"] for f in forms])),
            "r2": {f: fits[f]["r2"] for f in forms},
            "axis_summary": axis_summary,
            "forms": [{"form": f, "axis": a, "subspace_cos": sc, "procrustes_cv": proc, "linear_cka": cka}
                      for f, a, sc, proc, cka in form_rows],
        })

    # --- peak table ---
    print("\n" + "=" * 70)
    print(f"LAYER SWEEP (subspace_cos vs '{ref}', floor={floor:.3f})")
    print("-" * 70)
    print(f"  {'layer':>5}{'R^2':>8}{'script':>10}{'notation':>10}{'language':>10}")
    for pl in per_layer:
        a = pl["axis_summary"]
        def g(ax): return a.get(ax, {}).get("subspace_cos", float("nan"))
        print(f"  {pl['layer']:>5}{pl['r2_mean']:>8.3f}{g('script'):>10.3f}{g('notation'):>10.3f}{g('language'):>10.3f}")
    # best layer by mean over available axes
    def score(pl):
        vals = [pl["axis_summary"][ax]["subspace_cos"] for ax in AXES if ax in pl["axis_summary"]]
        return float(np.mean(vals)) if vals else -1
    best = max(per_layer, key=score)
    print("-" * 70)
    print(f"  peak sharing at layer {best['layer']}  (mean subspace_cos={score(best):.3f}, R^2={best['r2_mean']:.3f})")
    print("=" * 70)

    # --- plot ---
    tag = args.model.split("/")[-1]
    xs = [pl["layer"] for pl in per_layer]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(xs, [pl["r2_mean"] for pl in per_layer], color="#111", lw=2)
    ax1.set_ylabel("helix $R^2$\n(mean over forms)")
    ax1.set_title(f"{tag}  |  pooling={args.pooling}  |  helix geometry & cross-form sharing by layer")
    ax1.grid(alpha=0.25)
    for ax in AXES:
        ys = [pl["axis_summary"].get(ax, {}).get("subspace_cos", np.nan) for pl in per_layer]
        if np.isnan(ys).all():
            continue
        ax2.plot(xs, ys, color=AXIS_COLORS[ax], lw=2, marker="o", ms=3, label=ax)
    ax2.axhline(floor, ls="--", color="#888", lw=1, label=f"random floor ({floor:.02f})")
    ax2.axvline(best["layer"], ls=":", color="#aaa", lw=1)
    ax2.set_ylabel("subspace_cos\n(vs en_digit)")
    ax2.set_xlabel("layer")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"sweep_{tag}_{args.pooling}.png")
    fig.savefig(png, dpi=130)

    out = {"model_revision": model_revision(model, args.model), "model": args.model, "pooling": args.pooling, "reference": ref,
           "n_numbers": len(numbers), "d_model": d_model, "n_layers": n_layers,
           "random_subspace_floor": floor, "best_layer": best["layer"], "per_layer": per_layer}
    js = os.path.join(args.out_dir, f"sweep_{tag}_{args.pooling}.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {js}\n         {png}\n")


if __name__ == "__main__":
    main()

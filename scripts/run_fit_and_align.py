#!/usr/bin/env python
"""Step 1+2: fit the number helix per surface-form, then measure cross-form alignment.

This is the minimal first experiment. For each non-reference form it reports subspace_cos
(primary), procrustes_cv, and linear_CKA vs the en_digit reference -- against a random-subspace
floor and a shuffled-label control -- then a PER-AXIS summary (script/notation/language),
which is the headline H2 contrast (value-driven sharing => script >= notation >= language).

Usage:
    python scripts/run_fit_and_align.py
    python scripts/run_fit_and_align.py --model Qwen/Qwen2.5-7B --forms en_digit en_word es_word
    python scripts/run_fit_and_align.py --layer 14 --max-num 99
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import load_model, extract_form_activations
from src.helix import fit_helix, shuffled_control_r2
from src.alignment import (
    subspace_alignment, linear_cka, random_subspace_floor, orthogonal_procrustes_cv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=C.FORMS or D.DEFAULT_FORMS)
    p.add_argument("--reference", default=None, help="reference form (default: first in --forms)")
    p.add_argument("--layer", default=str(C.LAYER), help="'scan' or an int")
    p.add_argument("--pooling", default=C.POOLING, choices=["last", "mean"])
    p.add_argument("--max-num", type=int, default=max(C.NUMBERS))
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    return p.parse_args()


def choose_layer(acts_by_form, numbers, k_pca, candidate_layers):
    """Pick the layer maximizing mean helix R^2 across forms."""
    best, best_r2 = None, -1e9
    for layer in candidate_layers:
        r2s = [fit_helix(acts_by_form[f][layer], numbers, k_pca=k_pca)["r2"] for f in acts_by_form]
        m = float(np.mean(r2s))
        if m > best_r2:
            best, best_r2 = layer, m
    return best, best_r2


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    numbers = list(range(0, args.max_num + 1))
    forms = list(args.forms)
    ref = args.reference or forms[0]
    if ref not in forms:
        forms = [ref] + forms

    print(f"\nModel: {args.model}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"Device: {device} | d_model: {d_model} | layers: {n_layers}\n")

    # --- extract activations for every form ---
    acts_by_form = {}
    for f in forms:
        print(f"Extracting: {f} ({D.FORMS[f].axis} axis)")
        prompts = D.build_prompts(f, numbers)
        acts_by_form[f] = extract_form_activations(model, tok, device, prompts, pooling=args.pooling)

    # --- choose layer ---
    if args.layer == "scan":
        # scan the middle-to-late layers, where number value tends to be represented
        candidates = list(range(max(1, n_layers // 3), n_layers + 1))
        layer, mean_r2 = choose_layer(acts_by_form, numbers, args.k_pca, candidates)
        print(f"\nChosen layer (max mean R^2): {layer}  (mean R^2={mean_r2:.3f})")
    else:
        layer = int(args.layer)
        print(f"\nUsing layer: {layer}")

    # --- fit helix per form at the chosen layer ---
    fits, r2s, shuf = {}, {}, {}
    for f in forms:
        H = acts_by_form[f][layer]
        fits[f] = fit_helix(H, numbers, k_pca=args.k_pca)
        r2s[f] = fits[f]["r2"]
        shuf[f] = shuffled_control_r2(H, numbers, k_pca=args.k_pca)

    # --- alignment vs reference ---
    ref_dirs = fits[ref]["helix_dirs_model"]
    ref_H = acts_by_form[ref][layer]
    floor = random_subspace_floor(ref_dirs, d_model)

    rows = []
    for f in forms:
        if f == ref:
            continue
        sa = subspace_alignment(ref_dirs, fits[f]["helix_dirs_model"])
        cka = linear_cka(ref_H, acts_by_form[f][layer])
        proc = orthogonal_procrustes_cv(ref_H, acts_by_form[f][layer])
        rows.append((f, D.FORMS[f].axis, sa["mean_cos"], cka, proc))

    # --- report ---
    print("\n" + "=" * 74)
    print(f"HELIX FIT QUALITY (layer {layer})   [R^2; shuffled-label control should be ~0]")
    print("-" * 74)
    print(f"  {'form':<20}{'axis':<10}{'R^2':>10}{'R^2(shuffled)':>18}")
    for f in forms:
        print(f"  {f:<20}{D.FORMS[f].axis:<10}{r2s[f]:>10.3f}{shuf[f]:>18.3f}")

    print("\n" + "=" * 86)
    print(f"CROSS-FORM ALIGNMENT vs reference '{ref}'")
    print(f"  random-subspace floor (subspace_cos): {floor:.3f}   <- subspace_cos must beat this")
    print("-" * 86)
    print(f"  {'form':<20}{'axis':<10}{'subspace_cos':>14}{'procrustes_cv':>16}{'linear_CKA':>14}")
    for f, axis, mc, cka, proc in rows:
        print(f"  {f:<20}{axis:<10}{mc:>14.3f}{proc:>16.3f}{cka:>14.3f}")
    print("=" * 86)

    # --- per-axis summary: the headline H2 contrast (script vs notation vs language) ---
    axis_summary = {}
    for axis in ["script", "notation", "language"]:
        vals = [(mc, proc, cka) for _, a, mc, cka, proc in rows if a == axis]
        if vals:
            arr = np.array(vals)
            axis_summary[axis] = {
                "n": len(vals),
                "subspace_cos": float(arr[:, 0].mean()),
                "procrustes_cv": float(arr[:, 1].mean()),
                "linear_cka": float(arr[:, 2].mean()),
            }
    print("\nPER-AXIS MEANS (H2: value-driven sharing => script >= notation >= language)")
    print("-" * 86)
    print(f"  {'axis':<12}{'n':>4}{'subspace_cos':>16}{'procrustes_cv':>16}{'linear_CKA':>14}")
    for axis, s in axis_summary.items():
        print(f"  {axis:<12}{s['n']:>4}{s['subspace_cos']:>16.3f}{s['procrustes_cv']:>16.3f}{s['linear_cka']:>14.3f}")
    print(f"  {'floor':<12}{'':>4}{floor:>16.3f}")
    print("=" * 86)

    print("\nDecision table (subspace_cos is the primary, transport-relevant metric):")
    print("  subspace_cos >> floor                  -> SAME literal directions: direct patch works (step 3).")
    print("  subspace_cos ~floor BUT procrustes high -> same shape, rotated: transport needs an align map.")
    print("  both low                                -> different geometry OR a tokenization artifact.")
    print("  (CKA is a weak sanity check: high for ANY two number-encoders; necessary, not sufficient.)")
    print("  Signature of value-driven sharing: script-axis forms align more than language-axis forms.\n")

    # --- save ---
    result = {
        "model": args.model, "layer": layer, "reference": ref, "pooling": args.pooling,
        "n_numbers": len(numbers), "d_model": d_model,
        "r2": r2s, "r2_shuffled": shuf, "random_subspace_floor": floor,
        "axis_summary": axis_summary,
        "alignment": [
            {"form": f, "axis": axis, "subspace_mean_cos": mc,
             "procrustes_cv_r2": proc, "linear_cka": cka}
            for f, axis, mc, cka, proc in rows
        ],
    }
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"align_{tag}_L{layer}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

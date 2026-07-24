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
from src.extract import load_model, extract_form_activations, model_revision
from src.helix import (fit_helix, shuffled_control_r2, heldout_r2,
                       select_layer_independent, discovery_evaluation_split)
from src.alignment import (
    subspace_alignment, linear_cka, random_subspace_floor, orthogonal_procrustes_cv,
    canonical_map_cosines, permutation_alignment_null,
)
from src.provenance import stamp, VALIDATED, E_GEOMETRY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=C.FORMS or D.DEFAULT_FORMS)
    p.add_argument("--reference", default=None, help="reference form (default: first in --forms)")
    p.add_argument("--layer", default=str(C.LAYER), help="'scan' or an int")
    p.add_argument("--pooling", default=C.POOLING, choices=["last", "mean", "prompt_last"])
    p.add_argument("--max-num", type=int, default=max(C.NUMBERS))
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--discovery-frac", type=float, default=0.5,
                   help="fraction of numbers reserved for layer discovery (disjoint from evaluation)")
    return p.parse_args()


# NOTE: the old choose_layer() maximized MEAN IN-SAMPLE R^2 across ALL forms, so the target forms
# and the evaluation values both influenced the layer that later scored their own alignment. It is
# replaced by select_layer_independent (en_digit only, discovery values, held-out R^2) -- audit r6 #1.


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    numbers = list(range(0, args.max_num + 1))
    discovery, evaluation = discovery_evaluation_split(numbers, frac=args.discovery_frac, seed=0)
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
        # INDEPENDENT selection: reference form only, discovery values only, held-out R^2, frozen
        # before any other form is scored (audit r6 blocker #1).
        candidates = list(range(max(1, n_layers // 3), n_layers + 1))
        sel = select_layer_independent({L: acts_by_form[ref][L][[numbers.index(v) for v in discovery]]
                                        for L in candidates},
                                       discovery, k_pca=args.k_pca, candidate_layers=candidates)
        layer = sel["selected_layer"]
        layer_selection = {"method": "en_digit_heldout_r2", "discovery_numbers": discovery,
                           "evaluation_numbers": evaluation, "candidate_layers": candidates,
                           "selected_layer": layer,
                           "selection_frozen_before_crossform_evaluation": True,
                           "geometry_uses_discovery_values": False,
                           "per_layer": sel["per_layer"]}
        print(f"\nChosen layer {layer} via INDEPENDENT protocol "
              f"({ref} only, {len(discovery)} discovery values, held-out R^2)")
    else:
        layer = int(args.layer)
        layer_selection = {"method": "cli_argument", "selected_layer": layer,
                           "selection_frozen_before_crossform_evaluation": False}
        print(f"\nUsing layer: {layer}")

    # --- fit helix per form at the chosen layer, on EVALUATION VALUES ONLY (audit r7 blocker #2) ---
    # The layer was chosen because en_digit generalized well on the DISCOVERY values; reusing those
    # values in the reported geometry would not be fully held out. Everything below sees `evaluation`.
    eval_idx = [numbers.index(v) for v in evaluation]
    fits, r2s, r2ho, shuf = {}, {}, {}, {}
    for f in forms:
        H = acts_by_form[f][layer][eval_idx]
        fits[f] = fit_helix(H, evaluation, k_pca=args.k_pca)
        r2s[f] = fits[f]["r2"]
        r2ho[f] = heldout_r2(H, evaluation, k_pca=args.k_pca)[0]   # honest generalization R^2
        shuf[f] = shuffled_control_r2(H, evaluation, k_pca=args.k_pca)

    # --- alignment vs reference ---
    ref_dirs = fits[ref]["helix_dirs_model"]
    ref_H = acts_by_form[ref][layer][eval_idx]
    floor = random_subspace_floor(ref_dirs, d_model)

    rows = []
    for f in forms:
        if f == ref:
            continue
        sa = subspace_alignment(ref_dirs, fits[f]["helix_dirs_model"])
        cka = linear_cka(ref_H, acts_by_form[f][layer][eval_idx])
        proc = orthogonal_procrustes_cv(ref_H, acts_by_form[f][layer][eval_idx])
        coord = canonical_map_cosines(fits[ref], fits[f])         # audit #1: coordinate-level identity
        perm = permutation_alignment_null(ref_H, acts_by_form[f][layer][eval_idx], evaluation, k_pca=args.k_pca)
        rows.append((f, D.FORMS[f].axis, sa["mean_cos"], cka, proc,
                     coord["mean_abs_cos"], coord["mean_signed_cos"], perm["null_q95"]))

    # --- report ---
    print("\n" + "=" * 74)
    print(f"HELIX FIT QUALITY (layer {layer})   [R^2; shuffled-label control should be ~0]")
    print("-" * 74)
    print(f"  {'form':<20}{'axis':<10}{'R^2':>8}{'R^2(held-out)':>15}{'R^2(shuffled)':>15}")
    for f in forms:
        print(f"  {f:<20}{D.FORMS[f].axis:<10}{r2s[f]:>8.3f}{r2ho[f]:>15.3f}{shuf[f]:>15.3f}")
    print("  (held-out R^2 = fit on train numbers, scored on held-out numbers; shuffled should be ~0)")

    print("\n" + "=" * 108)
    print(f"CROSS-FORM ALIGNMENT vs reference '{ref}'")
    print(f"  random floor (subspace_cos): {floor:.3f}; perm_q95 = pipeline-matched null (audit #7) -- subspace_cos must beat it")
    print("  coord_|cos| / coord_scos = coordinate-level identity (audit #1): does each Fourier feature point the SAME way?")
    print("-" * 108)
    print(f"  {'form':<18}{'axis':<9}{'subspace_cos':>13}{'perm_q95':>10}{'coord_|cos|':>12}{'coord_scos':>11}{'procrustes':>11}{'CKA':>8}")
    for f, axis, mc, cka, proc, cabs, cscos, permq in rows:
        print(f"  {f:<18}{axis:<9}{mc:>13.3f}{permq:>10.3f}{cabs:>12.3f}{cscos:>11.3f}{proc:>11.3f}{cka:>8.3f}")
    print("=" * 108)

    # --- per-axis summary: the headline H2 contrast (script vs notation vs language) ---
    axis_summary = {}
    for axis in ["script", "notation", "language"]:
        vals = [(mc, proc, cka, cabs) for _, a, mc, cka, proc, cabs, cscos, permq in rows if a == axis]
        if vals:
            arr = np.array(vals)
            axis_summary[axis] = {
                "n": len(vals),
                "subspace_cos": float(arr[:, 0].mean()),
                "procrustes_cv": float(arr[:, 1].mean()),
                "linear_cka": float(arr[:, 2].mean()),
                "coord_abs_cos": float(arr[:, 3].mean()),
            }
    print("\nPER-AXIS MEANS (H2: value-driven sharing => script >= notation >= language)")
    print("-" * 90)
    print(f"  {'axis':<12}{'n':>4}{'subspace_cos':>16}{'coord_|cos|':>14}{'procrustes_cv':>16}{'linear_CKA':>14}")
    for axis, s in axis_summary.items():
        print(f"  {axis:<12}{s['n']:>4}{s['subspace_cos']:>16.3f}{s['coord_abs_cos']:>14.3f}"
              f"{s['procrustes_cv']:>16.3f}{s['linear_cka']:>14.3f}")
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
        **stamp(C.SCHEMA_VERSION, "align", estimand=E_GEOMETRY, analysis_status=VALIDATED),
        "model_revision": model_revision(model, args.model), "model": args.model, "layer": layer, "reference": ref, "pooling": args.pooling,
        "n_numbers": len(numbers), "d_model": d_model, "layer_selection": layer_selection,
        "layer_discovery_values": discovery, "geometry_fit_values": evaluation,
        "geometry_evaluation_values": evaluation, "geometry_uses_discovery_values": False,
        "r2": r2s, "r2_heldout": r2ho, "r2_shuffled": shuf, "random_subspace_floor": floor,
        # CONFOUNDED: every form vs en_digit, so "language" changes notation AND language.
        # The authoritative H2 contrasts are run_structure.py clean_contrasts (audit r7 geometry #1).
        "axis_summary_confounded_vs_en_digit": axis_summary,
        "authoritative_h2_source": "run_structure.py clean_contrasts",
        "alignment": [
            {"form": f, "axis": axis, "subspace_mean_cos": mc,
             "procrustes_cv_r2": proc, "linear_cka": cka,
             "coord_abs_cos": cabs, "coord_signed_cos": cscos, "perm_null_q95": permq}
            for f, axis, mc, cka, proc, cabs, cscos, permq in rows
        ],
    }
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"align_{tag}_{args.pooling}_L{layer}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

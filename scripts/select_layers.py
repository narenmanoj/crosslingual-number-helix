#!/usr/bin/env python
"""Freeze causal layers via the INDEPENDENT selection protocol (audit r6 blockers #1 + #7).

Production must not use hand-edited layer lists carried over from pre-correction runs. This runs the
approved discovery protocol per model and writes a schema-versioned, commit-stamped layer manifest
that the causal runners consume via --layer-manifest.

Protocol (frozen before any cross-form evaluation):
  * en_digit activations ONLY -- target forms cannot influence the choice;
  * DISCOVERY numbers only, disjoint from the numbers used to evaluate geometry;
  * scored by HELD-OUT R^2 (not in-sample, which rewards overfitting);
  * ties break to the shallower layer.

    python scripts/select_layers.py --models Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407 \
        --out experiments/2026-07-23_<commit>_pilot01/layers.json
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
from src.helix import select_layer_independent
from src.provenance import git_metadata

PROTOCOL = "en_digit_heldout_r2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", required=True, help="path to write layers.json")
    p.add_argument("--discovery", default="10:59", help="discovery numbers (disjoint from evaluation)")
    p.add_argument("--evaluation", default="60:99", help="numbers reserved for geometry evaluation")
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--allow-dirty", action="store_true", default=False)
    return p.parse_args()


def _rng(spec):
    lo, hi = spec.split(":")
    return list(range(int(lo), int(hi) + 1))


def main():
    args = parse_args()
    g = git_metadata()
    if not args.allow_dirty and (g["code_commit"] is None or g["dirty_worktree"]):
        raise SystemExit(f"\nRefusing to freeze layers from a dirty/unknown worktree ({g}).\n"
                         "Commit first, or pass --allow-dirty for a scratch run.\n")
    discovery, evaluation = _rng(args.discovery), _rng(args.evaluation)
    if not set(discovery).isdisjoint(evaluation):
        raise SystemExit("discovery and evaluation number sets must be disjoint")

    models = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        model, tok, device = load_model(name, args.device)
        rev = getattr(model, '_pinned_revision', None)
        n_layers = model.config.num_hidden_layers
        # en_digit ONLY, discovery numbers ONLY -- nothing about other forms is visible here
        acts = extract_form_activations(model, tok, device,
                                        D.build_prompts("en_digit", discovery), pooling="last")
        # exclude the FINAL hidden state: for many HF models it is post-final-norm while the
        # last-block hook is pre-norm, so it cannot be patched consistently (audit r8 non-blocking).
        candidates = list(range(1, n_layers))
        sel = select_layer_independent({L: acts[L] for L in candidates}, discovery,
                                       k_pca=args.k_pca, candidate_layers=candidates)
        best = [d for d in sel["per_layer"] if d["layer"] == sel["selected_layer"]][0]
        print(f"  selected layer {sel['selected_layer']} (held-out R^2={best['heldout_r2']:.3f}) "
              f"from {len(candidates)} candidates")
        models[name] = {"selected_layer": sel["selected_layer"],
                        "heldout_r2": best["heldout_r2"],
                        "model_revision": model_revision(model, name, revision=rev),
                        "per_layer": sel["per_layer"]}
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    out = {"schema_version": C.SCHEMA_VERSION, **g,
           "selection_protocol": PROTOCOL,
           "discovery_numbers": discovery, "evaluation_numbers": evaluation,
           "selection_frozen_before_crossform_evaluation": True,
           "k_pca": args.k_pca, "models": models}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nFrozen layer manifest -> {args.out}")
    for m, v in models.items():
        print(f"  {m}: L{v['selected_layer']}")
    print("\nUse it:  --layer-manifest " + args.out + "\n")


if __name__ == "__main__":
    main()

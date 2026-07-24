#!/usr/bin/env python
"""Does the CAUSAL transport effect peak where the CORRELATIONAL sharing peaks?

⚠ STALE / EXPLORATORY (audit r3 #5) -- NOT FOR HEADLINE. This sweep still uses the pre-audit design:
absolute RECONSTRUCTED targets (not matched-arithmetic delta), a single UN-matched random subspace,
no per-case statistics, and a `subspace/full` normalization whose denominator is itself a
context-mismatched intervention (so it does not cleanly remove propagation-depth effects). Use the
single-layer run_transport.py (delta transport, norm-matched controls, CIs) for causal claims; treat
this only as a qualitative "does the causal effect track the sharing band" picture.

Only subspace + random modes are swept. Cost = cases x layers x 2 forward passes.

Usage:
    python scripts/run_transport_sweep.py --model Qwen/Qwen2.5-7B
    python scripts/run_transport_sweep.py --model CohereLabs/aya-23-8B --layer-stride 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import (load_model, extract_form_activations, _number_token_indices, model_revision,
                         continuation_answer_ids)
from src.helix import fit_helix
from src.provenance import stamp, EXPLORATORY, E_ABSOLUTE
from src.patching import (
    helix_reconstruct, helix_subspace_basis, random_subspace_basis,
    make_patched_vector, patch_residual,
)

MODES = ["subspace", "full", "random"]  # full = layer-normalizer (ceiling intervention at each L)
FORM_COLORS = {"en_digit": "#111111", "devanagari_digit": "#2563eb", "arabic_indic_digit": "#7c3aed",
               "fullwidth_digit": "#0891b2", "en_word": "#059669", "es_word": "#dc2626", "fr_word": "#ea580c"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    # Default = CROSS-FORM only. en_digit (within-form) is excluded: at early layers, patching a
    # bare digit just swaps token identity (a trivial monotone-decaying effect that dominates the
    # y-axis), so it doesn't test transport of the *shared* geometry across the layer sweep.
    p.add_argument("--forms", nargs="+", default=["es_word", "fr_word", "devanagari_digit", "arabic_indic_digit"])
    p.add_argument("--layer-stride", type=int, default=2, help="1 = every layer (slower)")
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--pairs-per-form", type=int, default=20)
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--pooling", default="mean", help="only used to find the matching correlational sweep json")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def forward_logits(model, tok, device, prompt, want_hidden=False):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    hs = None
    if want_hidden:
        hs = [h[0].float().cpu().numpy() for h in out.hidden_states]  # list[L+1] of [seq, d]
    return logits, hs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\nModel: {args.model}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    sweep_layers = list(range(1, n_layers + 1, args.layer_stride))
    print(f"Device: {device} | d_model: {d_model} | sweeping {len(sweep_layers)} layers\n")

    # fit en_digit helix at ALL layers (one extraction), precompute per-layer geometry
    fit_numbers = list(range(0, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers),
                                    pooling="last")
    targets = list(range(0, args.max_sum + 1))
    fitL, Q_L, recon_L = {}, {}, {}
    for L in sweep_layers:
        f = fit_helix(acts[L], fit_numbers, k_pca=args.k_pca)
        fitL[L] = f
        Q_L[L] = helix_subspace_basis(f)
        recon_L[L] = helix_reconstruct(f, targets)
    r = Q_L[sweep_layers[0]].shape[1]
    Q_rand = random_subspace_basis(r, d_model, seed=args.seed)
    ans_ids = continuation_answer_ids(tok, targets)  # audit #2/#9: fail-fast

    # per (form, layer, mode) accumulators
    data = {form: {L: {m: [] for m in MODES} for L in sweep_layers} for form in args.forms}
    for form in args.forms:
        cases = []
        for b in args.addends:
            vals = [a for a in targets if a + b <= args.max_sum]
            cases += [(a, ap, b) for a in vals for ap in vals if a != ap]
        rng.shuffle(cases)
        cases = cases[: args.pairs_per_form]
        print(f"{form}: {len(cases)} cases x {len(sweep_layers)} layers")

        for (a, ap, b) in cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            Lc, hidden = forward_logits(model, tok, device, prompt, want_hidden=True)
            base = Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]
            for L in sweep_layers:
                h_orig = hidden[L][pos]
                target_vec = recon_L[L][ap]
                for mode in MODES:
                    Qm = Q_L[L] if mode == "subspace" else Q_rand
                    new_h = make_patched_vector(h_orig, target_vec, Q=Qm, mode=mode)
                    handle = patch_residual(model, L - 1, pos,
                                            torch.tensor(new_h, dtype=torch.float32, device=device))
                    try:
                        Lp, _ = forward_logits(model, tok, device, prompt)
                    finally:
                        handle.remove()
                    shift = (Lp[ans_ids[ap + b]] - Lp[ans_ids[a + b]]) - base
                    data[form][L][mode].append(float(shift))

    # aggregate raw curves
    curves = {form: {m: [float(np.mean(data[form][L][m])) if data[form][L][m] else np.nan
                         for L in sweep_layers] for m in MODES}
              for form in args.forms}

    # LAYER-NORMALIZED metric: fraction of the full-patch effect carried by the helix subspace.
    # Raw mean_shift is confounded -- an intervention at an earlier layer propagates through more
    # layers and yields a bigger logit shift regardless of sharing. Dividing by the full-patch
    # effect at the SAME layer (the ceiling any intervention there can reach) cancels that
    # amplification, leaving 'how much of the transportable value the helix subspace carries'.
    MIN_FULL = 0.3
    frac = {}
    for form in args.forms:
        frac[form] = []
        for i in range(len(sweep_layers)):
            s, f = curves[form]["subspace"][i], curves[form]["full"][i]
            frac[form].append(s / f if (f is not None and not np.isnan(f) and f > MIN_FULL) else np.nan)

    # correlational overlay
    tag = args.model.split("/")[-1]
    corr = None
    corr_path = os.path.join(args.out_dir, f"sweep_{tag}_{args.pooling}.json")
    if os.path.exists(corr_path):
        with open(corr_path) as fh:
            corr = json.load(fh)

    npanel = 2 if corr else 1
    fig, axes = plt.subplots(npanel, 1, figsize=(8, 3.4 * npanel), sharex=True)
    axes = np.atleast_1d(axes)
    if corr:
        cl = [pl["layer"] for pl in corr["per_layer"]]
        for ax_name in ["script", "language"]:
            ys = [pl["axis_summary"].get(ax_name, {}).get("subspace_cos", np.nan) for pl in corr["per_layer"]]
            axes[0].plot(cl, ys, lw=2, marker="o", ms=2, label=f"{ax_name} (corr.)")
        axes[0].set_ylabel("subspace_cos\n(correlational)")
        axes[0].set_title(f"{tag}: correlational sharing vs helix-specific causal transport")
        axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axc = axes[-1]
    for form in args.forms:
        axc.plot(sweep_layers, frac[form], lw=2, marker="o", ms=3,
                 color=FORM_COLORS.get(form, None), label=form)
    axc.axhline(0, color="#ccc", lw=0.8)
    axc.set_ylabel("subspace / full\n(helix-specific fraction)")
    axc.set_xlabel("layer")
    axc.grid(alpha=0.25); axc.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"transport_sweep_{tag}.png")
    fig.savefig(png, dpi=130)

    out = {**stamp(C.SCHEMA_VERSION, "transport_sweep", estimand=E_ABSOLUTE, analysis_status=EXPLORATORY), "stale": True, "stale_reason": "reconstructed targets; unmatched single random control; no per-case stats (audit r3 #5)",
           "model_revision": model_revision(model, args.model), "model": args.model, "layers": sweep_layers, "r": r, "forms": args.forms,
           "curves": curves, "frac_subspace_over_full": frac,
           "fit_r2": {L: fitL[L]["r2"] for L in sweep_layers}}
    js = os.path.join(args.out_dir, f"transport_sweep_{tag}.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)

    # peak table (layer-normalized)
    print("\n" + "=" * 72)
    print("HELIX-SPECIFIC TRANSPORT PEAK  (subspace/full, layer-normalized)  per form")
    print("-" * 72)
    for form in args.forms:
        ys = np.array(frac[form], dtype=float)
        raw_pk = sweep_layers[int(np.nanargmax(curves[form]["subspace"]))]
        if np.isnan(ys).all():
            print(f"  {form:<20} (no layers with reliable full-patch effect)")
            continue
        pk = sweep_layers[int(np.nanargmax(ys))]
        print(f"  {form:<20} norm-peak L{pk:<3} frac={np.nanmax(ys):.2f}   (raw subspace peak was L{raw_pk})")
    print("=" * 72)
    print(f"Saved -> {js}\n         {png}\n")


if __name__ == "__main__":
    main()

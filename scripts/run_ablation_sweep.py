#!/usr/bin/env python
"""Ablation-LAYER sweep: WHERE (if anywhere) does the model rely on the shared helix subspace?

run_necessity.py ablates at a single layer (the representational sharing peak). But a model may
*read* the number value at a different depth than where the value is most *shared*, so a single-
layer null (as on Mistral-Nemo @ L22) is ambiguous: genuine redundancy, or just the wrong layer?

This sweep resolves it. At every layer L it mean-ablates the en_digit-helix subspace from a source
number inside "a + b = " and measures the arithmetic-accuracy DROP, vs a multi-seed random-subspace
ablation. Read the per-form curve of Delta = acc(random-ablate) - acc(helix-ablate) vs layer:
  - a PEAK at some layer  => the model DOES rely on the shared subspace, read at that depth
                             (necessity is real, just localized -- report where each model reads)
  - FLAT ~0 at all layers => genuine redundancy: value is distributed, subspace is bypassable

Overlays the correlational sweep_{tag}_mean.json if present, so you can see whether the necessity
peak aligns with the representational sharing peak.

Usage:
    python scripts/run_ablation_sweep.py --model Qwen/Qwen2.5-7B
    python scripts/run_ablation_sweep.py --model mistralai/Mistral-Nemo-Base-2407 --layer-stride 1
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
from src.extract import load_model, extract_form_activations, _number_token_indices
from src.helix import fit_helix
from src.patching import (
    helix_subspace_basis, random_subspace_basis, make_patched_vector, patch_residual,
)

FORM_COLORS = {"en_digit": "#111111", "devanagari_digit": "#2563eb", "arabic_indic_digit": "#7c3aed",
               "fullwidth_digit": "#0891b2", "en_word": "#059669", "es_word": "#dc2626",
               "fr_word": "#ea580c", "de_word": "#a16207"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "devanagari_digit", "es_word"])
    p.add_argument("--layer-stride", type=int, default=2, help="1 = every layer (slower)")
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--pooling", default="mean", help="only to locate the matching correlational sweep json")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def answer_token_id(tok, v):
    for s in (f"{v}", f" {v}"):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tok.encode(f"{v}", add_special_tokens=False)[0]


@torch.no_grad()
def forward_logits(model, tok, device, prompt, want_hidden=False):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    hs = [h[0].float().cpu().numpy() for h in out.hidden_states] if want_hidden else None
    return logits, hs


@torch.no_grad()
def patched_correct(model, tok, device, prompt, hook_layer, pos, new_h, ans_ids, max_sum, target):
    handle = patch_residual(model, hook_layer, pos, torch.tensor(new_h, dtype=torch.float32, device=device))
    try:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        logits = model(**enc).logits[0, -1, :].float().cpu().numpy()
    finally:
        handle.remove()
    pred = int(np.argmax([logits[ans_ids[v]] for v in range(0, max_sum + 1)]))
    return int(pred == target)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nModel: {args.model}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    sweep_layers = list(range(1, n_layers + 1, args.layer_stride))

    fit_numbers = list(range(0, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers),
                                   pooling="last")
    fitL, Q_L, mean_L = {}, {}, {}
    for L in sweep_layers:
        f = fit_helix(acts[L], fit_numbers, k_pca=args.k_pca)
        fitL[L], Q_L[L], mean_L[L] = f, helix_subspace_basis(f), f["mean"]
    r = Q_L[sweep_layers[0]].shape[1]
    Q_rands = [random_subspace_basis(r, d_model, seed=args.seed + i) for i in range(args.n_seeds)]
    ans_ids = {v: answer_token_id(tok, v) for v in range(0, args.max_sum + 1)}
    print(f"Device: {device} | d_model {d_model} | sweeping {len(sweep_layers)} layers | r={r}\n")

    curves = {}   # form -> {"clean": acc, "helix": [per layer], "rand": [per layer]}
    for form in args.forms:
        cases = [(a, b) for b in args.addends for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
        clean_ok = 0
        helix_ok = {L: 0 for L in sweep_layers}
        rand_ok = {L: 0 for L in sweep_layers}
        n = 0
        for (a, b) in cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            n += 1
            Lc, hidden = forward_logits(model, tok, device, prompt, want_hidden=True)
            clean_ok += int(int(np.argmax([Lc[ans_ids[v]] for v in range(0, args.max_sum + 1)])) == a + b)
            for L in sweep_layers:
                h_orig = hidden[L][pos]
                h_hel = make_patched_vector(h_orig, mean_L[L], Q=Q_L[L], mode="subspace")
                helix_ok[L] += patched_correct(model, tok, device, prompt, L - 1, pos, h_hel, ans_ids, args.max_sum, a + b)
                rr = [patched_correct(model, tok, device, prompt, L - 1, pos,
                                      make_patched_vector(h_orig, mean_L[L], Q=Qr, mode="subspace"),
                                      ans_ids, args.max_sum, a + b) for Qr in Q_rands]
                rand_ok[L] += int(round(np.mean(rr)))  # per-case majority; averaged below
        n = max(n, 1)
        curves[form] = {
            "clean": clean_ok / n,
            "helix": [helix_ok[L] / n for L in sweep_layers],
            "rand": [rand_ok[L] / n for L in sweep_layers],
        }
        print(f"  done {form} (clean_acc={curves[form]['clean']:.2f})")

    # necessity Delta per layer = acc(random-ablate) - acc(helix-ablate)
    delta = {f: [curves[f]["rand"][i] - curves[f]["helix"][i] for i in range(len(sweep_layers))]
             for f in args.forms}

    # correlational overlay
    tag = args.model.split("/")[-1]
    corr_path = os.path.join(args.out_dir, f"sweep_{tag}_{args.pooling}.json")
    corr = json.load(open(corr_path)) if os.path.exists(corr_path) else None

    npanel = 2 if corr else 1
    fig, axes = plt.subplots(npanel, 1, figsize=(8, 3.4 * npanel), sharex=True)
    axes = np.atleast_1d(axes)
    if corr:
        cl = [pl["layer"] for pl in corr["per_layer"]]
        for ax_name in ["script", "language"]:
            ys = [pl["axis_summary"].get(ax_name, {}).get("subspace_cos", np.nan) for pl in corr["per_layer"]]
            axes[0].plot(cl, ys, lw=2, marker="o", ms=2, label=f"{ax_name} (corr.)")
        axes[0].set_ylabel("subspace_cos\n(correlational)")
        axes[0].set_title(f"{tag}: sharing peak vs where the model RELIES on the shared subspace")
        axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axc = axes[-1]
    for f in args.forms:
        axc.plot(sweep_layers, delta[f], lw=2, marker="o", ms=3,
                 color=FORM_COLORS.get(f), label=f)
    axc.axhline(0, color="#ccc", lw=0.8)
    axc.set_ylabel("necessity Δ\nacc(rand-abl) − acc(helix-abl)")
    axc.set_xlabel("ablation layer")
    axc.grid(alpha=0.25); axc.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"ablation_sweep_{tag}.png")
    fig.savefig(png, dpi=130)

    out = {"model": args.model, "layers": sweep_layers, "r": r, "forms": args.forms,
           "curves": curves, "necessity_delta": delta}
    js = os.path.join(args.out_dir, f"ablation_sweep_{tag}.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)

    # peak table
    print("\n" + "=" * 66)
    print("NECESSITY PEAK per form  (max Δ over layers; ~0 everywhere => redundant)")
    print("-" * 66)
    for f in args.forms:
        d = np.array(delta[f])
        pk = sweep_layers[int(np.argmax(d))]
        print(f"  {f:<20} clean={curves[f]['clean']:.2f}  peak-Δ={d.max():.2f} @ L{pk:<3}  (mean Δ={d.mean():.2f})")
    print("=" * 66)
    print(f"Saved -> {js}\n         {png}\n")


if __name__ == "__main__":
    main()

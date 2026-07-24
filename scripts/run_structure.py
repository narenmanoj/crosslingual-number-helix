#!/usr/bin/env python
"""Review items #6 + #7 in one model load.

(#6) PAIRWISE form x form alignment matrix -- the full compatibility structure, not just
     everything-vs-en_digit. Shows e.g. whether es/fr/de number-words cluster, whether the
     digit-scripts cluster, etc. Saved as a heatmap.

(#7) GEOMETRY <-> BEHAVIOR link -- does a form's representational sharing with en_digit predict
     how reliably the model does ARITHMETIC in that form? Correlate per-form subspace_cos (vs
     en_digit) with per-form single-digit addition accuracy. Optionally also vs the necessity
     ablation-Delta if a necessity_*.json is present. Base model needed for the arithmetic readout.

Usage:
    python scripts/run_structure.py --model Qwen/Qwen2.5-7B --layer 14
    python scripts/run_structure.py --model meta-llama/Llama-3.1-8B --layer <sweep-peak>
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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import (load_model, extract_form_activations, _number_token_indices, model_revision,
                         continuation_answer_ids)
from src.helix import fit_helix
from src.alignment import subspace_alignment, random_subspace_floor, subspace_overlap
from src.provenance import stamp, VALIDATED, E_GEOMETRY

AXIS_COLORS = {"script": "#2563eb", "notation": "#059669", "language": "#dc2626"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+",
                   default=["en_digit", "devanagari_digit", "arabic_indic_digit", "fullwidth_digit",
                            "en_word", "es_word", "fr_word", "de_word"])
    p.add_argument("--layer", default="scan", help="'scan' (max mean R^2) or an int")
    p.add_argument("--pooling", default="mean", choices=["last", "mean", "prompt_last"])
    p.add_argument("--max-num", type=int, default=99)
    p.add_argument("--acc-addends", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    return p.parse_args()


@torch.no_grad()
def arithmetic_accuracy(model, tok, device, form, addends, max_sum, ans_ids):
    """Fraction of single-token-answer additions the model gets right, with the number in `form`."""
    cases = [(a, b) for b in addends for a in range(0, max_sum + 1) if a + b <= max_sum]
    ok = 0
    n = 0
    for a, b in cases:
        a_str = D.FORMS[form].render(a)
        prompt = f"{a_str} + {b} = "
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        logits = model(**enc).logits[0, -1, :].float().cpu().numpy()
        pred = int(np.argmax([logits[ans_ids[v]] for v in range(0, max_sum + 1)]))
        ok += int(pred == a + b)
        n += 1
    return ok / max(n, 1), n


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    forms = list(args.forms)

    print(f"\nModel: {args.model} | pooling {args.pooling}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    numbers = list(range(0, args.max_num + 1))

    acts = {}
    for f in forms:
        print(f"Extracting: {f}")
        acts[f] = extract_form_activations(model, tok, device, D.build_prompts(f, numbers), pooling=args.pooling)

    # choose layer
    if args.layer == "scan":
        cands = list(range(max(1, n_layers // 3), n_layers + 1))
        best, bestr2 = cands[0], -1e9
        for L in cands:
            m = float(np.mean([fit_helix(acts[f][L], numbers, k_pca=args.k_pca)["r2"] for f in forms]))
            if m > bestr2:
                best, bestr2 = L, m
        layer = best
        print(f"\nChosen layer {layer} (mean R^2={bestr2:.3f})")
    else:
        layer = int(args.layer)

    fits = {f: fit_helix(acts[f][layer], numbers, k_pca=args.k_pca) for f in forms}
    floor = random_subspace_floor(fits[forms[0]]["helix_dirs_model"], d_model)

    # ---------- (#6) pairwise subspace_cos matrix ----------
    N = len(forms)
    M = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            c = subspace_alignment(fits[forms[i]]["helix_dirs_model"], fits[forms[j]]["helix_dirs_model"])["mean_cos"]
            M[i, j] = M[j, i] = c

    # ---------- (#7) arithmetic accuracy per form ----------
    ans_ids = continuation_answer_ids(tok, range(0, args.max_sum + 1))  # audit #2/#9: fail-fast
    acc = {}
    for f in forms:
        a, n = arithmetic_accuracy(model, tok, device, f, args.acc_addends, args.max_sum, ans_ids)
        acc[f] = a
    ref = "en_digit"
    ref_i = forms.index(ref)
    share = {f: float(M[forms.index(f), ref_i]) for f in forms}  # subspace_cos vs en_digit

    # correlation over NON-reference forms (en_digit's self-cos=1 would anchor trivially)
    corr_forms = [f for f in forms if f != ref]
    xs = np.array([share[f] for f in corr_forms])
    ys = np.array([acc[f] for f in corr_forms])
    pear = pearsonr(xs, ys)
    spear = spearmanr(xs, ys)

    # optional necessity-Delta correlation (necessity_{tag}_L{layer}_{pos}.json; take newest match)
    tag = args.model.split("/")[-1]
    import glob
    nec_files = sorted(glob.glob(os.path.join(args.out_dir, f"necessity_{tag}_L{layer}*.json")))
    nec_corr = None
    if nec_files:
        nec = json.load(open(nec_files[-1]))["ablation"]
        common = [f for f in corr_forms if f in nec and "controls" in nec[f]]
        if len(common) >= 3:
            nx = np.array([share[f] for f in common])
            nd = np.array([nec[f]["controls"]["random"]["acc_mean"] - nec[f]["acc_helix_ablate"] for f in common])
            nec_corr = {"forms": common, "pearson_r": float(pearsonr(nx, nd)[0])}

    # ---------- CLEAN CONTRASTS (fix the reference-form confound) ----------
    # "language" must compare number-WORDS to number-WORDS (en_word <-> foreign), not en_digit <-> words.
    idx = {f: i for i, f in enumerate(forms)}

    def mean_cell(ref, targets):
        vals = [M[idx[ref], idx[t]] for t in targets if t in idx and t != ref]
        return float(np.mean(vals)) if vals else float("nan")

    # rank-aware overlap (audit r3 #6): mean_cos ignores unmatched dimensions, so a rank-deficient
    # form can score high. Report rank-penalized overlap ||Qa^T Qb||_F^2 / max(r_a, r_b) too.
    def mean_overlap(ref, targets):
        vals = [subspace_overlap(fits[ref]["helix_dirs_model"], fits[t]["helix_dirs_model"])["overlap_rank_penalized"]
                for t in targets if t in idx and t != ref]
        return float(np.mean(vals)) if vals else float("nan")

    form_ranks = {f: int(subspace_overlap(fits[f]["helix_dirs_model"], fits[f]["helix_dirs_model"])["rank_a"])
                  for f in forms}

    scripts = [f for f in forms if D.FORMS[f].axis == "script"]        # en_digit + digit-scripts
    notation = [f for f in forms if D.FORMS[f].axis == "notation"]     # en_word
    langs = [f for f in forms if D.FORMS[f].axis == "language"]        # es/fr/de words
    clean_contrasts = {
        "script (en_digit <-> digit-scripts)": mean_cell("en_digit", [f for f in scripts if f != "en_digit"]),
        "notation (en_digit <-> en_word)": mean_cell("en_digit", notation),
        "language (en_word <-> foreign words)": mean_cell("en_word", langs) if "en_word" in idx else float("nan"),
    }
    clean_contrasts_rank_penalized = {
        "script": mean_overlap("en_digit", [f for f in scripts if f != "en_digit"]),
        "notation": mean_overlap("en_digit", notation),
        "language": mean_overlap("en_word", langs) if "en_word" in idx else float("nan"),
    }
    # direct word-to-word cells reviewers asked to see
    def cell(a, b):
        return float(M[idx[a], idx[b]]) if a in idx and b in idx else float("nan")
    word_cells = {f"{a}<->{b}": cell(a, b) for (a, b) in
                  [("en_word", "es_word"), ("en_word", "fr_word"), ("es_word", "fr_word"),
                   ("es_word", "de_word"), ("fr_word", "de_word")]}

    # ---------- report ----------
    print("\n" + "=" * 78)
    print(f"(#6) PAIRWISE subspace_cos matrix @ L{layer}   (floor={floor:.3f})")
    print("-" * 78)
    print("       " + "".join(f"{f[:7]:>9}" for f in forms))
    for i, f in enumerate(forms):
        print(f"  {f[:7]:>7}" + "".join(f"{M[i, j]:>9.2f}" for j in range(N)))
    print("=" * 78)
    print("\nCLEAN CONTRASTS (correct reference per axis -- avoids the en_digit reference confound)")
    print("-" * 78)
    for k, v in clean_contrasts.items():
        print(f"  {k:<44}{v:>8.3f}")
    print("  word-to-word cells: " + "  ".join(f"{k}={v:.2f}" for k, v in word_cells.items()))
    print("=" * 78)
    print(f"\n(#7) GEOMETRY <-> BEHAVIOR   (subspace_cos vs en_digit  ~  arithmetic accuracy)")
    print("-" * 78)
    print(f"  {'form':<20}{'axis':<10}{'share(cos)':>12}{'arith_acc':>11}")
    for f in forms:
        print(f"  {f:<20}{D.FORMS[f].axis:<10}{share[f]:>12.3f}{acc[f]:>11.2f}")
    print("-" * 78)
    print(f"  Pearson r = {pear[0]:.3f} (p={pear[1]:.3f}) | Spearman r = {spear[0]:.3f}  (n={len(corr_forms)} non-ref forms)")
    if nec_corr:
        print(f"  subspace_cos vs necessity-Delta: Pearson r = {nec_corr['pearson_r']:.3f} (n={len(nec_corr['forms'])})")
    print("=" * 78)

    # ---------- plots ----------
    fig, ax = plt.subplots(figsize=(1.1 * N + 1, 1.0 * N))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(N)); ax.set_xticklabels(forms, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(forms, fontsize=8)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title(f"{tag}: pairwise number-helix subspace_cos (L{layer})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    hm = os.path.join(args.out_dir, f"pairwise_{tag}_L{layer}.png")
    fig.savefig(hm, dpi=130)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for f in corr_forms:
        ax2.scatter(share[f], acc[f], color=AXIS_COLORS[D.FORMS[f].axis], s=60)
        ax2.annotate(f, (share[f], acc[f]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax2.axvline(floor, ls="--", color="#aaa", lw=1, label=f"floor {floor:.02f}")
    ax2.set_xlabel("subspace_cos vs en_digit (representational sharing)")
    ax2.set_ylabel("single-digit arithmetic accuracy (behavioral)")
    ax2.set_title(f"{tag}: geometry predicts numeracy?  r={pear[0]:.2f}")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    fig2.tight_layout()
    sc = os.path.join(args.out_dir, f"geombehav_{tag}_L{layer}.png")
    fig2.savefig(sc, dpi=130)

    out = {**stamp(C.SCHEMA_VERSION, "structure", estimand=E_GEOMETRY, analysis_status=VALIDATED),
           "model_revision": model_revision(model, args.model), "model": args.model, "layer": layer, "pooling": args.pooling, "floor": floor,
           "forms": forms, "pairwise_subspace_cos": M.tolist(), "form_ranks": form_ranks,
           # audit #8: this file is the AUTHORITATIVE H2 source. Its clean_contrasts use the correct
           # reference per axis; the everything-vs-en_digit axis_summary in align_*.json is confounded.
           "contrast_definition": {"script": "en_digit_vs_other_digit_scripts",
                                    "notation": "en_digit_vs_en_word",
                                    "language": "en_word_vs_foreign_words"},
           "clean_contrasts": clean_contrasts,
           "clean_contrasts_rank_penalized": clean_contrasts_rank_penalized, "word_to_word_cells": word_cells,
           "share_vs_en_digit": share, "arithmetic_acc": acc,
           "geometry_behavior": {"pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                                 "spearman_r": float(spear[0]), "n": len(corr_forms)},
           "necessity_corr": nec_corr}
    js = os.path.join(args.out_dir, f"structure_{tag}_L{layer}.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {js}\n         {hm}\n         {sc}\n")


if __name__ == "__main__":
    main()

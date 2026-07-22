#!/usr/bin/env python
"""Step 3b: NECESSITY with matched controls + multi-token interventions (review-hardened).

Sufficiency (run_transport.py) shows the circuit *can* read an injected direction. This asks
whether the model *relies* on the shared subspace, with the controls a skeptic demands:

  (A) ABLATION. Mean-ablate the en_digit-helix subspace from a source number in "a + b = " and
      measure the restricted digit-choice accuracy drop, vs THREE nulls (all multi-seed, per-seed
      curves kept, no rounding; removed L2 energy reported so the nulls remove comparable energy):
        - random        : Haar-random subspace in the full residual (weak null)
        - cov_matched   : random subspace inside the top-k activation PCA (energy-matched null)
        - shuf_fourier  : helix fit through the SAME pipeline on SHUFFLED labels (structured null)
      Intervention position is configurable (--intervention-pos): the number's last token, its whole
      span, or the token just after it (words span several tokens). Token count is recorded.

  (B) MATCHED-SOURCE INTERCHANGE. Patch the model's real en_digit activation for a', subspace-only,
      vs a NORM-MATCHED random subspace (so "does it steer" isn't confounded by "it perturbs less").

Readout = restricted digit-choice accuracy (argmax over the ten 0..9 answer tokens), single-digit
sums -- a controlled proxy, not unrestricted arithmetic. Base model required.

Usage:
    python scripts/run_necessity.py --model Qwen/Qwen2.5-7B --layer 14 --intervention-pos last
    python scripts/run_necessity.py --model Qwen/Qwen2.5-7B --layer 14 --intervention-pos span
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import load_model, extract_form_activations, _number_token_indices, model_revision
from src.helix import fit_helix
from src.patching import (
    helix_subspace_basis, random_subspace_basis, covariance_matched_basis, shuffled_fourier_basis,
    make_patched_vector, matched_injection, subspace_energy, patch_residual_multi,
)

CONTROLS = ["random", "cov_matched", "shuf_fourier"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "devanagari_digit", "es_word", "fr_word"])
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--intervention-pos", default="last", choices=["last", "span", "after"])
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--pairs-per-form", type=int, default=40)
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--n-seeds", type=int, default=8, help="control-null seeds; more = tighter null bands")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def answer_token_id(tok, v):
    for s in (f"{v}", f" {v}"):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tok.encode(f"{v}", add_special_tokens=False)[-1]  # LAST token = the digit (skip SP metaspace)


@torch.no_grad()
def forward(model, tok, device, prompt, layer=None, want_hidden=False):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    h = out.hidden_states[layer][0].float().cpu().numpy() if want_hidden else None
    return logits, h, enc["input_ids"].shape[1]


@torch.no_grad()
def patched_logits(model, tok, device, prompt, hook_layer, pos_to_vec):
    tvecs = {p: torch.tensor(v, dtype=torch.float32, device=device) for p, v in pos_to_vec.items()}
    handle = patch_residual_multi(model, hook_layer, tvecs)
    try:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        return model(**enc).logits[0, -1, :].float().cpu().numpy()
    finally:
        handle.remove()


def intervention_positions(idxs, mode, seq_len):
    if mode == "last":
        return [idxs[-1]]
    if mode == "span":
        return list(idxs)
    if mode == "after":
        return [min(idxs[-1] + 1, seq_len - 1)]
    raise ValueError(mode)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    hook_layer = args.layer - 1

    print(f"\nModel: {args.model} | layer {args.layer} | intervention-pos {args.intervention_pos}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size

    fit_numbers = list(range(0, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers), pooling="last")
    HL = acts[args.layer]
    fit = fit_helix(HL, fit_numbers, k_pca=args.k_pca)
    Q = helix_subspace_basis(fit)
    r = Q.shape[1]
    mean_vec = fit["mean"]
    en_real = HL
    # control-subspace banks (multi-seed)
    ctrl_bases = {
        "random": [random_subspace_basis(r, d_model, seed=args.seed + i) for i in range(args.n_seeds)],
        "cov_matched": [covariance_matched_basis(HL, r, seed=args.seed + i) for i in range(args.n_seeds)],
        "shuf_fourier": [shuffled_fourier_basis(HL, fit_numbers, k_pca=args.k_pca, seed=args.seed + i) for i in range(args.n_seeds)],
    }
    ans_ids = {v: answer_token_id(tok, v) for v in range(0, args.max_sum + 1)}
    print(f"en_digit helix @ L{args.layer}: R^2={fit['r2']:.3f}, r={r}\n")

    def argmax_ans(logits):
        return int(np.argmax([logits[ans_ids[v]] for v in range(0, args.max_sum + 1)]))

    rng = np.random.default_rng(args.seed)
    ablation, interchange = {}, {}
    for form in args.forms:
        # ---------- (A) ABLATION ----------
        ab_cases = [(a, b) for b in args.addends for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
        clean_ok, helix_ok = 0, 0
        ctrl_ok = {c: np.zeros(args.n_seeds) for c in CONTROLS}
        energy = {"helix": []} | {c: [] for c in CONTROLS}
        # per-case correctness (0/1 for clean+helix; per-case mean-over-seeds 0..1 for each null)
        clean_case, helix_case = [], []
        ctrl_case = {c: [] for c in CONTROLS}
        tok_counts, n = [], 0
        for (a, b) in ab_cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                idxs = _number_token_indices(tok, prompt, a_str)
            except ValueError:
                continue
            Lc, hidden, seq_len = forward(model, tok, device, prompt, layer=args.layer, want_hidden=True)
            positions = intervention_positions(idxs, args.intervention_pos, seq_len)
            n += 1
            tok_counts.append(len(idxs))
            cc = int(argmax_ans(Lc) == a + b); clean_ok += cc; clean_case.append(cc)
            # helix ablation (mean-ablate the helix subspace at each chosen position)
            p2v = {p: make_patched_vector(hidden[p], mean_vec, Q=Q, mode="subspace") for p in positions}
            hc = int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, p2v)) == a + b)
            helix_ok += hc; helix_case.append(hc)
            energy["helix"].append(subspace_energy(Q, hidden[positions[-1]], mean_vec))
            for c in CONTROLS:
                seed_correct = []
                for si, Qc in enumerate(ctrl_bases[c]):
                    p2v = {p: make_patched_vector(hidden[p], mean_vec, Q=Qc, mode="subspace") for p in positions}
                    sc = int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, p2v)) == a + b)
                    ctrl_ok[c][si] += sc; seed_correct.append(sc)
                ctrl_case[c].append(float(np.mean(seed_correct)))
                energy[c].append(subspace_energy(ctrl_bases[c][0], hidden[positions[-1]], mean_vec))
        n = max(n, 1)
        ablation[form] = {
            "axis": D.FORMS[form].axis, "n": n,
            "mean_tok_count": float(np.mean(tok_counts)),
            "clean_acc": clean_ok / n,
            "acc_helix_ablate": helix_ok / n,
            "controls": {c: {"acc_mean": float((ctrl_ok[c] / n).mean()),
                             "acc_std": float((ctrl_ok[c] / n).std()),
                             "delta_vs_helix": float((ctrl_ok[c] / n).mean() - helix_ok / n)}
                         for c in CONTROLS},
            "removed_energy": {k: float(np.mean(v)) for k, v in energy.items()},
            # per-case arrays for bootstrap CIs + paired tests (helix vs each null, per case)
            "per_case": {"clean": clean_case, "helix": helix_case, "controls": ctrl_case},
        }

        # ---------- (B) MATCHED-SOURCE INTERCHANGE ----------
        ic_cases = []
        for b in args.addends:
            vals = [a for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
            ic_cases += [(a, ap, b) for a in vals for ap in vals if a != ap]
        rng.shuffle(ic_cases)
        ic_cases = ic_cases[: args.pairs_per_form]
        sub_shift, matched_shift = [], []
        for (a, ap, b) in ic_cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            Lc, hidden, _ = forward(model, tok, device, prompt, layer=args.layer, want_hidden=True)
            base = Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]
            h_orig, target = hidden[pos], en_real[ap]
            h_sub = make_patched_vector(h_orig, target, Q=Q, mode="subspace")
            Ls = patched_logits(model, tok, device, prompt, hook_layer, {pos: h_sub})
            sub_shift.append(float((Ls[ans_ids[ap + b]] - Ls[ans_ids[a + b]]) - base))
            # NORM-MATCHED random control: same perturbation magnitude, random directions
            h_mr = matched_injection(h_orig, target, Q_signal=Q, Q_control=ctrl_bases["random"][0])
            Lm = patched_logits(model, tok, device, prompt, hook_layer, {pos: h_mr})
            matched_shift.append(float((Lm[ans_ids[ap + b]] - Lm[ans_ids[a + b]]) - base))
        interchange[form] = {
            "axis": D.FORMS[form].axis, "n": len(sub_shift),
            "subspace_shift": float(np.mean(sub_shift)) if sub_shift else float("nan"),
            "matched_random_shift": float(np.mean(matched_shift)) if matched_shift else float("nan"),
            # per-case (aligned) for bootstrap CI + paired subspace-vs-matched-random test
            "per_case": {"subspace": sub_shift, "matched_random": matched_shift},
        }
        print(f"  done {form}")

    # ---------- report ----------
    print("\n" + "=" * 96)
    print(f"(A) ABLATION necessity  (pos={args.intervention_pos})  -- helix-ablate acc << every null (~clean) => used")
    print("-" * 96)
    hdr = f"  {'form':<20}{'axis':<9}{'tok':>4}{'clean':>7}{'helix':>7}"
    for c in CONTROLS:
        hdr += f"{c[:9]:>11}"
    hdr += f"{'E:helix/cov':>13}"
    print(hdr)
    for form in args.forms:
        A = ablation[form]
        row = f"  {form:<20}{A['axis']:<9}{A['mean_tok_count']:>4.1f}{A['clean_acc']:>7.2f}{A['acc_helix_ablate']:>7.2f}"
        for c in CONTROLS:
            row += f"{A['controls'][c]['acc_mean']:>8.2f}±{A['controls'][c]['acc_std']:.2f}"
        row += f"{A['removed_energy']['helix']:>7.1f}/{A['removed_energy']['cov_matched']:.1f}"
        print(row)
    print("  (each null column = restricted-digit-choice acc after ablating that subspace, mean±std over seeds)")
    print("=" * 96)
    print(f"\n(B) MATCHED-SOURCE INTERCHANGE  -- subspace_shift >> norm-matched-random_shift => real, not energy artifact")
    print("-" * 96)
    print(f"  {'form':<20}{'axis':<9}{'subspace_shift':>16}{'matched_random':>16}")
    for form in args.forms:
        I = interchange[form]
        print(f"  {form:<20}{I['axis']:<9}{I['subspace_shift']:>16.3f}{I['matched_random_shift']:>16.3f}")
    print("=" * 96 + "\n")

    out = {"model_revision": model_revision(model, args.model), "layer": args.layer,
           "intervention_pos": args.intervention_pos, "r": r, "n_seeds": args.n_seeds,
           "readout": "restricted_digit_choice_accuracy (argmax over 0..9, single-digit sums)",
           "fit_r2": fit["r2"], "ablation": ablation, "interchange": interchange}
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"necessity_{tag}_L{args.layer}_{args.intervention_pos}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

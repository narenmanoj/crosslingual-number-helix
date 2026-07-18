#!/usr/bin/env python
"""Step 3b: NECESSITY, not just sufficiency (review item #4).

The plain transport experiment (run_transport.py) shows the arithmetic circuit *can* read an
injected helix direction (sufficiency). It does NOT show the model *naturally uses* that subspace
when processing a Spanish/French/Devanagari number. Two tests here close that gap, on a BASE model
(the addition readout needs a base model -- instruct models score ~0 clean_acc):

  (A) ABLATION (necessity). Mean-ablate the en_digit-helix subspace from a form-B number inside
      "a + b = " and measure whether the model can still add it. If ablating the *shared* subspace
      collapses form-B accuracy while ablating a random equal-dim subspace does not, the model
      relies on that subspace to read the value -> it is causally necessary, cross-form.

  (B) MATCHED-SOURCE INTERCHANGE (naturalness). Transport using the model's OWN en_digit activation
      for a' (not our Fourier reconstruction), restricted to the helix subspace. If this steers the
      answer toward a'+b while a random subspace does not, the shared subspace carries a genuine,
      model-produced value across forms -- independent of the fit's reconstruction quality.

Both use multi-seed random subspaces as the control (mean +/- std reported).

Usage:
    python scripts/run_necessity.py --model Qwen/Qwen2.5-7B --layer 14
    python scripts/run_necessity.py --model meta-llama/Llama-3.1-8B --layer <sweep-peak>
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
from src.extract import load_model, extract_form_activations, _number_token_indices
from src.helix import fit_helix
from src.patching import (
    helix_subspace_basis, random_subspace_basis, make_patched_vector, patch_residual,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "es_word", "fr_word", "devanagari_digit"])
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--pairs-per-form", type=int, default=40, help="interchange (a,a',b) triples")
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--n-seeds", type=int, default=5, help="random-subspace control seeds")
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
def forward_logits(model, tok, device, prompt, want_hidden=False, layer=None):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    h = out.hidden_states[layer][0].float().cpu().numpy() if want_hidden else None
    return logits, h


@torch.no_grad()
def patched_logits(model, tok, device, prompt, hook_layer, pos, new_h):
    handle = patch_residual(model, hook_layer, pos, torch.tensor(new_h, dtype=torch.float32, device=device))
    try:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        return model(**enc).logits[0, -1, :].float().cpu().numpy()
    finally:
        handle.remove()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    hook_layer = args.layer - 1

    print(f"\nModel: {args.model} | layer {args.layer}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size

    fit_numbers = list(range(0, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers),
                                   pooling="last")
    fit = fit_helix(acts[args.layer], fit_numbers, k_pca=args.k_pca)
    en_real = acts[args.layer]                    # real en_digit activations, [fit_max+1, d_model]
    Q = helix_subspace_basis(fit)
    r = Q.shape[1]
    Q_rands = [random_subspace_basis(r, d_model, seed=args.seed + i) for i in range(args.n_seeds)]
    mean_vec = fit["mean"]
    ans_ids = {v: answer_token_id(tok, v) for v in range(0, args.max_sum + 1)}
    print(f"en_digit helix @ L{args.layer}: R^2={fit['r2']:.3f}, subspace dim r={r}\n")

    def argmax_ans(logits):
        return int(np.argmax([logits[ans_ids[v]] for v in range(0, args.max_sum + 1)]))

    rng = np.random.default_rng(args.seed)
    ablation, interchange = {}, {}
    for form in args.forms:
        # ---------- (A) ABLATION ----------
        ab_cases = [(a, b) for b in args.addends for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
        clean_ok, helix_ok, rand_ok = 0, 0, [0] * args.n_seeds
        n_ab = 0
        for (a, b) in ab_cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            n_ab += 1
            Lc, hidden = forward_logits(model, tok, device, prompt, want_hidden=True, layer=args.layer)
            clean_ok += int(argmax_ans(Lc) == a + b)
            h_orig = hidden[pos]
            # mean-ablate the helix subspace (value-agnostic): keep orth complement, set in-Q comp to mean's
            h_hel = make_patched_vector(h_orig, mean_vec, Q=Q, mode="subspace")
            helix_ok += int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, pos, h_hel)) == a + b)
            for si, Qr in enumerate(Q_rands):
                h_rnd = make_patched_vector(h_orig, mean_vec, Q=Qr, mode="subspace")
                rand_ok[si] += int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, pos, h_rnd)) == a + b)
        n_ab = max(n_ab, 1)
        rand_accs = np.array(rand_ok) / n_ab
        ablation[form] = {
            "axis": D.FORMS[form].axis, "n": n_ab,
            "clean_acc": clean_ok / n_ab,
            "acc_helix_ablate": helix_ok / n_ab,
            "acc_random_ablate_mean": float(rand_accs.mean()),
            "acc_random_ablate_std": float(rand_accs.std()),
        }

        # ---------- (B) MATCHED-SOURCE INTERCHANGE ----------
        ic_cases = []
        for b in args.addends:
            vals = [a for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
            ic_cases += [(a, ap, b) for a in vals for ap in vals if a != ap]
        rng.shuffle(ic_cases)
        ic_cases = ic_cases[: args.pairs_per_form]
        sub_shift, rnd_shift = [], []
        for (a, ap, b) in ic_cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            Lc, hidden = forward_logits(model, tok, device, prompt, want_hidden=True, layer=args.layer)
            base = Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]
            h_orig = hidden[pos]
            target = en_real[ap]                              # REAL en_digit activation for a'
            h_sub = make_patched_vector(h_orig, target, Q=Q, mode="subspace")
            Lp = patched_logits(model, tok, device, prompt, hook_layer, pos, h_sub)
            sub_shift.append(float((Lp[ans_ids[ap + b]] - Lp[ans_ids[a + b]]) - base))
            rs = []
            for Qr in Q_rands:
                h_r = make_patched_vector(h_orig, target, Q=Qr, mode="subspace")
                Lr = patched_logits(model, tok, device, prompt, hook_layer, pos, h_r)
                rs.append(float((Lr[ans_ids[ap + b]] - Lr[ans_ids[a + b]]) - base))
            rnd_shift.append(float(np.mean(rs)))
        interchange[form] = {
            "axis": D.FORMS[form].axis, "n": len(sub_shift),
            "subspace_shift": float(np.mean(sub_shift)) if sub_shift else float("nan"),
            "subspace_pos_rate": float(np.mean([s > 0 for s in sub_shift])) if sub_shift else float("nan"),
            "random_shift": float(np.mean(rnd_shift)) if rnd_shift else float("nan"),
        }
        print(f"  done {form}")

    # ---------- report ----------
    print("\n" + "=" * 84)
    print(f"(A) NECESSITY via ABLATION  (mean-ablate helix subspace from the SOURCE number @ L{args.layer})")
    print("    helix-ablate acc << random-ablate acc (~clean)  =>  model USES the shared subspace")
    print("-" * 84)
    print(f"  {'source form':<20}{'axis':<9}{'clean_acc':>10}{'helix_abl':>11}{'rand_abl':>11}{'Δ(rand-helix)':>15}")
    for form in args.forms:
        A = ablation[form]
        drop = A["acc_random_ablate_mean"] - A["acc_helix_ablate"]
        print(f"  {form:<20}{A['axis']:<9}{A['clean_acc']:>10.2f}{A['acc_helix_ablate']:>11.2f}"
              f"{A['acc_random_ablate_mean']:>10.2f}±{A['acc_random_ablate_std']:.2f}{drop:>10.2f}")
    print("=" * 84)
    print(f"\n(B) NATURALNESS via MATCHED-SOURCE INTERCHANGE  (patch REAL en_digit activation, subspace-only)")
    print("    subspace_shift >> random_shift  =>  the shared subspace carries a genuine value cross-form")
    print("-" * 84)
    print(f"  {'source form':<20}{'axis':<9}{'subspace_shift':>16}{'pos_rate':>10}{'random_shift':>14}")
    for form in args.forms:
        I = interchange[form]
        print(f"  {form:<20}{I['axis']:<9}{I['subspace_shift']:>16.3f}{I['subspace_pos_rate']:>10.2f}{I['random_shift']:>14.3f}")
    print("=" * 84 + "\n")

    out = {"model": args.model, "layer": args.layer, "r": r, "n_seeds": args.n_seeds,
           "fit_r2": fit["r2"], "ablation": ablation, "interchange": interchange}
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"necessity_{tag}_L{args.layer}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

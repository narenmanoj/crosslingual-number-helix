#!/usr/bin/env python
"""Step 3: causal cross-form helix transport (H3).

Test whether the model USES the shared en_digit helix regardless of surface form. For a source
number a written in form B, inside an addition prompt "a + b =", we overwrite a's residual vector
(at the fitted layer) with the en_digit helix's encoding of a DIFFERENT value a', and check
whether the model's predicted answer moves from (a+b) toward (a'+b).

Controls (all reported side by side):
  - mode=full      : replace the whole vector with en_digit's reconstruction of a'
  - mode=subspace  : swap ONLY the helix-subspace component (the localized, defensible claim)
  - mode=random    : swap an equal-dim RANDOM subspace -> MUST NOT steer (illusion control)
  - form en_digit  : within-form positive control (must work before cross-form is meaningful)

Readout: single-token answers only, so values are restricted to a+b, a'+b in [0, max_sum].
This mirrors the addition-based causal test in Kantamneni & Tegmark (2502.00873).

Usage:
    python scripts/run_transport.py --model Qwen/Qwen2.5-7B --layer 14
    python scripts/run_transport.py --forms en_digit es_word devanagari_digit --layer 12
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
from src.extract import (load_model, extract_form_activations, _number_token_indices, model_revision,
                         validate_single_token_answers)
from src.helix import fit_helix
from src.patching import (
    helix_reconstruct, helix_subspace_basis, random_subspace_basis,
    make_patched_vector, patch_residual, verify_hook_layer,
)

MODES = ["full", "subspace", "random"]
# Delta transport (audit #2): instead of REPLACING the subspace component with a reconstruction
# (which also imports form/carrier/context offset), ADD only the matched-arithmetic value DISPLACEMENT
#   h_B(a,b)  ->  h_B(a,b) + QQ^T ( h_en(a',b) - h_en(a,b) )
# from two English-DIGIT arithmetic prompts at the same position. This holds addend, syntax, output
# format, and form/context offset fixed and transports only the a->a' change. delta_rand = same with
# a random subspace (control). This is the decisive value-transport test.
DELTA_MODES = ["delta", "delta_rand"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "es_word", "fr_word", "devanagari_digit"],
                   help="source forms to transport FROM (en_digit = within-form positive control)")
    p.add_argument("--layer", type=int, default=14, help="hidden_states index to fit + patch (7B~14, 1.5B~12)")
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9, help="keep answers single-token")
    p.add_argument("--pairs-per-form", type=int, default=80, help="raise for tighter CIs (cap ~all valid triples)")
    p.add_argument("--fit-max", type=int, default=99, help="fit the en_digit helix on 0..fit_max")
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--delta", dest="delta", action="store_true", default=True,
                   help="also run matched-arithmetic DELTA transport (audit #2; the decisive value test)")
    p.add_argument("--no-delta", dest="delta", action="store_false")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def answer_token_id(tok, v: int) -> int:
    # prompt ends with "= " (trailing space), so the model emits the BARE digit next.
    for s in (f"{v}", f" {v}"):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tok.encode(f"{v}", add_special_tokens=False)[-1]  # LAST token = the digit (skip SP metaspace)


@torch.no_grad()
def logits_last(model, tok, device, prompt, want_hidden=False, layer=None):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    h = None
    if want_hidden:
        h = out.hidden_states[layer][0].float().cpu().numpy()  # [seq, d_model]
    return logits, h, enc["input_ids"].shape[1]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\nModel: {args.model} | layer(hidden_states): {args.layer}")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    if args.layer < 1:
        raise SystemExit("--layer must be >= 1 (need a decoder block to hook)")
    hook_layer = args.layer - 1  # hidden_states[L] == output of decoder block L-1
    print(f"Device: {device} | d_model: {d_model} | hooking decoder block {hook_layer}")
    hook_err = verify_hook_layer(model, tok, device, hook_layer)  # audit #4: assumption -> tested
    flag = "  <-- WARN: hook != recorded hidden state; patches at this layer may be untrustworthy" if hook_err > 1e-3 else ""
    print(f"hook-equivalence rel-error @ block {hook_layer}: {hook_err:.2e}{flag}\n")

    # --- fit the en_digit helix at the target layer (pooling='last' -> a single patchable position) ---
    fit_numbers = list(range(0, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers),
                                    pooling="last")
    fit = fit_helix(acts[args.layer], fit_numbers, k_pca=args.k_pca)
    print(f"en_digit helix fit at layer {args.layer}: R^2={fit['r2']:.3f}")

    Q = helix_subspace_basis(fit)                          # [d_model, r]
    r = Q.shape[1]
    Q_rand = random_subspace_basis(r, d_model, seed=args.seed)
    recon = helix_reconstruct(fit, list(range(0, args.max_sum + 1)))  # target vectors for a'
    ans_ids = {v: answer_token_id(tok, v) for v in range(0, args.max_sum + 1)}
    ans_id_list = [ans_ids[v] for v in range(0, args.max_sum + 1)]
    bad_ans = validate_single_token_answers(tok, range(0, args.max_sum + 1))  # audit #9
    if bad_ans:
        print(f"  WARN: {len(bad_ans)} answer value(s) are not clean single tokens for this model: "
              f"{bad_ans[:5]}{'...' if len(bad_ans) > 5 else ''}\n  -> restricted digit-choice readout may be unreliable here.")

    def argmax_answer(logits):
        sub = np.array([logits[ans_ids[v]] for v in range(0, args.max_sum + 1)])
        return int(sub.argmax())

    # --- build ONE shared case set (a, a', b), reused for EVERY form ---
    # The triples are form-independent, so building/shuffling them once (not per form) means every
    # form is scored on the identical cases -> cross-form differences are effect differences, not
    # case-composition differences.
    cases = []
    for b in args.addends:
        vals = [a for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
        pairs = [(a, ap) for a in vals for ap in vals if a != ap]
        cases += [(a, ap, b) for (a, ap) in pairs]
    rng.shuffle(cases)
    cases = cases[: args.pairs_per_form]

    # --- delta transport cache: en_digit ARITHMETIC activation h_en(v,b) for every (v,b) we need ---
    all_modes = MODES + (DELTA_MODES if args.delta else [])
    en_arith = {}
    if args.delta:
        for (v, b) in {(a, b) for (a, ap, b) in cases} | {(ap, b) for (a, ap, b) in cases}:
            vs = str(v)
            eprompt = f"{vs} + {b} = "
            try:
                epos = _number_token_indices(tok, eprompt, vs)[-1]
            except ValueError:
                continue
            _, eh, _ = logits_last(model, tok, device, eprompt, want_hidden=True, layer=args.layer)
            en_arith[(v, b)] = eh[epos]
        print(f"delta transport: cached {len(en_arith)} en_digit arithmetic activations")

    results = {}
    for form in args.forms:
        per_mode = {m: {"shift": [], "flip": [], "n": 0} for m in all_modes}
        clean_correct = 0
        n_proc = 0  # cases that survived token-span identification (the honest denominator)
        for (a, ap, b) in cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "  # trailing space: next token is the answer digit itself
            try:
                idxs = _number_token_indices(tok, prompt, a_str)
            except ValueError:
                continue
            pos = idxs[-1]
            n_proc += 1
            Lc, hidden, _ = logits_last(model, tok, device, prompt, want_hidden=True, layer=args.layer)
            clean_correct += int(argmax_answer(Lc) == (a + b))
            h_orig = hidden[pos]                              # [d_model]
            target_vec = recon[ap]                            # en_digit helix vector for a'

            for mode in MODES:
                Qm = Q if mode == "subspace" else (Q_rand if mode == "random" else None)
                new_h = make_patched_vector(h_orig, target_vec, Q=Qm, mode=mode)
                handle = patch_residual(model, hook_layer, pos,
                                        torch.tensor(new_h, dtype=torch.float32, device=device))
                try:
                    Lp, _, _ = logits_last(model, tok, device, prompt)
                finally:
                    handle.remove()
                # shift toward a'+b relative to a+b, patched minus clean
                shift = ((Lp[ans_ids[ap + b]] - Lp[ans_ids[a + b]])
                         - (Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]))
                flip = int(argmax_answer(Lc) == (a + b) and argmax_answer(Lp) == (ap + b))
                per_mode[mode]["shift"].append(float(shift))
                per_mode[mode]["flip"].append(flip)
                per_mode[mode]["n"] += 1

            # --- delta transport: add ONLY the matched-arithmetic value displacement (audit #2) ---
            if args.delta and (a, b) in en_arith and (ap, b) in en_arith:
                diff = en_arith[(ap, b)] - en_arith[(a, b)]         # h_en(a',b) - h_en(a,b)
                for dmode, Qd in (("delta", Q), ("delta_rand", Q_rand)):
                    dvec = Qd @ (Qd.T @ diff)                       # project onto (helix | random) subspace
                    new_h = h_orig + dvec
                    handle = patch_residual(model, hook_layer, pos,
                                            torch.tensor(new_h, dtype=torch.float32, device=device))
                    try:
                        Lp, _, _ = logits_last(model, tok, device, prompt)
                    finally:
                        handle.remove()
                    shift = ((Lp[ans_ids[ap + b]] - Lp[ans_ids[a + b]])
                             - (Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]))
                    flip = int(argmax_answer(Lc) == (a + b) and argmax_answer(Lp) == (ap + b))
                    per_mode[dmode]["shift"].append(float(shift))
                    per_mode[dmode]["flip"].append(flip)
                    per_mode[dmode]["n"] += 1

        n = max(n_proc, 1)
        results[form] = {
            "axis": D.FORMS[form].axis,
            "n_cases": n_proc,
            "clean_acc": clean_correct / n,
            "modes": {m: {"mean_shift": float(np.mean(per_mode[m]["shift"])) if per_mode[m]["shift"] else float("nan"),
                          "pos_shift_rate": float(np.mean([s > 0 for s in per_mode[m]["shift"]])) if per_mode[m]["shift"] else float("nan"),
                          "flip_rate": float(np.mean(per_mode[m]["flip"])) if per_mode[m]["flip"] else float("nan"),
                          "n": per_mode[m]["n"]}
                      for m in all_modes},
            # per-case arrays (aligned across modes) for bootstrap CIs + paired significance tests
            "per_case_shift": {m: [float(s) for s in per_mode[m]["shift"]] for m in all_modes},
        }

    # --- report (long format: one row per form x mode) ---
    print("\n" + "=" * 82)
    print(f"CAUSAL TRANSPORT  (en_digit helix @ L{args.layer}, r={r})")
    print("  primary = mean_shift (logits toward a'+b) & pos_rate; random mode is the illusion control")
    print("-" * 82)
    print("  delta/delta_rand = matched-arithmetic value transport (audit #2): delta >> delta_rand => value is portable")
    print(f"  {'source form':<20}{'axis':<9}{'mode':<10}{'clean_acc':>10}{'mean_shift':>12}{'pos_rate':>10}{'flip':>7}")
    for form in args.forms:
        R = results[form]
        for m in all_modes:
            M = R["modes"][m]
            print(f"  {form:<20}{R['axis']:<9}{m:<10}{R['clean_acc']:>10.2f}"
                  f"{M['mean_shift']:>12.3f}{M['pos_shift_rate']:>10.2f}{M['flip_rate']:>7.2f}")
    print("=" * 82)
    print("Works iff: full/subspace mean_shift large & positive (pos_rate>>0.5), AND random")
    print("mean_shift ~0 -- judge the random control by MAGNITUDE (>>10x smaller), not pos_rate:")
    print("a random subspace leaks a tiny consistent nudge (~r/d_model), so its pos_rate is unreliable.")
    print("en_digit (within-form) should be strongest. flip is a strict lower bound (helix R^2~0.5 =>")
    print("partial reconstruction -> shift moves the answer without always flipping the argmax).\n")

    out = {"model_revision": model_revision(model, args.model), "model": args.model, "layer": args.layer, "r": r, "max_sum": args.max_sum,
           "addends": args.addends, "fit_r2": fit["r2"], "results": results}
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"transport_{tag}_L{args.layer}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

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

import hashlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import (load_model, extract_form_activations, _number_token_indices, model_revision,
                         continuation_answer_ids)
from src.helix import fit_helix
from src.patching import (
    helix_reconstruct, helix_subspace_basis, random_subspace_basis, top_pca_span_basis,
    shuffled_fourier_basis, make_patched_vector, patch_residual, assert_hook_equivalence,
    subspace_delta, norm_match, norm_match_diag, energy_matched_bank, ALPHA_LO, ALPHA_HI,
)
from src.provenance import stamp, resolve_layer, VALIDATED, LEGACY, E_DELTA, E_ABSOLUTE

# LEGACY absolute-target modes (estimand = absolute_carrier_reconstruction). These replace the
# subspace component with a reconstruction fitted on a *carrier* prompt, so they move value TOGETHER
# WITH prompt context / form offset / token position. Kept only as an opt-in diagnostic (audit r4 #2);
# they are NOT the default and must never share the "sufficiency" heading with the delta estimand.
MODES = ["full", "subspace", "random"]
# Structured control families for the DELTA estimand (audit r4 #5). Norm matching equalizes magnitude
# but not manifold plausibility, so we also project the SAME displacement into a top-PCA-span subspace
# and a shuffled-Fourier subspace (fit through the whole pipeline on permuted labels).
DELTA_FAMILIES = ["haar", "pca_span", "shuf_fourier"]
# Delta transport (the PRIMARY estimand): instead of REPLACING the subspace component with a
# reconstruction (which also imports form/carrier/context offset), ADD only the matched-arithmetic
# value DISPLACEMENT
#   h_B(a,b)  ->  h_B(a,b) + QQ^T ( h_en(a',b) - h_en(a,b) )
# from two English-DIGIT arithmetic prompts at the same position. This holds addend, syntax, output
# format, and form/context offset fixed and transports only the a->a' change.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "es_word", "fr_word", "devanagari_digit"],
                   help="source forms to transport FROM (en_digit = within-form positive control)")
    p.add_argument("--layer", type=int, default=None, help="hidden_states index to fit + patch (7B~14, 1.5B~12)")
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9, help="keep answers single-token")
    p.add_argument("--pairs-per-form", type=int, default=80, help="0 = ALL valid triples (recommended for production); >0 samples that many")
    # DISJOINT value sets (audit r5 blocker #2): Q is fitted on fit_min..fit_max (default 10..99) while
    # the causal test uses 0..max_sum (default 0..9) -- so the intervention subspace never saw the exact
    # values it is tested on. Pass --fit-min 0 to reproduce the older overlapping-fit behaviour.
    p.add_argument("--fit-min", type=int, default=10, help="low end of the helix fit range (10 => disjoint from 0..9)")
    p.add_argument("--fit-max", type=int, default=99, help="fit the en_digit helix on fit_min..fit_max")
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--delta", dest="delta", action="store_true", default=True,
                   help="matched-arithmetic DELTA transport -- the DEFAULT/primary estimand")
    p.add_argument("--no-delta", dest="delta", action="store_false")
    p.add_argument("--delta-ctrl-seeds", type=int, default=5,
                   help="seeds PER control family (haar/pca_span/shuf_fourier), all norm-matched + retained")
    p.add_argument("--energy-matched-controls", dest="energy_matched_controls", action="store_true",
                   default=True, help="select control subspaces with helix-like natural projected "
                                      "energy so norm-matching stays admissible (audit r5 #6)")
    p.add_argument("--no-energy-matched-controls", dest="energy_matched_controls", action="store_false")
    p.add_argument("--ctrl-candidates", type=int, default=60, help="candidate bank size per family")
    p.add_argument("--include-legacy-absolute-patching", dest="legacy_absolute", action="store_true",
                   default=False, help="ALSO run the legacy absolute-target modes (full/subspace/random) "
                                       "as a labelled legacy_diagnostic -- off by default (audit r4 #2)")
    p.add_argument("--layer-manifest", default=None,
                   help="frozen layers.json from scripts/select_layers.py (REQUIRED in --production)")
    p.add_argument("--production", action="store_true", default=False,
                   help="production mode: require a frozen layer manifest + clean worktree")
    p.add_argument("--allow-dirty", action="store_true", default=True,
                   help="allow writing results from a dirty/unknown worktree (set false for production)")
    p.add_argument("--no-allow-dirty", dest="allow_dirty", action="store_false")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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

    # BLOCKER 7: in production the layer must come from a frozen, current-commit layer manifest
    layer, layer_prov = resolve_layer(args.model, args.layer, args.layer_manifest,
                                      schema_version=C.SCHEMA_VERSION, production=args.production)
    args.layer = layer
    print(f"\nModel: {args.model} | layer(hidden_states): {args.layer} [{layer_prov['layer_source']}]")
    model, tok, device = load_model(args.model, args.device)
    d_model = model.config.hidden_size
    if args.layer < 1:
        raise SystemExit("--layer must be >= 1 (need a decoder block to hook)")
    hook_layer = args.layer - 1  # hidden_states[L] == output of decoder block L-1
    print(f"Device: {device} | d_model: {d_model} | hooking decoder block {hook_layer}")
    hook_err = assert_hook_equivalence(model, tok, device, hook_layer)  # audit #4: fail-fast, saved to JSON
    print(f"hook-equivalence rel-error @ block {hook_layer}: {hook_err:.2e}\n")

    # --- fit the en_digit helix at the target layer (pooling='last' -> a single patchable position) ---
    fit_numbers = list(range(args.fit_min, args.fit_max + 1))
    causal_values = list(range(0, args.max_sum + 1))
    disjoint = set(fit_numbers).isdisjoint(causal_values)
    print(f"fit values: {fit_numbers[0]}..{fit_numbers[-1]} | causal values: 0..{args.max_sum} | "
          f"disjoint: {disjoint}{'' if disjoint else '  <-- Q saw the causal test values'}")
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers),
                                    pooling="last")
    fit = fit_helix(acts[args.layer], fit_numbers, k_pca=args.k_pca)
    print(f"en_digit helix fit at layer {args.layer}: R^2={fit['r2']:.3f}")

    HL = acts[args.layer]
    Q = helix_subspace_basis(fit)                          # [d_model, r]
    r = Q.shape[1]
    Q_rand = random_subspace_basis(r, d_model, seed=args.seed)
    ctrl_seeds = list(range(args.delta_ctrl_seeds))
    # three control FAMILIES for the delta estimand, every seed retained (audit r4 #4/#5).
    # Candidates are ENERGY-MATCHED (audit r5 #6): a Haar subspace in d~1500 captures almost none of an
    # 8-d displacement, so naive draws need alpha~8 and are inadmissible by the predefined band. We
    # draw a larger bank and keep the subspaces whose natural projected energy resembles the helix's,
    # then still norm-match exactly. The selection procedure is recorded in the output.
    sample_vecs = HL[: min(32, len(HL))] - fit["mean"]           # representative displacements
    bank_reports = {}
    if args.energy_matched_controls:
        builders = {"haar": lambda sd: random_subspace_basis(r, d_model, seed=sd),
                    "pca_span": lambda sd: top_pca_span_basis(HL, r, seed=sd),
                    "shuf_fourier": lambda sd: shuffled_fourier_basis(HL, fit_numbers, k_pca=args.k_pca, seed=sd)}
        ctrl_banks = {}
        for fam, build in builders.items():
            ctrl_banks[fam], bank_reports[fam] = energy_matched_bank(
                sample_vecs, Q, r, d_model, n_keep=args.delta_ctrl_seeds,
                n_candidates=args.ctrl_candidates, seed=args.seed + 1, builder=build)
    else:
        ctrl_banks = {
            "haar": [random_subspace_basis(r, d_model, seed=args.seed + 1 + i) for i in ctrl_seeds],
            "pca_span": [top_pca_span_basis(HL, r, seed=args.seed + 1 + i) for i in ctrl_seeds],
            "shuf_fourier": [shuffled_fourier_basis(HL, fit_numbers, k_pca=args.k_pca, seed=args.seed + 1 + i)
                             for i in ctrl_seeds],
        }
    for fam, rep in bank_reports.items():
        print(f"  control bank [{fam}]: kept {rep['n_kept']}/{rep['n_candidates']} candidates, "
              f"implied alpha {np.round(rep['implied_alpha'], 2).tolist()}")
    recon = helix_reconstruct(fit, list(range(0, args.max_sum + 1)))  # target vectors for a' (legacy modes)
    ans_ids = continuation_answer_ids(tok, range(0, args.max_sum + 1))  # audit #2/#9: fail-fast, no fallback

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
    all_cases = list(cases)
    if args.pairs_per_form and args.pairs_per_form > 0:
        rng.shuffle(cases)
        cases = cases[: args.pairs_per_form]
    else:                       # 0 => EXHAUSTIVE: every valid (a, a', b), no sampling variance (r6 #8)
        cases = sorted(all_cases)
    case_set_hash = hashlib.sha256(repr(sorted(cases)).encode()).hexdigest()[:16]
    print(f"cases: {len(cases)} of {len(all_cases)} valid triples "
          f"({'exhaustive' if len(cases) == len(all_cases) else 'sampled'}) | hash {case_set_hash}")

    # --- delta transport cache: en_digit ARITHMETIC activation h_en(v,b) for every (v,b) we need ---
    FAM2MODE = {"haar": "delta_rand", "pca_span": "delta_pca_span", "shuf_fourier": "delta_shuf_fourier"}
    delta_modes = (["delta"] + [FAM2MODE[f] for f in DELTA_FAMILIES]) if args.delta else []
    legacy_modes = MODES if args.legacy_absolute else []
    all_modes = legacy_modes + delta_modes
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
        ctrl_by_seed = {f: [] for f in DELTA_FAMILIES}   # per-case list of per-seed shifts (audit r4 #4)
        alphas = {f: [] for f in DELTA_FAMILIES}         # flat alphas for the summary
        # FULL case x seed diagnostics -- retained, not summarized away (audit r5 #6)
        diag = {f: {k: [] for k in ("alpha", "raw_norm", "matched_norm", "admissible")}
                for f in DELTA_FAMILIES}
        case_keys, delta_keys = [], []     # audit #12: (a, a', b) per case, aligned to the per-case arrays
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
            case_keys.append((a, ap, b))
            Lc, hidden, _ = logits_last(model, tok, device, prompt, want_hidden=True, layer=args.layer)
            clean_correct += int(argmax_answer(Lc) == (a + b))
            h_orig = hidden[pos]                              # [d_model]
            target_vec = recon[ap]                            # en_digit helix vector for a'

            for mode in legacy_modes:   # LEGACY absolute-target diagnostic (opt-in, audit r4 #2)
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
            # control is NORM-MATCHED to the helix delta and averaged over delta_ctrl_seeds random
            # subspaces (audit #3), so "does it steer" is not confounded by "it perturbs more/less".
            if args.delta and (a, b) in en_arith and (ap, b) in en_arith:
                diff = en_arith[(ap, b)] - en_arith[(a, b)]         # h_en(a',b) - h_en(a,b)
                dvec_h = subspace_delta(diff, Q)                    # helix-subspace value displacement
                nh = np.linalg.norm(dvec_h)

                def shift_of(new_h):
                    handle = patch_residual(model, hook_layer, pos,
                                            torch.tensor(new_h, dtype=torch.float32, device=device))
                    try:
                        Lp, _, _ = logits_last(model, tok, device, prompt)
                    finally:
                        handle.remove()
                    return float((Lp[ans_ids[ap + b]] - Lp[ans_ids[a + b]])
                                 - (Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]])), \
                           int(argmax_answer(Lc) == (a + b) and argmax_answer(Lp) == (ap + b))

                sh, fl = shift_of(h_orig + dvec_h)
                per_mode["delta"]["shift"].append(sh); per_mode["delta"]["flip"].append(fl); per_mode["delta"]["n"] += 1
                for fam in DELTA_FAMILIES:
                    seed_sh, seed_fl, seed_alpha, seed_raw, seed_matched, seed_adm = [], [], [], [], [], []
                    for Qc in ctrl_banks[fam]:
                        # NORM-MATCH to the helix delta, keeping the FULL per-(case,seed) diagnostics
                        dc, dg = norm_match_diag(subspace_delta(diff, Qc), nh)
                        s, f = shift_of(h_orig + dc)
                        seed_sh.append(s); seed_fl.append(f)
                        seed_alpha.append(dg["alpha"]); seed_raw.append(dg["raw_norm"])
                        seed_matched.append(dg["matched_norm"]); seed_adm.append(dg["admissible"])
                    m = FAM2MODE[fam]
                    per_mode[m]["shift"].append(float(np.mean(seed_sh)))
                    per_mode[m]["flip"].append(float(np.mean(seed_fl)))   # MEASURED, not hard-coded (audit r4 #6)
                    per_mode[m]["n"] += 1
                    ctrl_by_seed[fam].append([float(s) for s in seed_sh])
                    diag[fam]["alpha"].append([float(x) for x in seed_alpha])
                    diag[fam]["raw_norm"].append([float(x) for x in seed_raw])
                    diag[fam]["matched_norm"].append([float(x) for x in seed_matched])
                    diag[fam]["admissible"].append([bool(x) for x in seed_adm])
                    alphas[fam].extend(seed_alpha)
                delta_keys.append((a, ap, b))

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
            # per-case (a, a', b) keys for exact pairing + clustered inference (audit #11/#12)
            "per_case_keys": {"modes": case_keys, "delta": delta_keys},
            # full case x seed control matrices -- never only the mean (audit r4 #4)
            "delta_control_by_seed": ctrl_by_seed,
            "control_seeds": ctrl_seeds,
            # FULL per-(case, seed) norm-match diagnostics + the predefined admissibility flag, so the
            # analysis can run primary-on-admissible / sensitivity-on-all (audit r5 #6)
            "control_diagnostics": diag,
            "alpha_range": [ALPHA_LO, ALPHA_HI],
            "delta_alpha": {f: {"median": float(np.nanmedian(alphas[f])) if alphas[f] else float("nan"),
                                "frac_out_of_range": float(np.mean([not (ALPHA_LO <= x <= ALPHA_HI)
                                                                    for x in alphas[f] if np.isfinite(x)]))
                                if alphas[f] else float("nan")}
                             for f in DELTA_FAMILIES},
        }

    # --- report (long format: one row per form x mode) ---
    print("\n" + "=" * 82)
    print(f"CAUSAL TRANSPORT  (en_digit helix @ L{args.layer}, r={r})")
    print(f"  PRIMARY estimand = {E_DELTA}: delta vs norm-matched controls in 3 families")
    print("  (delta_rand=Haar, delta_pca_span=top-PCA-span, delta_shuf_fourier=shuffled-pipeline)")
    if args.legacy_absolute:
        print(f"  full/subspace/random = LEGACY {E_ABSOLUTE} (diagnostic only -- NOT a sufficiency claim)")
    print("-" * 92)
    print(f"  {'source form':<20}{'axis':<9}{'mode':<19}{'clean_acc':>10}{'mean_shift':>12}{'pos_rate':>10}{'flip':>7}")
    for form in args.forms:
        R = results[form]
        for m in all_modes:
            M = R["modes"][m]
            print(f"  {form:<20}{R['axis']:<9}{m:<19}{R['clean_acc']:>10.2f}"
                  f"{M['mean_shift']:>12.3f}{M['pos_shift_rate']:>10.2f}{M['flip_rate']:>7.2f}")
    print("=" * 92)
    print("Works iff: delta mean_shift large & positive AND every norm-matched control family ~0.")
    for form in args.forms:
        al = results[form].get("delta_alpha", {})
        bad = [f"{f}:{al[f]['frac_out_of_range']:.2f}" for f in al if np.isfinite(al[f]["frac_out_of_range"])
               and al[f]["frac_out_of_range"] > 0.2]
        if bad:
            print(f"  WARN {form}: control norm-match scale alpha outside [0.25,4] for {', '.join(bad)} "
                  "-> those 'matched' controls are extrapolations, read them with care.")
    print("en_digit (within-form) should be strongest. flip is a strict lower bound (helix R^2~0.5 =>")
    print("partial reconstruction -> shift moves the answer without always flipping the argmax).\n")

    out = {**stamp(C.SCHEMA_VERSION, "transport",
                   estimand=E_DELTA if args.delta else E_ABSOLUTE,
                   analysis_status=VALIDATED if args.delta else LEGACY,
                   allow_dirty=args.allow_dirty),
           "legacy_absolute_included": args.legacy_absolute,
           "legacy_absolute_estimand": E_ABSOLUTE if args.legacy_absolute else None,
           "intervention_position": "source_last_token",
           "layer_selection": layer_prov,
           "delta_control_families": DELTA_FAMILIES,
           "control_bank_selection": bank_reports,
           "energy_matched_controls": args.energy_matched_controls,
           "model_revision": model_revision(model, args.model),
           "model": args.model, "layer": args.layer, "r": r, "max_sum": args.max_sum,
           "addends": args.addends, "fit_r2": fit["r2"], "hook_rel_error": hook_err,
           "delta_ctrl_seeds": args.delta_ctrl_seeds,
           "fit_values": [args.fit_min, args.fit_max], "causal_values": [0, args.max_sum],
           "value_sets_disjoint": disjoint,
           "case_set_hash": case_set_hash, "case_set_exhaustive": len(cases) == len(all_cases),
           "results": results}
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"transport_{tag}_L{args.layer}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

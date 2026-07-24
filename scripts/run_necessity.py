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

  (B) MATCHED-ARITHMETIC DELTA INTERCHANGE. Add the real en_digit arithmetic displacement, subspace-only,
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
    helix_subspace_basis, random_subspace_basis, covariance_matched_basis, shuffled_fourier_basis,
    make_patched_vector, norm_matched_ablation, subspace_delta, norm_match, subspace_energy,
    patch_residual, patch_residual_multi, assert_hook_equivalence, _proj,
    norm_match_diag, ALPHA_LO, ALPHA_HI,
)
from src.provenance import stamp, resolve_layer, VALIDATED, E_ABLATION

CONTROLS = ["random", "cov_matched", "shuf_fourier"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=["en_digit", "devanagari_digit", "es_word", "fr_word"])
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--intervention-pos", default="last", choices=["last", "span", "after"])
    p.add_argument("--ablation-baseline", default="form_arith", choices=["form_arith", "carrier"],
                   help="mean-ablation target: 'form_arith' = this form's OWN arithmetic-context mean "
                        "(context-matched, audit #12); 'carrier' = the en_digit fit mean (legacy)")
    p.add_argument("--baseline-crossfit", default="source_value", choices=["source_value", "none"],
                   help="cross-fit the ablation baseline so a case never contributes to its own target "
                        "(audit r4 #8): 'source_value' = leave-one-source-value-out")
    p.add_argument("--layer-manifest", default=None,
                   help="frozen layers.json from scripts/select_layers.py (REQUIRED in --production)")
    p.add_argument("--production", action="store_true", default=False,
                   help="production mode: require a frozen layer manifest + clean worktree")
    p.add_argument("--allow-dirty", action="store_true", default=True)
    p.add_argument("--no-allow-dirty", dest="allow_dirty", action="store_false")
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--pairs-per-form", type=int, default=40)
    # DISJOINT value sets (audit r5 #2): Q fitted on 10..99, causal test on 0..max_sum.
    p.add_argument("--fit-min", type=int, default=10, help="10 => helix fit disjoint from the 0..9 causal values")
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--on-baseline-fallback", default="skip", choices=["skip", "error", "label"],
                   help="what to do when a cross-fit baseline bucket has no out-of-source examples "
                        "(audit r5 #7): skip the case (default), fail, or use it with an explicit label")
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--n-seeds", type=int, default=8, help="control-null seeds; more = tighter null bands")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--out-dir", default=C.OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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
        nxt = idxs[-1] + 1
        if nxt > seq_len - 1:
            # clamping would silently turn an "after" intervention into a "last" one, making the two
            # positions incomparable while still being labelled differently.
            raise ValueError("no token after the number span; 'after' is undefined for this prompt")
        return [nxt]
    raise ValueError(mode)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    layer, layer_prov = resolve_layer(args.model, args.layer, args.layer_manifest,
                                      schema_version=C.SCHEMA_VERSION, production=args.production)
    args.layer = layer
    hook_layer = args.layer - 1

    print(f"\nModel: {args.model} | layer {args.layer} | intervention-pos {args.intervention_pos}")
    model, tok, device = load_model(args.model, args.device,
                                    revision=layer_prov.get('frozen_model_revision'))
    d_model = model.config.hidden_size
    hook_err = assert_hook_equivalence(model, tok, device, hook_layer)  # audit #4: fail-fast, saved to JSON
    print(f"hook-equivalence rel-error @ block {hook_layer}: {hook_err:.2e}")

    fit_numbers = list(range(args.fit_min, args.fit_max + 1))
    acts = extract_form_activations(model, tok, device, D.build_prompts("en_digit", fit_numbers), pooling="last")
    HL = acts[args.layer]
    fit = fit_helix(HL, fit_numbers, k_pca=args.k_pca)
    Q = helix_subspace_basis(fit)
    r = Q.shape[1]
    mean_vec = fit["mean"]
    # control-subspace banks (multi-seed)
    ctrl_bases = {
        "random": [random_subspace_basis(r, d_model, seed=args.seed + i) for i in range(args.n_seeds)],
        "cov_matched": [covariance_matched_basis(HL, r, seed=args.seed + i) for i in range(args.n_seeds)],
        "shuf_fourier": [shuffled_fourier_basis(HL, fit_numbers, k_pca=args.k_pca, seed=args.seed + i) for i in range(args.n_seeds)],
    }
    ans_ids = continuation_answer_ids(tok, range(0, args.max_sum + 1))  # audit #2/#9: fail-fast, no fallback
    print(f"en_digit helix @ L{args.layer}: R^2={fit['r2']:.3f}, r={r} | ablation baseline={args.ablation_baseline}\n")

    def argmax_ans(logits):
        return int(np.argmax([logits[ans_ids[v]] for v in range(0, args.max_sum + 1)]))

    rng = np.random.default_rng(args.seed)
    # ONE shared case set per experiment, reused for EVERY form (triples are form-independent), so
    # cross-form differences are effect differences, not case-composition differences. Ablation cases
    # are already deterministic; the interchange set is shuffled ONCE here, not per form.
    ab_cases = [(a, b) for b in args.addends for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
    ic_cases = []
    for b in args.addends:
        vals = [a for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
        ic_cases += [(a, ap, b) for a in vals for ap in vals if a != ap]
    all_ic = list(ic_cases)
    if args.pairs_per_form and args.pairs_per_form > 0:
        rng.shuffle(ic_cases)
        ic_cases = ic_cases[: args.pairs_per_form]
    else:
        ic_cases = sorted(all_ic)
    case_set_hash = hashlib.sha256(repr((sorted(ab_cases), sorted(ic_cases))).encode()).hexdigest()[:16]
    print(f"cases: ablation {len(ab_cases)}, interchange {len(ic_cases)}/{len(all_ic)} | hash {case_set_hash}")

    # en_digit ARITHMETIC activations for the DELTA interchange (audit r3 #2): the sufficiency check
    # now transports a matched-arithmetic value displacement h_en(a',b)-h_en(a,b), NOT an absolute
    # carrier activation en_real[ap] (which also imported carrier/context/offset).
    en_arith = {}
    for (v, b) in {(a, b) for (a, ap, b) in ic_cases} | {(ap, b) for (a, ap, b) in ic_cases}:
        vs = str(v)
        ep = f"{vs} + {b} = "
        try:
            epos = _number_token_indices(tok, ep, vs)[-1]
        except ValueError:
            continue
        _, eh, _ = forward(model, tok, device, ep, layer=args.layer, want_hidden=True)
        en_arith[(v, b)] = eh[epos]

    ablation, interchange = {}, {}
    for form in args.forms:
        # ---------- (A) ABLATION ----------
        # Context-matched baseline (audit #12/#9): pull the value-subspace toward THIS form's own
        # arithmetic-context mean, CONDITIONED on (token_count, relative token position) so a span
        # token is ablated toward the mean at its own position/length, not a form-wide pooled mean.
        # Falls back to the global fit mean for any (tc, rel) bucket with no data.
        # Buckets keep the CONTRIBUTING SOURCE VALUE with each vector so the baseline can be cross-fit
        # (leave-one-source-value-out), i.e. a case never helps estimate its own intervention target.
        buckets = {}
        if args.ablation_baseline == "form_arith":
            for (a, b) in ab_cases:
                a_str = D.FORMS[form].render(a)
                prompt = f"{a_str} + {b} = "
                try:
                    idxs = _number_token_indices(tok, prompt, a_str)
                except ValueError:
                    continue
                _, hid, sl = forward(model, tok, device, prompt, layer=args.layer, want_hidden=True)
                for p in intervention_positions(idxs, args.intervention_pos, sl):
                    buckets.setdefault((len(idxs), p - idxs[0]), []).append((a, hid[p]))

        def baseline_at(idxs, p, src):
            """Per-(token-count, relative-position) mean, EXCLUDING cases whose source value == src.

            Returns (vector, meta). NEVER silently relaxes the rule (audit r5 #7): if the cross-fit
            bucket has no out-of-source examples the caller decides (skip / error / explicitly label),
            and every patched position records which baseline estimand it actually used."""
            if args.ablation_baseline != "form_arith":
                return mean_vec, {"baseline_source": "carrier_fit_mean", "n_calibration_examples": 0,
                                  "fallback_used": False, "excluded_source_value": None}
            entries = buckets.get((len(idxs), p - idxs[0]), [])
            if args.baseline_crossfit == "source_value":
                held = [v for (s, v) in entries if s != src]
                if held:
                    return np.mean(held, axis=0), {"baseline_source": "crossfit_form_arith",
                                                   "n_calibration_examples": len(held),
                                                   "fallback_used": False, "excluded_source_value": src}
                # no admissible cross-fit calibration for this (token-count, position, source)
                return None, {"baseline_source": "unavailable", "n_calibration_examples": 0,
                              "fallback_used": True, "excluded_source_value": src}
            if entries:
                return np.mean([v for (_, v) in entries], axis=0), {
                    "baseline_source": "form_arith_insample", "n_calibration_examples": len(entries),
                    "fallback_used": False, "excluded_source_value": None}
            return None, {"baseline_source": "unavailable", "n_calibration_examples": 0,
                          "fallback_used": True, "excluded_source_value": src}

        clean_ok, helix_ok = 0, 0
        ctrl_ok = {c: np.zeros(args.n_seeds) for c in CONTROLS}
        energy = {"helix": []} | {c: [] for c in CONTROLS}
        clean_case, helix_case, ab_keys = [], [], []
        ctrl_case = {c: [] for c in CONTROLS}                              # per-case MEAN over seeds
        ctrl_case_by_seed = {c: [] for c in CONTROLS}                     # per-case list of per-seed 0/1 (audit r3 #3)
        ctrl_alpha = {c: [] for c in CONTROLS}                            # flat alphas for the summary
        # FULL per-(case, seed) norm-match diagnostics + admissibility (audit r5 #6)
        ctrl_diag = {c: {k: [] for k in ("alpha", "raw_norm", "matched_norm", "admissible")}
                     for c in CONTROLS}
        baseline_meta = []          # per-case, per-position baseline provenance (audit r5 #7)
        n_skipped_baseline = 0
        tok_counts, n = [], 0
        skipped_keys = []
        for (a, b) in ab_cases:
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                idxs = _number_token_indices(tok, prompt, a_str)
            except ValueError:
                # BLOCKER 10: never silently drop a case in production -- an unprocessed case changes
                # the effective case set for this form only, breaking cross-form comparability.
                skipped_keys.append([a, b])
                if args.production:
                    raise SystemExit(f"token-span extraction failed for {form} (a={a}, b={b}); "
                                     "production requires every expected case to be processed")
                continue
            Lc, hidden, seq_len = forward(model, tok, device, prompt, layer=args.layer, want_hidden=True)
            positions = intervention_positions(idxs, args.intervention_pos, seq_len)
            # cross-fit per-position baseline; a missing calibration bucket is NEVER silently relaxed
            bres = {p: baseline_at(idxs, p, a) for p in positions}
            if any(v is None for v, _ in bres.values()):
                if args.on_baseline_fallback == "error":
                    raise SystemExit(f"no cross-fit baseline for {form} (a={a}, b={b}); "
                                     "use --on-baseline-fallback skip|label or add calibration data")
                if args.on_baseline_fallback == "skip":
                    n_skipped_baseline += 1
                    continue
                # 'label': fall back to the pooled bucket mean but RECORD it as a different estimand
                for p, (v, meta) in list(bres.items()):
                    if v is None:
                        ent = buckets.get((len(idxs), p - idxs[0]), [])
                        pooled = np.mean([x for (_, x) in ent], axis=0) if ent else mean_vec
                        meta.update(baseline_source="labeled_fallback_pooled" if ent else "labeled_fallback_carrier")
                        bres[p] = (pooled, meta)
            base = {p: v for p, (v, _) in bres.items()}
            baseline_meta.append({str(p): m for p, (_, m) in bres.items()})
            n += 1
            tok_counts.append(len(idxs)); ab_keys.append((a, b))
            cc = int(argmax_ans(Lc) == a + b); clean_ok += cc; clean_case.append(cc)
            # helix ablation (mean-ablate the helix subspace at each chosen position -> per-position baseline)
            p2v = {p: make_patched_vector(hidden[p], base[p], Q=Q, mode="subspace") for p in positions}
            hc = int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, p2v)) == a + b)
            helix_ok += hc; helix_case.append(hc)
            # WHOLE-SPAN helix removed energy: sqrt(sum_p ||QQ^T(h_p - base_p)||^2)
            energy["helix"].append(float(np.sqrt(sum(subspace_energy(Q, hidden[p], base[p]) ** 2 for p in positions))))
            for c in CONTROLS:
                seed_correct, s_alpha, s_raw, s_matched, s_adm = [], [], [], [], []
                for si, Qc in enumerate(ctrl_bases[c]):
                    # NORM-MATCHED ablation: remove the SAME energy as the helix at each position (audit r3 #3)
                    p2v = {p: norm_matched_ablation(hidden[p], base[p], Q_signal=Q, Q_control=Qc) for p in positions}
                    sc = int(argmax_ans(patched_logits(model, tok, device, prompt, hook_layer, p2v)) == a + b)
                    ctrl_ok[c][si] += sc; seed_correct.append(sc)
                    # per-(case, seed) norm-match diagnostics, aggregated over the patched span
                    a_list, r_list, m_list = [], [], []
                    for p in positions:
                        nrs = float(np.linalg.norm(_proj(Q, hidden[p] - base[p])))
                        _, dg = norm_match_diag(_proj(Qc, hidden[p] - base[p]), nrs)
                        a_list.append(dg["alpha"]); r_list.append(dg["raw_norm"]); m_list.append(dg["matched_norm"])
                    # BLOCKER 6: each position is scaled INDEPENDENTLY, so a mean alpha can hide two
                    # pathological positions (0.1 and 5.0 average to 2.55, "in range", though neither
                    # is). Admissible only if EVERY patched position is in band; keep the full array.
                    s_alpha.append([float(x) for x in a_list])
                    s_raw.append(float(np.mean(r_list))); s_matched.append(float(np.mean(m_list)))
                    s_adm.append(bool(all(ALPHA_LO <= x <= ALPHA_HI for x in a_list)))
                    ctrl_alpha[c].extend(a_list)
                ctrl_case[c].append(float(np.mean(seed_correct)))
                ctrl_case_by_seed[c].append([int(s) for s in seed_correct])
                ctrl_diag[c]["alpha"].append(s_alpha); ctrl_diag[c]["raw_norm"].append(s_raw)
                ctrl_diag[c]["matched_norm"].append(s_matched); ctrl_diag[c]["admissible"].append(s_adm)
                # after norm-matching, per-seed removed energy ~ helix energy by construction (report helix's)
                energy[c].append(energy["helix"][-1])
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
            "controls_norm_matched": True,
            "control_diagnostics": ctrl_diag,
            "alpha_range": [ALPHA_LO, ALPHA_HI],
            "baseline_meta": baseline_meta,
            "n_skipped_no_baseline": n_skipped_baseline,
            "control_alpha": {c: {"median": float(np.nanmedian(ctrl_alpha[c])) if ctrl_alpha[c] else float("nan"),
                                  "frac_out_of_range": float(np.mean([not (ALPHA_LO <= x <= ALPHA_HI)
                                                                      for x in ctrl_alpha[c] if np.isfinite(x)]))
                                  if ctrl_alpha[c] else float("nan")}
                              for c in CONTROLS},
            # per-case arrays for bootstrap CIs + paired tests + full per-seed fidelity + case keys
            "per_case": {"clean": clean_case, "helix": helix_case, "controls": ctrl_case,
                         "controls_by_seed": ctrl_case_by_seed, "keys": ab_keys},
        }

        # ---------- (B) MATCHED-ARITHMETIC DELTA INTERCHANGE (audit r3 #2) ----------
        # Sufficiency via the REAL en_digit arithmetic displacement h_en(a',b)-h_en(a,b) added at the
        # source token (NOT an absolute carrier activation en_real[a'], which also imported carrier /
        # context / surface-form offset). Control = norm-matched random delta, averaged over ALL seeds.
        sub_shift, matched_shift, ic_keys = [], [], []
        ic_ctrl_by_seed = []          # per-case list of per-seed control shifts (audit r4 #4)
        for (a, ap, b) in ic_cases:
            if (a, b) not in en_arith or (ap, b) not in en_arith:
                continue
            a_str = D.FORMS[form].render(a)
            prompt = f"{a_str} + {b} = "
            try:
                pos = _number_token_indices(tok, prompt, a_str)[-1]
            except ValueError:
                continue
            Lc, hidden, _ = forward(model, tok, device, prompt, layer=args.layer, want_hidden=True)
            base = Lc[ans_ids[ap + b]] - Lc[ans_ids[a + b]]
            h_orig = hidden[pos]
            diff = en_arith[(ap, b)] - en_arith[(a, b)]
            dh = subspace_delta(diff, Q); nh = np.linalg.norm(dh)
            Ls = patched_logits(model, tok, device, prompt, hook_layer, {pos: h_orig + dh})
            sub_shift.append(float((Ls[ans_ids[ap + b]] - Ls[ans_ids[a + b]]) - base))
            seed_sh = []
            for Qc in ctrl_bases["random"]:                              # ALL seeds, each norm-matched
                dc = norm_match(subspace_delta(diff, Qc), nh)
                Lm = patched_logits(model, tok, device, prompt, hook_layer, {pos: h_orig + dc})
                seed_sh.append(float((Lm[ans_ids[ap + b]] - Lm[ans_ids[a + b]]) - base))
            matched_shift.append(float(np.mean(seed_sh)))
            ic_ctrl_by_seed.append([float(s) for s in seed_sh])
            ic_keys.append((a, ap, b))
        interchange[form] = {
            "axis": D.FORMS[form].axis, "n": len(sub_shift), "estimand": "matched_arithmetic_delta",
            "intervention_position": "source_last_token",     # NOT the ablation position (audit r4 #9)
            "subspace_shift": float(np.mean(sub_shift)) if sub_shift else float("nan"),
            "matched_random_shift": float(np.mean(matched_shift)) if matched_shift else float("nan"),
            "per_case": {"subspace": sub_shift, "matched_random": matched_shift, "keys": ic_keys},
            "control_by_seed": ic_ctrl_by_seed, "control_seeds": list(range(args.n_seeds)),
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
    print(f"\n(B) MATCHED-ARITHMETIC DELTA INTERCHANGE (inject @ source last token)  -- subspace_shift >> norm-matched control => real")
    print("-" * 96)
    print(f"  {'form':<20}{'axis':<9}{'subspace_shift':>16}{'matched_random':>16}")
    for form in args.forms:
        I = interchange[form]
        print(f"  {form:<20}{I['axis']:<9}{I['subspace_shift']:>16.3f}{I['matched_random_shift']:>16.3f}")
    print("=" * 96 + "\n")

    out = {**stamp(C.SCHEMA_VERSION, "necessity", estimand=E_ABLATION,
                   analysis_status=VALIDATED, allow_dirty=args.allow_dirty),
           "model_revision": model_revision(model, args.model, revision=getattr(model, "_pinned_revision", None)),
           "layer": args.layer,
           # positions are recorded SEPARATELY: the ablation honors --intervention-pos, the delta
           # interchange always injects at the source's last token (audit r4 #9)
           "layer_selection": layer_prov,
           "ablation_position": args.intervention_pos, "interchange_position": "source_last_token",
           "interchange_estimand": "matched_arithmetic_delta",
           "r": r, "n_seeds": args.n_seeds, "ablation_baseline": args.ablation_baseline,
           "baseline_fit_split": "in_run", "baseline_crossfit_group": args.baseline_crossfit,
           "baseline_policy": "in_run_leave_one_source_value_out" if args.baseline_crossfit == "source_value" else "in_run_pooled",
           "on_baseline_fallback": args.on_baseline_fallback,
           "fit_values": [args.fit_min, args.fit_max], "causal_values": [0, args.max_sum],
           "value_sets_disjoint": set(range(args.fit_min, args.fit_max + 1)).isdisjoint(range(0, args.max_sum + 1)),
           "case_set_hash": case_set_hash, "case_set_exhaustive": len(ic_cases) == len(all_ic),
           "all_cases_processed": all(v.get("all_cases_processed", True) for v in ablation.values()),
           "skipped_case_keys": {f: v.get("skipped_case_keys", []) for f, v in ablation.items()
                                 if v.get("skipped_case_keys")},
           "controls_norm_matched": True, "hook_rel_error": hook_err,
           "readout": "restricted_digit_choice_accuracy (argmax over 0..9, single-digit sums)",
           "fit_r2": fit["r2"], "ablation": ablation, "interchange": interchange}
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out_dir, f"necessity_{tag}_L{args.layer}_{args.intervention_pos}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {path}\n")


if __name__ == "__main__":
    main()

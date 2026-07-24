#!/usr/bin/env python
"""Register the EXACT expected FILE cells into a run manifest (audit r8/r9).

Called by run_overnight.sh after layers are frozen and eligibility is measured. It:
  * reads the frozen layer + pinned model revision per model (r8 #6/#7);
  * STRICTLY validates the eligibility artifact -- schema, commit, revision, and that EVERY requested
    (model, form) is present with finite accuracy and exact case coverage; a missing entry FAILS
    registration rather than silently becoming ineligible (r9 #3);
  * writes ONE FILE-level expected cell per (model, experiment[, position]) with its own
    `expected_forms` attached (r9 #1/#2) -- never one duplicate cell per form;
  * separates preregistered-ineligible necessity forms (kept out, recorded) from testable ones (r9 #6);
  * marks each cell required_primary / required_secondary (issue #7);
  * freezes the writer-side experiment policy (issue #8).

Prints the jobs the runner should launch (tab-delimited), one per line:
    transport <model> <layer> <space-separated forms>
    necessity <model> <layer> <position> <space-separated eligible forms>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.provenance import E_DELTA, E_ABLATION, git_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--eligibility", required=True)
    p.add_argument("--transport-forms", nargs="+", required=True)
    p.add_argument("--necessity-forms", nargs="+", required=True)
    # writer-side experiment policy to freeze (issue #8)
    p.add_argument("--fit-min", type=int, default=10)
    p.add_argument("--fit-max", type=int, default=99)
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--k-pca", type=int, default=C.K_PCA)
    p.add_argument("--transport-ctrl-seeds", type=int, default=8)
    p.add_argument("--necessity-seeds", type=int, default=10)
    p.add_argument("--ctrl-candidates", type=int, default=60)
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    man_path = os.path.join(args.run_dir, "manifest.json")
    man = json.load(open(man_path))
    lman = json.load(open(args.layers))
    elig = json.load(open(args.eligibility))
    g = git_metadata()

    # ---- STRICT eligibility-artifact validation (r9 #3) ----
    if elig.get("schema_version") != C.SCHEMA_VERSION:
        raise SystemExit(f"eligibility schema {elig.get('schema_version')} != {C.SCHEMA_VERSION}")
    if elig.get("experiment_type") != "behavioral_eligibility":
        raise SystemExit("eligibility file is not a behavioral_eligibility artifact")
    if g["code_commit"] and elig.get("code_commit") not in (None, g["code_commit"]):
        raise SystemExit(f"eligibility commit {elig.get('code_commit')} != current {g['code_commit']}")
    if elig.get("max_sum") != args.max_sum or list(elig.get("addends", [])) != list(args.addends):
        raise SystemExit("eligibility calibration config (addends/max_sum) does not match this run")

    pol = man["analysis_policy"]
    thr = pol["clean_accuracy_threshold"]
    primary_pos = pol["primary_necessity_position"]
    positions = [primary_pos] + [p for p in pol["secondary_necessity_positions"] if p != primary_pos]

    cells, jobs, ineligible = [], [], {}
    for model in man["expected_models"]:
        lentry = (lman.get("models") or {}).get(model)
        if lentry is None:
            raise SystemExit(f"no frozen layer for {model}")
        layer = int(lentry["selected_layer"])
        layer_rev = (lentry.get("model_revision") or {}).get("revision")
        m_elig = (elig.get("models") or {}).get(model)
        if m_elig is None:
            raise SystemExit(f"eligibility artifact has no entry for expected model {model}")
        if m_elig.get("model_revision") != layer_rev:
            raise SystemExit(f"{model}: eligibility revision {m_elig.get('model_revision')} != "
                             f"layer-manifest revision {layer_rev}")

        # transport: every requested form (logit steering meaningful even at imperfect accuracy)
        cells.append({"experiment_type": "transport", "model": model, "estimand": E_DELTA,
                      "layer": layer, "expected_forms": list(args.transport_forms),
                      "requirement": "required_primary"})
        jobs.append(f"transport\t{model}\t{layer}\t{' '.join(args.transport_forms)}")

        # necessity: only behaviourally competent forms, decided on measured accuracy BEFORE effects
        eligible = []
        for f in args.necessity_forms:
            fe = m_elig["forms"].get(f)
            if fe is None:                                   # NEVER default to zero (r9 #3)
                raise SystemExit(f"eligibility artifact missing {model}:{f} (requested necessity form)")
            acc = fe["clean_acc"]
            if acc is None or not (acc == acc):              # finite check (NaN != NaN)
                raise SystemExit(f"eligibility {model}:{f} accuracy is not finite")
            if fe.get("n_processed") != fe.get("n_expected"):
                raise SystemExit(f"eligibility {model}:{f} processed {fe.get('n_processed')} of "
                                 f"{fe.get('n_expected')} cases")
            if acc >= thr:
                eligible.append(f)
            else:
                ineligible[f"{model}:{f}"] = {"clean_acc": acc, "reason": f"clean_accuracy_below_{thr}"}
        for pos in positions:
            if eligible:
                cells.append({"experiment_type": "necessity", "model": model, "estimand": E_ABLATION,
                              "layer": layer, "ablation_position": pos,
                              "expected_forms": list(eligible),
                              "requirement": "required_primary" if pos == primary_pos else "required_secondary"})
                jobs.append(f"necessity\t{model}\t{layer}\t{pos}\t{' '.join(eligible)}")

    man["expected_cells"] = cells
    man["primary_hypothesis_families"] = C.PRIMARY_FAMILIES
    man["secondary_families"] = C.SECONDARY_FAMILIES
    man["transport_forms"] = list(args.transport_forms)
    man["necessity_forms"] = list(args.necessity_forms)
    man["necessity_ineligible_forms"] = ineligible
    man["eligibility_hash"] = __import__("hashlib").sha256(
        json.dumps(elig, sort_keys=True).encode()).hexdigest()[:16]
    man["required_fallback_count"] = 0
    # freeze the writer-side experiment design (issue #8)
    man["experiment_policy"] = {
        "fit_values": [args.fit_min, args.fit_max], "causal_values": [0, args.max_sum],
        "addends": args.addends, "exhaustive_cases": True, "k_pca": args.k_pca,
        "transport_control_seeds": args.transport_ctrl_seeds,
        "necessity_control_seeds": args.necessity_seeds,
        "transport_control_candidates": args.ctrl_candidates,
        "energy_matched_controls": True, "rng_seed": args.rng_seed,
        "readout": "restricted_single_token_digit_choice"}
    json.dump(man, open(man_path, "w"), indent=2)

    print(f"# registered {len(cells)} FILE cells; primary_necessity_position={primary_pos}; "
          f"{len(ineligible)} ineligible necessity form(s)", file=sys.stderr)
    print("\n".join(jobs))


if __name__ == "__main__":
    main()

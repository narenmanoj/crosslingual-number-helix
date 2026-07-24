#!/usr/bin/env python
"""Register the EXACT expected analysis cells into a run manifest (audit r8 blockers #2, #3, #4, #6).

Called by run_overnight.sh after layers are frozen and after a behavioural-eligibility pass. It:
  * reads the frozen layer per model from layers.json (so each expected cell pins its layer, #6);
  * reads the per-(model, form) clean accuracy measured in the eligibility pass;
  * marks necessity forms below the frozen clean-accuracy threshold `not testable` (#2), keeping them
    OUT of the expected-cell set and recording WHY in the manifest;
  * uses separate transport / necessity form lists including en_word so notation is isolated (#3);
  * registers ONE primary necessity position and the secondaries (#4).

Writes the updated manifest and prints the jobs the runner should actually launch, one per line:
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
from src.provenance import E_DELTA, E_ABLATION


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--eligibility", required=True, help="json: {model: {form: clean_acc}}")
    p.add_argument("--transport-forms", nargs="+", required=True)
    p.add_argument("--necessity-forms", nargs="+", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    man_path = os.path.join(args.run_dir, "manifest.json")
    man = json.load(open(man_path))
    layers = json.load(open(args.layers))["models"]
    elig = json.load(open(args.eligibility))
    pol = man["analysis_policy"]
    thr = pol["clean_accuracy_threshold"]
    primary_pos = pol["primary_necessity_position"]
    positions = [primary_pos] + [p for p in pol["secondary_necessity_positions"] if p != primary_pos]

    cells, jobs, ineligible = [], [], {}
    for model in man["expected_models"]:
        layer = int(layers[model]["selected_layer"])
        # transport: every requested form (logit steering is meaningful even at imperfect accuracy)
        cells.append({"experiment_type": "transport", "model": model, "estimand": E_DELTA, "layer": layer})
        jobs.append(f"transport\t{model}\t{layer}\t{' '.join(args.transport_forms)}")
        # necessity: only behaviourally competent forms (frozen threshold, decided before effects)
        accs = elig.get(model, {})
        eligible = [f for f in args.necessity_forms if float(accs.get(f, 0.0)) >= thr]
        for f in args.necessity_forms:
            if f not in eligible:
                ineligible[f"{model}:{f}"] = {"clean_acc": accs.get(f), "reason": f"clean_accuracy_below_{thr}"}
        for pos in positions:
            for f in eligible:
                cells.append({"experiment_type": "necessity", "model": model, "estimand": E_ABLATION,
                              "layer": layer, "ablation_position": pos})
            if eligible:
                jobs.append(f"necessity\t{model}\t{layer}\t{pos}\t{' '.join(eligible)}")

    man["expected_cells"] = cells
    man["primary_hypothesis_families"] = C.PRIMARY_FAMILIES
    man["secondary_families"] = C.SECONDARY_FAMILIES
    man["transport_forms"] = args.transport_forms
    man["necessity_forms"] = args.necessity_forms
    man["necessity_ineligible_forms"] = ineligible
    man["required_fallback_count"] = 0
    json.dump(man, open(man_path, "w"), indent=2)

    print(f"# registered {len(cells)} expected cells; primary_necessity_position={primary_pos}; "
          f"{len(ineligible)} ineligible necessity form(s)", file=sys.stderr)
    print("\n".join(jobs))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pre-intervention behavioural calibration (audit r8 #2, r9 #3).

Measures restricted digit-choice CLEAN accuracy per (model, form) BEFORE any intervention, at the
frozen layer manifest's PINNED model revision, so necessity eligibility is decided on behaviour alone
-- never after seeing intervention effects.

The output is a STAMPED production artifact (schema/commit/worktree/revision/config/case coverage), so
a stale, partial, or hand-edited calibration file cannot silently remove primary necessity tests
(a missing entry must fail registration, not default to zero).

    python scripts/measure_clean.py --layers layers.json --forms en_digit es_word ... --out elig.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D
from src.extract import load_model, continuation_answer_ids, model_revision
from src.provenance import git_metadata, model_commit

READOUT = "restricted_single_token_digit_choice"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", required=True, help="frozen layers.json (models + pinned revisions)")
    p.add_argument("--forms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--allow-dirty", action="store_true", default=False)
    return p.parse_args()


@torch.no_grad()
def clean_accuracy(model, tok, device, form, cases, max_sum, ans_ids):
    ok = n = 0
    for a, b in cases:
        a_str = D.FORMS[form].render(a)
        prompt = f"{a_str} + {b} = "
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        logits = model(**enc).logits[0, -1, :].float().cpu().numpy()
        pred = int(np.argmax([logits[ans_ids[v]] for v in range(0, max_sum + 1)]))
        ok += int(pred == a + b); n += 1
    return ok / max(n, 1), n


def main():
    args = parse_args()
    g = git_metadata()
    if not args.allow_dirty and (g["code_commit"] is None or g["dirty_worktree"]):
        raise SystemExit(f"\nRefusing to write an eligibility artifact from a dirty worktree ({g}).\n")
    lman = json.load(open(args.layers))
    if lman.get("schema_version") != C.SCHEMA_VERSION:
        raise SystemExit(f"layers.json schema {lman.get('schema_version')} != {C.SCHEMA_VERSION}")
    layers_hash = hashlib.sha256(json.dumps(lman, sort_keys=True).encode()).hexdigest()[:16]

    cases = [(a, b) for b in args.addends for a in range(0, args.max_sum + 1) if a + b <= args.max_sum]
    expected_keys = [[a, b] for (a, b) in cases]

    models = {}
    for model_id, entry in lman["models"].items():
        pinned = (entry.get("model_revision") or {}).get("revision")
        model, tok, device = load_model(model_id, args.device, revision=pinned)
        used_rev = model_commit({"model_revision": model_revision(model, model_id,
                                                                  revision=getattr(model, "_pinned_revision", None))})
        forms = {}
        for f in args.forms:
            ans_ids = continuation_answer_ids(tok, range(0, args.max_sum + 1))  # fail-fast readout check
            acc, n = clean_accuracy(model, tok, device, f, cases, args.max_sum, ans_ids)
            forms[f] = {"clean_acc": acc, "n_expected": len(cases), "n_processed": n}
            print(f"  {model_id} {f}: clean_acc={acc:.2f}")
        models[model_id] = {"model_revision": used_rev, "forms": forms}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {"schema_version": C.SCHEMA_VERSION, "experiment_type": "behavioral_eligibility", **g,
           "layers_manifest_hash": layers_hash, "readout": READOUT,
           "addends": args.addends, "max_sum": args.max_sum,
           "expected_case_keys": expected_keys, "models": models}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"eligibility -> {args.out}")


if __name__ == "__main__":
    main()

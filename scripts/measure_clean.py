#!/usr/bin/env python
"""Pre-intervention behavioural calibration (audit r8 blocker #2).

Measures restricted digit-choice CLEAN accuracy per (model, form) BEFORE any intervention, at the
frozen causal layer's model revision, so necessity eligibility is decided on behaviour alone -- never
after seeing intervention effects. Writes {model: {form: clean_acc}} for register_cells.py.

    python scripts/measure_clean.py --layers layers.json --forms en_digit es_word ... --out elig.json
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
from src.extract import load_model, continuation_answer_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", required=True, help="frozen layers.json (for models + pinned revisions)")
    p.add_argument("--forms", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--addends", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-sum", type=int, default=9)
    p.add_argument("--device", default=C.DEVICE)
    return p.parse_args()


@torch.no_grad()
def clean_accuracy(model, tok, device, form, addends, max_sum, ans_ids):
    cases = [(a, b) for b in addends for a in range(0, max_sum + 1) if a + b <= max_sum]
    ok = n = 0
    for a, b in cases:
        a_str = D.FORMS[form].render(a)
        prompt = f"{a_str} + {b} = "
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        logits = model(**enc).logits[0, -1, :].float().cpu().numpy()
        pred = int(np.argmax([logits[ans_ids[v]] for v in range(0, max_sum + 1)]))
        ok += int(pred == a + b); n += 1
    return ok / max(n, 1)


def main():
    args = parse_args()
    layers = json.load(open(args.layers))["models"]
    out = {}
    for model_id, entry in layers.items():
        rev = (entry.get("model_revision") or {}).get("revision")
        model, tok, device = load_model(model_id, args.device, revision=rev)
        ans_ids = continuation_answer_ids(tok, range(0, args.max_sum + 1))
        out[model_id] = {}
        for f in args.forms:
            acc = clean_accuracy(model, tok, device, f, args.addends, args.max_sum, ans_ids)
            out[model_id][f] = acc
            print(f"  {model_id} {f}: clean_acc={acc:.2f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"eligibility -> {args.out}")


if __name__ == "__main__":
    main()

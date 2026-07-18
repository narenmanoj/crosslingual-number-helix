#!/usr/bin/env python
"""Diagnostic: how does each surface form tokenize, and what does pooling='last' grab?

The cross-form comparison assumes we're comparing the model's representation of the same
QUANTITY. But forms tokenize very differently -- "37" (maybe 2 digit tokens) vs "thirty-seven"
(several word-piece tokens) vs Devanagari "३७" (multi-byte). pooling='last' takes only the
final sub-token of the number span, so if the language-axis alignment drops, we need to know
whether that's geometry or just "the last sub-token of a long word carries less of the value".

Usage:
    python scripts/inspect_tokenization.py
    python scripts/inspect_tokenization.py --model Qwen/Qwen2.5-7B --samples 0 7 37 90
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src import data as D


def number_span_tokens(tok, text: str, number_str: str):
    """Return (token_strings_for_number_span, last_token_string)."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    start = text.find(number_str)
    end = start + len(number_str)
    idxs = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    toks = [tok.decode([ids[i]]) for i in idxs]
    last = toks[-1] if toks else ""
    return toks, last


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--forms", nargs="+", default=C.FORMS or D.DEFAULT_FORMS)
    p.add_argument("--samples", type=int, nargs="+", default=[7, 37, 42, 90])
    p.add_argument("--max-num", type=int, default=99)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    numbers = list(range(0, args.max_num + 1))

    # --- per-form mean number-span token length over the full range ---
    print(f"\nTokenizer: {args.model}")
    print("=" * 78)
    print(f"NUMBER-SPAN TOKEN LENGTH over 0..{args.max_num}  (pooling='last' uses only the LAST)")
    print("-" * 78)
    print(f"  {'form':<20}{'axis':<10}{'mean_tok':>10}{'max_tok':>9}{'%>1tok':>9}")
    lengths = {}
    for f in args.forms:
        counts = []
        for n in numbers:
            rendered = D.FORMS[f].render(n)
            prompt = D.FORMS[f].template.format(x=rendered)
            toks, _ = number_span_tokens(tok, prompt, rendered)
            counts.append(len(toks))
        counts = np.array(counts)
        lengths[f] = counts
        print(f"  {f:<20}{D.FORMS[f].axis:<10}{counts.mean():>10.2f}{counts.max():>9d}"
              f"{100*(counts>1).mean():>8.0f}%")

    # --- concrete examples: what tokens does each form produce, and the 'last' one ---
    print("\n" + "=" * 78)
    print("EXAMPLE TOKENIZATIONS  (| separates tokens; [last] = what pooling='last' pools)")
    print("-" * 78)
    for n in args.samples:
        print(f"\n  n = {n}")
        for f in args.forms:
            rendered = D.FORMS[f].render(n)
            prompt = D.FORMS[f].template.format(x=rendered)
            toks, last = number_span_tokens(tok, prompt, rendered)
            shown = "|".join(repr(t)[1:-1] for t in toks)
            print(f"    {f:<20}{rendered!r:<22} -> {shown}    [last]={last!r}")

    print("\n" + "=" * 78)
    print("Read: if language/notation forms have mean_tok >> 1 while digit forms ~= 1-2,")
    print("the 'last sub-token' is representing a word-fragment, not the whole quantity ->")
    print("that alone can depress cross-form alignment. Compare against --pooling mean.\n")


if __name__ == "__main__":
    main()

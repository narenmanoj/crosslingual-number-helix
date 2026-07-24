#!/usr/bin/env python
"""Create an ISOLATED production run directory + manifest (audit r5 blocker #5).

A shared experiments/ directory lets a stale-but-schema-compatible file from an earlier job silently
enter a final report, and lets a partially-failed run be analyzed as if complete. A production run
gets its own directory, a manifest declaring what it MUST produce, and a commit it must all share.

    python scripts/new_run.py --run-id pilot01 \
        --models Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407 \
        --forms en_digit devanagari_digit arabic_indic_digit es_word fr_word

Prints the created directory; pass it to the experiment scripts via --out-dir, then validate with:

    python scripts/analyze_stats.py --out-dir <dir> --production
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.provenance import new_run_dir, write_manifest, git_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="short label, e.g. pilot01")
    p.add_argument("--root", default=C.OUT_DIR)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--experiments", nargs="+", default=["transport", "necessity", "structure"])
    p.add_argument("--forms", nargs="+",
                   default=["en_digit", "devanagari_digit", "arabic_indic_digit", "es_word", "fr_word"])
    p.add_argument("--allow-dirty", action="store_true", default=False,
                   help="permit a dirty/unknown worktree (NOT for production reports)")
    return p.parse_args()


def main():
    args = parse_args()
    g = git_metadata()
    if not args.allow_dirty and (g["code_commit"] is None or g["dirty_worktree"]):
        raise SystemExit(
            f"\nRefusing to open a production run: worktree is dirty or unknown ({g}).\n"
            "Commit your changes first so every result file shares one identifiable build,\n"
            "or pass --allow-dirty for a non-production scratch run.\n")
    run_dir = new_run_dir(args.root, args.run_id)
    man = write_manifest(run_dir, run_id=args.run_id, schema_version=C.SCHEMA_VERSION,
                         expected_models=args.models, expected_experiments=args.experiments,
                         expected_forms=args.forms, allow_dirty=args.allow_dirty)
    print(f"\nRun directory: {run_dir}")
    print(f"  commit  : {man['code_commit']}  (dirty={man['dirty_worktree']})")
    print(f"  schema  : {man['schema_version']}")
    print(f"  models  : {len(man['expected_models'])}   experiments: {man['expected_experiments']}")
    print(f"\nUse it:  --out-dir {run_dir}"
          f"\nThen:    python scripts/analyze_stats.py --out-dir {run_dir} --production\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ============================================================================
# MANIFEST-DRIVEN production runner (audit r6 blocker #3).
#
# The old version wrote into a shared experiments/ directory with hard-coded layers and then analyzed
# every admissible file it found -- so a stale JSON could silently stand in for a failed current job.
# This version:
#   1. requires a clean worktree and creates an ISOLATED run directory;
#   2. FREEZES layers with the independent protocol (scripts/select_layers.py);
#   3. writes a manifest enumerating the exact expected cells BEFORE running;
#   4. records success/failure for every job;
#   5. analyzes ONLY that directory, with --production (which rejects incomplete/mixed/stale runs).
# Exploratory sweeps are NOT part of the production job (set RUN_SWEEPS=1 to add them separately).
#
# Usage:
#   RUN_ID=pilot01 MODELS="Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407" bash scripts/run_overnight.sh
#   RUN_ID=smoke   MODELS="Qwen/Qwen2.5-1.5B" PAIRS=0 ALLOW_DIRTY=1 bash scripts/run_overnight.sh
# SCOPE: this is a CAUSAL-ONLY production run (transport + necessity). Geometry/structure jobs
# are a separate run; the manifest declares exactly that so the two cannot be confused.
# ============================================================================
set -uo pipefail
# NOTE: with `set -u`, an empty array must be expanded as "${arr[@]+"${arr[@]}"}" --
# plain "${arr[@]}" is an unbound-variable error on bash 3.2 (macOS default).
cd "$(dirname "$0")/.." || exit 1

if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi
RUN_ID="${RUN_ID:-run$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-experiments}"
MODELS="${MODELS:-Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407}"
FORMS="${FORMS:-en_digit devanagari_digit arabic_indic_digit es_word fr_word}"
POSITIONS="${POSITIONS:-last span after}"   # audit r6 #9: necessity at all three positions
export POSITIONS                             # the manifest builder reads this from the environment
NSEEDS="${NSEEDS:-10}"
CTRL_SEEDS="${CTRL_SEEDS:-8}"
PAIRS="${PAIRS:-0}"                          # 0 => ALL valid triples (audit r6 #8)
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
RUN_SWEEPS="${RUN_SWEEPS:-0}"
CLEAN_CACHE="${CLEAN_CACHE:-1}"

# PRODUCTION is the default. ALLOW_DIRTY=1 declares a SCRATCH run (not a reportable result), which
# also turns off the writers' --production contract -- layers frozen from a dirty tree cannot satisfy
# it, so leaving --production on would make every scratch job fail for the wrong reason.
DIRTY_FLAGS=(--no-allow-dirty); NEWRUN_FLAGS=(); PROD_FLAGS=(--production); MODE=production
if [[ "$ALLOW_DIRTY" == "1" ]]; then
  DIRTY_FLAGS=(--allow-dirty); NEWRUN_FLAGS=(--allow-dirty); PROD_FLAGS=(); MODE=scratch
fi

# ---------------------------------------------------------------- 1. isolated run dir + manifest
RUN_DIR="$("$PY" scripts/new_run.py --run-id "$RUN_ID" --root "$ROOT" \
            --models $MODELS --forms $FORMS --experiments transport necessity "${NEWRUN_FLAGS[@]+"${NEWRUN_FLAGS[@]}"}" \
            | sed -n 's/^Run directory: //p')"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "FATAL: could not create the run directory (dirty worktree? see scripts/new_run.py)"; exit 1
fi
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/run.log"
echo "=== $MODE run $RUN_ID -> $RUN_DIR ($(date)) ===" | tee "$LOG"
[[ "$MODE" == "scratch" ]] && echo "NOTE: scratch mode -- results are NOT reportable "\
  "(dirty worktree; writer --production contract disabled)" | tee -a "$LOG"

# ---------------------------------------------------------------- 2. freeze layers (independent)
LAYERS="$RUN_DIR/layers.json"
echo ">> freezing layers via the independent protocol" | tee -a "$LOG"
if ! "$PY" scripts/select_layers.py --models $MODELS --out "$LAYERS" \
      $([[ "$ALLOW_DIRTY" == "1" ]] && echo --allow-dirty) 2>&1 | tee -a "$LOG"; then
  echo "FATAL: layer discovery failed -- refusing to run causal jobs on unfrozen layers" | tee -a "$LOG"
  exit 1
fi

# ---------------------------------------------------------------- 3. register + run every job
"$PY" - "$RUN_DIR" $MODELS <<'PYCELLS' 2>&1 | tee -a "$LOG"
import json, sys, os
run_dir, models = sys.argv[1], sys.argv[2:]
sys.path.insert(0, os.getcwd())
import config as C
from src.provenance import E_DELTA, E_ABLATION
man = json.load(open(os.path.join(run_dir, "manifest.json")))
cells, positions = [], os.environ.get("POSITIONS", "last span after").split()
for m in models:
    cells.append({"experiment_type": "transport", "model": m, "estimand": E_DELTA})
    for p in positions:
        cells.append({"experiment_type": "necessity", "model": m, "estimand": E_ABLATION,
                      "ablation_position": p})
man["expected_cells"] = cells
man["primary_hypothesis_families"] = C.PRIMARY_FAMILIES
man["secondary_families"] = C.SECONDARY_FAMILIES
man["required_fallback_count"] = 0
json.dump(man, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
print(f"manifest: {len(cells)} expected cells; primary families {C.PRIMARY_FAMILIES}")
PYCELLS

record() {  # record(job_id, status, detail)
  "$PY" -c "import sys,os; sys.path.insert(0,os.getcwd());
from src.provenance import record_completion; record_completion(*sys.argv[1:])" "$RUN_DIR" "$1" "$2" "$3"
}

# PAIRS=0 means EXHAUSTIVE -- it must be passed through explicitly. Omitting the flag would
# silently fall back to the writer's own default (80 sampled triples), quietly defeating the
# "use every valid case" requirement.
PAIRS_FLAG=(--pairs-per-form "$PAIRS")

for MODEL in $MODELS; do
  echo "" | tee -a "$LOG"; echo "#### $MODEL ($(date +%H:%M:%S))" | tee -a "$LOG"

  echo ">> transport" | tee -a "$LOG"
  if "$PY" scripts/run_transport.py --model "$MODEL" --layer-manifest "$LAYERS" "${PROD_FLAGS[@]+"${PROD_FLAGS[@]}"}" \
        --forms $FORMS "${PAIRS_FLAG[@]+"${PAIRS_FLAG[@]}"}" --delta-ctrl-seeds "$CTRL_SEEDS" \
        --out-dir "$RUN_DIR" "${DIRTY_FLAGS[@]+"${DIRTY_FLAGS[@]}"}" 2>&1 | tee -a "$LOG"; then
    record "transport:$MODEL" ok ""
  else
    record "transport:$MODEL" failed "see logs/run.log"; echo "!! transport FAILED for $MODEL" | tee -a "$LOG"
  fi

  for POS in $POSITIONS; do          # audit r6 #9: last / span / after
    echo ">> necessity ($POS)" | tee -a "$LOG"
    if "$PY" scripts/run_necessity.py --model "$MODEL" --layer-manifest "$LAYERS" "${PROD_FLAGS[@]+"${PROD_FLAGS[@]}"}" \
          --intervention-pos "$POS" --forms $FORMS "${PAIRS_FLAG[@]+"${PAIRS_FLAG[@]}"}" --n-seeds "$NSEEDS" \
          --on-baseline-fallback error --out-dir "$RUN_DIR" "${DIRTY_FLAGS[@]+"${DIRTY_FLAGS[@]}"}" 2>&1 | tee -a "$LOG"; then
      record "necessity:$MODEL:$POS" ok ""
    else
      record "necessity:$MODEL:$POS" failed "see logs/run.log"
      echo "!! necessity($POS) FAILED for $MODEL" | tee -a "$LOG"
    fi
  done

  if [[ "$RUN_SWEEPS" == "1" ]]; then     # EXPLORATORY -- never part of the default production job
    "$PY" scripts/run_ablation_sweep.py --model "$MODEL" --forms $FORMS \
      --out-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
  fi

  if [[ "$CLEAN_CACHE" == "1" ]]; then
    CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL//\//--}"
    [[ -d "$CACHE" ]] && { echo ">> clearing $CACHE" | tee -a "$LOG"; rm -rf "$CACHE"; }
  fi
done

# ---------------------------------------------------------------- 4. production analysis
echo "" | tee -a "$LOG"
echo "=== production analysis (rejects incomplete / mixed / stale runs) ===" | tee -a "$LOG"
ANALYZE_FLAGS=(--global-fdr); [[ "$MODE" == "production" ]] && ANALYZE_FLAGS+=(--production)
"$PY" scripts/analyze_stats.py --out-dir "$RUN_DIR" "${ANALYZE_FLAGS[@]+"${ANALYZE_FLAGS[@]}"}" 2>&1 | tee -a "$LOG"
STATUS=$?
echo "" | tee -a "$LOG"
if [[ $STATUS -eq 0 ]]; then
  echo "=== RUN COMPLETE: $RUN_DIR ($(date)) ===" | tee -a "$LOG"
else
  echo "=== RUN REJECTED by production analysis -- see the message above ===" | tee -a "$LOG"
fi
exit $STATUS

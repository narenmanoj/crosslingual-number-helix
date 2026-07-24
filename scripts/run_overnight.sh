#!/usr/bin/env bash
# ============================================================================
# MANIFEST-DRIVEN, ELIGIBILITY-GATED production runner (audit r6/r7/r8).
#
# Pipeline:
#   1. clean worktree check -> ISOLATED run directory + manifest;
#   2. FREEZE layers with the independent protocol (pins each model's HF revision too);
#   3. behavioural CALIBRATION: measure clean accuracy per (model, form) BEFORE any intervention;
#   4. REGISTER exact expected cells -- layer-pinned; necessity only for behaviourally-eligible forms;
#      transport and necessity use SEPARATE form lists (transport includes en_word for notation);
#   5. run only the registered jobs, recording success/failure for each;
#   6. analyze ONLY that directory with --production (rejects incomplete/mixed/stale/dropped-cell runs).
# SCOPE: causal-only (transport + necessity). Geometry is a separate run.
#
#   RUN_ID=pilot01 MODELS="Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407" bash scripts/run_overnight.sh
#   RUN_ID=smoke MODELS="Qwen/Qwen2.5-1.5B" ALLOW_DIRTY=1 bash scripts/run_overnight.sh   # scratch
# ============================================================================
# NOTE: with `set -u`, an empty array must be expanded as "${arr[@]+"${arr[@]}"}" --
# plain "${arr[@]}" is an unbound-variable error on bash 3.2 (macOS default).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi
RUN_ID="${RUN_ID:-run$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-experiments}"
MODELS="${MODELS:-Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407}"
# SEPARATE form sets (audit r8 #2/#3). Transport includes en_word so the notation axis is isolated;
# necessity's set is behaviourally gated further, at run time, by measured clean accuracy.
TRANSPORT_FORMS="${TRANSPORT_FORMS:-$("$PY" -c 'import config as C; print(" ".join(C.TRANSPORT_FORMS))')}"
NECESSITY_FORMS="${NECESSITY_FORMS:-$("$PY" -c 'import config as C; print(" ".join(C.NECESSITY_FORMS))')}"
NSEEDS="${NSEEDS:-10}"
CTRL_SEEDS="${CTRL_SEEDS:-8}"
PAIRS="${PAIRS:-0}"                          # 0 => ALL valid triples
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
RUN_SWEEPS="${RUN_SWEEPS:-0}"
CLEAN_CACHE="${CLEAN_CACHE:-1}"
ALL_FORMS="$(printf '%s\n' $TRANSPORT_FORMS $NECESSITY_FORMS | sort -u | tr '\n' ' ')"

# PRODUCTION is default. ALLOW_DIRTY=1 = SCRATCH (not reportable; writer --production contract off).
DIRTY_FLAGS=(--no-allow-dirty); NEWRUN_FLAGS=(); PROD_FLAGS=(--production); MODE=production
if [[ "$ALLOW_DIRTY" == "1" ]]; then
  DIRTY_FLAGS=(--allow-dirty); NEWRUN_FLAGS=(--allow-dirty); PROD_FLAGS=(); MODE=scratch
fi

# ----------------------------------------------------------- 1. isolated dir + manifest
RUN_DIR="$("$PY" scripts/new_run.py --run-id "$RUN_ID" --root "$ROOT" \
            --models $MODELS --forms $ALL_FORMS --experiments transport necessity \
            "${NEWRUN_FLAGS[@]+"${NEWRUN_FLAGS[@]}"}" | sed -n 's/^Run directory: //p')"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "FATAL: could not create the run directory (dirty worktree? see scripts/new_run.py)"; exit 1
fi
mkdir -p "$RUN_DIR/logs"; LOG="$RUN_DIR/logs/run.log"; LAYERS="$RUN_DIR/layers.json"; ELIG="$RUN_DIR/eligibility.json"
echo "=== $MODE run $RUN_ID -> $RUN_DIR ($(date)) ===" | tee "$LOG"
[[ "$MODE" == "scratch" ]] && echo "NOTE: scratch mode -- results are NOT reportable" | tee -a "$LOG"
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true   # record the environment

# ----------------------------------------------------------- 2. freeze layers (+ pin revisions)
echo ">> freezing layers via the independent protocol" | tee -a "$LOG"
if ! "$PY" scripts/select_layers.py --models $MODELS --out "$LAYERS" \
      $([[ "$ALLOW_DIRTY" == "1" ]] && echo --allow-dirty) 2>&1 | tee -a "$LOG"; then
  echo "FATAL: layer discovery failed" | tee -a "$LOG"; exit 1
fi

# ----------------------------------------------------------- 3. behavioural calibration
echo ">> measuring clean accuracy (eligibility, pre-intervention)" | tee -a "$LOG"
if ! "$PY" scripts/measure_clean.py --layers "$LAYERS" --forms $NECESSITY_FORMS --out "$ELIG" \
      2>&1 | tee -a "$LOG"; then
  echo "FATAL: eligibility calibration failed" | tee -a "$LOG"; exit 1
fi

# ----------------------------------------------------------- 4. register exact expected cells
JOBS="$("$PY" scripts/register_cells.py --run-dir "$RUN_DIR" --layers "$LAYERS" --eligibility "$ELIG" \
        --transport-forms $TRANSPORT_FORMS --necessity-forms $NECESSITY_FORMS 2> >(tee -a "$LOG" >&2))"
[[ -z "$JOBS" ]] && { echo "FATAL: no jobs registered" | tee -a "$LOG"; exit 1; }

record() { "$PY" -c "import sys,os; sys.path.insert(0,os.getcwd());
from src.provenance import record_completion; record_completion(*sys.argv[1:])" "$RUN_DIR" "$1" "$2" "$3"; }
PAIRS_FLAG=(--pairs-per-form "$PAIRS")   # always explicit; 0 = exhaustive

# ----------------------------------------------------------- 5. run the registered jobs
while IFS=$'\t' read -r KIND MODEL LAYER A B; do
  [[ -z "${KIND:-}" ]] && continue
  echo "" | tee -a "$LOG"; echo "#### $KIND $MODEL @L$LAYER ($(date +%H:%M:%S))" | tee -a "$LOG"
  if [[ "$KIND" == "transport" ]]; then
    FORMS_JOB="$A"
    if "$PY" scripts/run_transport.py --model "$MODEL" --layer-manifest "$LAYERS" \
          "${PROD_FLAGS[@]+"${PROD_FLAGS[@]}"}" --forms $FORMS_JOB "${PAIRS_FLAG[@]}" \
          --delta-ctrl-seeds "$CTRL_SEEDS" --out-dir "$RUN_DIR" \
          "${DIRTY_FLAGS[@]+"${DIRTY_FLAGS[@]}"}" 2>&1 | tee -a "$LOG"; then
      record "transport:$MODEL" ok ""
    else record "transport:$MODEL" failed "see run.log"; echo "!! transport FAILED $MODEL" | tee -a "$LOG"; fi
  else                              # necessity: A=position  B=forms
    POS="$A"; FORMS_JOB="$B"
    if "$PY" scripts/run_necessity.py --model "$MODEL" --layer-manifest "$LAYERS" \
          "${PROD_FLAGS[@]+"${PROD_FLAGS[@]}"}" --intervention-pos "$POS" --forms $FORMS_JOB \
          "${PAIRS_FLAG[@]}" --n-seeds "$NSEEDS" --on-baseline-fallback error --out-dir "$RUN_DIR" \
          "${DIRTY_FLAGS[@]+"${DIRTY_FLAGS[@]}"}" 2>&1 | tee -a "$LOG"; then
      record "necessity:$MODEL:$POS" ok ""
    else record "necessity:$MODEL:$POS" failed "see run.log"; echo "!! necessity($POS) FAILED $MODEL" | tee -a "$LOG"; fi
  fi
done <<< "$JOBS"

for MODEL in $MODELS; do
  if [[ "$CLEAN_CACHE" == "1" ]]; then
    CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL//\//--}"
    [[ -d "$CACHE" ]] && { echo ">> clearing $CACHE" | tee -a "$LOG"; rm -rf "$CACHE"; }
  fi
done

# ----------------------------------------------------------- 6. production analysis
echo "" | tee -a "$LOG"; echo "=== production analysis ===" | tee -a "$LOG"
ANALYZE_FLAGS=(--global-fdr); [[ "$MODE" == "production" ]] && ANALYZE_FLAGS+=(--production)
"$PY" scripts/analyze_stats.py --out-dir "$RUN_DIR" "${ANALYZE_FLAGS[@]}" 2>&1 | tee -a "$LOG"
STATUS=$?
echo "" | tee -a "$LOG"
[[ $STATUS -eq 0 ]] && echo "=== RUN COMPLETE: $RUN_DIR ($(date)) ===" | tee -a "$LOG" \
                    || echo "=== RUN REJECTED by production analysis ===" | tee -a "$LOG"
exit $STATUS

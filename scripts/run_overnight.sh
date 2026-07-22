#!/usr/bin/env bash
# ============================================================================
# Overnight significance run: for every BASE (causal) model, collect the
# per-case data the three causal legs now log, then bootstrap CIs + paired
# significance tests over all of it. Designed to run unattended and survive a
# single model failing (gated/OOM) without killing the night.
#
#   legs per model:  transport (sufficiency)  |  necessity span (necessity +
#                    matched-source interchange)  |  ablation-layer sweep
#                    (held-out necessity Δ with its own bootstrap CI)
#   at the end:      scripts/analyze_stats.py -> experiments/stats_summary.json
#                    + stats_forest.png + a printed table with 95% CIs and p.
#
# HF cache is cleared PER MODEL (disk was the bottleneck: 250 GB fills fast).
# Instruct-only families (Aya) and the least-tested gated multimodal path
# (Gemma-4) are omitted here -- add them by hand once verified.
#
# Usage:
#   bash scripts/run_overnight.sh                 # full set, experiments/
#   OUT_DIR=exp_night2 bash scripts/run_overnight.sh
#   MODELS="Qwen/Qwen2.5-7B ibm-granite/granite-4.0-h-tiny-base" bash scripts/run_overnight.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${python}"
OUT_DIR="${OUT_DIR:-experiments}"
PAIRS="${PAIRS:-120}"          # transport / interchange cases per form (tighter CIs)
NSEEDS="${NSEEDS:-10}"         # necessity control-null seeds
SWEEP_SEEDS="${SWEEP_SEEDS:-5}"
STRIDE="${STRIDE:-2}"          # ablation-sweep layer stride (1 = every layer, slower)
CLEAN_CACHE="${CLEAN_CACHE:-1}"  # 0 to keep HF cache between models
mkdir -p "$OUT_DIR"

# model : causal-leg layer (from the representational sharing peak; see README H2 table).
# Format "org/name=LAYER". Edit LAYER if a fresh fit-and-align moves the peak.
MODEL_LAYERS_DEFAULT=(
  "Qwen/Qwen2.5-7B=14"
  "mistralai/Mistral-Nemo-Base-2407=22"
  "Qwen/Qwen3-8B-Base=18"
  "Qwen/Qwen3-14B-Base=20"
  "allenai/Olmo-3-1025-7B=16"
  "utter-project/EuroLLM-9B-2512=24"
  "ibm-granite/granite-4.0-h-tiny-base=20"
  "tiiuae/Falcon-H1-7B-Base=20"
  "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base=24"
)
# Olmo is English-only: script+notation forms only (no foreign number-words).
declare -A FORMS_OVERRIDE=(
  ["allenai/Olmo-3-1025-7B"]="en_digit devanagari_digit arabic_indic_digit fullwidth_digit en_word"
)
DEFAULT_FORMS="en_digit devanagari_digit arabic_indic_digit es_word fr_word"

# Optional MODELS="id1 id2" filter (matched against the ids above).
FILTER="${MODELS:-}"

hf_cache_dir() {  # org/name -> ~/.cache/huggingface/hub/models--org--name
  echo "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"
}

LOG="$OUT_DIR/overnight_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
echo "=== overnight run start: $(date) | OUT_DIR=$OUT_DIR PAIRS=$PAIRS NSEEDS=$NSEEDS ===" | tee "$LOG"

for entry in "${MODEL_LAYERS_DEFAULT[@]}"; do
  MODEL="${entry%%=*}"; LAYER="${entry##*=}"
  if [[ -n "$FILTER" && " $FILTER " != *" $MODEL "* ]]; then continue; fi
  FORMS="${FORMS_OVERRIDE[$MODEL]:-$DEFAULT_FORMS}"
  echo "" | tee -a "$LOG"
  echo "################################################################" | tee -a "$LOG"
  echo "## $MODEL  @ L$LAYER  ($(date +%H:%M:%S))" | tee -a "$LOG"
  echo "##   forms: $FORMS" | tee -a "$LOG"
  echo "################################################################" | tee -a "$LOG"

  # --- (1) SUFFICIENCY: cross-form transport ---
  echo ">> transport" | tee -a "$LOG"
  "$PY" scripts/run_transport.py --model "$MODEL" --layer "$LAYER" \
    --forms $FORMS --pairs-per-form "$PAIRS" --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

  # --- (2) NECESSITY (whole-span) + matched-source interchange ---
  echo ">> necessity (span)" | tee -a "$LOG"
  "$PY" scripts/run_necessity.py --model "$MODEL" --layer "$LAYER" --intervention-pos span \
    --forms $FORMS --pairs-per-form "$PAIRS" --n-seeds "$NSEEDS" --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

  # --- (3) NECESSITY layer sweep (held-out Δ with bootstrap CI) ---
  echo ">> ablation sweep" | tee -a "$LOG"
  "$PY" scripts/run_ablation_sweep.py --model "$MODEL" \
    --forms $FORMS --layer-stride "$STRIDE" --n-seeds "$SWEEP_SEEDS" --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

  # --- free the disk before the next model ---
  if [[ "$CLEAN_CACHE" == "1" ]]; then
    CACHE="$(hf_cache_dir "$MODEL")"
    if [[ -d "$CACHE" ]]; then
      echo ">> clearing HF cache: $CACHE" | tee -a "$LOG"
      rm -rf "$CACHE"
    fi
  fi
  echo "## done $MODEL ($(date +%H:%M:%S))" | tee -a "$LOG"
done

# ---------------------------------------------------------------------------
# Aggregate: bootstrap 95% CIs + paired significance across everything we ran.
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== analyze_stats (B=20000) ===" | tee -a "$LOG"
"$PY" scripts/analyze_stats.py --out-dir "$OUT_DIR" --b 20000 2>&1 | tee -a "$LOG"

echo "=== overnight run done: $(date) ===" | tee -a "$LOG"
echo "Results in $OUT_DIR/  (stats_summary.json, stats_forest.png, per-model *_L*.json)"

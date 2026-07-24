"""Default experiment configuration. Override any of these on the CLI."""

# Bump on any change to an estimand / control / intervention so pre- and post-change result JSONs are
# never silently aggregated (audit r3 #1). Every output file stamps `schema_version`.
#   1.x = pre-overhaul (centering bug, reconstruction transport, legacy interchange, unmatched controls)
#   2.0 = post round-2 (centered fit, delta transport, fail-fast tokens, hook asserts, FDR-on-perm)
#   2.1 = post round-3 (norm-matched necessity controls + per-seed, delta interchange, case keys,
#         rank-aware overlap propagated, schema versioning, sweeps marked stale)
#   2.2 = post round-4 (ENFORCED schema/experiment_type/estimand/analysis_status, delta transport is
#         the default estimand + legacy absolute patching opt-in, structured delta controls with
#         per-seed retention, cross-fit ablation baseline, cluster-level permutation p, git provenance)
SCHEMA_VERSION = "2.2"

# Base models are cleaner for number geometry than chat-tuned ones. The causal legs need BASE
# models (instruct models score ~0 clean_acc on "a + b = "); the representational legs (H2 +
# layer sweep) also run on instruct models.
# Fast iteration default (runs on a laptop GPU/MPS, even CPU for a smoke test):
MODEL = "Qwen/Qwen2.5-1.5B"

# --- verified model registry (2026-07-18; exact HF ids, all fit a 32 GB GPU in bf16) ---
# Each entry: id, family/org, gated?, arch, and which axes it supports.
MODELS = {
    # already run (2 causal + 1 representational-only families):
    "Qwen/Qwen2.5-7B":                        dict(org="Alibaba",  gated=False, arch="transformer", causal=True,  langs="multi"),
    "mistralai/Mistral-Nemo-Base-2407":       dict(org="Mistral",  gated=False, arch="transformer", causal=True,  langs="multi"),
    "CohereLabs/aya-23-8B":                   dict(org="Cohere",   gated=True,  arch="transformer", causal=False, langs="multi"),  # instruct -> repr. only
    # --- NEW: recent (2025-26) DROP-IN transformers (model.model.layers) ---
    "utter-project/EuroLLM-9B-2512":          dict(org="EU-consortium", gated=True,  arch="transformer", causal=True, langs="multi(EU)"),   # Dec 2025
    "Qwen/Qwen3-8B-Base":                     dict(org="Alibaba",  gated=False, arch="transformer", causal=True,  langs="multi(119)"),      # same-org control
    "Qwen/Qwen3-14B-Base":                    dict(org="Alibaba",  gated=False, arch="transformer", causal=True,  langs="multi(119)"),
    "allenai/Olmo-3-1025-7B":                 dict(org="AI2",      gated=False, arch="transformer", causal=True,  langs="EN-only"),         # script+notation axes only
    # --- NEW: hybrid / SSM (architecture diversity; residual stream still hookable) ---
    "ibm-granite/granite-4.0-h-tiny-base":    dict(org="IBM",      gated=False, arch="hybrid-mamba-moe",  causal=True, langs="multi(12)", layers="model.model.layers"),
    "tiiuae/Falcon-H1-7B-Base":               dict(org="TII",      gated=False, arch="hybrid-parallel",   causal=True, langs="multi(18)", layers="model.model.layers"),
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base":dict(org="NVIDIA",   gated=False, arch="hybrid-mamba",      causal=True, langs="multi(9)",  layers="model.backbone.layers"),  # trust_remote_code
    # --- NEW: Gemma 4 (newest + best multilingual + overlaps Gupta 2026); multimodal wrapper ---
    "google/gemma-4-E4B":                     dict(org="Google",   gated=True,  arch="transformer(multimodal-nested)", causal=True, langs="multi(140)", layers="model.model.language_model.layers"),
    "google/gemma-4-12B":                     dict(org="Google",   gated=True,  arch="transformer(multimodal-nested)", causal=True, langs="multi(140)", layers="model.model.language_model.layers"),
}
# src/extract.load_model handles trust_remote_code + the multimodal load class; src/patching
# get_decoder_layers resolves the layer path automatically. Gated models need `hf auth login`
# + license acceptance. Gemma-4 (multimodal) is the least-tested path -- verify a smoke run first.

NUMBERS = list(range(0, 100))          # 0..99, matching the original helix work
FORMS = None                           # None -> src.data.DEFAULT_FORMS
LAYER = "scan"                         # "scan" picks the layer with best mean helix R^2, or an int
POOLING = "mean"                       # "mean" over number span (primary: apples-to-apples
                                       # across forms w/ different token counts), "last", or
                                       # "prompt_last" (final carrier token). Report all 3.
K_PCA = 20
DEVICE = "auto"
OUT_DIR = "experiments"

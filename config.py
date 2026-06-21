"""Default experiment configuration. Override any of these on the CLI."""

# Base models are cleaner for number geometry than chat-tuned ones.
# Fast iteration default (runs on a laptop GPU/MPS, even CPU for a smoke test):
MODEL = "Qwen/Qwen2.5-1.5B"
# Scale-ups for the real run (use your cluster):
#   "Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B", "CohereForAI/aya-23-8B", "google/gemma-2-9b"

NUMBERS = list(range(0, 100))          # 0..99, matching the original helix work
FORMS = None                           # None -> src.data.DEFAULT_FORMS
LAYER = "scan"                         # "scan" picks the layer with best mean helix R^2, or an int
POOLING = "last"                       # "last" token of the number span, or "mean"
K_PCA = 20
DEVICE = "auto"
OUT_DIR = "experiments"

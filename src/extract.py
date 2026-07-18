"""Load a causal LM and extract residual-stream activations at the number token."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def model_revision(model, name: str) -> dict:
    """Reproducibility stamp: model id + resolved commit hash (if available) + dtype."""
    cfg = getattr(model, "config", None)
    return {
        "name": name,
        "commit_hash": getattr(cfg, "_commit_hash", None),
        "dtype": str(getattr(model, "dtype", None)),
    }


def pick_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    # bf16 on CUDA; fp32 elsewhere (mps bf16 support is flaky for some ops).
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(name: str, device: str | None = "auto"):
    device = pick_device(device)
    dtype = pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(name)
    if not tok.is_fast:
        raise RuntimeError(
            f"Tokenizer for {name} is not a fast tokenizer; offset mapping is required. "
            "Pick a model with a fast tokenizer."
        )
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, output_hidden_states=True)
    model.to(device).eval()
    return model, tok, device


def _number_token_indices(tok, text: str, number_str: str) -> list[int]:
    """Token indices (into the no-special-tokens encoding) overlapping the number span."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    start = text.find(number_str)
    if start < 0:
        raise ValueError(f"Could not locate {number_str!r} in {text!r}")
    end = start + len(number_str)
    idxs = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not idxs:
        raise ValueError(f"No tokens overlap number span for {text!r}")
    return idxs


@torch.no_grad()
def extract_form_activations(
    model,
    tok,
    device: str,
    prompts: list[tuple[int, str, str]],
    pooling: str = "last",
) -> dict[int, "np.ndarray"]:
    """Run each prompt, pull hidden states at the number token(s).

    Returns {layer_index: array[n_numbers, d_model]} where layer 0 == embeddings,
    1..L == transformer block outputs.
    """
    import numpy as np

    n_layers = model.config.num_hidden_layers + 1
    per_layer: list[list] = [[] for _ in range(n_layers)]

    for _, number_str, prompt in tqdm(prompts, desc="extract", leave=False):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc)
        hs = out.hidden_states  # tuple length n_layers, each [1, seq, d_model]
        seq_len = hs[0].shape[1]
        if pooling == "prompt_last":
            # final token of the whole prompt (the carrier after the number). The number is
            # fully consumed here; number-varying signal = the integrated value. Sidesteps the
            # "which sub-token of a multi-token number holds the value" problem across forms.
            sel = [seq_len - 1]
        else:
            idxs = _number_token_indices(tok, prompt, number_str)
            sel = idxs[-1:] if pooling == "last" else idxs  # "last" span token, else "mean"
        for layer in range(n_layers):
            vecs = hs[layer][0, sel, :].float().mean(0)  # [d_model]
            per_layer[layer].append(vecs.cpu().numpy())

    return {layer: np.stack(per_layer[layer], axis=0) for layer in range(n_layers)}

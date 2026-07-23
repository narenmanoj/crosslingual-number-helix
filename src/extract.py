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
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if not tok.is_fast:
        raise RuntimeError(
            f"Tokenizer for {name} is not a fast tokenizer; offset mapping is required. "
            "Pick a model with a fast tokenizer."
        )
    kw = dict(torch_dtype=dtype, output_hidden_states=True, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        # multimodal wrappers (e.g. Gemma-4 *ForConditionalGeneration) aren't registered under
        # AutoModelForCausalLM; text-only forward still works via the image-text-to-text class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    # normalize nested (multimodal) configs so downstream .config.num_hidden_layers / .hidden_size work,
    # and force output_hidden_states at both levels (multimodal wrappers ignore the top-level flag).
    cfg = model.config
    cfg.output_hidden_states = True
    tcfg = getattr(cfg, "text_config", None)
    if tcfg is not None:
        tcfg.output_hidden_states = True
        for attr in ("num_hidden_layers", "hidden_size"):
            if getattr(cfg, attr, None) is None and getattr(tcfg, attr, None) is not None:
                setattr(cfg, attr, getattr(tcfg, attr))
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


def validate_single_token_answers(tok, values, prompt: str = "3 + 4 = ") -> list:
    """Audit #9: verify each candidate answer is a SINGLE continuation token after the real prompt
    ending and decodes to the intended digit. The causal readout takes argmax over these token ids;
    a bad id (byte fragment, SentencePiece metaspace, or a multi-token answer) silently corrupts
    clean_acc and every shift. Returns [(value, reason), ...] for values that FAIL, so the caller can
    warn or drop the model from the single-token task. Empty list => the readout is clean."""
    base = tok(prompt, add_special_tokens=True)["input_ids"]
    bad = []
    for v in values:
        full = tok(prompt + str(v), add_special_tokens=True)["input_ids"]
        if full[:len(base)] != base:
            bad.append((v, "prompt is not a token-prefix after appending the answer (retokenization)"))
            continue
        cont = full[len(base):]
        if len(cont) != 1:
            bad.append((v, f"answer is {len(cont)} continuation tokens, not 1"))
        elif tok.decode(cont).strip() != str(v):
            bad.append((v, f"answer token decodes to {tok.decode(cont)!r}, not {v!r}"))
    return bad


def continuation_answer_ids(tok, values, prompt: str = "3 + 4 = ") -> dict:
    """FAIL-FAST answer-token ids (audit #2): derive each value's id from the ACTUAL continuation
    after `prompt`, asserting it is exactly ONE token that decodes to the value. NO silent last-token
    fallback -- raises ValueError listing every value that fails, so a model whose digits are not
    single tokens is omitted from the restricted single-token task rather than scored on a fragment.
    Returns {value: token_id}. `prompt` should match the readout's real ending ('a + b = ')."""
    bad = validate_single_token_answers(tok, values, prompt=prompt)
    if bad:
        raise ValueError(
            f"answer tokens are not clean single-token continuations after {prompt!r}: {bad}. "
            "This model needs full-continuation scoring for the restricted digit readout -- omit it "
            "from the single-token task (do NOT fall back to the last sub-token).")
    base = tok(prompt, add_special_tokens=True)["input_ids"]
    return {v: tok(prompt + str(v), add_special_tokens=True)["input_ids"][len(base):][0] for v in values}


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
        out = model(**enc, output_hidden_states=True)
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

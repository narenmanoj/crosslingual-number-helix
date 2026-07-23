"""Step 3: causal cross-form helix transport.

Claim to test (H3): the model *uses* the shared number helix, not just represents it. If we take
a number written in form B (e.g. Spanish "siete"), overwrite its residual-stream vector with the
en_digit helix's encoding of a DIFFERENT value n', and the model's downstream arithmetic shifts
toward n', then the en_digit helix subspace is functionally read regardless of surface form.

Central threat (Makelov et al. 2311.17030, "interpretability illusion"): a subspace patch can
change behavior via a dormant parallel pathway even if the patched subspace isn't the mechanism.
So a raw transport success is necessary, NOT sufficient. The controls that make it defensible:
  - subspace vs full patch: does moving ONLY the helix-subspace component suffice? (localizes it)
  - random-direction control: an equal-dim RANDOM subspace must NOT steer (kills the illusion)
  - within-form positive control: en_digit -> en_digit transport must work first
  - value-distance curve: the output should move toward n', graded by |n' - n|, not just "some number"

`run_transport.py` drives the experiment; this module holds the machinery.

hidden_states indexing: hidden_states[L] (L=0..n_layers) is the residual AFTER decoder block L-1
(L=0 = embeddings). So to intervene on the stream we fit at hs[L], we hook decoder layer L-1.
"""
from __future__ import annotations

import numpy as np
import torch

from src.helix import fourier_basis, DEFAULT_PERIODS, fit_helix
from src.alignment import orthonormal_basis


_LAYER_PATHS = (
    "model.layers",                 # Llama/Qwen/Mistral/EuroLLM/OLMo/Granite/Falcon-H1 (standard)
    "model.language_model.layers",  # Gemma-4 (multimodal wrapper -> text submodule)
    "backbone.layers",              # Nemotron-H (hybrid Mamba/attn)
    "model.backbone.layers",
)


def get_decoder_layers(model):
    """Return the list of decoder blocks whose output IS the residual stream.

    Handles the common paths across architectures (standard transformers, Gemma multimodal
    nesting, Nemotron-H hybrid). For hybrid models (Nemotron/Granite/Falcon-H1) a block may be
    Mamba/MLP/attention, but each still does norm->mixer->residual-add, so hooking its output
    captures the residual stream all the same."""
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError(
        "could not locate decoder layers; add this model's path to _LAYER_PATHS in src/patching.py")


@torch.no_grad()
def verify_hook_layer(model, tok, device, hook_layer: int, prompt: str = "The number 42 is") -> float:
    """Validate the core hooking assumption for THIS architecture: the forward-hook output of decoder
    block `hook_layer` equals `out.hidden_states[hook_layer + 1]` (our convention: hidden_states[L] is
    the output of block L-1, so we hook block L-1 to intervene on hidden_states[L]).

    Returns the relative L2 error. ~0 => the hook point is exactly the recorded residual state, so
    patches there are trustworthy. Large (e.g. final-norm layers, wrapper remaps, hybrid blocks whose
    tuple[0] isn't the state) => do NOT trust patches at that layer. Cheap: one forward pass.
    The audit (#4) flagged that this was assumed, not tested -- especially load-bearing for the
    cross-architecture claim."""
    captured = {}
    layer = get_decoder_layers(model)[hook_layer]

    def hook(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layer.register_forward_hook(hook)
    try:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, output_hidden_states=True)
    finally:
        handle.remove()
    hs = out.hidden_states[hook_layer + 1]  # block hook_layer output == hidden_states[hook_layer+1]
    num = float(torch.linalg.norm((captured["h"] - hs).float()))
    den = float(torch.linalg.norm(hs.float())) + 1e-8
    return num / den


def patch_residual(model, layer_idx: int, position: int, new_vector: torch.Tensor):
    """Forward hook overwriting the residual stream at (decoder layer_idx, position) with
    new_vector. Returns the handle; call .remove(). To affect hidden_states[L], use layer_idx=L-1."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:, position, :] = new_vector.to(hidden.dtype).to(hidden.device)
        return out

    return layer.register_forward_hook(hook)


def patch_residual_multi(model, layer_idx: int, pos_to_vec: dict):
    """Overwrite the residual at several positions at once (for whole-span interventions).
    pos_to_vec maps position -> torch vector. To affect hidden_states[L], use layer_idx=L-1."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        for p, v in pos_to_vec.items():
            hidden[:, p, :] = v.to(hidden.dtype).to(hidden.device)
        return out

    return layer.register_forward_hook(hook)


# --- helix geometry helpers (operate on a fit_helix(...) result dict) ---

def helix_reconstruct(fit: dict, values) -> np.ndarray:
    """Model-space vector the form's helix predicts for each value. [len(values), d_model].
    h_hat(n) = mean + ((B(n) - B_mean) @ W) @ PCA.components_   (see helix.fit_helix).

    Reuses the fit's stored nmax + periods so the linear term is normalized consistently with the
    fit (critical when reconstructing on a narrower range than the fit -- see fourier_basis), and the
    stored B_mean so the design matrix is centered EXACTLY as during the fit (else the reconstructed
    vectors are offset and every transport magnitude is wrong)."""
    B = fourier_basis(values, fit["periods"], nmax=fit["nmax"])   # [m, d_fourier]
    Bc = B - fit["B_mean"]                                        # center as in the fit
    Z = Bc @ fit["W"]                                            # [m, k] (PCA space)
    return fit["mean"] + Z @ fit["pca"].components_               # [m, d_model]


def helix_subspace_basis(fit: dict) -> np.ndarray:
    """Orthonormal basis [d_model, r] of the helix subspace (r = #Fourier features)."""
    return orthonormal_basis(fit["helix_dirs_model"])


def random_subspace_basis(r: int, d_model: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return orthonormal_basis(rng.standard_normal((r, d_model)))


def _proj(Q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project v onto the subspace spanned by Q's columns (Q orthonormal)."""
    return Q @ (Q.T @ v)


def make_patched_vector(h_orig: np.ndarray, target_vec: np.ndarray,
                        Q: np.ndarray | None = None, mode: str = "full") -> np.ndarray:
    """Build the replacement residual vector.

    mode="full":     replace the whole vector with the target reconstruction.
    mode="subspace": keep h_orig's component orthogonal to Q, set the in-Q component to the
                     target's in-Q component -> (I-QQ^T) h_orig + QQ^T target_vec.
    mode="random":   same formula but caller passes a RANDOM Q (control: should not steer).
    """
    if mode == "full":
        return np.asarray(target_vec, dtype=float)
    if mode in ("subspace", "random"):
        if Q is None:
            raise ValueError(f"mode={mode} requires a subspace basis Q")
        h_orig = np.asarray(h_orig, dtype=float)
        return h_orig - _proj(Q, h_orig) + _proj(Q, np.asarray(target_vec, dtype=float))
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Stronger controls (review response): the point of a control subspace is to remove/inject a
# comparable amount of *task-relevant* energy WITHOUT being the number-value directions. Haar-random
# in the full residual is too weak (it barely overlaps the high-variance manifold). These are:
#   - covariance_matched_basis: random r-dim subspace drawn WITHIN the top-k PCA of activations
#     (matches activation energy/covariance, a much stronger null).
#   - shuffled_fourier_basis: a helix subspace fit through the SAME pipeline on SHUFFLED value labels
#     (same fitting machinery + sparsity, but no real value structure).
#   - norm-matched injection (matched_injection): scale a control-subspace swap to the SIGNAL-subspace
#     perturbation norm, so "does it steer" isn't confounded by "it perturbs less".
# ---------------------------------------------------------------------------

def covariance_matched_basis(H: np.ndarray, r: int, top_k: int = 64, seed: int = 0) -> np.ndarray:
    """Random r-dim subspace inside the top-k PCA span of activations H (covariance-matched null)."""
    from sklearn.decomposition import PCA
    H = np.asarray(H, dtype=float)
    k = min(top_k, H.shape[0] - 1, H.shape[1])
    V = PCA(n_components=k, svd_solver="full").fit(H).components_        # [k, d_model] (deterministic)
    rng = np.random.default_rng(seed)
    return orthonormal_basis(rng.standard_normal((r, k)) @ V)


def shuffled_fourier_basis(H: np.ndarray, numbers, k_pca: int = 20, seed: int = 0) -> np.ndarray:
    """Helix subspace fit through the SAME pipeline on SHUFFLED value labels (structured null)."""
    rng = np.random.default_rng(seed)
    shuf = list(numbers)
    rng.shuffle(shuf)
    return helix_subspace_basis(fit_helix(np.asarray(H, dtype=float), shuf, k_pca=k_pca))


def subspace_energy(Q: np.ndarray, h: np.ndarray, mean: np.ndarray) -> float:
    """L2 norm removed by a mean-ablation of subspace Q at h: ||QQ^T (h-mean)||. Report per control
    so reviewers can see the intervention removed comparable energy (not just fewer/more dims)."""
    return float(np.linalg.norm(_proj(Q, np.asarray(h, float) - np.asarray(mean, float))))


def matched_injection(h_orig: np.ndarray, target_vec: np.ndarray,
                      Q_signal: np.ndarray, Q_control: np.ndarray) -> np.ndarray:
    """Control injection scaled to the SIGNAL-subspace perturbation norm (norm-matched random)."""
    h_orig = np.asarray(h_orig, dtype=float); target_vec = np.asarray(target_vec, dtype=float)
    d_sig = _proj(Q_signal, target_vec) - _proj(Q_signal, h_orig)
    d_ctrl = _proj(Q_control, target_vec) - _proj(Q_control, h_orig)
    nc = np.linalg.norm(d_ctrl)
    if nc > 1e-8:
        d_ctrl = d_ctrl * (np.linalg.norm(d_sig) / nc)
    return h_orig + d_ctrl

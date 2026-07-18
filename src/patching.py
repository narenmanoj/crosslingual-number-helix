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

from src.helix import fourier_basis, DEFAULT_PERIODS
from src.alignment import orthonormal_basis


def get_decoder_layers(model):
    """Works for Llama/Qwen/Mistral/Gemma-style HF models."""
    return model.model.layers


def patch_residual(model, layer_idx: int, position: int, new_vector: torch.Tensor):
    """Forward hook overwriting the residual stream at (decoder layer_idx, position) with
    new_vector. Returns the handle; call .remove(). To affect hidden_states[L], use layer_idx=L-1."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:, position, :] = new_vector.to(hidden.dtype).to(hidden.device)
        return out

    return layer.register_forward_hook(hook)


# --- helix geometry helpers (operate on a fit_helix(...) result dict) ---

def helix_reconstruct(fit: dict, values, periods=DEFAULT_PERIODS) -> np.ndarray:
    """Model-space vector the form's helix predicts for each value. [len(values), d_model].
    h_hat(n) = mean + (B(n) @ W) @ PCA.components_   (see helix.fit_helix)."""
    B = fourier_basis(values, periods)         # [m, d_fourier]
    Z = B @ fit["W"]                           # [m, k] (PCA space)
    return fit["mean"] + Z @ fit["pca"].components_   # [m, d_model]


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

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
    h_hat(n) = mean + (B(n) @ W) @ PCA.components_   (see helix.fit_helix).

    Reuses the fit's stored nmax + periods so the linear term is normalized consistently with the
    fit (critical when reconstructing on a narrower range than the fit -- see fourier_basis)."""
    B = fourier_basis(values, fit["periods"], nmax=fit["nmax"])   # [m, d_fourier]
    Z = B @ fit["W"]                                              # [m, k] (PCA space)
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
    V = PCA(n_components=k).fit(H).components_        # [k, d_model]
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

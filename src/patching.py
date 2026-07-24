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


def assert_hook_equivalence(model, tok, device, hook_layer: int, tol: float = 1e-3) -> float:
    """FAIL-FAST wrapper around verify_hook_layer (audit #4): raise if the hook point is not the
    recorded residual state (>tol or non-finite), so a causal run never silently patches a different
    tensor from the one Q was fit on. Returns the measured rel-error (save it in the output JSON).
    Every causal script should call this at startup."""
    err = verify_hook_layer(model, tok, device, hook_layer)
    if not np.isfinite(err) or err > tol:
        raise RuntimeError(
            f"hook mismatch at block {hook_layer}: rel-error={err:.3e} > tol={tol:.0e}. The forward-hook "
            "output is not hidden_states[L] for this architecture -- patches would hit a different "
            "representation than the fit. Locate the correct hook point before running causal legs.")
    return err


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

def top_pca_span_basis(H: np.ndarray, r: int, top_k: int = 64, seed: int = 0) -> np.ndarray:
    """Random r-dim subspace drawn UNIFORMLY inside the top-k PCA span of activations H. This is an
    activation-manifold / top-PCA-span control: it lands in the high-variance directions but does NOT
    reproduce the covariance eigenvalue spectrum (audit #14 -- hence the honest name; the JSON key
    'cov_matched' is retained for continuity but means this). For true covariance matching, weight the
    coefficients by the PCA eigenvalues before orthogonalization."""
    from sklearn.decomposition import PCA
    H = np.asarray(H, dtype=float)
    k = min(top_k, H.shape[0] - 1, H.shape[1])
    V = PCA(n_components=k, svd_solver="full").fit(H).components_        # [k, d_model] (deterministic)
    rng = np.random.default_rng(seed)
    return orthonormal_basis(rng.standard_normal((r, k)) @ V)


# Backward-compatible alias: the function was formerly (mis)named covariance_matched_basis.
covariance_matched_basis = top_pca_span_basis


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


def norm_match(vec: np.ndarray, target_norm: float) -> np.ndarray:
    """Rescale `vec` to have L2 norm `target_norm` (no-op if `vec` is ~0). The single place norm-matching
    happens, so run_transport delta controls and the tests exercise the SAME code (audit r3 #9)."""
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    return vec * (target_norm / n) if n > 1e-8 else vec


def subspace_delta(diff: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Value displacement projected onto subspace Q: QQ^T diff (the delta-transport signal / control)."""
    return _proj(Q, np.asarray(diff, dtype=float))


ALPHA_LO, ALPHA_HI = 0.25, 4.0   # PREDEFINED admissibility band; declared before running (audit r5 #6)


def energy_matched_bank(sample_vectors, Q_signal, r: int, d_model: int, n_keep: int = 5,
                        n_candidates: int = 60, seed: int = 0, builder=None) -> tuple:
    """Select control subspaces whose NATURAL projected energy already resembles the signal's, so
    norm-matching is a mild rescale rather than an extrapolation (audit r5 #6, "better control
    generation").

    A Haar subspace in d~1500 captures almost none of an 8-d structured displacement, so alpha =
    ||signal proj|| / ||control proj|| is routinely ~8 and every control is inadmissible. Instead we
    draw `n_candidates` subspaces, score each by its mean projected norm over representative
    displacement vectors, and keep the `n_keep` closest to the signal's (log-ratio distance, so it is
    symmetric in over/under-shoot). The selection is reported, not hidden.

    builder(seed) -> [d_model, r] orthonormal basis; defaults to Haar.
    Returns (bases, report).
    """
    S = np.atleast_2d(np.asarray(sample_vectors, dtype=float))
    build = builder or (lambda sd: random_subspace_basis(r, d_model, seed=sd))
    target = float(np.mean([np.linalg.norm(_proj(Q_signal, v)) for v in S]))
    scored = []
    for i in range(n_candidates):
        Qc = build(seed + i)
        e = float(np.mean([np.linalg.norm(_proj(Qc, v)) for v in S]))
        dist = abs(np.log((e + 1e-12) / (target + 1e-12)))
        scored.append((dist, i, e, Qc))
    scored.sort(key=lambda t: t[0])
    keep = scored[:n_keep]
    return [t[3] for t in keep], {
        "selection": "energy_matched_bank",
        "n_candidates": n_candidates, "n_kept": len(keep),
        "signal_mean_proj_norm": target,
        "kept_seeds": [int(t[1]) for t in keep],
        "kept_mean_proj_norm": [float(t[2]) for t in keep],
        "implied_alpha": [float(target / t[2]) if t[2] > 1e-12 else float("inf") for t in keep],
    }


def norm_match_diag(raw: np.ndarray, target_norm: float) -> tuple:
    """Norm-match `raw` to `target_norm` AND return the full diagnostics needed to judge whether the
    'matched' control is a plausible intervention or an extrapolation (audit r5 #6).

    A control subspace that barely overlaps the displacement has a tiny raw norm, so alpha =
    target/raw explodes and the rescaled vector is far off-manifold. We keep raw_norm, matched_norm,
    alpha and an admissibility flag PER (case, seed) -- never just a median."""
    raw = np.asarray(raw, dtype=float)
    n_raw = float(np.linalg.norm(raw))
    alpha = (target_norm / n_raw) if n_raw > 1e-12 else float("inf")
    matched = norm_match(raw, target_norm)
    return matched, {"raw_norm": n_raw, "matched_norm": float(np.linalg.norm(matched)),
                     "alpha": alpha, "admissible": bool(ALPHA_LO <= alpha <= ALPHA_HI)}


def norm_matched_ablation(h: np.ndarray, mean: np.ndarray,
                          Q_signal: np.ndarray, Q_control: np.ndarray) -> np.ndarray:
    """Mean-ablate the Q_control subspace, but scaled so the REMOVED energy equals the helix
    (Q_signal) removed energy for this token (audit r3 #3). Removes ||QsQs^T(h-mean)|| worth of the
    control direction, so a control's effect can't be smaller merely because it removes less energy."""
    h, mean = np.asarray(h, float), np.asarray(mean, float)
    rs = _proj(Q_signal, h - mean)                      # helix removed vector
    rc = _proj(Q_control, h - mean)                     # control removed vector
    return h - norm_match(rc, np.linalg.norm(rs))       # ablate the control, energy-matched to helix

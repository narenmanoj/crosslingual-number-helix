"""Step-3 milestone skeleton: causal cross-form helix transport.

NOT the first experiment. Only build this once step-2 alignment looks promising.
This file gives you (a) a working residual-stream patch primitive, and (b) the
SCAFFOLD + control checklist for the transport experiment, so the design is in
front of you from the start.

The central threat (Makelov et al. 2311.17030, "interpretability illusion"): a
subspace patch can change behavior by activating a *dormant parallel pathway*
even when the patched subspace is NOT the model's real mechanism. So a successful
transport is necessary but NOT sufficient evidence of a shared helix. The controls
below are what make the causal claim defensible.
"""
from __future__ import annotations

import torch


def get_decoder_layers(model):
    """Works for Llama/Qwen/Mistral/Gemma-style HF models."""
    return model.model.layers


def patch_residual(model, layer_idx: int, position: int, new_vector: torch.Tensor):
    """Context-manager-style forward hook that overwrites the residual stream at
    (layer_idx, position) with `new_vector`. Returns the hook handle; call .remove()."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:, position, :] = new_vector.to(hidden.dtype).to(hidden.device)
        return out

    return layer.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# TRANSPORT EXPERIMENT (to implement once step 2 passes)
# ---------------------------------------------------------------------------
# Goal: fit the helix on form A (e.g. en_digit). Take a number n rendered in form B
# (e.g. es_word). Replace n's residual-stream vector with the form-A helix's
# coordinate for n, and check the model's downstream behavior (e.g. completing an
# arithmetic prompt) shifts to the value n -- using ONLY the A-helix subspace.
#
# REQUIRED CONTROLS (bake in from day one):
#   1. random-direction control: patch a random subspace of equal dim -> should NOT
#      reliably steer the value (guards the interpretability illusion).
#   2. full-activation patch vs subspace patch: if full-activation patching the A-vector
#      onto B works but the A-helix-subspace-only patch does not, the helix is not the
#      carrier (Makelov-style attribution check).
#   3. value specificity: patching toward n should move the output toward n, not just
#      "some other number" (report a value-distance curve, not a binary).
#   4. within-form sanity: A->A transport must work before A->B is meaningful.
#
# def transport_value(model, tok, helix_A, n, form_B_prompt, layer, position): ...

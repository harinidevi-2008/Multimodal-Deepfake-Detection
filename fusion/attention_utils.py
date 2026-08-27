"""
Non-causal summary of the fusion model's cross-attention weights.

IMPORTANT: attention weights describe which tokens the model attended
to when computing its output - they do NOT establish which modality
caused the prediction. Two modalities can be correlated (audio energy
and mouth movement both track the same speech, for instance) without
either one being "the reason" for a fake verdict, and attention can
concentrate on a modality that turns out not to matter. Treat the
output of this module as a descriptive "where did attention go"
summary only, never as a causal explanation. For a claim closer to
"which modality actually drove this prediction", use
modality_contribution.py's ablation-based method instead - and even
that is a heuristic, not a proof (see its own docstring).
"""

import numpy as np

MODALITY_ORDER_3 = ["visual", "audio", "semantic"]
MODALITY_ORDER_5 = ["visual", "audio", "semantic", "blink", "lipsync"]


def summarize_attention(attention_weights, modality_order):
    """
    attention_weights: torch.Tensor or np.ndarray, shape (B, N, N) or
        (N, N) - the second value FusionModel/EnhancedFusionModel's
        forward() returns (nn.MultiheadAttention's attention weights,
        already averaged over heads by PyTorch).
    modality_order: list of N modality names in the same stacking order
        the model used (MODALITY_ORDER_3 or MODALITY_ORDER_5).

    Returns {modality_name: mean_incoming_attention}: for each
    modality, the average attention every token paid TO it (mean over
    the "query" axis of that column) - the closest single number to
    "how much the fused representation drew on this modality's token"
    that attention weights alone can honestly support. This is a
    descriptive summary, not a causal attribution - see the module
    docstring above.
    """
    weights = attention_weights
    try:
        weights = weights.detach().cpu().numpy()
    except AttributeError:
        weights = np.asarray(weights)

    if weights.ndim == 3:
        weights = weights.mean(axis=0)  # average over the batch dimension

    n = len(modality_order)
    if weights.shape != (n, n):
        raise ValueError(
            f"attention_weights shape {weights.shape} does not match {n} modalities {modality_order}"
        )

    column_means = weights.mean(axis=0)  # incoming attention per key/modality
    return {name: round(float(v), 4) for name, v in zip(modality_order, column_means)}

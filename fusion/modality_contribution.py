"""
Ablation-based modality contribution: the preferred way to ask "how
much did modality X matter for this prediction", in contrast to
attention_utils.py's descriptive (explicitly non-causal) attention
summary.

Method: run the model once with every modality as given (the
baseline), then once more per modality with only that modality's input
zeroed out. The drop in predicted P(fake) when a modality is zeroed is
reported as that modality's contribution. This is still a heuristic,
not a ground-truth causal measurement - zeroing a projected input is
not the same as "this modality was never observed" (the
Linear->LayerNorm projection still runs on an all-zero vector and
produces some nonzero token, and the model was never trained on
zeroed inputs) - but it is a much more direct probe of the model's
actual sensitivity to each modality's input than reading off attention
weights, which only describe internal token interactions.
"""

import torch


@torch.no_grad()
def modality_contributions(model, inputs, modality_names):
    """
    model: the active three-stream FusionModel, already .eval()'d,
        whose forward(*inputs) returns (logits,
        attention_weights).
    inputs: list/tuple of tensors, one per modality in the exact order
        forward() expects, each shaped (1, dim) - batch size 1.
    modality_names: names matching `inputs`' order, e.g.
        visual/audio/semantic order.

    Returns:
        {
          "baseline_fake_probability": float,
          "contributions": {modality_name: fake_probability_drop, ...},
          "ablated_fake_probabilities": {modality_name: fake_probability_with_that_input_zeroed, ...}
        }
    A positive contribution means zeroing that modality REDUCED the
    predicted fake probability (its real input was pushing the
    prediction toward fake); a negative value means zeroing it
    INCREASED the fake probability (its real input was pushing toward
    real).
    """
    if len(inputs) != 3 or list(modality_names) != ["visual", "audio", "semantic"]:
        raise ValueError("Contribution diagnostics support visual/audio/semantic inputs only.")

    model.eval()

    baseline_logits, _ = model(*inputs)
    baseline_fake_prob = torch.softmax(baseline_logits, dim=1)[0, 1].item()

    contributions = {}
    ablated_fake_probabilities = {}
    for i, name in enumerate(modality_names):
        ablated_inputs = list(inputs)
        ablated_inputs[i] = torch.zeros_like(inputs[i])
        logits, _ = model(*ablated_inputs)
        fake_prob = torch.softmax(logits, dim=1)[0, 1].item()
        contributions[name] = round(baseline_fake_prob - fake_prob, 4)
        ablated_fake_probabilities[name] = round(fake_prob, 4)

    return {
        "baseline_fake_probability": round(baseline_fake_prob, 4),
        "contributions": contributions,
        "ablated_fake_probabilities": ablated_fake_probabilities,
    }

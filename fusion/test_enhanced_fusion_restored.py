"""
Checks that EnhancedFusionModel matches the documented design:
five 256-d tokens (visual, audio, semantic, blink, lip-sync), one 8-head
attention layer with a residual connection, mean pooling, and a
linear -> ReLU -> dropout -> linear head. No gates, no extra parameters.

Run directly:
    python fusion/test_enhanced_fusion_restored.py
"""

import json
import tempfile
from pathlib import Path

import torch

from enhanced_fusion_model import (
    MODALITY_ORDER,
    EnhancedFusionModel,
    check_checkpoint_architecture,
    default_reliability,
    load_enhanced_state_dict,
)


def make_inputs(batch=4):
    return (
        torch.randn(batch, 1280),
        torch.randn(batch, 768),
        torch.randn(batch, 384),
        torch.randn(batch, 4),
        torch.randn(batch, 2),
    )


def test_output_shapes_and_probabilities():
    model = EnhancedFusionModel().eval()
    logits, attention = model(*make_inputs())
    assert logits.shape == (4, 2)
    assert attention.shape == (4, 5, 5)
    probabilities = torch.softmax(logits, dim=1)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)


def test_parameter_names_have_no_gate_parameters():
    names = set(EnhancedFusionModel().state_dict())
    assert not any(name.startswith("reliability_gates") for name in names)
    assert "invalid_retention_logits" not in names
    assert "cross_attention.in_proj_weight" in names
    assert "classifier.3.weight" in names


def test_modality_order():
    assert MODALITY_ORDER == ["visual", "audio", "semantic", "blink", "lipsync"]


def test_legacy_gate_keys_are_dropped_and_output_is_unchanged():
    torch.manual_seed(0)
    reference = EnhancedFusionModel().eval()
    legacy_state = dict(reference.state_dict())
    for index in range(5):
        legacy_state[f"reliability_gates.{index}.weight"] = torch.zeros(1, 257)
        legacy_state[f"reliability_gates.{index}.bias"] = torch.zeros(1)
    legacy_state["invalid_retention_logits"] = torch.full((5,), -2.0)

    loaded = EnhancedFusionModel().eval()
    result, dropped = load_enhanced_state_dict(loaded, legacy_state, strict=True)
    assert len(dropped) == 11
    assert not result.missing_keys and not result.unexpected_keys

    inputs = make_inputs()
    with torch.no_grad():
        expected, _ = reference(*inputs)
        actual, _ = loaded(*inputs)
    assert torch.allclose(expected, actual)


def test_gated_checkpoint_is_rejected():
    with tempfile.TemporaryDirectory() as folder:
        weights = Path(folder) / "model.pt"
        Path(str(weights) + ".model_meta.json").write_text(
            json.dumps({"fusion_architecture": "gated"}), encoding="utf-8"
        )
        try:
            check_checkpoint_architecture(weights)
        except ValueError:
            return
        raise AssertionError("gated checkpoint was accepted")


def test_semantic_reliability_reporting_only():
    semantic = torch.zeros(2, 384)
    semantic[1, 0] = 1.0
    reliability = default_reliability(semantic)
    assert reliability.shape == (2, 5)
    assert reliability[0].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert reliability[1].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} checks passed")


if __name__ == "__main__":
    main()

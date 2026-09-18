import sys
from pathlib import Path

import numpy as np
import pytest
import torch

FUSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = FUSION_DIR.parent
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from enhanced_fusion_model import EnhancedFusionModel, MODALITY_ORDER, default_reliability, semantic_validity
from feature_normalization import apply_normalization, load_normalization


def _features(batch_size=2, semantic_zero=False):
    semantic = torch.zeros(batch_size, 384) if semantic_zero else torch.randn(batch_size, 384)
    return (
        torch.randn(batch_size, 1280),
        torch.randn(batch_size, 768),
        semantic,
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 2),
    )


def test_valid_semantic_vector_is_detected_and_gate_can_remain_active():
    visual, audio, semantic, blink, lipsync = _features(batch_size=1, semantic_zero=False)
    model = EnhancedFusionModel(fusion_architecture="gated")

    logits, attention, diagnostics = model(
        visual, audio, semantic, blink, lipsync, return_diagnostics=True
    )

    semantic_index = MODALITY_ORDER.index("semantic")
    assert attention is None
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    assert diagnostics["reliability"][0, semantic_index].item() == pytest.approx(1.0)
    assert diagnostics["gate_values"][0, semantic_index].item() > 0.0


def test_zero_semantic_vector_is_detected_and_reduced_without_nan():
    visual, audio, semantic, blink, lipsync = _features(batch_size=1, semantic_zero=True)
    model = EnhancedFusionModel(fusion_architecture="gated")

    valid_reliability = torch.ones(1, len(MODALITY_ORDER))
    _, _, valid_diag = model(
        visual, audio, semantic, blink, lipsync,
        reliability=valid_reliability, return_diagnostics=True,
    )
    logits, _, invalid_diag = model(
        visual, audio, semantic, blink, lipsync, return_diagnostics=True
    )

    semantic_index = MODALITY_ORDER.index("semantic")
    assert semantic_validity(semantic)[0, 0].item() == pytest.approx(0.0)
    assert invalid_diag["reliability"][0, semantic_index].item() == pytest.approx(0.0)
    assert invalid_diag["gate_values"][0, semantic_index] < valid_diag["gate_values"][0, semantic_index]
    assert torch.isfinite(logits).all()


def test_blink_remains_available_and_is_not_globally_suppressed():
    visual, audio, semantic, blink, lipsync = _features(batch_size=1)
    model = EnhancedFusionModel(fusion_architecture="gated")

    _, _, diagnostics = model(
        visual, audio, semantic, blink, lipsync, return_diagnostics=True
    )

    blink_index = MODALITY_ORDER.index("blink")
    assert diagnostics["reliability"][0, blink_index].item() == pytest.approx(1.0)
    assert diagnostics["gate_values"][0, blink_index].item() > 0.0


def test_attention_and_gated_full_five_modality_forward_passes_work():
    features = _features(batch_size=2)
    attention_model = EnhancedFusionModel()
    gated_model = EnhancedFusionModel(fusion_architecture="gated")

    attention_logits, attention = attention_model(*features)
    gated_logits, gated_attention = gated_model(*features)

    assert attention_logits.shape == (2, 2)
    assert attention.shape == (2, 5, 5)
    assert gated_logits.shape == (2, 2)
    assert gated_attention is None
    assert torch.isfinite(attention_logits).all()
    assert torch.isfinite(gated_logits).all()


def test_old_style_inference_call_still_works_for_attention_architecture():
    features = _features(batch_size=1, semantic_zero=True)
    model = EnhancedFusionModel()

    logits, attention = model(*features)

    assert logits.shape == (1, 2)
    assert attention.shape == (1, 5, 5)


def test_batch_inference_reliability_detection_works():
    visual, audio, semantic, blink, lipsync = _features(batch_size=3)
    semantic[1].zero_()
    reliability = default_reliability(semantic)
    model = EnhancedFusionModel(fusion_architecture="gated")

    logits, _, diagnostics = model(
        visual, audio, semantic, blink, lipsync,
        reliability=reliability, return_diagnostics=True,
    )

    semantic_index = MODALITY_ORDER.index("semantic")
    assert logits.shape == (3, 2)
    assert diagnostics["reliability"][:, semantic_index].tolist() == [1.0, 0.0, 1.0]


def test_runtime_failing_sample_is_accepted_and_semantic_is_reduced_if_present():
    job_root = REPO_ROOT / "runtime" / "jobs" / "0b918916502e41b5b5f42d0466bf5de2" / "features"
    required = {
        "visual": job_root / "visual" / "sample.npy",
        "audio": job_root / "audio" / "sample.npy",
        "semantic": job_root / "semantic" / "sample.npy",
        "blink": job_root / "blink" / "sample.npy",
        "lipsync": job_root / "lipsync" / "sample.npy",
    }
    if not all(path.exists() for path in required.values()):
        pytest.skip("runtime failing sample artifacts are not present in this checkout")

    normalization = load_normalization(REPO_ROOT / "fusion" / "feature_normalization.json")
    raw = {name: np.load(path).astype(np.float32) for name, path in required.items()}
    blink_raw = raw["blink"].copy()
    raw["blink"] = apply_normalization(raw["blink"], normalization["blink"])
    raw["lipsync"] = apply_normalization(raw["lipsync"], normalization["lipsync"])
    tensors = [torch.from_numpy(raw[name]).unsqueeze(0) for name in MODALITY_ORDER]

    model = EnhancedFusionModel(fusion_architecture="gated")
    logits, _, diagnostics = model(*tensors, return_diagnostics=True)

    semantic_index = MODALITY_ORDER.index("semantic")
    blink_index = MODALITY_ORDER.index("blink")
    assert raw["semantic"].shape == (384,)
    assert np.max(np.abs(raw["semantic"])) <= 1e-8
    assert diagnostics["reliability"][0, semantic_index].item() == pytest.approx(0.0)
    assert diagnostics["gate_values"][0, semantic_index].item() < diagnostics["gate_values"][0, blink_index].item()
    assert diagnostics["reliability"][0, blink_index].item() == pytest.approx(1.0)
    assert float(np.linalg.norm(blink_raw)) > 0.0
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_existing_attention_checkpoint_baseline_is_unchanged_if_present():
    weights = REPO_ROOT / "fusion" / "best_enhanced_fusion_model.pt"
    job_root = REPO_ROOT / "runtime" / "jobs" / "0b918916502e41b5b5f42d0466bf5de2" / "features"
    required = [job_root / name / "sample.npy" for name in MODALITY_ORDER]
    if not weights.exists() or not all(path.exists() for path in required):
        pytest.skip("current checkpoint or runtime failing sample artifacts are not present")

    normalization = load_normalization(REPO_ROOT / "fusion" / "feature_normalization.json")
    arrays = {name: np.load(job_root / name / "sample.npy").astype(np.float32) for name in MODALITY_ORDER}
    arrays["blink"] = apply_normalization(arrays["blink"], normalization["blink"])
    arrays["lipsync"] = apply_normalization(arrays["lipsync"], normalization["lipsync"])
    tensors = [torch.from_numpy(arrays[name]).unsqueeze(0) for name in MODALITY_ORDER]

    model = EnhancedFusionModel()
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    model.eval()
    with torch.inference_mode():
        logits, _ = model(*tensors)
        fake_probability = torch.softmax(logits, dim=1)[0, 1].item()

    assert fake_probability == pytest.approx(0.020487, abs=5e-4)

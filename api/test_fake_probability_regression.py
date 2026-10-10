import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.serializer import serialize_result
from api.errors import InferenceFailedError


def valid_result():
    return {
        "prediction": "DEEPFAKE", "final_fake_probability": 0.94,
        "final_real_probability": 0.06, "decision_threshold": 0.9,
        "visual_fake_probability": 0.87, "audio_fake_probability": 0.75,
        "semantic_fake_probability": 0.71, "blink_anomaly_score": 0.8,
        "blink_status": "Abnormal", "lip_sync_mismatch_score": 0.7,
        "lip_sync_status": "Inconsistent", "window_evidence": [],
        "frame_evidence": [], "_audio_evidence": {"status": "global_only", "envelope": []},
        "_blink_timeline": [], "_blink_events": [],
        "_feature_health": {"semantic": {"status": "valid"}},
        "_semantic_metadata": {"language": "en", "transcript": "", "segments": []},
        "evidence": {
            "visual_fake_probability": 0.87, "audio_fake_probability": 0.75,
            "semantic_fake_probability": 0.71, "blink_anomaly_score": 0.8,
            "blink_status": "Abnormal", "lip_sync_mismatch_score": 0.7,
            "lip_sync_status": "Inconsistent", "final_fake_probability": 0.94,
            "low_confidence": False, "evidence": ["Visual stream shows high fake probability"],
            "signals": [], "verdict": "Likely Deepfake", "evidence_consistency": "HIGH",
            "modality_disagreement": "LOW", "review_recommended": False,
            "confidence_level": "high",
        },
    }


def serialize(result):
    return serialize_result(
        result, {"fps": 24.0, "total_frames": 120, "duration_seconds": 5.0},
        job_id="job123", video_filename="demo.mp4", processing_time_seconds=1.25,
        analyzed_at="2026-09-24T00:00:00Z",
    )


def test_serializer_keeps_fused_fake_probability_not_real_probability():
    result = valid_result()
    response = serialize(result)

    assert response["final_verdict"] == "LIKELY_DEEPFAKE"
    assert (response["final_verdict"] == "LIKELY_DEEPFAKE") == (
        response["final_fake_probability"] >= response["decision_threshold"]
    )
    assert response["final_fake_probability"] == 0.94
    assert response["overall_fake_likelihood"] == response["final_fake_probability"]
    assert response["final_real_probability"] == 0.06
    assert math.isclose(response["final_fake_probability"] + response["final_real_probability"], 1.0)
    assert response["final_fake_probability"] > 0.5
    assert "not been calibrated" in response["model_confidence_note"]
    assert {key for key in response["modalities"] if response["modalities"][key]["kind"] == "learned_probability"} == {
        "visual", "audio", "semantic"
    }
    assert "attention_summary" not in response
    assert "modality_contributions" not in response
    assert "fusion_diagnostics" not in response
    assert response["supplementary_consistency"]["contributes_to_final_result"] is False
    assert {"video_filename", "video_duration_seconds", "analyzed_at", "processing_time_seconds",
            "prediction", "final_verdict", "final_fake_probability", "overall_fake_likelihood",
            "final_real_probability", "decision_threshold", "modalities", "evidence",
            "contextual_analysis"}.issubset(response)


def test_modality_scores_do_not_override_final_fusion_probability():
    first = valid_result()
    second = deepcopy(first)
    second.update(visual_fake_probability=0.01, audio_fake_probability=0.99, semantic_fake_probability=0.02)
    assert serialize(first)["overall_fake_likelihood"] == serialize(second)["overall_fake_likelihood"] == 0.94


@pytest.mark.parametrize("value", [None, "0.7", float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_or_missing_final_probability_raises_structured_error(value):
    result = valid_result()
    if value is None:
        result.pop("final_fake_probability")
    else:
        result["final_fake_probability"] = value
    with pytest.raises(InferenceFailedError, match="invalid final fake probability"):
        serialize(result)


def test_serializer_does_not_depend_on_obsolete_aggregation(monkeypatch):
    import api.serializer as serializer
    monkeypatch.setattr(serializer, "aggregate_learned_fake_probabilities", None, raising=False)
    assert serialize(valid_result())["overall_fake_likelihood"] == 0.94

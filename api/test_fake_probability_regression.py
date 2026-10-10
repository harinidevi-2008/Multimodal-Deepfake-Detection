import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.serializer import serialize_result


def test_serializer_keeps_fused_fake_probability_not_real_probability():
    result = {
        "prediction": "DEEPFAKE",
        "final_fake_probability": 0.94,
        "final_real_probability": 0.06,
        "decision_threshold": 0.9,
        "visual_fake_probability": 0.87,
        "audio_fake_probability": 0.75,
        "semantic_fake_probability": 0.71,
        "blink_anomaly_score": 0.8,
        "blink_status": "Abnormal",
        "lip_sync_mismatch_score": 0.7,
        "lip_sync_status": "Inconsistent",
        "window_evidence": [],
        "frame_evidence": [],
        "_audio_evidence": {"status": "global_only", "envelope": []},
        "_blink_timeline": [],
        "_blink_events": [],
        "_feature_health": {
            "visual": {"status": "valid"},
            "audio": {"status": "valid"},
            "semantic": {"status": "valid"},
            "blink": {"status": "valid"},
            "lipsync": {"status": "valid"},
        },
        "_semantic_metadata": {"language": "en", "transcript": "", "segments": []},
        "evidence": {
            "visual_fake_probability": 0.87,
            "audio_fake_probability": 0.75,
            "semantic_fake_probability": 0.71,
            "blink_anomaly_score": 0.8,
            "blink_status": "Abnormal",
            "lip_sync_mismatch_score": 0.7,
            "lip_sync_status": "Inconsistent",
            "final_fake_probability": 0.94,
            "low_confidence": False,
            "evidence": ["Visual stream shows high fake probability"],
            "signals": [],
            "verdict": "Likely Deepfake",
            "evidence_consistency": "HIGH",
            "modality_disagreement": "LOW",
            "review_recommended": False,
            "confidence_level": "high",
        },
    }

    response = serialize_result(
        result,
        {"fps": 24.0, "total_frames": 120, "duration_seconds": 5.0},
        job_id="job123",
        video_filename="demo.mp4",
        processing_time_seconds=1.25,
        analyzed_at="2026-09-24T00:00:00Z",
    )

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

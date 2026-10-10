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
        "visual_fake_probability": 0.87,
        "audio_fake_probability": 0.75,
        "semantic_fake_probability": 0.71,
        "blink_anomaly_score": 0.8,
        "blink_status": "Abnormal",
        "lip_sync_mismatch_score": 0.7,
        "lip_sync_status": "Inconsistent",
        "attention_summary": {"weights": {"visual": 0.4, "audio": 0.2, "semantic": 0.2, "eye_blink": 0.1, "lip_sync": 0.1}},
        "modality_contributions": {"contributions": {"visual": 0.2, "audio": 0.1, "semantic": 0.1, "blink": 0.1, "lipsync": 0.1}, "ablated_fake_probabilities": {"visual": 0.5, "audio": 0.6, "semantic": 0.7, "blink": 0.8, "lipsync": 0.9}},
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
    assert response["final_fake_probability"] == 0.94
    assert response["final_real_probability"] == 0.06
    assert math.isclose(response["final_fake_probability"] + response["final_real_probability"], 1.0)
    assert response["final_fake_probability"] > 0.5


def test_no_audio_serializer_excludes_audio_and_fusion_scores():
    result = {
        "prediction": "UNAVAILABLE",
        "final_fake_probability": None,
        "trained_fusion_fake_probability": None,
        "final_real_probability": None,
        "visual_fake_probability": 0.9,
        "audio_fake_probability": None,
        "semantic_fake_probability": 0.85,
        "blink_anomaly_score": 0.2,
        "blink_status": "Normal",
        "lip_sync_mismatch_score": None,
        "lip_sync_status": "unavailable",
        "attention_summary": {},
        "modality_contributions": {"contributions": {}, "ablated_fake_probabilities": {}},
        "window_evidence": "NOT AVAILABLE: no audio stream is available for lip-sync analysis.",
        "frame_evidence": [],
        "_no_audio_reason": "audio feature unavailable",
        "_audio_evidence": {"status": "unavailable", "reason": "No audio stream detected"},
        "_feature_health": {
            "visual": {"valid": True}, "audio": {"valid": False}, "semantic": {"valid": True},
            "blink": {"valid": True}, "lipsync": {"valid": False},
        },
        "_semantic_metadata": {"transcript": "", "language": "unknown", "segments": []},
        "_blink_timeline": [], "_blink_events": [],
        "evidence": {
            "low_confidence": True, "evidence_consistency": "UNKNOWN", "modality_disagreement": "UNKNOWN",
            "review_recommended": True, "confidence_level": "unavailable", "evidence": [], "signals": [],
        },
    }
    response = serialize_result(
        result, {"fps": 24.0, "duration_seconds": 5.0}, "job-no-audio", "silent.mp4", 1.0, "now"
    )
    assert 0.87 < response["overall_fake_likelihood"] < 0.88
    assert response["final_fake_probability"] is None
    assert response["trained_fusion_fake_probability"] is None
    assert response["modalities"]["audio"]["fake_probability"] is None
    assert response["modalities"]["lip_sync"]["mismatch_score"] is None
    assert response["evidence"]["audio_evidence_status"] == "unavailable"


def _overall_score(visual, audio, semantic, feature_health=None):
    from api.serializer import aggregate_learned_fake_probabilities

    return aggregate_learned_fake_probabilities(
        {"visual": visual, "audio": audio, "semantic": semantic},
        feature_health or {},
    )


def test_power_mean_scoring_synthetic_cases_and_bounds():
    agreed = _overall_score(0.99, 0.96, 0.94)
    disagreement = _overall_score(0.98, 0.78, 0.58)
    weak = _overall_score(0.30, 0.25, 0.20)
    missing_audio = _overall_score(0.90, None, 0.80)
    missing_semantic = _overall_score(0.90, 0.80, None)
    invalid_audio = _overall_score(0.90, 0.80, 0.70, {"audio": {"valid": False}})
    all_missing = _overall_score(None, None, None)

    assert agreed > 0.95
    assert 0.80 < disagreement < 0.98
    assert weak < 0.35
    assert missing_audio == _overall_score(0.90, None, 0.80)
    assert missing_semantic == _overall_score(0.90, 0.80, None)
    assert invalid_audio == _overall_score(0.90, None, 0.70)
    assert all_missing is None
    for value in (agreed, disagreement, weak, missing_audio, missing_semantic, invalid_audio):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert _overall_score(float("nan"), float("inf"), None) is None
    assert _overall_score(-1.0, None, 2.0) == _overall_score(0.0, None, 1.0)


def test_power_mean_scoring_handles_probability_endpoints():
    assert _overall_score(1.0, None, None) == 1.0
    assert _overall_score(0.0, None, None) == 0.0

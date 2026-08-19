"""
Demo of the evidence layer using example values, to show the exact
output shape - not a real inference run. Once the classifiers, blink
analyzer, lip-sync analyzer, and fusion model are all runnable on a
real video, replace these hardcoded numbers with their actual outputs.

Usage:
    python evidence/evidence_test.py
"""

import json

try:
    from .evidence_builder import build_evidence_report
except ImportError:
    from evidence_builder import build_evidence_report


def main():
    # Stand-ins for: classifiers/evaluate_classifier.py output per modality
    visual_probability = 0.81
    audio_probability = 0.67
    semantic_probability = 0.54

    # Stand-in for: eye_blink/blink_analyzer.py's analyze_blinks() output
    blink_result = {
        "blink_count": 3,
        "blink_rate_per_min": 4.2,
        "average_blink_duration_sec": 0.55,
        "blink_irregularity_score": 0.72,
        "blink_anomaly_score": 0.72,
        "blink_status": "Abnormal",
    }

    # Stand-in for: lipsync/lipsync_analyzer.py's analyze_lipsync() output
    lipsync_result = {
        "sync_score": 0.19,
        "mismatch_score": 0.81,
        "sync_status": "Inconsistent",
        "valid_frame_ratio": 0.94,
    }

    # Stand-in for: fusion/evaluate_fusion.py or evaluate_enhanced_fusion.py output
    final_fake_probability = 0.93

    report = build_evidence_report(
        visual_probability=visual_probability,
        audio_probability=audio_probability,
        semantic_probability=semantic_probability,
        blink_result=blink_result,
        lipsync_result=lipsync_result,
        final_fake_probability=final_fake_probability,
    )

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()

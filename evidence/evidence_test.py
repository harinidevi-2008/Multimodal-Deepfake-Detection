"""
Demo of the evidence layer using example values, to show the exact
output shape - not a real inference run. Once the classifiers, blink
analyzer, lip-sync analyzer, and fusion model are all runnable on a
real video, replace these hardcoded numbers with their actual outputs.

Updated alongside evidence_builder.py's terminology rename: parameter
names are now *_fake_probability, and the blink/lipsync stand-in dicts
below match blink_analyzer.py / lipsync_analyzer.py's current output
keys (blink_anomaly_score and sync_status were removed there as
duplicates of blink_irregularity_score and lipsync_status).

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
    visual_fake_probability = 0.81
    audio_fake_probability = 0.67
    semantic_fake_probability = 0.54

    # Stand-in for: eye_blink/blink_analyzer.py's analyze_blinks() output
    blink_result = {
        "blink_count": 3,
        "blink_rate_per_min": 4.2,
        "average_blink_duration_sec": 0.55,
        "blink_irregularity_score": 0.72,
        "blink_status": "Abnormal",
    }

    # Stand-in for: lipsync/lipsync_analyzer.py's analyze_lipsync() output
    lipsync_result = {
        "lipsync_consistency": 0.19,
        "sync_score": 0.19,
        "mismatch_score": 0.81,
        "lipsync_status": "Inconsistent",
        "valid_frame_ratio": 0.94,
    }

    # Stand-in for: fusion/evaluate_fusion.py or evaluate_enhanced_fusion.py output
    final_fake_probability = 0.93

    report = build_evidence_report(
        visual_fake_probability=visual_fake_probability,
        audio_fake_probability=audio_fake_probability,
        semantic_fake_probability=semantic_fake_probability,
        blink_result=blink_result,
        lipsync_result=lipsync_result,
        final_fake_probability=final_fake_probability,
    )

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()

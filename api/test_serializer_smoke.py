"""
Synthetic wiring smoke test for api/serializer.py, following the exact
same convention inference/test_full_pipeline_smoke.py already
established for full_pipeline.py: temporary synthetic checkpoints +
temporary synthetic precomputed features, so run_full_inference() and
serialize_result() are exercised for REAL - only the trained weights
and the five feature vectors are substituted, never the pipeline code
itself. The probabilities produced are meaningless; only the presence,
shape, and field-mapping of the result matter here.

This additionally uses a REAL (procedurally generated, faceless) test
video with a real audio track - not a real human face - to exercise
frame_evidence extraction, the blink/lip-sync video-based evidence
paths, and this module's own blink_timeline construction against
genuine cv2/mediapipe calls, not mocks. No face means blink events and
lip-sync windows come back empty, which is itself the correct, honest
behavior being verified (frame_evidence still falls back to
uniform_sample; window_evidence/blink_timeline are correctly empty/
omitted rather than fabricated) - see the print output for exactly
which evidence this run does and does not produce.

Usage: python api/test_serializer_smoke.py <path-to-a-real-video>
    (the video needs a real audio track; a real face is NOT required -
    see the module docstring above for what that does and doesn't prove)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))
sys.path.insert(0, str(REPO_ROOT / "visual" / "src"))

import full_pipeline  # noqa: E402
from full_pipeline import run_full_inference  # noqa: E402
from single_stream_classifier import SingleStreamClassifier  # noqa: E402
from enhanced_fusion_model import EnhancedFusionModel  # noqa: E402
from blink_analyzer import compute_ear_series, detect_blinks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.serializer import serialize_result  # noqa: E402

REQUIRED_TOP_LEVEL_KEYS = [
    "video_filename", "video_duration_seconds", "analyzed_at", "processing_time_seconds",
    "final_verdict", "final_fake_probability", "final_real_probability", "low_confidence",
    "model_confidence_note", "modalities", "attention_summary", "modality_contributions",
    "evidence", "contextual_analysis",
]


def _build_checkpoints_and_features(tmp_dir):
    tmp_dir = Path(tmp_dir)
    rel = Path("sample.npy")

    roots = {"visual": (tmp_dir / "visual", 1280), "audio": (tmp_dir / "audio", 768), "semantic": (tmp_dir / "semantic", 384)}
    for root, dim in roots.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / rel, np.random.randn(dim).astype(np.float32))

    blink_root = tmp_dir / "blink"
    (blink_root / rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(blink_root / rel, np.array([5.0, 18.0, 0.3, 0.62], dtype=np.float32))

    lipsync_root = tmp_dir / "lipsync"
    (lipsync_root / rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(lipsync_root / rel, np.array([0.35, 0.65], dtype=np.float32))

    checkpoints = {}
    for name, dim in [("visual", 1280), ("audio", 768), ("semantic", 384)]:
        model = SingleStreamClassifier(input_dim=dim)
        path = tmp_dir / f"best_{name}_classifier.pt"
        torch.save(model.state_dict(), path)
        checkpoints[name] = path
    fusion_path = tmp_dir / "best_enhanced_fusion_model.pt"
    torch.save(EnhancedFusionModel().state_dict(), fusion_path)
    checkpoints["enhanced_fusion"] = fusion_path

    return rel, roots, blink_root, lipsync_root, checkpoints


def main():
    if len(sys.argv) != 2:
        print("Usage: python api/test_serializer_smoke.py <path-to-a-real-video>", file=sys.stderr)
        sys.exit(2)
    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print(f"[FAIL] video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_seconds = (total_frames / fps) if fps else None
    meta = {"fps": fps, "total_frames": total_frames, "duration_seconds": duration_seconds}
    print(f"[INFO] probed video: fps={fps:.2f} total_frames={total_frames} duration={duration_seconds}")

    with tempfile.TemporaryDirectory() as tmp_dir_str, tempfile.TemporaryDirectory() as evidence_dir_str:
        rel, roots, blink_root, lipsync_root, checkpoints = _build_checkpoints_and_features(tmp_dir_str)
        evidence_dir = Path(evidence_dir_str)

        # Real compute_ear_series()/detect_blinks() pass, exactly like
        # api/pipeline_adapter.py's _extract_blink() does, to build a
        # genuine blink_events list (expected empty - no face in this
        # synthetic clip) and blink_timeline.
        ear_data = compute_ear_series(str(video_path))
        blink_events = detect_blinks(ear_data["ear_values"])
        blink_timeline = [
            {"timestamp_seconds": round(i / ear_data["fps"], 3) if ear_data["fps"] else None,
             "ear_value": round(float(v), 4), "is_blink": any(s <= i <= e for s, e in blink_events)}
            for i, v in enumerate(ear_data["ear_values"]) if v is not None
        ]
        print(f"[INFO] real compute_ear_series()/detect_blinks(): "
              f"{sum(1 for v in ear_data['ear_values'] if v is not None)}/{len(ear_data['ear_values'])} "
              f"frames had a detected face, {len(blink_events)} blink event(s), "
              f"blink_timeline has {len(blink_timeline)} entries.")

        result = run_full_inference(
            str(rel),
            visual_root=str(roots["visual"][0]), audio_root=str(roots["audio"][0]), semantic_root=str(roots["semantic"][0]),
            blink_root=str(blink_root), lipsync_root=str(lipsync_root),
            visual_classifier_weights=str(checkpoints["visual"]), audio_classifier_weights=str(checkpoints["audio"]),
            semantic_classifier_weights=str(checkpoints["semantic"]), enhanced_fusion_weights=str(checkpoints["enhanced_fusion"]),
            normalization_path=None,
            video_path=str(video_path), frame_output_dir=str(evidence_dir), blink_events=blink_events,
        )
        result["_blink_timeline"] = blink_timeline

        we = result["window_evidence"]
        we_desc = we if isinstance(we, str) else f"{len(we)} window(s)"
        print(f"[INFO] real run_full_inference(): frame_evidence={len(result['frame_evidence'] or [])} frame(s), "
              f"window_evidence={we_desc}")

        response = serialize_result(
            result, meta, job_id="smoketestjob00000000000000000000",
            video_filename=video_path.name, processing_time_seconds=1.234,
            analyzed_at="2026-08-27T00:00:00+00:00",
        )

        missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in response]
        if missing:
            print(f"[FAIL] serialize_result() output missing key(s): {missing}")
            sys.exit(1)
        print(f"[PASS] serialize_result() returned all {len(REQUIRED_TOP_LEVEL_KEYS)} required top-level keys.")

        for m in ("visual", "audio", "semantic", "eye_blink", "lip_sync"):
            if m not in response["modalities"]:
                print(f"[FAIL] modalities missing '{m}'")
                sys.exit(1)
        print("[PASS] modalities covers all 5 frontend modality keys (visual/audio/semantic/eye_blink/lip_sync).")

        if response["modalities"]["eye_blink"]["kind"] != "rule_based_score" or "anomaly_score" not in response["modalities"]["eye_blink"]:
            print("[FAIL] eye_blink modality is not shaped as a rule_based_score with anomaly_score.")
            sys.exit(1)
        if response["modalities"]["lip_sync"]["kind"] != "rule_based_score" or "mismatch_score" not in response["modalities"]["lip_sync"]:
            print("[FAIL] lip_sync modality is not shaped as a rule_based_score with mismatch_score.")
            sys.exit(1)
        if any("fake_probability" in response["modalities"][m] for m in ("eye_blink", "lip_sync")):
            print("[FAIL] eye_blink/lip_sync must NEVER carry a 'fake_probability' key (anomaly/mismatch score only).")
            sys.exit(1)
        print("[PASS] eye_blink/lip_sync are rule_based_score (anomaly_score/mismatch_score), never fake_probability - "
              "the terminology rule holds through the serializer.")

        for m in ("visual", "audio", "semantic"):
            if response["modalities"][m]["kind"] != "learned_probability" or "fake_probability" not in response["modalities"][m]:
                print(f"[FAIL] {m} modality is not shaped as a learned_probability with fake_probability.")
                sys.exit(1)
        print("[PASS] visual/audio/semantic are learned_probability with fake_probability.")

        weights = response["attention_summary"]["weights"]
        deltas = response["modality_contributions"]["deltas"]
        for m in ("visual", "audio", "semantic", "eye_blink", "lip_sync"):
            if m not in weights or m not in deltas:
                print(f"[FAIL] attention_summary/modality_contributions missing renamed key '{m}' "
                      f"(blink/lipsync -> eye_blink/lip_sync rename).")
                sys.exit(1)
        print("[PASS] attention_summary.weights and modality_contributions.deltas both use eye_blink/lip_sync "
              "(renamed from the pipeline's native blink/lipsync).")

        if response["evidence"].get("frames"):
            for f in response["evidence"]["frames"]:
                if not f["url"].startswith(f"/api/evidence/smoketestjob00000000000000000000/"):
                    print(f"[FAIL] evidence frame url does not point at this job's evidence route: {f['url']!r}")
                    sys.exit(1)
                if f["detected_artifact"] is not None:
                    print(f"[FAIL] detected_artifact must always be null (no real artifact detector exists), got {f['detected_artifact']!r}")
                    sys.exit(1)
            print(f"[PASS] evidence.frames has {len(response['evidence']['frames'])} frame(s), each with a "
                  f"job-scoped evidence URL and detected_artifact=null.")
        else:
            print("[INFO] evidence.frames is empty/omitted for this run (see frame_evidence count above).")

        modality_explanations = response.get("contextual_analysis", {}).get("modality_explanations", [])
        if not modality_explanations or len(modality_explanations) != 5:
            print(f"[FAIL] contextual_analysis.modality_explanations missing or incomplete: {modality_explanations!r}")
            sys.exit(1)
        for item in modality_explanations:
            required = {"modality", "label", "supports", "strength", "signal_meaning", "limitation", "explanation"}
            missing = sorted(required - set(item.keys()))
            if missing:
                print(f"[FAIL] modality explanation missing required keys {missing}: {item!r}")
                sys.exit(1)
        print("[PASS] contextual_analysis.modality_explanations includes all 5 modalities with honest signal explanations.")

        if "audio_waveform" in response["evidence"]:
            print("[FAIL] audio_waveform must be omitted entirely - no genuine envelope is computed anywhere in this pipeline.")
            sys.exit(1)
        print("[PASS] audio_waveform key correctly omitted (never fabricated).")

        print("\nNOTE: this run uses randomly-initialized, untrained checkpoints and synthetic random "
              "visual/audio/semantic/blink/lipsync feature vectors, plus a procedurally generated test "
              "video with no real human face. The final_fake_probability/verdict in this run are "
              "meaningless and must never be reported as an accuracy result - this verifies serializer "
              "wiring and field-mapping correctness only.")
        print("\nAll serializer synthetic smoke checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
Lightweight synthetic smoke test for inference/full_pipeline.py.

Does NOT run real FakeAVCeleb inference and does NOT use its result as
an accuracy measurement. The only purpose is to verify that
run_full_inference() - the one application-facing function this whole
hardening pass built up to - actually executes end to end and returns
every key the schema promises, using:
    - temporary synthetic .npy features (random vectors at the correct
      dimensions for visual/audio/semantic/blink/lipsync)
    - tiny, randomly-initialized SingleStreamClassifier / EnhancedFusionModel
      instances saved as temporary checkpoints (never trained - the
      probabilities produced are meaningless, only their presence and
      shape matter here)

No video_path is supplied, so this also exercises the "precomputed
features only" path: frame_evidence should be None, window_evidence
should be the explicit unavailable note (not fabricated), and
evidence_availability should reflect that honestly.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

# Importing full_pipeline has the side effect of inserting fusion/,
# classifiers/, evidence/, visual/src/lipsync, visual/src/eye_blink
# onto sys.path - see its own module docstring/top - so the plain
# imports below (matching how full_pipeline.py itself imports them)
# work afterwards without duplicating that setup here.
import full_pipeline  # noqa: E402
from full_pipeline import (  # noqa: E402
    run_full_inference,
    MissingFeatureFileError,
    FeatureShapeError,
)
from single_stream_classifier import SingleStreamClassifier  # noqa: E402
from enhanced_fusion_model import EnhancedFusionModel  # noqa: E402

REQUIRED_KEYS = [
    "final_fake_probability",
    "final_real_probability",
    "prediction",
    "visual_fake_probability",
    "audio_fake_probability",
    "semantic_fake_probability",
    "blink_anomaly_score",
    "blink_status",
    "lip_sync_mismatch_score",
    "lip_sync_status",
    "attention_summary",
    "modality_contributions",
    "evidence",
    "frame_evidence",
    "window_evidence",
    "normalization_applied",
    "evidence_availability",
]


def _build_fixture(tmp_dir):
    tmp_dir = Path(tmp_dir)
    rel = Path("FakeVideo-FakeAudio/African/men/id00001/00001_0_id00002_wavtolip.npy")

    roots = {
        "visual": (tmp_dir / "visual", 1280),
        "audio": (tmp_dir / "audio", 768),
        "semantic": (tmp_dir / "semantic", 384),
    }
    for name, (root, dim) in roots.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / rel, np.random.randn(dim).astype(np.float32))

    blink_root = tmp_dir / "blink"
    (blink_root / rel).parent.mkdir(parents=True, exist_ok=True)
    # [blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score]
    np.save(blink_root / rel, np.array([5.0, 18.0, 0.3, 0.62], dtype=np.float32))

    lipsync_root = tmp_dir / "lipsync"
    (lipsync_root / rel).parent.mkdir(parents=True, exist_ok=True)
    # [sync_score (=consistency), mismatch_score]
    np.save(lipsync_root / rel, np.array([0.35, 0.65], dtype=np.float32))

    checkpoints = {}
    for name, dim in [("visual", 1280), ("audio", 768), ("semantic", 384)]:
        model = SingleStreamClassifier(input_dim=dim)
        path = tmp_dir / f"best_{name}_classifier.pt"
        torch.save(model.state_dict(), path)
        checkpoints[name] = path

    fusion_model = EnhancedFusionModel()
    fusion_path = tmp_dir / "best_enhanced_fusion_model.pt"
    torch.save(fusion_model.state_dict(), fusion_path)
    checkpoints["enhanced_fusion"] = fusion_path

    return rel, roots, blink_root, lipsync_root, checkpoints


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        rel, roots, blink_root, lipsync_root, checkpoints = _build_fixture(tmp_dir)

        result = run_full_inference(
            str(rel),
            visual_root=str(roots["visual"][0]),
            audio_root=str(roots["audio"][0]),
            semantic_root=str(roots["semantic"][0]),
            blink_root=str(blink_root),
            lipsync_root=str(lipsync_root),
            visual_classifier_weights=str(checkpoints["visual"]),
            audio_classifier_weights=str(checkpoints["audio"]),
            semantic_classifier_weights=str(checkpoints["semantic"]),
            enhanced_fusion_weights=str(checkpoints["enhanced_fusion"]),
            normalization_path=None,  # no normalization file in this synthetic fixture - raw features
        )

        missing = [k for k in REQUIRED_KEYS if k not in result]
        if missing:
            print(f"[FAIL] run_full_inference() result is missing key(s): {missing}")
            sys.exit(1)
        print(f"[PASS] run_full_inference() returned all {len(REQUIRED_KEYS)} required schema keys.")

        if result["frame_evidence"] is not None:
            print(f"[FAIL] Expected frame_evidence=None with no video_path supplied, got "
                  f"{result['frame_evidence']!r}.")
            sys.exit(1)
        print("[PASS] frame_evidence is None when no video_path is supplied (honest, not fabricated).")

        if result["window_evidence"] != full_pipeline.WINDOW_EVIDENCE_UNAVAILABLE_NOTE:
            print(f"[FAIL] Expected the explicit window-evidence-unavailable note with no video_path, "
                  f"got {result['window_evidence']!r}.")
            sys.exit(1)
        print("[PASS] window_evidence is the explicit unavailable note when no video_path is supplied.")

        availability = result["evidence_availability"]
        expected_availability = {
            "numeric_stream_scores": True,
            "frame_evidence": False,
            "blink_event_frames": False,
            "lip_sync_window_evidence": False,
            "visual_artifact_detection": False,
        }
        if availability != expected_availability:
            print(f"[FAIL] evidence_availability mismatch.\n  expected: {expected_availability}\n"
                  f"  got:      {availability}")
            sys.exit(1)
        print("[PASS] evidence_availability correctly reflects precomputed-features-only execution.")

        if result["prediction"] not in ("REAL", "DEEPFAKE"):
            print(f"[FAIL] prediction should be 'REAL' or 'DEEPFAKE', got {result['prediction']!r}.")
            sys.exit(1)
        print(f"[PASS] prediction is a valid label ({result['prediction']!r}).")

        # ------------------------------------------------------------
        # Schema/consistency checks (item 12 of this hardening pass).
        # ------------------------------------------------------------
        prob_sum = result["final_fake_probability"] + result["final_real_probability"]
        if abs(prob_sum - 1.0) > 1e-3:
            print(f"[FAIL] final_fake_probability + final_real_probability = {prob_sum!r}, expected ~1.0.")
            sys.exit(1)
        print(f"[PASS] final_fake_probability + final_real_probability ≈ 1 ({prob_sum:.6f}).")

        expected_prediction = "DEEPFAKE" if result["final_fake_probability"] >= 0.5 else "REAL"
        if result["prediction"] != expected_prediction:
            print(f"[FAIL] prediction={result['prediction']!r} does not match the 0.5 threshold on "
                  f"final_fake_probability={result['final_fake_probability']!r} (expected {expected_prediction!r}).")
            sys.exit(1)
        print("[PASS] prediction agrees with the 0.5 threshold on final_fake_probability.")

        attention_summary = result["attention_summary"]
        contribution = result["modality_contributions"]
        five_modalities = {"visual", "audio", "semantic", "blink", "lipsync"}
        if not five_modalities.issubset(set(attention_summary.keys()) if isinstance(attention_summary, dict) else set()):
            print(f"[FAIL] attention_summary does not cover all five modalities: {attention_summary!r}")
            sys.exit(1)
        contribution_keys = set(contribution.get("contributions", {}).keys()) if isinstance(contribution, dict) else set()
        if not five_modalities.issubset(contribution_keys):
            print(f"[FAIL] modality_contributions['contributions'] does not cover all five modalities: {contribution!r}")
            sys.exit(1)
        print("[PASS] attention_summary and modality_contributions both cover all five modalities.")

        print("\nNOTE: this test uses randomly-initialized, untrained models and synthetic random "
              "features. The probabilities in this run are meaningless and must never be reported as "
              "an accuracy result - it verifies wiring/schema only.")

        # ------------------------------------------------------------
        # Input validation (item 13 of this hardening pass): an
        # incomplete or malformed sample must fail clearly and by name,
        # not reach a classifier or the fusion model silently.
        # ------------------------------------------------------------
        common_kwargs = dict(
            visual_root=str(roots["visual"][0]), audio_root=str(roots["audio"][0]),
            semantic_root=str(roots["semantic"][0]),
            visual_classifier_weights=str(checkpoints["visual"]),
            audio_classifier_weights=str(checkpoints["audio"]),
            semantic_classifier_weights=str(checkpoints["semantic"]),
            enhanced_fusion_weights=str(checkpoints["enhanced_fusion"]),
            normalization_path=None,
        )

        try:
            run_full_inference(
                str(rel), blink_root=str(Path(tmp_dir) / "does_not_exist_blink"),
                lipsync_root=str(lipsync_root), **common_kwargs,
            )
            print("[FAIL] Expected MissingFeatureFileError for a missing blink feature file, none was raised.")
            sys.exit(1)
        except MissingFeatureFileError as exc:
            if "blink" not in str(exc):
                print(f"[FAIL] MissingFeatureFileError was raised but does not name 'blink': {exc}")
                sys.exit(1)
            print("[PASS] A missing modality feature file raises MissingFeatureFileError naming that modality.")

        bad_shape_lipsync_root = Path(tmp_dir) / "bad_shape_lipsync"
        (bad_shape_lipsync_root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(bad_shape_lipsync_root / rel, np.array([0.35, 0.65, 0.1], dtype=np.float32))  # wrong: 3-d, not 2-d
        try:
            run_full_inference(
                str(rel), blink_root=str(blink_root), lipsync_root=str(bad_shape_lipsync_root), **common_kwargs,
            )
            print("[FAIL] Expected FeatureShapeError for a wrong-shape lipsync feature, none was raised.")
            sys.exit(1)
        except FeatureShapeError as exc:
            print(f"[PASS] A wrong-shape feature vector raises FeatureShapeError: {exc}")

        print("\nAll full_pipeline synthetic smoke checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

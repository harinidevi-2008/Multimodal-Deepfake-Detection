"""
Lightweight test: missing trained checkpoints -> a clean, structured
`missing_checkpoint` API error - without running any real video through
the pipeline (no face detection, no feature extraction, no fusion model
forward pass). This is the "model not trained yet" state this repo is
actually in today (only fusion/best_fusion_model.pt is shipped - see
fusion/env_defaults.py), so it must be provably handled, not just assumed.

Two checks:

1. `check_full_pipeline_raises_on_real_checkpoints()` - calls
   inference/full_pipeline.py's own `run_full_inference()` against this
   repo's REAL, currently-shipped checkpoint paths and confirms it still
   raises `MissingCheckpointError`. Proves the underlying "not trained
   yet" guard (`require_file()`) is intact and unmodified. Skips itself
   (does not fail) if a checkpoint has since actually been trained and
   placed there - this test verifies the guard, not a permanent absence.

2. `check_api_returns_structured_missing_checkpoint_error()` - a real
   HTTP request through the real FastAPI app (`api.server.app`) via
   TestClient, with `api.pipeline_adapter.run_raw_video_analysis`
   monkeypatched to raise `MissingCheckpointAPIError` directly (so this
   check doesn't need a real face-containing video or GPU time to
   exercise real feature extraction - that's covered separately by
   api/test_serializer_smoke.py and inference/test_full_pipeline_smoke.py).
   Confirms: HTTP 503, `error: "missing_checkpoint"`, and - important
   given this hardening pass - that the JSON body contains no traceback
   and no server filesystem path.

Run: python api/test_missing_checkpoint_error.py
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `import api.server` works when run as a script
sys.path.insert(0, str(REPO_ROOT / "inference"))

# Substrings that must never appear in a client-facing error body - any of
# these would mean a traceback or a server-side path leaked to the client.
FORBIDDEN_MARKERS = ('Traceback', 'File "', str(REPO_ROOT), '\\classifiers\\', 'C:\\')


def check_full_pipeline_raises_on_real_checkpoints():
    from full_pipeline import run_full_inference, MissingCheckpointError
    from env_defaults import (
        DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
        DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
        DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
        DEFAULT_ENHANCED_FUSION_WEIGHTS,
    )

    for name, rel in [
        ("visual classifier", DEFAULT_VISUAL_CLASSIFIER_WEIGHTS),
        ("audio classifier", DEFAULT_AUDIO_CLASSIFIER_WEIGHTS),
        ("semantic classifier", DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS),
        ("enhanced fusion", DEFAULT_ENHANCED_FUSION_WEIGHTS),
    ]:
        if (REPO_ROOT / rel).exists():
            print(f"[SKIP] {name} checkpoint is now present - training has evidently "
                  f"progressed since this repo's pre-training state; this specific "
                  f"pre-training check no longer applies.")
            return None

    try:
        run_full_inference(
            "sample.npy",
            visual_classifier_weights=str(REPO_ROOT / DEFAULT_VISUAL_CLASSIFIER_WEIGHTS),
            audio_classifier_weights=str(REPO_ROOT / DEFAULT_AUDIO_CLASSIFIER_WEIGHTS),
            semantic_classifier_weights=str(REPO_ROOT / DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS),
            enhanced_fusion_weights=str(REPO_ROOT / DEFAULT_ENHANCED_FUSION_WEIGHTS),
        )
    except MissingCheckpointError as exc:
        print("[PASS] run_full_inference() raises MissingCheckpointError against the real, "
              f"currently-shipped checkpoint paths: {str(exc).splitlines()[0]}")
        return True

    print("[FAIL] Expected MissingCheckpointError against the real checkpoint paths, none was raised.")
    return False


def check_api_returns_structured_missing_checkpoint_error():
    from fastapi.testclient import TestClient
    from api.server import app
    from api.errors import MissingCheckpointAPIError

    def fake_run_raw_video_analysis(video_path, job):
        # Stands in for the real pipeline call - only the error-mapping
        # and HTTP/JSON contract are under test here, not real feature
        # extraction or inference (see api/test_serializer_smoke.py and
        # inference/test_full_pipeline_smoke.py for those).
        raise MissingCheckpointAPIError(
            "The trained models required for analysis are not available yet.",
            details=(
                "Train the required stream classifiers and enhanced fusion model "
                "before running real inference."
            ),
        )

    client = TestClient(app)
    with patch("api.server.run_raw_video_analysis", side_effect=fake_run_raw_video_analysis):
        fake_video = io.BytesIO(b"stand-in bytes - this request never reaches real video decoding")
        response = client.post("/api/analyze", files={"video": ("clip.mp4", fake_video, "video/mp4")})

    if response.status_code != 503:
        print(f"[FAIL] expected HTTP 503, got {response.status_code}: {response.text}")
        return False

    body = response.json()
    if body.get("error") != "missing_checkpoint":
        print(f"[FAIL] expected error='missing_checkpoint', got {body!r}")
        return False
    if not body.get("message") or "details" not in body:
        print(f"[FAIL] response body missing a non-empty message and/or a details key: {body!r}")
        return False

    body_text = str(body)
    for marker in FORBIDDEN_MARKERS:
        if marker in body_text:
            print(f"[FAIL] response body leaked something that looks like a path/traceback "
                  f"({marker!r}): {body_text!r}")
            return False

    print(f"[PASS] POST /api/analyze returns a clean 503 missing_checkpoint JSON body, "
          f"no traceback or filesystem path: {body}")
    return True


def check_adapter_preflights_checkpoints():
    from api import pipeline_adapter
    from api.errors import MissingCheckpointAPIError
    from api.jobs import Job

    job = Job("preflight-test")
    with patch.object(
        pipeline_adapter,
        "_missing_checkpoint_paths",
        return_value=["classifiers/best_visual_classifier.pt"],
    ), patch.object(pipeline_adapter, "_extract_visual") as extract_visual:
        try:
            pipeline_adapter.run_raw_video_analysis(Path("does-not-exist.mp4"), job)
        except MissingCheckpointAPIError as exc:
            if exc.details != ["classifiers/best_visual_classifier.pt"]:
                print(f"[FAIL] preflight details were not preserved: {exc.details!r}")
                return False
        else:
            print("[FAIL] adapter did not reject missing checkpoints before extraction")
            return False

    if extract_visual.called:
        print("[FAIL] visual extraction ran before checkpoint preflight")
        return False
    print("[PASS] adapter rejects missing checkpoints before feature extraction")
    return True


def main():
    results = []

    r1 = check_full_pipeline_raises_on_real_checkpoints()
    if r1 is not None:
        results.append(r1)

    r2 = check_api_returns_structured_missing_checkpoint_error()
    results.append(r2)
    results.append(check_adapter_preflights_checkpoints())

    if all(results):
        print("\nAll missing-checkpoint error-path checks passed.")
        sys.exit(0)
    print("\nAt least one missing-checkpoint error-path check FAILED.")
    sys.exit(1)


if __name__ == "__main__":
    main()

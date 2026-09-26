"""Regression check: raw analysis computes lip-sync once and reuses it."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _job(root):
    roots = {name: root / name for name in ("visual", "audio", "semantic", "blink", "lipsync")}
    for directory in roots.values():
        directory.mkdir(parents=True, exist_ok=True)

    def feature_path(directory):
        return directory / "sample.npy"

    return SimpleNamespace(
        visual_root=roots["visual"], audio_root=roots["audio"], semantic_root=roots["semantic"],
        blink_root=roots["blink"], lipsync_root=roots["lipsync"], evidence_dir=root / "evidence",
        feature_path=feature_path,
    )


def main():
    from api import pipeline_adapter

    lipsync_result = {
        "lipsync_consistency": 0.3,
        "sync_score": 0.3,
        "mismatch_score": 0.7,
        "lipsync_status": "Inconsistent",
        "window_evidence": [{"start_frame": 0, "end_frame": 29, "mismatch_score": 0.7}],
    }
    captured = {}

    def fake_inference(*args, **kwargs):
        captured["precomputed_lipsync_result"] = kwargs["precomputed_lipsync_result"]
        return {}

    with tempfile.TemporaryDirectory() as directory:
        job = _job(Path(directory))
        with patch.object(pipeline_adapter, "_probe_video", return_value={"fps": 25.0, "total_frames": 50}), \
             patch.object(pipeline_adapter, "_extract_visual"), \
             patch.object(pipeline_adapter, "_extract_audio", return_value={"status": "unavailable"}), \
             patch.object(pipeline_adapter, "_extract_semantic", return_value={"reliable": True}), \
             patch.object(pipeline_adapter, "_extract_blink", return_value=([], [])), \
             patch.object(pipeline_adapter, "feature_health", return_value={"status": "valid"}), \
             patch.object(pipeline_adapter, "run_full_inference", side_effect=fake_inference), \
             patch.object(pipeline_adapter, "analyze_lipsync", return_value=lipsync_result) as analyze:
            result, _ = pipeline_adapter._run_raw_video_analysis(Path(directory) / "video.mp4", job)

            assert np.array_equal(
                np.load(job.lipsync_root / "sample.npy"), np.array([0.3, 0.7], dtype=np.float32)
            )

    assert analyze.call_count == 1, f"expected one lip-sync analysis, got {analyze.call_count}"
    assert captured["precomputed_lipsync_result"] is lipsync_result
    assert result["_blink_events"] == []
    print("[PASS] one lip-sync analysis generated both the feature vector and inference evidence.")


if __name__ == "__main__":
    main()

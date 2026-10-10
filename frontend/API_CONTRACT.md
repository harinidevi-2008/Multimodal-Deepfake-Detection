# Analysis API contract

`api/serializer.py` is the authoritative adapter from native inference results to this response. The final likelihood and verdict come from the three-stream `FusionModel` output. Visual, audio, and semantic cards show their individual classifier fake probabilities. Blink and lip-sync are supplementary only.

```json
{
  "video_filename": "upload.mp4",
  "video_duration_seconds": 12.4,
  "final_verdict": "LIKELY_REAL",
  "final_fake_probability": 0.12,
  "overall_fake_likelihood": 0.12,
  "final_real_probability": 0.88,
  "decision_threshold": 0.5,
  "modalities": {
    "visual": {"kind": "learned_probability", "fake_probability": 0.1},
    "audio": {"kind": "learned_probability", "fake_probability": 0.2},
    "semantic": {"kind": "learned_probability", "fake_probability": 0.3},
    "eye_blink": {"kind": "rule_based_score", "anomaly_score": null, "status": "Unavailable"},
    "lip_sync": {"kind": "rule_based_score", "mismatch_score": null, "status": "Unavailable"}
  },
  "supplementary_consistency": {"contributes_to_final_result": false},
  "evidence": {
    "frames": [],
    "audio_waveform": {"status": "global_only", "localized_windows": []},
    "lip_sync_window_evidence": [],
    "blink_timeline": []
  },
  "transcript": "",
  "transcript_segments": [],
  "contextual_analysis": {"detected_mismatch": null}
}
```

The audio waveform is a global envelope; it must not be described as a suspicious interval unless the backend returns localized windows with timestamps. Sampled frames are supporting context unless an actual visual artifact detector supplies a finding. Missing learned-stream features or incompatible checkpoints fail inference; they are never represented as zero scores.

Existing job endpoints remain unchanged: `POST /api/analyze`, `GET /api/analyze/{job_id}/status`, and `GET /api/media/{job_id}`.

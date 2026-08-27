# Frontend ⇄ Backend API contract

This is the contract the frontend already codes against (every component was
built and is still wired against this shape — see `src/mock/mockResult.*.json`
for worked examples). It was **not** derived from the real backend, because
this integration pass did not have access to the backend repository
(`inference/full_pipeline.py`, `fusion/*`, `evidence/*`, etc.) — only the
frontend side of the integration could be built and verified this pass. When
the backend is available, write **one** adapter/serializer that converts
`run_full_inference()`'s actual return value into this shape — don't scatter
field-name conversions across React components.

If it turns out easier to change field names on the backend than to keep
matching this exact shape, that's fine — just update this file and the
frontend in the same change, in one place (ideally `src/api/analyzeVideo.js`
plus a thin normalizer function), not by touching every component.

## `POST /api/analyze`

**Request:** `multipart/form-data`, field name `video`.

**Success response:** `200 OK`, `application/json`, shape below.

**Error response:** non-2xx, `application/json`:

```json
{
  "error": "missing_checkpoint",
  "message": "Human-readable, safe to show in the UI.",
  "details": "Optional extra context, also safe to show — no tracebacks."
}
```

`src/api/analyzeVideo.js` recognizes these `error` codes with built-in
copy (falls back to `message` for anything else): `missing_checkpoint`,
`unsupported_media_type`, `corrupt_video`, `feature_extraction_failed`,
`inference_failed`. Use these where they fit; add new codes freely — the
frontend shows `message`/`details` verbatim for unrecognized codes rather
than breaking.

## Success response shape

```jsonc
{
  "video_filename": "clip_0417_interview.mp4",
  "video_duration_seconds": 14.8,
  "analyzed_at": "2026-08-27T10:42:00+05:30",
  "processing_time_seconds": 9.4,

  "final_verdict": "LIKELY_DEEPFAKE",   // or "LIKELY_REAL" — map from `prediction` (DEEPFAKE/REAL)
  "final_fake_probability": 0.718,
  "final_real_probability": 0.282,
  "low_confidence": false,
  "model_confidence_note": "Model confidence reflects distance from the 0.5 decision boundary...",

  "modalities": {
    "visual":   { "label": "Visual",   "kind": "learned_probability", "fake_probability": 0.82 },
    "audio":    { "label": "Audio",    "kind": "learned_probability", "fake_probability": 0.64 },
    "semantic": { "label": "Semantic", "kind": "learned_probability", "fake_probability": 0.60 },
    "eye_blink": {
      "label": "Eye Blink", "kind": "rule_based_score",
      "anomaly_score": 0.75, "status": "Irregular"   // or "Normal" / "Abnormal" — see ModalityCard.jsx
    },
    "lip_sync": {
      "label": "Lip Sync", "kind": "rule_based_score",
      "mismatch_score": 0.67, "status": "Inconsistent"  // or "Consistent"
    }
  },

  "attention_summary": {
    "weights": { "visual": 0.32, "audio": 0.18, "semantic": 0.16, "eye_blink": 0.18, "lip_sync": 0.16 },
    "note": "Descriptive cross-attention weights... not a causal measure of importance."
  },
  "modality_contributions": {
    "deltas": { "visual": 0.25, "audio": 0.17, "semantic": 0.09, "eye_blink": 0.12, "lip_sync": 0.11 },
    "note": "Ablation-based: each value is the drop in the final fake probability when that modality's input is zeroed out..."
  },

  "evidence": {
    // Omit a key (or send an empty array / null) rather than fabricating —
    // EvidenceViewer.jsx renders an explicit "<X> evidence unavailable"
    // state per tab when a key is missing/empty. It never crashes on this.
    "frames": [
      {
        "frame_index": 42,
        "timestamp_seconds": 1.4,
        "url": "/api/evidence/<job_id>/frame_000042_eye_blink_event.jpg",
        "frame_selection_reason": "eye_blink_event",   // or "uniform_sample"
        "detected_artifact": null   // ALWAYS null unless a real artifact detector produced this
      }
    ],
    "audio_waveform": {
      "sample_rate_hz": 16000,
      "duration_seconds": 14.8,
      "envelope": [0.35, 0.41, "... genuine amplitude envelope, or omit this whole key"]
    },
    "lip_sync_window_evidence": [
      { "window_start_seconds": 3.2, "window_end_seconds": 4.0, "mismatch_score": 0.62 }
    ],
    "blink_timeline": [
      { "timestamp_seconds": 1.4, "ear_value": 0.09, "is_blink": true }
    ]
  },

  "contextual_analysis": {
    "detected_mismatch": "Lip movements do not consistently track the audio track between 3.2s and 5.6s.",
    "explanation": "Only ever built from evidence that actually exists (see §17 of the integration spec) — never a specific visual-artifact claim like 'the face boundary is warped' unless a real detector produced it."
  }
}
```

### Field notes / mapping from the pipeline's native names

- `final_verdict`: map from `prediction` (`"DEEPFAKE"` / `"REAL"`) →
  `"LIKELY_DEEPFAKE"` / `"LIKELY_REAL"`. (`OverallResultPanel.jsx` reads
  `final_verdict`.)
- `modalities.visual/audio/semantic.fake_probability` ← the three
  independently-trained single-stream classifiers' `*_fake_probability`
  outputs. Never derived from the fusion model internals.
- `modalities.eye_blink.anomaly_score` / `modalities.lip_sync.mismatch_score`
  ← the rule-based `blink_anomaly_score` / `lip_sync_mismatch_score`. **Never**
  relabel these as a "fake probability" — every component that renders them
  (`ModalityCard.jsx`, `modalityMeta.js`, evidence tab copy) is written
  around that distinction on purpose.
- `attention_summary` ← `summarize_attention()`'s output; keep its
  "descriptive, not causal" framing in `note` verbatim-ish.
- `modality_contributions` ← the ablation-based `modality_contribution.py`
  output (ΔP(fake) per zeroed modality); keep the "ablation, not ground
  truth" framing in `note`.
- `evidence.frames[].url` ← don't send a server filesystem path
  (`saved_path` / `C:\...`). Serve it through an evidence route (e.g.
  `GET /api/evidence/{job_id}/{filename}`) and put that route's URL here.
  A `data:` URI also works (that's what the bundled mock fixtures use).
- `evidence.frames[].detected_artifact` ← must stay `null` unless a real
  visual-artifact detector produced a value — matches
  `evidence_availability["visual_artifact_detection"] = False` in the
  pipeline today.
- `evidence.lip_sync_window_evidence` ← only include if
  `analyze_lipsync(video_path, compute_windows=True)` actually succeeded for
  this upload. If it's unavailable, omit the key (or send `[]`); do not
  fabricate windows. Same idea for `evidence.blink_timeline` (from
  `compute_ear_series()` / `detect_blinks()`) and `evidence.audio_waveform`.

### What the frontend does if a field is missing

- Missing/empty `evidence.*` arrays → that Evidence tab shows an explicit
  "___ evidence unavailable" message instead of an empty or broken chart.
- Missing `modalities.<x>.status` → the status badge just doesn't render;
  the score still does.
- A non-2xx response → the whole app shows `ErrorScreen` with the `error`
  code, `message`, and `details` from the JSON body — never a partial/blank
  results dashboard.

## `GET /api/health`

Not yet called by the frontend, but referenced in the integration spec as a
basic liveness check — implement as `{"status": "ok"}` when the backend
exists.

## Local dev wiring already in place on the frontend side

- `vite.config.js` proxies `/api/*` → `http://localhost:8000` (override with
  `BACKEND_URL` env var) so the dev server needs no CORS config.
- `VITE_API_BASE_URL` — set for a prod build where frontend/backend are on
  different origins; leave unset for local dev (relative `/api` + proxy).
- `VITE_USE_MOCK_API=true` — bypasses the network entirely and resolves
  `src/mock/mockResult.fake.json`, for UI work with no backend running. Off
  by default; the shipped/default path always calls the real API.

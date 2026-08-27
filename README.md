# Multimodal Deepfake Detection

A multimodal deepfake detection application: a Python/FastAPI backend that runs a real
five-stream (visual / audio / semantic / eye-blink / lip-sync) analysis pipeline over an
uploaded video, and a React frontend that displays the result.

**Integration status:** the application is fully wired end to end — upload a video in the
browser, it goes to the real backend, the real pipeline code runs, and the frontend renders
whatever the pipeline genuinely produces. **Detection accuracy is a separate, unfinished
concern** — see [Checkpoints and current limitations](#checkpoints-and-current-limitations)
below. This README describes the integrated system as it stands today, not a finished,
production-accurate detector.

## Contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [The five modalities](#the-five-modalities)
- [Fusion](#fusion)
- [Evidence](#evidence)
- [API](#api)
- [Installation](#installation)
- [Environment configuration](#environment-configuration)
- [Running it (Windows PowerShell)](#running-it-windows-powershell)
- [Testing an upload](#testing-an-upload)
- [Checkpoints and current limitations](#checkpoints-and-current-limitations)
- [Troubleshooting](#troubleshooting)

## Architecture

```
 Browser (React, Vite dev server: http://localhost:5173)
      │  POST /api/analyze  (multipart/form-data, field "video")
      │  proxied by Vite to the backend — see "Environment configuration"
      ▼
 FastAPI backend (http://127.0.0.1:8000)
   api/server.py            HTTP routes, upload validation, job lifecycle, CORS
   api/pipeline_adapter.py  orchestrates the real per-stream extraction + fusion
   api/serializer.py        ONE place that maps the pipeline's native output to
                             the frontend's response contract
   api/errors.py            structured JSON errors (never a raw traceback)
   api/jobs.py               per-request temp directories under runtime/jobs/<job_id>/
      │
      ▼
 Existing ML pipeline (unmodified — see docs/ML_PIPELINE.md)
   visual/   audio/   semantic/   fusion/   classifiers/   evidence/   inference/
```

The `api/` layer is a thin adapter: it calls the existing extraction and inference
functions in `visual/`, `audio/`, `semantic/`, `fusion/`, `classifiers/`, and `evidence/`
directly. None of that code was rewritten or duplicated to build this integration.

## Project layout

```
Multimodal-Deepfake-Detection/
├── api/                    FastAPI adapter layer (new, built for this integration)
│   ├── server.py           routes: GET /api/health, POST /api/analyze, GET /api/evidence/...
│   ├── pipeline_adapter.py raw-video -> five feature files -> run_full_inference()
│   ├── serializer.py       pipeline output -> frontend response contract
│   ├── errors.py           structured error types + FastAPI exception handlers
│   ├── jobs.py             per-request job directories (runtime/jobs/<job_id>/...)
│   ├── runners/
│   │   └── run_audio_extract.py   audio extraction, isolated in its own subprocess
│   │                               (see "Known environment gotcha" in docs/ML_PIPELINE.md)
│   └── test_serializer_smoke.py   synthetic wiring test for the serializer
│
├── frontend/               React + Vite UI (untouched by this integration pass)
│   ├── src/
│   ├── API_CONTRACT.md     the exact request/response shape the UI reads from
│   └── README.md           frontend-specific dev notes
│
├── inference/full_pipeline.py   run_full_inference() — the application-facing entry point
├── visual/                 face extraction (EfficientNet-B0), eye-blink + lip-sync analyzers
├── audio/                  Wav2Vec2 audio feature extraction
├── semantic/                Whisper transcription + Sentence-BERT embedding
├── classifiers/            per-modality SingleStreamClassifier
├── fusion/                 EnhancedFusionModel, attention/contribution utilities,
│                            feature normalization, and the ONLY checkpoint shipped
│                            with this repo (fusion/best_fusion_model.pt — see below)
├── evidence/               plain-language evidence report + frame-selection logic
├── eval/                   dataset-scale evaluation scripts (not part of the API path)
├── docs/ML_PIPELINE.md     the original per-module ML documentation (training protocol,
│                            checkpoint policy, dataset conventions) — read this for the
│                            ML internals; this README covers running the application
├── requirements.txt        pinned ML dependencies (torch, mediapipe, whisper, ...)
├── requirements-api.txt    additive: fastapi, uvicorn, python-multipart
└── .gitignore
```

Nothing under `visual/`, `audio/`, `semantic/`, `fusion/`, `classifiers/`, `evidence/`,
`inference/`, or `eval/` was modified for this integration — only `api/` was added, and
`frontend/` is the existing UI unchanged.

## The five modalities

| Modality | Kind | What the frontend shows |
|---|---|---|
| Visual (EfficientNet-B0, 1280-d) | learned probability | `fake_probability` |
| Audio (Wav2Vec2, 768-d) | learned probability | `fake_probability` |
| Semantic (Whisper + Sentence-BERT, 384-d) | learned probability | `fake_probability` |
| Eye Blink (rule-based EAR analysis, 4-d) | rule-based score | `anomaly_score` |
| Lip Sync (rule-based mouth/audio correlation, 2-d) | rule-based score | `mismatch_score` |

Eye-blink and lip-sync are never labeled or treated as a "fake probability" anywhere in the
API or the UI — they are fixed, hand-designed rules, not outputs of a trained classifier.
This distinction is enforced in `api/serializer.py` and mirrored in the frontend's
`modalityMeta.js`.

## Fusion

`fusion/enhanced_fusion_model.py`'s `EnhancedFusionModel` projects all five modality vectors
into a shared 256-d space and combines them with cross-attention (`nn.MultiheadAttention`),
exactly as it already existed in this repo — the architecture was not changed for this
integration. Its output produces `final_fake_probability` / `final_real_probability` /
`final_verdict`, using the same 0.5 decision threshold and the same low-confidence-margin
logic (`evidence/evidence_builder.py`) already established in the pipeline. No second,
conflicting threshold system was introduced.

`attention_summary` (descriptive, non-causal) and `modality_contributions` (ablation-based)
come straight from `fusion/attention_utils.py` and `fusion/modality_contribution.py`.

## Evidence

Evidence is only ever included when it was genuinely produced:

- **Frames** — `evidence/frame_evidence.py` selects real blink-event or uniformly-sampled
  frames from the uploaded video and saves them under the job's own evidence directory;
  the frontend gets a URL like `/api/evidence/<job_id>/<filename>` — never a filesystem path.
- **Lip-sync window evidence** — only present when `analyze_lipsync(video_path,
  compute_windows=True)` actually produced windows for this clip; otherwise the key is
  omitted (never fabricated timestamps).
- **Blink timeline** — built from a real `compute_ear_series()` / `detect_blinks()` pass
  over the uploaded video; omitted where no face/landmarks were found.
- **Audio waveform** — always `null`. No genuine sample-level amplitude envelope is computed
  anywhere in this pipeline (the audio stream only produces a pooled Wav2Vec2 embedding), so
  this is never fabricated from that embedding.
- **Contextual analysis** — built only from `evidence/evidence_builder.py`'s real
  plain-language reasons (which streams crossed their threshold, low-confidence margin,
  etc.) — never an invented low-level artifact claim like "face boundary distortion".

## API

### `GET /api/health`

```json
{"status": "ok"}
```

No model is loaded for this check — it's a plain liveness probe.

### `POST /api/analyze`

`multipart/form-data`, field name `video`. Accepts `.mp4`, `.avi`, `.mov` (matching
`audio/src/config.py`'s own `SUPPORTED_EXTENSIONS` — the audio stream, and therefore the
whole 5-modal pipeline, is not verified beyond these three containers), up to 500 MB.

Success: `200 OK` with the response shape below. Failure: a non-2xx status with
`{"error", "message", "details"}` — see `frontend/API_CONTRACT.md` for the full list of
`error` codes (`missing_video`, `unsupported_media_type`, `file_too_large`, `corrupt_video`,
`feature_extraction_failed`, `inference_failed`, `missing_checkpoint`, `not_found`).

```jsonc
{
  "video_filename": "clip.mp4",
  "video_duration_seconds": 3.0,
  "analyzed_at": "2026-08-27T10:42:00+00:00",
  "processing_time_seconds": 9.4,
  "final_verdict": "LIKELY_DEEPFAKE",
  "final_fake_probability": 0.718,
  "final_real_probability": 0.282,
  "low_confidence": false,
  "model_confidence_note": "...",
  "modalities": {
    "visual":   { "label": "Visual",   "kind": "learned_probability", "fake_probability": 0.82 },
    "audio":    { "label": "Audio",    "kind": "learned_probability", "fake_probability": 0.64 },
    "semantic": { "label": "Semantic", "kind": "learned_probability", "fake_probability": 0.60 },
    "eye_blink": { "label": "Eye Blink", "kind": "rule_based_score", "anomaly_score": 0.75, "status": "Irregular" },
    "lip_sync":  { "label": "Lip Sync",  "kind": "rule_based_score", "mismatch_score": 0.67, "status": "Inconsistent" }
  },
  "attention_summary": { "weights": { "...": 0.0 }, "note": "..." },
  "modality_contributions": { "deltas": { "...": 0.0 }, "note": "..." },
  "evidence": {
    "frames": [ { "frame_index": 42, "timestamp_seconds": 1.4, "url": "/api/evidence/<job_id>/<file>.jpg", "frame_selection_reason": "eye_blink_event", "detected_artifact": null } ],
    "audio_waveform": null,
    "lip_sync_window_evidence": [ { "window_start_seconds": 3.2, "window_end_seconds": 4.0, "mismatch_score": 0.62 } ],
    "blink_timeline": [ { "timestamp_seconds": 1.4, "ear_value": 0.09, "is_blink": true } ]
  },
  "contextual_analysis": { "detected_mismatch": "...", "explanation": "..." }
}
```

Full field-by-field mapping notes live in `frontend/API_CONTRACT.md`.

### `GET /api/evidence/{job_id}/{filename}`

Serves one evidence file (a `.jpg` frame) for a completed job. Path-traversal-checked —
`filename` must be a bare name (no `/`, `\`, `..`, or leading `.`), and the resolved path is
confirmed to still be inside that job's own evidence directory before it's served.

## Installation

Requires Python 3.11 and Node.js 18+.

```powershell
# Backend
cd Multimodal-Deepfake-Detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-api.txt

# Frontend
cd frontend
npm install
```

`requirements.txt` pins the full ML stack (torch, torchvision, torchaudio, mediapipe,
opencv, facenet-pytorch, whisper, sentence-transformers, transformers, librosa, moviepy).
`requirements-api.txt` adds only `fastapi`, `uvicorn`, `python-multipart` on top of it.
`ffmpeg` must also be on your `PATH` (used by the lip-sync analyzer's audio extraction).

## Environment configuration

No machine-specific absolute paths are required. `fusion/env_defaults.py` defines every
path the pipeline uses, each overridable by an environment variable — set these only if you
need to point at feature/checkpoint locations different from the repo-relative defaults:

| Variable | Default (relative to repo root) | Purpose |
|---|---|---|
| `DFD_VISUAL_ROOT` | `visual/data/features_aligned` | precomputed visual feature root |
| `DFD_AUDIO_ROOT` | `audio/data/features` | precomputed audio feature root |
| `DFD_SEMANTIC_ROOT` | `semantic/data/features` | precomputed semantic feature root |
| `DFD_BLINK_ROOT` | `visual/data/blink_features` | precomputed blink feature root |
| `DFD_LIPSYNC_ROOT` | `visual/data/lipsync_features` | precomputed lip-sync feature root |
| `DFD_VISUAL_CLASSIFIER_WEIGHTS` | `classifiers/best_visual_classifier.pt` | visual classifier checkpoint |
| `DFD_AUDIO_CLASSIFIER_WEIGHTS` | `classifiers/best_audio_classifier.pt` | audio classifier checkpoint |
| `DFD_SEMANTIC_CLASSIFIER_WEIGHTS` | `classifiers/best_semantic_classifier.pt` | semantic classifier checkpoint |
| `DFD_ENHANCED_FUSION_WEIGHTS` | `fusion/best_enhanced_fusion_model.pt` | 5-modal fusion checkpoint |
| `DFD_FUSION_WEIGHTS` | `fusion/best_fusion_model.pt` | original 3-modal fusion checkpoint (the one currently shipped) |

These are the actual variable names already defined in `fusion/env_defaults.py` — the API
layer (`api/pipeline_adapter.py`) reads the same defaults, it doesn't introduce a second
config system. The API's own upload limits (allowed extensions, 500 MB cap) and CORS origins
are set directly in `api/server.py` if you need to change them for your setup.

Frontend environment variables (`frontend/.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | unset | absolute backend URL, only needed if frontend/backend are on different origins |
| `BACKEND_URL` | `http://localhost:8000` | dev-only: where Vite's proxy forwards `/api/*` |
| `VITE_USE_MOCK_API` | `false` | **must stay `false`** to use the real backend — `true` bypasses the network entirely |

The default (unset `VITE_API_BASE_URL`, `VITE_USE_MOCK_API=false`) is what ships: the
frontend calls relative `/api/...` paths, Vite's dev proxy forwards them to
`http://localhost:8000`, and no CORS configuration is needed for that path at all.
`api/server.py` also configures CORS for `http://localhost:5173` / `http://127.0.0.1:5173`
as a second, independent safety net in case you point the frontend directly at the backend's
origin instead of going through the proxy — pick one approach; both are already wired.

## Running it (Windows PowerShell)

**Terminal 1 — backend:**

```powershell
cd Multimodal-Deepfake-Detection
.\venv\Scripts\Activate.ps1
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend:**

```powershell
cd Multimodal-Deepfake-Detection\frontend
npm run dev
```

Then open **http://localhost:5173**, upload a video, and click Analyze.

(Linux/macOS: replace `.\venv\Scripts\Activate.ps1` with `source venv/bin/activate`.)

## Testing an upload

1. `GET http://127.0.0.1:8000/api/health` should return `{"status": "ok"}`.
2. From the UI, upload a small `.mp4`/`.avi`/`.mov` file and click Analyze.
3. **Today, with the checkpoints currently shipped in this repo, every real analysis
   request will end in one of two honest outcomes** (see the next section for why):
   - If no face is detected in any sampled frame: a clean `500 feature_extraction_failed`
     error, shown by the frontend's error screen.
   - If a face is detected and visual/audio/semantic/blink/lip-sync extraction all
     succeed: a clean `503 missing_checkpoint` error, because the trained single-stream
     and enhanced-fusion checkpoints don't exist yet in this repo.
4. Neither of those is a bug — both are the pipeline's own `require_file()` /
   no-faces-detected checks working as designed, surfaced honestly instead of a fabricated
   result. `api/test_serializer_smoke.py` and `inference/test_full_pipeline_smoke.py`
   (synthetic-checkpoint wiring tests) demonstrate the full pipeline mechanically
   completes end-to-end once real checkpoints exist.

## Checkpoints and current limitations

**The application integration is complete. Model accuracy depends on the currently
available trained checkpoints, which is a separate, not-yet-finished effort.**

Only `fusion/best_fusion_model.pt` (the original 3-modal checkpoint) is currently shipped
with this repo. The four checkpoints the 5-modal pipeline needs do not exist yet:

- `classifiers/best_visual_classifier.pt`
- `classifiers/best_audio_classifier.pt`
- `classifiers/best_semantic_classifier.pt`
- `fusion/best_enhanced_fusion_model.pt`

`inference/full_pipeline.py`'s `require_file()` check deliberately raises
`MissingCheckpointError` — never a random/fabricated prediction — when any of these are
missing, and `api/pipeline_adapter.py` maps that straight to a `503 missing_checkpoint`
JSON response. Training these checkpoints (see `docs/ML_PIPELINE.md`'s "Training and
evaluation protocol") is out of scope for this integration pass and was not attempted here —
per this task's own instructions, that's a later, separate effort. Once those four files
exist at the paths above (or the env vars in the previous section point at them), the same
running server will start producing real fused predictions with no code changes.

## Troubleshooting

- **`ModuleNotFoundError` on backend startup** — activate the venv and confirm
  `pip install -r requirements.txt -r requirements-api.txt` completed without errors.
- **`ffmpeg` errors from the lip-sync analyzer** — install ffmpeg and ensure it's on `PATH`.
- **CORS error in the browser console** — you're likely pointing the frontend directly at
  the backend's origin; either go through the Vite proxy (leave `VITE_API_BASE_URL` unset)
  or confirm your origin matches `ALLOWED_ORIGINS` in `api/server.py`.
- **`503 missing_checkpoint`** — expected right now; see
  [Checkpoints and current limitations](#checkpoints-and-current-limitations).
- **`500 feature_extraction_failed`, "No face was detected..."** — the uploaded video needs
  a visible, front-facing face in at least some sampled frames.
- **Frontend shows a generic network error, not a specific one** — confirm the backend is
  actually running on the port the frontend is configured for (`BACKEND_URL` / `VITE_API_BASE_URL`).

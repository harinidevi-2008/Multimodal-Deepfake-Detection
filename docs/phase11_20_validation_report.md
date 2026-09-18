# Phase 11-20 Validation Report

## Dataset

- Dataset: FakeAVCeleb_v1.2
- Total samples: 21,544
- Split counts: train 15,714; validation 2,195; test 3,635
- Identity groups: 10

## Feature Audit

- Modalities audited: visual, audio, semantic, eye-blink, lip-sync
- Feature files: 21,544 in each modality root
- Alignment: full alignment across all five modality roots
- NaN/Inf status: no NaN and no Inf found
- Shapes/dtypes: expected shapes and dtypes verified

## Normalization

- Normalization is train-only.
- Training normalization sample count: 15,714
- Recomputed normalization matched the stored normalization artifact exactly.

## Checkpoints

- Individual classifiers: visual, audio, and semantic checkpoints are used by the raw-video pipeline.
- 3-stream baseline: retained for baseline evaluation.
- 5-stream attention: current strict-loading default checkpoint, `fusion/best_enhanced_fusion_model_attention.pt`.
- 5-stream gated: retained as a current enhanced fusion checkpoint.
- Legacy note: `fusion/best_enhanced_fusion_model.pt` is legacy and requires compatibility handling; it is not the default.

## Calibration

- Threshold calibration uses validation labels only.
- Selection criterion: balanced accuracy.
- Tie-breaks: F1 first, then closeness to 0.5.
- Test labels were not used for threshold calibration.

## Calibrated Test Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Accuracy | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Visual | 0.8787 | 0.9764 | 0.8932 | 0.9329 | 0.8614 | 0.9885 | 0.7616 | 0.6300 |
| Audio | 0.8985 | 0.9700 | 0.9211 | 0.9449 | 0.8293 | 0.9863 | 0.7156 | 0.5100 |
| Semantic | 0.8638 | 0.9474 | 0.9063 | 0.9264 | 0.5556 | 0.9566 | 0.5206 | 0.1350 |
| 3-stream Fusion | 0.9392 | 0.9788 | 0.9563 | 0.9675 | 0.9199 | 0.9943 | 0.8007 | 0.6450 |
| 5-stream Enhanced | 0.9015 | 0.9824 | 0.9121 | 0.9460 | 0.9231 | 0.9948 | 0.8160 | 0.7200 |
| 5-stream Attention | 0.8575 | 0.9860 | 0.8614 | 0.9195 | 0.9055 | 0.9932 | 0.8257 | 0.7900 |
| 5-stream Gated | 0.8751 | 0.9847 | 0.8815 | 0.9303 | 0.9153 | 0.9941 | 0.8233 | 0.7650 |

## Inference

- Raw-video API flow: upload is saved to a job directory, OpenCV probes video readability and duration, visual/audio/semantic/blink/lip-sync features are extracted into job-scoped feature roots, then `run_full_inference()` runs on those generated features.
- Final prediction uses the calibrated threshold associated with the selected enhanced fusion checkpoint.
- Diagnostics expose `decision_threshold` and `decision_threshold_source`.
- Failed feature extraction paths raise structured errors; modality scores are not fabricated when a modality fails.

## API

- `POST /api/analyze` accepts `.mp4`, `.avi`, and `.mov`.
- Blocking raw-video analysis is offloaded from the async FastAPI endpoint through `run_in_threadpool`.
- A spawned worker process runs the expensive raw-video pipeline and is terminated on timeout.
- Structured errors are returned as JSON with `error`, `message`, and `details`.
- IPC error preservation was fixed so child-process `AnalysisAPIError` cases, such as corrupt video and feature extraction failures, keep their specific API error code instead of becoming generic `inference_failed`.

## Evidence

- Learned signals: visual, audio, and semantic fake-class probabilities.
- Analytical signals: blink anomaly score and lip-sync mismatch score.
- Frame evidence is supporting context unless a real detector provides a localized artifact; the current pipeline does not provide visual artifact localization.
- Audio and semantic classifier evidence are global unless the backend explicitly returns localized evidence.

## Frontend

- Current section order: Final Result, Stream Analysis, Why This Result, Evidence, Advanced Diagnostics.
- Stream cards render all five modalities and show unavailable states when a modality is missing.
- Advanced Diagnostics displays the calibrated threshold and threshold source when returned.
- User-facing wording distinguishes final fusion result, individual learned stream outputs, rule-based signals, evidence, and diagnostics.

## Error Handling

- Unsupported file type: tested, returns `415 unsupported_media_type`.
- Malformed/missing request: tested missing video, returns `400 missing_video`.
- Empty upload: tested, returns `400 missing_video`.
- Corrupted video: tested, returns `422 corrupt_video`.
- Missing checkpoint: tested, returns structured `503 missing_checkpoint`.
- Missing audio, missing face/visual information, semantic extraction failure, blink failure, and lip-sync failure are mapped through `FeatureExtractionFailedError` without fabricated modality scores.
- Inference timeout is implemented with `DFD_ANALYSIS_TIMEOUT_SECONDS` and returns `inference_timeout`; no timeout test was run.

## Testing

- `.\venv\Scripts\python.exe inference\test_full_pipeline_smoke.py`: passed.
- `.\venv\Scripts\python.exe evidence\evidence_test.py`: passed.
- `.\venv\Scripts\python.exe api\test_missing_checkpoint_error.py`: passed.
- `.\venv\Scripts\python.exe -m py_compile api\pipeline_adapter.py api\server.py api\serializer.py inference\test_full_pipeline_smoke.py`: passed.
- FastAPI `TestClient` error probe for missing, unsupported, empty, and corrupt uploads: passed.
- `npm.cmd run build` in `frontend/`: passed.
- `api/test_serializer_smoke.py` requires a real video path; no `.mp4`, `.avi`, or `.mov` file was present under `C:\Deepfake_Detection`, so it was not run.

## Performance

- Frontend production build measured by Vite: 4.02s.
- Corrupt-video API validation path through `TestClient` and worker startup returned in about 15.1s.
- No successful full raw-video inference latency was measured because no existing local video sample was available.

## Limitations

- Test split is imbalanced: 200 real vs 3,435 fake.
- Evaluation covers only 10 identity groups.
- Results are dataset-specific and do not establish real-world generalization.
- Blink and lip-sync are analytical signals, not trained probabilities.
- The current pipeline does not localize visual artifacts.
- Audio and semantic evidence are global unless localized evidence is explicitly returned.
- Full successful API smoke testing still needs an existing suitable raw video sample.

## Retraining Decision

Retraining is not technically necessary from this continuation. The required fixes were calibration usage, response-field alignment, frontend display correctness, and structured error handling, not a model-training defect.

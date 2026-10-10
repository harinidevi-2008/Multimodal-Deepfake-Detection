# Restoration changelog: branch `integration_changes`

Base: branch `integration`, commit `c2e3eb0e77dd28d46c369e90a10d89bef0cbeae3`.
This snapshot is meant to become the branch `integration_changes`. It was prepared from a ZIP/clone working copy; nothing was pushed to GitHub from here, and `integration` was not modified.

## 1. What was compared

The original methodology document was compared stream by stream with the code on `integration` and with the history of the branch.

| Component | Document | `integration` code | Verdict |
|-----------|----------|--------------------|---------|
| Visual stream (2 fps, MTCNN 224x224 margin 20, EfficientNet-B0, mean pool, 1280-D) | as described | same | matches |
| Audio stream (16 kHz mono, wav2vec2-base-960h, mean-pooled last hidden state, 768-D) | as described | same; the 80-band mel-spectrogram is computed and discarded | matches; `audio_stream.py` preserved |
| Semantic stream (Whisper base, all-MiniLM-L6-v2, 384-D, zero vector for unusable speech) | "score below -1 is unreliable" | zero vector below -1.5, metadata label `reliable` above -1.0, present since the first commit | document wording corrected; code unchanged (see 4) |
| Blink / lip-sync (EAR 0.21, 10-30 per minute, 0.5; MAR vs RMS, +/-5 frames, 0.4) | as described | same | matches |
| Base fusion (3 tokens, 256-D, 8 heads, residual, LayerNorm, mean pool, MLP head) | as described | `fusion/fusion_model.py` unchanged since the start | matches |
| Enhanced fusion (5 tokens, same block) | as described | extended in commit `4ec8195` (Sep 19) with a `gated` reliability-gate architecture, gate parameters, an `invalid_retention_logits` parameter, a `reliability` input and a diagnostics return | **diverged: restored** |
| Evidence layer (0.6 per stream, 0.75 / 0.5 verdict bands) | as described | same, plus a validation-calibrated decision threshold (commit `4ec8195`) | kept; documented (see 4) |

## 2. Architecture changes

Only the enhanced fusion model diverged. It is restored to the documented design: five projected tokens (visual, audio, semantic, blink, lip-sync), one 8-head attention layer, residual connection, LayerNorm, mean pooling and the Linear-ReLU-Dropout-Linear head. `forward(visual, audio, semantic, blink, lipsync)` returns `(logits, attention_weights)` again.

| File | Change | Why |
|------|--------|-----|
| `fusion/enhanced_fusion_model.py` | Removed the `gated` architecture, `use_reliability_gates`, the gate modules, `invalid_retention_logits`, the `reliability` argument and the diagnostics return. Kept `MODALITY_ORDER`, `semantic_validity` and `default_reliability` as reporting helpers. Added `load_enhanced_state_dict` (drops the unused gate keys of old checkpoints) and `check_checkpoint_architecture` (refuses checkpoints marked `gated`) | restore the documented block; keep old attention checkpoints loadable |
| `fusion/train_enhanced_fusion.py` | Removed `--fusion-architecture` and the reliability tensor; model metadata sidecar now records `fusion_architecture: attention` and the modality order | follow the model |
| `fusion/evaluate_enhanced_fusion.py` | Removed the architecture option and the reliability input; loads through `load_enhanced_state_dict` | follow the model |
| `fusion/enhanced_fusion_inference.py` | Same | follow the model |
| `inference/full_pipeline.py` | Loads through the new helpers, calls the model without reliability, always summarises attention weights; removed the `enhanced_gated` threshold key and the `modality_gates` diagnostic. Kept `modality_reliability` (computed from the semantic zero-vector flag) and `fusion_architecture` (now always `attention`) | follow the model; no API or frontend field was consumed from the removed items |
| `eval/calibrated_evaluation.py` | Removed the gated job and the reliability input | follow the model |
| `diagnostics/scan_demo_candidates.py` | Updated model construction and call | follow the model |
| `fusion/test_enhanced_fusion_restored.py` | New test file | covers the restored design |

Public API routes, the response schema (`api/serializer.py`) and the frontend were not changed.

## 3. Files deliberately preserved

- `audio/src/audio_stream.py`: byte-identical to the supplied snapshot (sha256 `d69af70b7d553abf8034c31d06216dd9a9fa938215085cc6367b5c3955973893`, checked with `cmp`). `audio/src/config.py` and `audio/src/extractor.py` are also unchanged and compatible with it.
- `fusion/fusion_model.py` (3-token model), all feature-extraction scripts, split and normalisation files, `evidence/`, `api/`, `frontend/`, and the calibrated threshold step (`eval/results/thresholds.json`, `evidence/evidence_builder.py`).
- FaceForensics++ support (`DATASET_NAME=ffpp`) in the visual, audio and semantic scripts is kept as optional code, but FaceForensics++ is not a selected dataset. Removing it was not needed for the restoration.
- Recorded evaluation artifacts in `eval/results/*.json` and the old training histories of the attention model are unchanged. They still contain an `enhanced_gated` entry from an earlier run; it is historical.

## 4. Decisions that need your confirmation

1. **Semantic threshold.** The document said a Whisper score below -1 is unreliable and encoded as zeros. The code has always zeroed below -1.5 and uses -1.0 only for the `reliable` metadata label. This was in the first commit, not an Innovation Day change, and every existing feature file was extracted with -1.5. Changing the constant would make the code disagree with the features and checkpoints, so the code was left alone and the revised document now describes the code. If you want -1.0 for zeroing, the features must be re-extracted and the models retrained and recalibrated.
2. **Calibrated decision threshold.** The code predicts DEEPFAKE only when the fusion probability reaches a threshold calibrated on the validation split (about 0.97 for the attention checkpoint), while the original document used 0.5 / 0.75 bands. It was added in commit `4ec8195` and affects only the REAL / DEEPFAKE cut-off, not the probability. It was kept because removing it changes API behaviour and the document does not forbid it, and it is described in the revised methodology. To drop it, delete the threshold lookup in `inference/full_pipeline.py` and `fusion/enhanced_fusion_inference.py`.
3. **Old checkpoints.** Attention checkpoints trained while the gated code existed load correctly (their unused gate keys are dropped; output is identical in the equivalence test). A checkpoint trained as `gated` can no longer be used.

## 5. Dataset-selection updates

Selected experimental datasets: LAV-DF, FakeAVCeleb, AV-Deepfake1M++. FaceForensics++ and DFDC are no longer presented as experimental datasets; they appear only as related work. The repository README has a new "Selected datasets" section.

- FakeAVCeleb: the pipeline and the existing split (21,544 samples, 10 identity groups) were built on it.
- LAV-DF: restartable preprocessing scripts and a 100-video dev pilot exist; no full extraction is claimed. `scripts/preprocess_lavdf_full.py` still has a Windows default for `--root`; it can be overridden on the command line and was not changed.
- AV-Deepfake1M++: no preprocessing or results exist. The document proposes a bounded-subset plan and labels it as a plan. No script was changed to process a whole dataset.

## 6. Methodology document

`Multimodal_Deepfake_Detection_System_Methodology_Revised.docx` (delivered separately) was rebuilt from the original: abstract, introduction, a thematic literature survey of exactly 15 papers (selection in `docs/PAPER_SELECTION.md`), research gap, framework overview with a figure, modules A-H aligned with the code, dataset descriptions, evaluation plan, expected contributions, limitations and 15 references. Corrections to match the code: the Whisper confidence definition and thresholds, the discarded mel-spectrogram, the blink and lip-sync formulas, blink/lip-sync normalisation, the attention description (self-attention over modality tokens), the calibrated threshold, and the removal of the unsupported claim that SSL encoders "significantly outperform" hand-crafted features.

Open citation points: the mixture-of-experts paper is cited as arXiv:2409.11909 (2024) following the summary table, while the old document gave ICASSP 2025; the LAV-DF and AV-Deepfake1M++ descriptions are not covered by the supplied summaries and are marked in the document for verification; Whisper and Sentence-BERT have no citation because no supplied summary covers them.

## 7. Tests

See `TEST_REPORT.md`. Summary: compile check, 13 pytest tests, the new restored-architecture tests, the pipeline smoke test and four script tests passed; an equivalence check showed identical outputs to the previous attention mode; tests needing real videos, datasets, whisper or mediapipe could not run.

## 8. Excluded from this snapshot

Not included, per the task rules: `audio/src/audio_stream.py.bak`, `fusion/best_fusion_model.pt` (the 3.6 MB tracked checkpoint), `.git`, virtual environments, caches, and the removed gated artifacts (`fusion/test_enhanced_fusion_gating.py`, `fusion/training_history_gated.json`, `fusion/training_gated_log.txt`, the two gated sidecar files). The first two and the gated files still exist on `integration`; see section 9 for how to treat them.

## 9. Applying the snapshot to the real repository

Run these in your local clone (VS Code terminal). They create the new branch from `integration`, overlay the snapshot, and remove only the files this change deliberately drops. `integration` is not touched.

```bash
git fetch origin
git checkout -b integration_changes origin/integration
unzip -o /path/to/integration_changes.zip -d /tmp/ic_unzip      # folder: integration_changes/
cp -r /tmp/ic_unzip/integration_changes/. .                      # overlay (same layout)
git rm -f audio/src/audio_stream.py.bak \
          fusion/test_enhanced_fusion_gating.py \
          fusion/training_history_gated.json fusion/training_gated_log.txt \
          fusion/best_enhanced_fusion_model_gated.pt.model_meta.json \
          fusion/best_enhanced_fusion_model_gated.pt.normalization_meta.json
git status                      # review; fusion/best_fusion_model.pt stays as on integration
git diff origin/integration --stat -- audio/src/audio_stream.py   # must print nothing
git add -A
git commit -m "Restore documented enhanced fusion architecture; update README and docs"
git push -u origin integration_changes
```

`fusion/best_fusion_model.pt` and `data/lavdf_full/**` are unchanged on the branch because the overlay does not delete them. If the push is rejected, you need write access to the repository (the account used here only had read access).

## 10. Known limitations and manual steps

- No result in the document is new; no accuracy claim is made.
- The attention checkpoint, classifier checkpoints and feature files are not in the repository, so `inference/full_pipeline.py` still raises `MissingCheckpointError` until they are present.
- Verify the three flagged bibliographic and dataset points in section 6 before submission.
- Decide on the two items in section 4 (semantic threshold, calibrated decision threshold).

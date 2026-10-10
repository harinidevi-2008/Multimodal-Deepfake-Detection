# Test report (integration_changes)

All commands were run from the repository root of the restored tree, on CPU, with synthetic data only.
No dataset was downloaded or processed, and no network access was used for the tests.
Python 3.13, torch 2.14.1, numpy, scikit-learn, pytest, fastapi, opencv-python-headless.
Not installed in the test environment: whisper, sentence-transformers, mediapipe, librosa, transformers (see "Could not run").

## Passed

| Check | Command | Result |
|-------|---------|--------|
| Byte-compile of all Python files | `python3 -m compileall -q .` | exit 0 |
| Pytest collection that runs without heavy dependencies (13 tests: 4 serializer/probability regression, 3 enhanced dataset, 6 restored-architecture) | `python3 -m pytest -q --ignore=...` (the six ignored files are listed below) | 13 passed |
| Restored-architecture checks (output shapes, finite softmax rows, no gate parameters, modality order, legacy gate keys dropped with unchanged output, gated checkpoint refused, reliability flag) | `python3 fusion/test_enhanced_fusion_restored.py` | 6 checks passed |
| Equivalence with the previous attention mode: the same random weights run through the old `EnhancedFusionModel(fusion_architecture="attention")` and through the restored class (checkpoint loaded with `load_enhanced_state_dict`, 11 gate keys dropped) | ad hoc script, 8 random samples including an all-zero semantic vector | max logit difference 0.0, max attention difference 0.0 |
| Full pipeline wiring with random weights and synthetic features, including missing-file and wrong-shape error paths | `python3 inference/test_full_pipeline_smoke.py` | all checks passed |
| Missing-checkpoint error path of the API adapter | `python3 api/test_missing_checkpoint_error.py` | passed |
| Checkpoint saved at F1 = 0 (includes `train_enhanced_fusion.py`) | `python3 fusion/test_checkpoint_at_zero_f1.py` | all checks passed |
| Lip-sync consistency boundary rule | `python3 fusion/test_lipsync_consistency_boundary.py` | all checks passed |
| 3-token base `FusionModel` forward pass | `python3 fusion/test_fusion.py` | passed (logits (4, 2), attention (4, 3, 3)) |
| Audio stream path handling | `python3 -m pytest -q audio/src/test_audio_stream_paths.py` | 4 passed (run on its own; it fails to collect in a whole-repo pytest run because `config` module names of different streams collide, as in the original branch) |
| `audio/src/audio_stream.py` identical to the supplied snapshot | `cmp` and `sha256sum` | byte-identical, sha256 `d69af70b7d553abf8034c31d06216dd9a9fa938215085cc6367b5c3955973893` |
| No hard-coded paths or credentials added | diff of added lines against the original snapshot, searched for `C:\`, `/home/`, `/tmp/`, `password`, `secret`, `api_key` | none |
| No stale references to removed code (`return_diagnostics`, `fusion_architecture=`, `modality_gates`, `test_enhanced_fusion_gating`) | `grep -rn` | none in code; the recorded `enhanced_gated` entries in `eval/results/*.json` are kept as historical artifacts |

## Could not run (missing dependencies or data)

| Check | Reason |
|-------|--------|
| `api/test_serializer_smoke.py` | needs a real video file as an argument |
| `fusion/test_batch.py` | needs precomputed feature files (the dataset has zero samples here) |
| `semantic/src/run_test.py`, `semantic/src/whisper_test.py` | need whisper and model downloads |
| `visual/src/eye_blink/eye_blink_test.py`, `visual/src/lipsync/lipsync_test.py` | need mediapipe and a real face video |
| `test_ffpp_visual_extraction.py` | needs FaceForensics++ videos |
| Any end-to-end run on a real video, training, or evaluation with real checkpoints | the trained checkpoints and datasets are not part of the repository |
| Frontend build | no frontend file was changed, so it was not rebuilt |

## Failed

None of the checks that were run failed.

## Not tested

The behaviour of the real trained `best_enhanced_fusion_model_attention.pt` checkpoint was not exercised, because it is not in the repository. The equivalence test above shows that a checkpoint containing the old gate parameters loads and gives identical output, but it uses random weights.

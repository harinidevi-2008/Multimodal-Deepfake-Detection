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

## Follow-up verification (2026-10-10)

These results were first run when the active branch was `codex_three_stream`; the
final review reported `integration_changes`. No branch-switch command was issued by
this task, so the branch changed during the work session. The final focused regression
run and source checks completed after that change.

| Check | Command | Result |
|-------|---------|--------|
| Serializer regression coverage | `.\\.venv\\Scripts\\python.exe -m pytest api\\test_fake_probability_regression.py -q` | 10 passed |
| Serializer contract smoke | `.\\.venv\\Scripts\\python.exe api\\test_serializer_smoke.py` | exit 0 |
| Three-stream checkpoint and UI smoke | `.\\.venv\\Scripts\\python.exe inference\\test_full_pipeline_smoke.py` | 2 passed |
| Python compilation of modified Python files | `.\\.venv\\Scripts\\python.exe -m py_compile api\\serializer.py api\\test_fake_probability_regression.py api\\test_serializer_smoke.py` | exit 0 |
| Frontend production build | `npm run build` (from `frontend`) | passed; Vite 5.4.21 built 48 modules |
| Obsolete serializer aggregation search | `rg -n "aggregate_learned_fake_probabilities|learned_reliability|learned_scores" api\\serializer.py` | no matches |
| Undefined frontend variable search | `rg -n "fakePct" frontend\\src\\components\\OverallResultPanel.jsx` | no matches after the follow-up fix |
| Protected audio file | `Get-FileHash audio/src/audio_stream.py -Algorithm SHA256` and `git diff -- audio/src/audio_stream.py` | working-tree SHA-256 `f77e20007ef936d24f74112809977040e220744cf225bedcc1a1c1d4249eb3eb`; normalized Git diff empty; HEAD blob SHA-256 matches expected `d69af70b7d553abf8034c31d06216dd9a9fa938215085cc6367b5c3955973893`; `core.autocrlf=true` converts line endings |

No frontend test framework is configured in `frontend/package.json`; the existing
three-stream smoke test checks the three primary modality cards and supplementary
section placement. The audio working-tree hash observed at task start was
`e53b43a21d9c2646f89916ba01b228ecd013642372b62604ac3ab67fa873e767`; it later read
`f77e20007ef936d24f74112809977040e220744cf225bedcc1a1c1d4249eb3eb`. The file was
not edited by this task or rewritten to force a hash match. No real-checkpoint or
real-video end-to-end inference was run.
# Follow-up implementation check (2026-10-12)

This follow-up preserves the report above as historical output. Changes in this task were checked with `python -m compileall -q api` (passed) and `npm run build` from `frontend/` (passed). The Python regression, serializer smoke, and backend tests could not run because `pytest` is not installed in the active Python 3.11 environment (`No module named pytest`). Running npm at repository root also failed because the package manifest is under `frontend/`; the correct frontend build then passed.

The no-audio adapter branch now selects the independent visual and semantic classifiers and leaves fusion unavailable; the serializer accepts and exposes a null fusion score as unavailable. Lip-sync extraction is skipped when the audio feature file was not produced. Full modality-failure handling, three-level risk bands, and ablation execution remain incomplete and are not covered by this check.

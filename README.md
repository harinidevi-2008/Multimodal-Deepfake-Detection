# Multimodal Deepfake Detection Using Visual, Audio, and Semantic Inconsistencies

Mini project — SSN College of Engineering, IT Dept. Guide: Dr. Anjali T.
Team: Kruthika C D (visual), Varshana M (audio), Harini Devi B (fusion), Napster (semantic).

## Pipeline

Video → three parallel streams → cross-attention fusion transformer → real/fake classifier.

| Stream   | Path              | Backbone                              | Output dim |
|----------|-------------------|----------------------------------------|-----------|
| Visual   | `visual/src/`     | EfficientNet-B0                        | 1280      |
| Audio    | `audio/src/`      | Wav2Vec 2.0                            | 768       |
| Semantic | `semantic/src/`   | Whisper (base) → Sentence-BERT (MiniLM)| 384       |
| Fusion   | `fusion/`         | Cross-attention transformer            | → 2 classes |

Each stream saves raw, un-projected features to disk as `.npy`. The 384→256 / 768→256 / 1280→256
projections happen **inside `fusion/fusion_model.py`**, not in the individual stream scripts —
keep it that way; don't pre-project features before saving.

## Repo layout

```
visual/src/           frame extraction, face alignment, EfficientNet feature extraction
visual/src/eye_blink/ landmark-based blink detection and irregularity scoring (rule-based)
visual/src/lipsync/   mouth-motion vs audio-energy consistency scoring (rule-based)
audio/src/            audio extraction, mel-spectrogram, Wav2Vec2 feature extraction
semantic/src/         transcription (Whisper) + embedding (Sentence-BERT) + precompute pipeline
fusion/                FusionModel, FusionDataset, train_fusion.py, best_fusion_model.pt
requirements.txt       single merged environment for all streams + fusion
```

### Eye-blink and lip-sync modules

Both are lightweight, rule-based (no trainable weights) and independent of the
EfficientNet visual pipeline and the Wav2Vec2 audio pipeline — neither imports
from nor modifies the existing feature extraction code.

Run them directly:

```bash
python visual/src/eye_blink/eye_blink_test.py path/to/video.mp4
python visual/src/lipsync/lipsync_test.py path/to/video.mp4
```

- **Eye-blink**: face-mesh landmarks (MediaPipe) → Eye Aspect Ratio per frame →
  blink events → blink count, rate, and an irregularity score (0-1, higher =
  more abnormal). Samples the video at its own higher frame rate rather than
  reusing the main pipeline's low `DEFAULT_FPS_TARGET`, since individual
  blinks (100-400ms) would fall between sparsely-sampled frames.
- **Lip-sync**: mouth-opening ratio per frame (same landmark model) correlated
  against the audio's RMS energy envelope, with a small search over time-lag
  to tolerate minor start-offset misalignment. Outputs a consistency score
  (0-1, higher = better synced). Pass an already-extracted wav via
  `--audio path.wav` (e.g. from the audio stream's `temp_audio/` folder), or
  let it auto-extract one with ffmpeg.

Both currently return a per-video scalar score/status — not yet wired into
`fusion/fusion_model.py`. See the "further steps" discussion for how to feed
them in as two extra tokens/features alongside visual, audio, and semantic.

### Individual modality classifiers

`classifiers/` holds a small MLP head trained per-modality on the existing
frozen features — visual (1280-d), audio (768-d), or semantic (384-d) — so
each stream can report its own fake-probability independently of the fusion
model. One generic script handles all three modalities via CLI flags:

```bash
python classifiers/train_classifier.py --modality visual \
    --feature-root visual/data/features_aligned --input-dim 1280

python classifiers/evaluate_classifier.py --modality visual \
    --feature-root visual/data/features_aligned --input-dim 1280 \
    --weights classifiers/best_visual_classifier.pt --save-probs classifiers/visual_probs.npy
```

Evaluation reports accuracy, precision, recall, F1, ROC-AUC, and a confusion
matrix over the full feature set (not a single sample), and `--save-probs`
writes per-sample fake-probabilities to `.npy` for later use by the evidence
layer.

**Note:** no trained weights (`best_visual_classifier.pt` etc.) are included
in this repo — training requires the real precomputed `.npy` features, which
only exist on the machine that ran the feature-extraction pipelines. Run
`train_classifier.py` yourself once you have those features in place.

### Blink and lip-sync feature precomputation

Before the enhanced fusion model can use blink/lip-sync evidence, those
scores need to exist as `.npy` files mirroring the dataset structure, same
as the other three streams. `visual/src/blink_lipsync_precompute.py` runs
`analyze_blinks()` and `analyze_lipsync()` across every video in a dataset
folder and saves two small feature vectors per video:

```bash
python visual/src/blink_lipsync_precompute.py \
    --dataset-root FakeAVCeleb_v1.2 \
    --blink-output visual/data/blink_features \
    --lipsync-output visual/data/lipsync_features
```

Blink vector (4-d): `[blink_count, blink_rate_per_min, average_blink_duration_sec, blink_irregularity_score]`
Lip-sync vector (2-d): `[sync_score, mismatch_score]`

Safe to re-run after an interruption — already-processed videos are skipped.

### Enhanced fusion model (visual + audio + semantic + blink + lip-sync)

`fusion/enhanced_fusion_model.py` and `fusion/enhanced_fusion_dataset.py` are
new, separate files — the original `FusionModel`/`FusionDataset` are
untouched and still usable on their own. The enhanced model projects blink
and lip-sync vectors into the same shared 256-d space as the other three
modalities and treats them as two additional tokens in the cross-attention
block (5 tokens total instead of 3), rather than only combining them at the
final decision layer.

```bash
python fusion/train_enhanced_fusion.py \
    --visual-root visual/data/features_aligned --audio-root audio/data/features \
    --semantic-root semantic/data/features --blink-root visual/data/blink_features \
    --lipsync-root visual/data/lipsync_features
```

### Evidence / explanation layer

`evidence/evidence_builder.py` composes the classifier probabilities, the
blink/lip-sync results, and the fusion model's final probability into one
structured report with plain-language reasons — e.g. "Abnormal blinking
pattern", "Audio-visual synchronization inconsistency". It's a pure function,
not a pipeline — it doesn't run any model itself, just assembles scores you
already computed. See `evidence/evidence_test.py` for example usage and
output shape.

### Full evaluation

`eval/run_full_evaluation.py` runs every evaluation script in sequence
(visual/audio/semantic classifiers, original fusion, enhanced fusion, and
the rule-based blink/lip-sync scores) and prints one consolidated report.
Gracefully skips any section whose weights or feature roots don't exist yet.

Blink and lip-sync are evaluated separately from the trained classifiers
(`fusion/evaluate_blink_lipsync.py`) using ROC-AUC and their own analyzers'
fixed thresholds — deliberately not framed as "trained classifier accuracy",
since they're rule-based (EAR thresholds, correlation) with no learned
decision boundary or train/val split of their own.

### What's still open

- **External dataset evaluation (DFDC, FaceForensics++)** — deliberately not
  started. These are a separate generalization-testing stage once the full
  backend above is validated on FakeAVCeleb; the training/evaluation
  protocol (what gets trained on what, what's held out for generalization
  testing) needs to be decided before mixing datasets in.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt --break-system-packages   # if needed on your platform
```

## Running each stream

Each stream expects the dataset (e.g. `FakeAVCeleb_v1.2/`) placed next to that stream's folder,
or update the path constant at the top of the relevant script/config:

- Visual: `visual/src/config.py` → `VIDEO_FOLDER`
- Audio: `audio/src/config.py`
- Semantic: `semantic/src/precompute.py` → `DATASET_PATH` (auto-detects a few common locations,
  edit directly if your dataset lives elsewhere)

Feature `.npy` files are **not** committed to this repo — they're regenerated locally per machine
(21k+ files, too large for git). Each stream mirrors the dataset's folder structure inside its own
`features/` directory so `fusion/fusion_dataset.py` can align samples by relative path.

## Training fusion

```bash
cd fusion
python train_fusion.py
```

Edit the three root paths at the top of `train_fusion.py`'s `FusionDataset(...)` call to point at
wherever each teammate's precomputed features actually live on your machine.

## Known environment gotcha

`transformers==5.x` breaks on `torch==2.2.2`. Stick to `transformers==4.38.2` +
`sentence-transformers==2.7.0`. If you `pip freeze` after installing something new and it bumps
these versions, do not commit that requirements.txt without testing all three streams first.

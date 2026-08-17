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
visual/src/       frame extraction, face alignment, EfficientNet feature extraction
audio/src/        audio extraction, mel-spectrogram, Wav2Vec2 feature extraction
semantic/src/     transcription (Whisper) + embedding (Sentence-BERT) + precompute pipeline
fusion/           FusionModel, FusionDataset, train_fusion.py, best_fusion_model.pt
requirements.txt  single merged environment for all three streams + fusion
```

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

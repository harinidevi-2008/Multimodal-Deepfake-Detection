"""
precompute.py
=============
Runs Whisper + Sentence-BERT on every FakeAVCeleb video
and saves the 384-d semantic embedding as a .npy file.

HOW TO RUN:
    python precompute.py

    The script auto-detects paths relative to its own location.
    No hardcoded paths. Works on any machine.

OUTPUT:
    features/semantic/
    ├── metadata.json
    ├── RealVideo-RealAudio/African/men/id00076/00109.npy
    └── ...  (one .npy per video, shape=(384,), dtype=float32)

NOTE:
    This saves 384-d raw Sentence-BERT embeddings.
    The projection (384→256) happens inside the fusion model
    during training — NOT here.

RESUME:
    Safe to stop with Ctrl+C and restart.
    Already-processed videos are skipped automatically.
"""

import os
import json
import time
import numpy as np
import torch
import whisper
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── AUTO-DETECT PATHS ─────────────────────────────────────────────
# All paths are relative to the location of this script.
# No hardcoded C:\Users\... paths.
SCRIPT_DIR    = Path(__file__).resolve().parent
DATASET_PATH  = SCRIPT_DIR / "FakeAVCeleb_v1.2"
FEATURES_DIR  = SCRIPT_DIR / "features" / "semantic"
METADATA_FILE = FEATURES_DIR / "metadata.json"

# ── CONFIGURATION ─────────────────────────────────────────────────
WHISPER_SIZE      = "base"
CONFIDENCE_THRESH = -1.5
MIN_WORDS         = 3

CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
    "RealVideo-FakeAudio": 1,
}

# ── VALIDATE PATHS ────────────────────────────────────────────────
if not DATASET_PATH.exists():
    # Try common alternative locations
    alternatives = [
        SCRIPT_DIR / "FakeAVCeleb_v1.2",
        SCRIPT_DIR.parent / "FakeAVCeleb_v1.2",
        SCRIPT_DIR / "datasets" / "FakeAVCeleb_v1.2",
    ]
    for alt in alternatives:
        if alt.exists():
            DATASET_PATH = alt
            break
    else:
        print(f"ERROR: Dataset not found.")
        print(f"Expected at: {DATASET_PATH}")
        print(f"Please place FakeAVCeleb_v1.2 next to this script")
        print(f"or update DATASET_PATH at the top of this file.")
        exit(1)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ── SETUP ─────────────────────────────────────────────────────────
print("=" * 60)
print("SEMANTIC STREAM  —  PRECOMPUTE")
print("=" * 60)
print(f"Dataset  : {DATASET_PATH}")
print(f"Features : {FEATURES_DIR}")
print(f"Saving   : 384-d Sentence-BERT embeddings (float32)")
print()

# ── LOAD MODELS ───────────────────────────────────────────────────
print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_SIZE)
print(f"  Whisper ready — "
      f"{sum(p.numel() for p in whisper_model.parameters()):,} params")

print("Loading Sentence-BERT...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
for param in sbert_model.parameters():
    param.requires_grad = False
print(f"  Sentence-BERT ready — frozen")
print()

# ── COLLECT ALL VIDEO PATHS ───────────────────────────────────────
print("Scanning dataset...")
all_videos = []

for category, label in CATEGORIES.items():
    cat_path = DATASET_PATH / category
    if not cat_path.exists():
        print(f"  WARNING: {category} not found — skipping")
        continue
    vids = list(cat_path.rglob("*.mp4"))
    for v in vids:
        all_videos.append({
            "video_path": str(v),
            "label":      label,
            "category":   category,
            "rel_path":   str(v.relative_to(DATASET_PATH))
        })
    print(f"  {category:<30} {len(vids):>6} videos")

total = len(all_videos)
print(f"\n  Total: {total:,} videos")
print()

# ── LOAD EXISTING METADATA ────────────────────────────────────────
if METADATA_FILE.exists():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Resuming — {len(metadata):,} done, "
          f"{total - len(metadata):,} remaining")
else:
    metadata = {}
    print("Starting fresh")
print()

# ── HELPERS ───────────────────────────────────────────────────────
def transcribe(video_path):
    try:
        result = whisper_model.transcribe(
            str(video_path),
            language=None,
            task="transcribe",
            verbose=False,
            fp16=False
        )
        text = result["text"].strip()
        segs = result.get("segments", [])
        conf = (sum(s["avg_logprob"] for s in segs) / len(segs)
                if segs else -2.0)
        if len(text.split()) < MIN_WORDS:
            conf -= 0.5
        return text, conf
    except Exception as e:
        return "", -2.0


def encode(transcript, confidence):
    """
    Returns a 384-d Sentence-BERT embedding.
    Returns zeros if transcript is unreliable.
    The projection (384→256) happens in the fusion model — NOT here.
    """
    if not transcript or confidence < CONFIDENCE_THRESH:
        return np.zeros(384, dtype=np.float32)

    with torch.no_grad():
        emb = sbert_model.encode(
            transcript,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).float().detach().cpu().numpy()

    return emb.astype(np.float32)


def verify_embedding(emb, video_stem):
    """
    Checks the embedding meets the expected contract.
    """
    assert emb.shape == (384,), \
        f"[{video_stem}] Wrong shape: {emb.shape} — expected (384,)"
    assert emb.dtype == np.float32, \
        f"[{video_stem}] Wrong dtype: {emb.dtype} — expected float32"


# ── MAIN LOOP ─────────────────────────────────────────────────────
done    = 0
skipped = 0
start_t = time.time()

print("Starting precompute loop...")
print("Press Ctrl+C to stop — progress saves every 200 videos.")
print()

try:
    for item in all_videos:
        video_path = item["video_path"]
        rel_path   = item["rel_path"]

        # Skip if already done
        if rel_path in metadata:
            skipped += 1
            continue

        video_stem = Path(video_path).stem

        # Save path mirrors dataset structure exactly
        # Filename = video filename (stem) + .npy
        save_path = FEATURES_DIR / Path(rel_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Transcribe + encode
        transcript, confidence = transcribe(video_path)
        embedding = encode(transcript, confidence)

        # Verify before saving
        verify_embedding(embedding, video_stem)

        # Save 384-d embedding
        np.save(str(save_path), embedding)

        # Confirm saved file is correct
        loaded = np.load(str(save_path))
        assert loaded.shape == (384,), \
            f"Saved file has wrong shape: {loaded.shape}"
        assert loaded.dtype == np.float32, \
            f"Saved file has wrong dtype: {loaded.dtype}"

        # Record metadata
        metadata[rel_path] = {
            "video_path":   video_path,
            "video_stem":   video_stem,
            "embedding":    str(save_path),
            "label":        item["label"],
            "category":     item["category"],
            "transcript":   transcript,
            "confidence":   round(float(confidence), 4),
            "reliable":     bool(confidence > -1.0),
            "zero_vector":  bool(transcript == "" or
                                  confidence < CONFIDENCE_THRESH),
            "shape":        [384],
            "dtype":        "float32",
        }

        done += 1

        # Progress every 50 videos
        if done % 50 == 0:
            elapsed   = time.time() - start_t
            rate      = done / max(elapsed, 1)
            remaining = (total - done - skipped) / max(rate, 0.001)
            hrs       = int(remaining // 3600)
            mins      = int((remaining % 3600) // 60)
            pct       = (done + skipped) / total * 100
            print(f"[{done+skipped:>6}/{total}] {pct:>5.1f}%  "
                  f"done={done}  skipped={skipped}  "
                  f"ETA={hrs}h {mins}m  |  {Path(video_path).name[:40]}")

        # Checkpoint every 200
        if done % 200 == 0:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

except KeyboardInterrupt:
    print("\nStopped — saving progress...")

# ── FINAL SAVE ────────────────────────────────────────────────────
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

elapsed_total = time.time() - start_t
hrs  = int(elapsed_total // 3600)
mins = int((elapsed_total % 3600) // 60)

print()
print("=" * 60)
print("PRECOMPUTE COMPLETE")
print("=" * 60)
print(f"Total          : {total:,}")
print(f"Processed now  : {done:,}")
print(f"Already done   : {skipped:,}")
print(f"Time taken     : {hrs}h {mins}m")
print(f"Output shape   : (384,)  float32")
print(f"Features saved : {FEATURES_DIR}")
print(f"Metadata saved : {METADATA_FILE}")
print()

real_count = sum(1 for v in metadata.values() if v["label"] == 0)
fake_count = sum(1 for v in metadata.values() if v["label"] == 1)
reliable   = sum(1 for v in metadata.values() if v["reliable"])
zeros      = sum(1 for v in metadata.values() if v["zero_vector"])

print(f"Real    : {real_count:,}")
print(f"Fake    : {fake_count:,}")
print(f"Reliable transcripts : {reliable:,}")
print(f"Zero vectors         : {zeros:,}")

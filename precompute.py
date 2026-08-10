import os
import json
import time
import numpy as np
import torch
import whisper
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── CONFIGURATION ─────────────────────────────────────────────
DATASET_PATH  = r"C:\Users\ASUS\semantic_stream\FakeAVCeleb_v1.2"
FEATURES_DIR  = r"C:\Users\ASUS\semantic_stream\features\semantic"
METADATA_FILE = r"C:\Users\ASUS\semantic_stream\features\semantic\metadata.json"

WHISPER_SIZE      = "base"
CONFIDENCE_THRESH = -1.5
MIN_WORDS         = 3

CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
    "RealVideo-FakeAudio": 1,
}

# ── SETUP ──────────────────────────────────────────────────────
dataset_path  = Path(DATASET_PATH)
features_path = Path(FEATURES_DIR)
features_path.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SEMANTIC STREAM — PRECOMPUTE")
print("=" * 60)
print("Dataset  :", DATASET_PATH)
print("Features :", FEATURES_DIR)
print()

# ── LOAD MODELS ────────────────────────────────────────────────
print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_SIZE)
whisper_params = sum(p.numel() for p in whisper_model.parameters())
print("  Whisper ready —", whisper_params, "params")

print("Loading Sentence-BERT...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
for param in sbert_model.parameters():
    param.requires_grad = False
print("  Sentence-BERT ready — frozen")
print()

# ── COLLECT ALL VIDEO PATHS ────────────────────────────────────
print("Scanning dataset...")
all_videos = []

for category, label in CATEGORIES.items():
    cat_path = dataset_path / category
    if not cat_path.exists():
        print("  WARNING:", category, "not found — skipping")
        continue
    vids = list(cat_path.rglob("*.mp4"))
    for v in vids:
        all_videos.append({
            "video_path": str(v),
            "label":      label,
            "category":   category,
            "rel_path":   str(v.relative_to(dataset_path))
        })
    print(" ", category, "->", len(vids), "videos")

total = len(all_videos)
print()
print("  Total:", total, "videos")
print()

# ── LOAD EXISTING METADATA ─────────────────────────────────────
if Path(METADATA_FILE).exists():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Resuming —", len(metadata), "already done,", total - len(metadata), "remaining")
else:
    metadata = {}
    print("Starting fresh")
print()

# ── HELPERS ───────────────────────────────────────────────────
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
        segments = result.get("segments", [])

        if segments:
            confidence = sum(s["avg_logprob"] for s in segments) / len(segments)
        else:
            confidence = -2.0

        if len(text.split()) < MIN_WORDS:
            confidence -= 0.5

        return text, confidence

    except Exception as e:
        print("  Whisper error on", Path(video_path).name, ":", e)
        return "", -2.0


def encode(transcript, confidence):
    if not transcript or confidence < CONFIDENCE_THRESH:
        return np.zeros(384, dtype=np.float32)

    with torch.no_grad():
        embedding = sbert_model.encode(
            transcript,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embedding = embedding.float().detach().cpu().numpy()

    return embedding.astype(np.float32)


# ── MAIN LOOP ─────────────────────────────────────────────────
done    = 0
skipped = 0
start_t = time.time()

print("Starting precompute loop...")
print("Safe to stop with Ctrl+C — restarts from where it stopped.")
print()

try:
    for i, item in enumerate(all_videos):
        video_path = item["video_path"]
        rel_path   = item["rel_path"]

        if rel_path in metadata:
            skipped += 1
            continue

        save_path = features_path / Path(rel_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        transcript, confidence = transcribe(video_path)
        embedding = encode(transcript, confidence)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

        np.save(str(save_path), embedding)

        metadata[rel_path] = {
            "video_path":  video_path,
            "embedding":   str(save_path),
            "label":       item["label"],
            "category":    item["category"],
            "transcript":  transcript,
            "confidence":  round(float(confidence), 4),
            "reliable":    bool(confidence > -1.0),
            "zero_vector": bool(transcript == "" or confidence < CONFIDENCE_THRESH)
        }

        done += 1

        if done % 50 == 0:
            elapsed   = time.time() - start_t
            rate      = done / max(elapsed, 1)
            remaining = (total - done - skipped) / max(rate, 0.001)
            hrs       = int(remaining // 3600)
            mins      = int((remaining % 3600) // 60)
            pct       = (done + skipped) / total * 100
            print("[" + str(done + skipped) + "/" + str(total) + "]",
                  str(round(pct, 1)) + "%",
                  "done=" + str(done),
                  "skipped=" + str(skipped),
                  "ETA=" + str(hrs) + "h " + str(mins) + "m",
                  "|", Path(video_path).name[:40])

        if done % 200 == 0:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

except KeyboardInterrupt:
    print()
    print("Stopped by user — saving progress...")

# ── FINAL SAVE ────────────────────────────────────────────────
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ── FINAL REPORT ──────────────────────────────────────────────
elapsed_total = time.time() - start_t
hrs  = int(elapsed_total // 3600)
mins = int((elapsed_total % 3600) // 60)

print()
print("=" * 60)
print("PRECOMPUTE COMPLETE")
print("=" * 60)
print("Total videos         :", total)
print("Processed this run   :", done)
print("Already done before  :", skipped)
print("Time taken           :", str(hrs) + "h " + str(mins) + "m")
print("Features saved to    :", FEATURES_DIR)
print("Metadata saved to    :", METADATA_FILE)
print()

real_count = sum(1 for v in metadata.values() if v["label"] == 0)
fake_count = sum(1 for v in metadata.values() if v["label"] == 1)
reliable   = sum(1 for v in metadata.values() if v["reliable"])
zero_vecs  = sum(1 for v in metadata.values() if v["zero_vector"])

print("Real embeddings      :", real_count)
print("Fake embeddings      :", fake_count)
print("Reliable transcripts :", reliable)
print("Zero vectors         :", zero_vecs)
print()
print("Next step: build dataset.py")

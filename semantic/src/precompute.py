"""Precompute semantic features for FakeAVCeleb or core FaceForensics++.

Select the dataset explicitly with DATASET_NAME=fakeavceleb|ffpp and optionally
override its root with DATASET_ROOT. Importing this module never starts work.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import whisper
from sentence_transformers import SentenceTransformer

# Keep semantic model loading local/offline, as in the existing pipeline.
os.environ["HF_HUB_OFFLINE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASET_NAME = os.environ.get("DATASET_NAME", "fakeavceleb").strip().lower()
FFPP_SOURCES = (
    ("original_sequences/youtube", 0),
    ("original_sequences/actors", 0),
    ("manipulated_sequences/Deepfakes", 1),
    ("manipulated_sequences/Face2Face", 1),
    ("manipulated_sequences/FaceSwap", 1),
    ("manipulated_sequences/NeuralTextures", 1),
)
DEFAULT_DATASET_ROOT = (PROJECT_ROOT / "Datasets" / "FaceForensics++" / "data"
                        if DATASET_NAME == "ffpp" else PROJECT_ROOT / "datasets" / "FakeAVCeleb_v1.2")
DATASET_PATH = Path(os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT))
FEATURES_BASE = SCRIPT_DIR.parent / "data" / "features"
# Keep FF++ metadata and outputs separate from the existing FakeAVCeleb tree.
FEATURES_DIR = FEATURES_BASE / "ffpp" if DATASET_NAME == "ffpp" else FEATURES_BASE
METADATA_FILE = FEATURES_DIR / "metadata.json"

WHISPER_SIZE = "base"
CONFIDENCE_THRESH = -1.5
MIN_WORDS = 3
FAKEAV_CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
    "RealVideo-FakeAudio": 1,
}


def scan_videos():
    """Return video records with root-relative paths and labels."""
    records = []
    if DATASET_NAME == "ffpp":
        categories = FFPP_SOURCES
    elif DATASET_NAME == "fakeavceleb":
        categories = FAKEAV_CATEGORIES
    else:
        raise ValueError("DATASET_NAME must be 'fakeavceleb' or 'ffpp'")

    for category, label in (categories if DATASET_NAME == "ffpp" else categories.items()):
        category_root = DATASET_PATH / category
        if DATASET_NAME == "ffpp":
            category_root = category_root / "c23" / "videos"
        videos = sorted(category_root.rglob("*.mp4")) if category_root.is_dir() else []
        for video in videos:
            rel_path = video.relative_to(DATASET_PATH)
            records.append({
                "video_path": str(video),
                "label": label,
                "category": category,
                "dataset": DATASET_NAME,
                "rel_path": rel_path.as_posix(),
            })
        print(f"  {category:<30} {len(videos):>6} videos")
    return records


def transcribe(whisper_model, video_path):
    try:
        result = whisper_model.transcribe(str(video_path), language=None, task="transcribe",
                                          verbose=False, fp16=False)
        text = result["text"].strip()
        segments = result.get("segments", [])
        confidence = (sum(s["avg_logprob"] for s in segments) / len(segments)
                      if segments else -2.0)
        if len(text.split()) < MIN_WORDS:
            confidence -= 0.5
        return text, confidence, result.get("language", "unknown")
    except Exception:
        return "", -2.0, "unknown"


def encode(transcript, confidence, sbert_model):
    if not transcript or confidence < CONFIDENCE_THRESH:
        return np.zeros(384, dtype=np.float32)
    with torch.no_grad():
        embedding = sbert_model.encode(transcript, convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=False)
    return embedding.float().detach().cpu().numpy().astype(np.float32)


def main():
    if DATASET_NAME not in {"ffpp", "fakeavceleb"}:
        raise ValueError("DATASET_NAME must be 'fakeavceleb' or 'ffpp'")
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_PATH}")
    records = scan_videos()
    print(f"Dataset : {DATASET_PATH}")
    print(f"Features: {FEATURES_DIR}")
    print(f"Total   : {len(records):,} videos")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA_FILE.exists():
        with METADATA_FILE.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    print(f"Resuming: {len(metadata):,} metadata entries")

    print("Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_SIZE)
    print("Loading Sentence-BERT...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    for parameter in sbert_model.parameters():
        parameter.requires_grad = False

    done = skipped = 0
    started = time.time()
    try:
        for item in records:
            rel_path = item["rel_path"]
            output_path = FEATURES_DIR / Path(rel_path).with_suffix(".npy")
            if rel_path in metadata and output_path.exists():
                skipped += 1
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            transcript, confidence, language = transcribe(whisper_model, item["video_path"])
            embedding = encode(transcript, confidence, sbert_model)
            assert embedding.shape == (384,) and embedding.dtype == np.float32
            np.save(str(output_path), embedding)
            metadata[rel_path] = {
                "video_path": item["video_path"],
                "relative_path": rel_path,
                "video_stem": Path(item["video_path"]).stem,
                "embedding": str(output_path),
                "label": item["label"],
                "category": item["category"],
                "dataset": item["dataset"],
                "transcript": transcript,
                "confidence": round(float(confidence), 4),
                "language": language,
                "reliable": bool(confidence > -1.0),
                "zero_vector": bool(not transcript or confidence < CONFIDENCE_THRESH),
                "shape": [384],
                "dtype": "float32",
            }
            done += 1
            if done % 200 == 0:
                with METADATA_FILE.open("w", encoding="utf-8") as stream:
                    json.dump(metadata, stream, indent=2, ensure_ascii=False)
                print(f"Processed {done:,}; resumed {skipped:,}")
    except KeyboardInterrupt:
        print("Stopped; saving progress...")
    with METADATA_FILE.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(f"Complete: processed {done:,}, resumed {skipped:,} in {time.time() - started:.1f}s")
    print(f"Features: {FEATURES_DIR}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()

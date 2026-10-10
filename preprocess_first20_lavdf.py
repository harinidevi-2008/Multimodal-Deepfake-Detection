
import os
import json
import time
from pathlib import Path

# Set paths before importing project modules.
ROOT = Path(r"C:\LAV-DF\dev")
os.environ["DATASET_ROOT"] = str(ROOT)
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import whisper
from sentence_transformers import SentenceTransformer

from visual.src.process_video import process_video as process_visual
from visual.src.config import OUTPUT_FOLDER as VISUAL_OUT
from audio.src.audio_stream import process_video as process_audio
from audio.src.config import OUTPUT_FOLDER as AUDIO_OUT
from visual.src.eye_blink.blink_analyzer import analyze_blinks
from visual.src.lipsync.lipsync_analyzer import analyze_lipsync
from visual.src.blink_lipsync_precompute import (
    blink_result_to_vector,
    lipsync_result_to_vector,
)

BLINK_OUT = Path("visual/data/blink_features_lavdf")
LIPSYNC_OUT = Path("visual/data/lipsync_features_lavdf")
SEMANTIC_OUT = Path("semantic/data/features_lavdf")
METADATA_OUT = SEMANTIC_OUT / "metadata.json"

def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

    videos = sorted(
        p for p in ROOT.rglob("*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov"}
    )[:20]

    if not videos:
        raise RuntimeError(f"No videos found under {ROOT}")

    print(f"Testing {len(videos)} videos")
    print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load semantic models once, not once per video.
    print("Loading Whisper base model...")
    whisper_model = whisper.load_model("base")

    print("Loading Sentence-BERT...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    sbert.eval()

    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if METADATA_OUT.exists():
        metadata = json.loads(METADATA_OUT.read_text(encoding="utf-8"))

    start = time.time()

    for index, video in enumerate(videos, start=1):
        rel = video.relative_to(ROOT)
        feature_rel = rel.with_suffix(".npy")

        print(f"\n[{index}/{len(videos)}] {rel}")

        # 1. Visual features
        try:
            out = VISUAL_OUT / feature_rel
            if not out.exists():
                process_visual(str(video), str(out))
            print("  Visual: done")
        except Exception as e:
            print(f"  Visual FAILED: {e}")

        # 2. Audio features
        try:
            out = AUDIO_OUT / feature_rel
            if not out.exists():
                process_audio(video, out)
            print("  Audio: done")
        except Exception as e:
            print(f"  Audio FAILED: {e}")

        # 3. Blink features
        try:
            out = BLINK_OUT / feature_rel
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                result = analyze_blinks(video)
                np.save(out, blink_result_to_vector(result))
            print("  Blink: done")
        except Exception as e:
            print(f"  Blink FAILED: {e}")

        # 4. Lip-sync features
        try:
            out = LIPSYNC_OUT / feature_rel
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                result = analyze_lipsync(video)
                np.save(out, lipsync_result_to_vector(result))
            print("  Lip-sync: done")
        except Exception as e:
            print(f"  Lip-sync FAILED: {e}")

        # 5. Semantic features
        try:
            out = SEMANTIC_OUT / feature_rel
            if not out.exists():
                result = whisper_model.transcribe(
                    str(video), task="transcribe", verbose=False, fp16=False
                )
                transcript = result["text"].strip()
                segments = result.get("segments", [])
                confidence = (
                    sum(s["avg_logprob"] for s in segments) / len(segments)
                    if segments else -2.0
                )
                if len(transcript.split()) < 3:
                    confidence -= 0.5

                if not transcript or confidence < -1.5:
                    embedding = np.zeros(384, dtype=np.float32)
                else:
                    with torch.no_grad():
                        embedding = sbert.encode(
                            transcript,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        ).astype(np.float32)

                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, embedding)
                metadata[rel.as_posix()] = {
                    "relative_path": rel.as_posix(),
                    "transcript": transcript,
                    "confidence": float(confidence),
                    "language": result.get("language", "unknown"),
                }
                METADATA_OUT.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            print("  Semantic: done")
        except Exception as e:
            print(f"  Semantic FAILED: {e}")

        print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")

    print("\n20-video test finished.")
    print(f"Total time: {(time.time() - start) / 60:.1f} minutes")
    print("Review the FAILED messages and output counts before scaling up.")

if __name__ == "__main__":
    main()

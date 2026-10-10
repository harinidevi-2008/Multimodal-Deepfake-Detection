import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import json
import time
import traceback
from pathlib import Path

import numpy as np

from visual.src.process_video import process_video as process_visual
from audio.src.audio_stream import process_video as process_audio
from visual.src.eye_blink.blink_analyzer import analyze_blinks
from visual.src.lipsync.lipsync_analyzer import analyze_lipsync
from visual.src.blink_lipsync_precompute import (
    blink_result_to_vector,
    lipsync_result_to_vector,
)
from semantic.src.semantic_stream import SemanticStream

ROOT = Path(r"C:\LAV-DF\dev")
PROJECT = Path(__file__).resolve().parents[1]
LIMIT = 100

OUT = {
    "visual": PROJECT / "visual/data/features_aligned_lavdf",
    "audio": PROJECT / "audio/data/features_lavdf",
    "blink": PROJECT / "visual/data/blink_features_lavdf",
    "lipsync": PROJECT / "visual/data/lipsync_features_lavdf",
    "semantic": PROJECT / "semantic/data/features_lavdf",
}
METADATA = OUT["semantic"] / "metadata.json"


def save_vector(path, vector, expected_dim):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (expected_dim,):
        raise ValueError(
            f"Expected ({expected_dim},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains NaN or infinity")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"LAV-DF folder not found: {ROOT}")

    videos = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4"
    )[:LIMIT]

    if not videos:
        raise RuntimeError(f"No MP4 videos found under {ROOT}")

    for path in OUT.values():
        path.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA.exists():
        try:
            with METADATA.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                metadata = {}
        except (OSError, json.JSONDecodeError):
            print("Existing metadata could not be read; starting a new file.")

    semantic_model = None
    counts = {
        stream: {"created": 0, "skipped": 0, "failed": 0}
        for stream in OUT
    }
    failures = []
    started = time.perf_counter()

    print(f"LAV-DF root: {ROOT}")
    print(f"Videos selected: {len(videos)}")
    print("Outputs are kept in dataset-specific folders.")

    for index, video in enumerate(videos, start=1):
        # LAV-DF/dev is currently a flat folder, so the video stem
        # uniquely identifies these files within this dataset.
        key = video.relative_to(ROOT).as_posix()
        stem = video.stem
        print(f"\n[{index}/{len(videos)}] {video.name}")

        # Visual features: 1280 dimensions
        out = OUT["visual"] / f"{stem}.npy"
        if out.exists():
            counts["visual"]["skipped"] += 1
        else:
            try:
                result = process_visual(str(video), str(out))
                if result is None or not out.exists():
                    raise RuntimeError("No visual feature was saved")
                save_vector(out, np.load(out), 1280)
                counts["visual"]["created"] += 1
            except Exception as exc:
                counts["visual"]["failed"] += 1
                failures.append((key, "visual", str(exc)))
                print(f"  VISUAL FAILED: {exc}")

        # Audio features: 768 dimensions
        out = OUT["audio"] / f"{stem}.npy"
        if out.exists():
            counts["audio"]["skipped"] += 1
        else:
            try:
                process_audio(video, out)
                if not out.exists():
                    raise RuntimeError("No audio feature was saved")
                save_vector(out, np.load(out), 768)
                counts["audio"]["created"] += 1
            except Exception as exc:
                counts["audio"]["failed"] += 1
                failures.append((key, "audio", str(exc)))
                print(f"  AUDIO FAILED: {exc}")

        # Blink features: 4 dimensions
        out = OUT["blink"] / f"{stem}.npy"
        if out.exists():
            counts["blink"]["skipped"] += 1
        else:
            try:
                result = analyze_blinks(video)
                save_vector(out, blink_result_to_vector(result), 4)
                counts["blink"]["created"] += 1
            except Exception as exc:
                counts["blink"]["failed"] += 1
                failures.append((key, "blink", str(exc)))
                print(f"  BLINK FAILED: {exc}")

        # Lip-sync features: 2 dimensions
        out = OUT["lipsync"] / f"{stem}.npy"
        if out.exists():
            counts["lipsync"]["skipped"] += 1
        else:
            try:
                result = analyze_lipsync(video)
                save_vector(out, lipsync_result_to_vector(result), 2)
                counts["lipsync"]["created"] += 1
            except Exception as exc:
                counts["lipsync"]["failed"] += 1
                failures.append((key, "lipsync", str(exc)))
                print(f"  LIP-SYNC FAILED: {exc}")

        # Semantic features: 384 dimensions
        out = OUT["semantic"] / f"{stem}.npy"
        if out.exists():
            counts["semantic"]["skipped"] += 1
        else:
            try:
                if semantic_model is None:
                    semantic_model = SemanticStream(whisper_size="base")
                result = semantic_model.extract_features(video)
                embedding = result["embedding_384"].detach().cpu().numpy()
                save_vector(out, embedding, 384)
                metadata[key] = {
                    "video_path": str(video),
                    "relative_path": key,
                    "video_stem": stem,
                    "embedding": str(out),
                    "dataset": "lavdf",
                    "transcript": result.get("transcript", ""),
                    "confidence": float(result.get("confidence", -2.0)),
                    "language": result.get("language", "unknown"),
                    "reliable": bool(result.get("reliable", False)),
                    "shape": [384],
                    "dtype": "float32",
                }
                counts["semantic"]["created"] += 1
            except Exception as exc:
                counts["semantic"]["failed"] += 1
                failures.append((key, "semantic", str(exc)))
                print(f"  SEMANTIC FAILED: {exc}")

        # Save progress after every video, so an interruption is recoverable.
        with METADATA.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - started
    report = {
        "dataset_root": str(ROOT),
        "videos_selected": len(videos),
        "elapsed_seconds": round(elapsed, 2),
        "streams": counts,
        "failures": [
            {"video": v, "stream": s, "error": e}
            for v, s, e in failures
        ],
    }

    report_path = PROJECT / "scripts" / "lavdf_100_pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== PILOT SUMMARY ==========")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    for stream, stats in counts.items():
        print(
            f"{stream:8s}: created={stats['created']}, "
            f"skipped={stats['skipped']}, failed={stats['failed']}"
        )
    print(f"Report: {report_path}")
    print(f"Failures: {len(failures)}")
    print("Inspect the report before processing the full dataset.")


if __name__ == "__main__":
    main()

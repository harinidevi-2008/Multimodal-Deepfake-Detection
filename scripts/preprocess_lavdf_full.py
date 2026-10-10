"""Restartable, split-specific LAV-DF feature extraction.

Outputs are isolated under <project>/data/lavdf_full/<split>/<stream>/.
Existing valid vectors are retained; invalid or missing vectors are rebuilt.
"""

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DEFAULT_ROOT = Path(
    r"C:\Deepfake_Detection\Multimodal-Deepfake-Detection\datasets\LAV-DF\complete_extract\LAV-DF"
)
SPLITS = ("train", "dev", "test")
DIMENSIONS = {"visual": 1280, "audio": 768, "blink": 4, "lipsync": 2, "semantic": 384}
CHECKPOINT_EVERY = 25


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def validate_vector(path, expected_dim):
    try:
        import numpy as np
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (expected_dim,):
            return False, f"expected shape ({expected_dim},), got {vector.shape}"
        if not np.issubdtype(vector.dtype, np.number):
            return False, f"expected numeric dtype, got {vector.dtype}"
        if not np.isfinite(vector).all():
            return False, "contains NaN or infinity"
        return True, "ok"
    except Exception as exc:
        return False, f"cannot read feature: {exc}"


def atomic_save_vector(path, vector, expected_dim):
    import numpy as np
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (expected_dim,):
        raise ValueError(f"Expected vector shape ({expected_dim},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains NaN or infinity")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            np.save(f, vector, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        # Verify the serialized bytes before publishing the final name.
        valid, reason = validate_vector(temp, expected_dim)
        if not valid:
            raise ValueError(f"Serialized feature failed validation: {reason}")
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def import_streams():
    """Import extractors individually, preserving their established APIs."""
    specs = {
        "visual": ("visual.src.process_video", "process_video"),
        "audio": ("audio.src.audio_stream", "process_video"),
        "blink": ("visual.src.eye_blink.blink_analyzer", "analyze_blinks"),
        "lipsync": ("visual.src.lipsync.lipsync_analyzer", "analyze_lipsync"),
        "vectors": ("visual.src.blink_lipsync_precompute", None),
        "semantic": ("semantic.src.semantic_stream", "SemanticStream"),
    }
    modules, errors = {}, {}
    for key, (module_name, _) in specs.items():
        try:
            modules[key] = importlib.import_module(module_name)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
    return modules, errors


def stream_requirements(modules, errors, streams):
    """Fail fast only for selected streams whose imports/models are unavailable."""
    issues = []
    for stream in streams:
        dependency = "vectors" if stream in ("blink", "lipsync") else stream
        if dependency in errors:
            issues.append(f"{stream} extractor import failed: {errors[dependency]}")
    # These checks provide clear guidance before a potentially long run.
    required_modules = {
        "visual": ("cv2", "torch", "torchvision"),
        "audio": ("torch", "librosa", "moviepy", "transformers", "safetensors"),
        "blink": ("cv2", "mediapipe"),
        "lipsync": ("cv2", "mediapipe", "librosa"),
        "semantic": ("torch", "whisper", "sentence_transformers", "huggingface_hub"),
    }
    for stream in streams:
        for name in required_modules[stream]:
            try:
                importlib.import_module(name)
            except Exception as exc:
                issues.append(f"{stream} dependency '{name}' unavailable: {exc}")
    if "audio" in streams:
        try:
            import audio.src.audio_stream as audio_stream
            from transformers import Wav2Vec2Config
            Wav2Vec2Config.from_pretrained(audio_stream.MODEL_NAME, local_files_only=True)
            cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--wav2vec2-base-960h" / "snapshots"
            if not list(cache.glob("*/model.safetensors")):
                issues.append(f"audio Wav2Vec2 checkpoint missing under {cache}")
        except Exception as exc:
            issues.append(f"audio Wav2Vec2 local model check failed: {exc}")
    if "semantic" in streams:
        # The semantic module forces Hugging Face offline mode. Check its
        # expected local assets without constructing either model twice.
        whisper_cache = Path.home() / ".cache" / "whisper" / "base.pt"
        if not whisper_cache.is_file():
            issues.append(f"semantic Whisper base checkpoint not found: {whisper_cache}")
        try:
            from huggingface_hub import try_to_load_from_cache
            cached_config = try_to_load_from_cache(
                "sentence-transformers/all-MiniLM-L6-v2", "config.json"
            )
            if not isinstance(cached_config, str) or not Path(cached_config).is_file():
                issues.append("semantic Sentence-BERT all-MiniLM-L6-v2 is not available in the local Hugging Face cache")
        except Exception as exc:
            issues.append(f"semantic Sentence-BERT cache check failed: {exc}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Preprocess one LAV-DF split into isolated feature directories.")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"LAV-DF dataset root (default: {DEFAULT_ROOT})")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos (smoke tests).")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="Atomically checkpoint metadata and progress every N videos (default: 25).")
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401 - required by validation/extraction helpers
    except Exception as exc:
        parser.error(f"Required dependency 'numpy' is unavailable: {exc}")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be a positive integer")
    root = args.root.expanduser().resolve()
    split_root = root / args.split
    if not root.is_dir():
        parser.error(f"Dataset root does not exist or is not a directory: {root}")
    if not split_root.is_dir():
        parser.error(f"Selected split folder does not exist: {split_root}")

    videos = sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        parser.error(f"No MP4 videos found in selected split: {split_root}")

    output_root = PROJECT / "data" / "lavdf_full" / args.split
    output_dirs = {s: output_root / s for s in DIMENSIONS}
    metadata_path = output_root / "semantic_metadata.json"
    progress_path = output_root / "progress.json"
    report_path = PROJECT / "scripts" / f"lavdf_{args.split}_full_report.json"
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    streams = list(DIMENSIONS)
    modules, import_errors = import_streams()
    issues = stream_requirements(modules, import_errors, streams)
    if issues:
        parser.error("Startup dependency/model checks failed:\n  - " + "\n  - ".join(issues))

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata root is not an object")
    except Exception as exc:
        parser.error(f"Cannot read semantic metadata {metadata_path}: {exc}")

    counts = {stream: {"created": 0, "skipped": 0, "failed": 0} for stream in streams}
    failures = []
    semantic_model = None
    started = time.perf_counter()
    print(f"Dataset root: {root}\nSelected split: {args.split} ({split_root})\nVideos selected: {len(videos)}")
    print(f"Output root: {output_root}\nCheckpoint interval: every {args.checkpoint_every} videos")

    def record_failure(video_key, stream, exc):
        failure = {"video": video_key, "stream": stream, "error": f"{type(exc).__name__}: {exc}"}
        failures.append(failure)
        counts[stream]["failed"] += 1
        print(f"  {stream.upper()} FAILED: {failure['error']}")

    for index, video in enumerate(videos, 1):
        key = video.relative_to(split_root).as_posix()
        rel_output = Path(*Path(key).with_suffix(".npy").parts)
        print(f"\n[{index}/{len(videos)}] {key}")
        for stream, expected_dim in DIMENSIONS.items():
            out = output_dirs[stream] / rel_output
            if out.exists():
                valid, reason = validate_vector(out, expected_dim)
                if valid:
                    counts[stream]["skipped"] += 1
                    if stream == "semantic" and key not in metadata:
                        metadata[key] = {"video_path": str(video), "relative_path": key,
                                         "video_stem": video.stem, "embedding": str(out),
                                         "dataset": "lavdf", "split": args.split,
                                         "shape": [384], "dtype": "float32",
                                         "metadata_recovered": True}
                    continue
                print(f"  {stream.upper()} existing file invalid ({reason}); rebuilding")
            try:
                if stream == "visual":
                    result = modules["visual"].process_video(str(video), str(out))
                    if result is None:
                        raise RuntimeError("No faces/features produced")
                    if not out.exists():
                        raise RuntimeError("Extractor returned without saving a feature")
                    # Re-serialize atomically after validating the extractor output.
                    valid, reason = validate_vector(out, expected_dim)
                    if not valid:
                        raise ValueError(f"Generated vector invalid: {reason}")
                    generated = np.load(out, allow_pickle=False)
                    atomic_save_vector(out, generated, expected_dim)
                elif stream == "audio":
                    modules["audio"].process_video(video, out)
                    if not out.exists():
                        raise RuntimeError("Extractor returned without saving a feature")
                    valid, reason = validate_vector(out, expected_dim)
                    if not valid:
                        raise ValueError(f"Generated vector invalid: {reason}")
                    generated = np.load(out, allow_pickle=False)
                    atomic_save_vector(out, generated, expected_dim)
                elif stream == "blink":
                    result = modules["blink"].analyze_blinks(video)
                    vector = modules["vectors"].blink_result_to_vector(result)
                    atomic_save_vector(out, vector, expected_dim)
                elif stream == "lipsync":
                    result = modules["lipsync"].analyze_lipsync(video)
                    vector = modules["vectors"].lipsync_result_to_vector(result)
                    atomic_save_vector(out, vector, expected_dim)
                else:
                    if semantic_model is None:
                        semantic_model = modules["semantic"].SemanticStream(whisper_size="base")
                    result = semantic_model.extract_features(video)
                    embedding = result["embedding_384"].detach().cpu().numpy()
                    atomic_save_vector(out, embedding, expected_dim)
                    metadata[key] = {
                        "video_path": str(video), "relative_path": key, "video_stem": video.stem,
                        "embedding": str(out), "dataset": "lavdf", "split": args.split,
                        "transcript": result.get("transcript", ""),
                        "confidence": float(result.get("confidence", -2.0)),
                        "language": result.get("language", "unknown"),
                        "reliable": bool(result.get("reliable", False)),
                        "shape": [384], "dtype": "float32",
                    }
                valid, reason = validate_vector(out, expected_dim)
                if not valid:
                    raise ValueError(f"Generated vector failed final validation: {reason}")
                counts[stream]["created"] += 1
            except Exception as exc:
                record_failure(key, stream, exc)
                # Remove incomplete/corrupt outputs so a restart retries them.
                try:
                    out.unlink(missing_ok=True)
                except OSError:
                    pass

        if index % args.checkpoint_every == 0 or index == len(videos):
            atomic_json(metadata_path, metadata)
            atomic_json(progress_path, {
                "dataset_root": str(root), "split": args.split, "videos_selected": len(videos),
                "last_completed_index": index, "last_video": key,
                "streams": counts, "updated_unix_time": time.time(),
            })

    elapsed = time.perf_counter() - started
    report = {
        "dataset_root": str(root), "split": args.split, "split_root": str(split_root),
        "videos_selected": len(videos), "elapsed_seconds": round(elapsed, 2),
        "output_root": str(output_root), "streams": counts, "failures": failures,
    }
    atomic_json(report_path, report)
    print(f"\n========== {args.split.upper()} SUMMARY ==========")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    for stream, stats in counts.items():
        print(f"{stream:8s}: created={stats['created']}, skipped={stats['skipped']}, failed={stats['failed']}")
    print(f"Report: {report_path}\nFailures: {len(failures)}")


if __name__ == "__main__":
    main()

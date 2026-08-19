"""
Batch-precompute eye-blink and lip-sync feature vectors across a whole
dataset (e.g. FakeAVCeleb), mirroring the same RealVideo*/FakeVideo*
folder structure the other streams use, so EnhancedFusionDataset can
align samples by relative path exactly like FusionDataset does.

Each video produces two small .npy files (not one - blink and lipsync
are conceptually separate evidence sources, kept separate on disk so
either can be swapped, re-run, or evaluated independently):

    <blink_output>/<RealVideo-.../.../clip>.npy   -> 4-d vector
    <lipsync_output>/<RealVideo-.../.../clip>.npy -> 2-d vector

Blink vector:   [blink_count, blink_rate_per_min, average_blink_duration_sec, blink_irregularity_score]
Lipsync vector: [sync_score, mismatch_score]

Usage:
    python visual/src/blink_lipsync_precompute.py \
        --dataset-root FakeAVCeleb_v1.2 \
        --blink-output visual/data/blink_features \
        --lipsync-output visual/data/lipsync_features

Skips videos that already have both output files, so it's safe to
re-run after an interrupted pass (same convention as the semantic
stream's precompute.py).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from eye_blink.blink_analyzer import analyze_blinks
    from lipsync.lipsync_analyzer import analyze_lipsync
except ImportError:
    from .eye_blink.blink_analyzer import analyze_blinks
    from .lipsync.lipsync_analyzer import analyze_lipsync

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def blink_result_to_vector(result):
    return np.array(
        [
            result["blink_count"],
            result["blink_rate_per_min"],
            result["average_blink_duration_sec"],
            result["blink_irregularity_score"],
        ],
        dtype=np.float32,
    )


def lipsync_result_to_vector(result):
    return np.array(
        [result["sync_score"], result["mismatch_score"]],
        dtype=np.float32,
    )


def process_dataset(dataset_root, blink_output, lipsync_output, max_blink_fps=None):
    dataset_root = Path(dataset_root)
    blink_output = Path(blink_output)
    lipsync_output = Path(lipsync_output)

    video_files = sorted(
        p for p in dataset_root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    logger.info("Found %d videos under %s", len(video_files), dataset_root)

    processed, skipped, failed = 0, 0, 0

    for video_path in video_files:
        relative_path = video_path.relative_to(dataset_root).with_suffix(".npy")
        blink_out = blink_output / relative_path
        lipsync_out = lipsync_output / relative_path

        if blink_out.exists() and lipsync_out.exists():
            skipped += 1
            continue

        try:
            if not blink_out.exists():
                blink_result = analyze_blinks(video_path, max_fps=max_blink_fps)
                blink_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(blink_out, blink_result_to_vector(blink_result))

            if not lipsync_out.exists():
                lipsync_result = analyze_lipsync(video_path)
                lipsync_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(lipsync_out, lipsync_result_to_vector(lipsync_result))

            processed += 1
            if processed % 25 == 0:
                logger.info("Processed %d / %d videos", processed, len(video_files))

        except Exception as exc:  # noqa: BLE001 - keep the batch going on a bad clip
            logger.warning("Failed on %s: %s", video_path, exc)
            failed += 1

    logger.info(
        "Done. processed=%d skipped=%d failed=%d total=%d",
        processed, skipped, failed, len(video_files),
    )


def main():
    parser = argparse.ArgumentParser(description="Batch precompute blink and lip-sync feature vectors.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--blink-output", required=True)
    parser.add_argument("--lipsync-output", required=True)
    parser.add_argument("--max-blink-fps", type=float, default=None)
    args = parser.parse_args()

    process_dataset(args.dataset_root, args.blink_output, args.lipsync_output, args.max_blink_fps)


if __name__ == "__main__":
    main()

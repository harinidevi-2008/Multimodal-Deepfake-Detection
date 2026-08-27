"""
Quick manual runner for the lip-sync module.

Renamed from lipsync_test.py to lipsync_manual.py so pytest's default
`test_*.py` / `*_test.py` collection patterns no longer match this
file. Importing this file previously pulled in visual/src/__init__.py
(via the package chain visual/src/lipsync -> visual/src), which
imports feature_extractor.py and face_detector.py - both of which used
to build their models (EfficientNet-B0, MTCNN) eagerly at import time,
triggering a real network weights download just from pytest
discovering this file. That eager-loading has separately been fixed
(see feature_extractor.py / face_detector.py - both now lazy-load on
first actual use), but this file never needed to be pytest-collectable
in the first place: it is a manual CLI runner, not an automated test.

Usage:
    python visual/src/lipsync/lipsync_manual.py path/to/video.mp4
    python visual/src/lipsync/lipsync_manual.py path/to/video.mp4 --audio path/to/audio.wav

If --audio is omitted, audio is auto-extracted from the video with ffmpeg.
If no video path is given, edit DEFAULT_VIDEO below to point at a sample clip.
"""

import argparse
import logging
from pathlib import Path

try:
    from .lipsync_analyzer import analyze_lipsync
except ImportError:
    from lipsync_analyzer import analyze_lipsync

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_VIDEO = Path(__file__).resolve().parents[3] / "sample_videos" / "sample.mp4"


def main():
    parser = argparse.ArgumentParser(description="Run lip-sync consistency analysis on a video.")
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Path to video file.")
    parser.add_argument("--audio", default=None, help="Optional path to a pre-extracted wav file.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return

    logger.info("Analyzing: %s", video_path)
    result = analyze_lipsync(video_path, audio_path=args.audio)

    print(f"Lip-sync consistency: {result['lipsync_consistency']}")
    print(f"Lip-sync status: {result['lipsync_status']}")
    print(f"Valid frame ratio: {result['valid_frame_ratio']}")


if __name__ == "__main__":
    main()

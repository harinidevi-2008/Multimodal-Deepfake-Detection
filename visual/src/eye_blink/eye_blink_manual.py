"""
Quick manual runner for the eye-blink module.

Renamed from eye_blink_test.py to eye_blink_manual.py so pytest's
default `test_*.py` / `*_test.py` collection patterns no longer match
this file. Importing this file previously pulled in
visual/src/__init__.py (via the package chain visual/src/eye_blink ->
visual/src), which imports feature_extractor.py and face_detector.py -
both of which used to build their models (EfficientNet-B0,
MTCNN) eagerly at import time, triggering a real network weights
download just from pytest discovering this file. That eager-loading
has separately been fixed (see feature_extractor.py / face_detector.py -
both now lazy-load on first actual use), but this file never needed to
be pytest-collectable in the first place: it is a manual CLI runner,
not an automated test.

Usage:
    python visual/src/eye_blink/eye_blink_manual.py path/to/video.mp4
    python visual/src/eye_blink/eye_blink_manual.py path/to/video.mp4 --max-fps 15

If no path is given, edit DEFAULT_VIDEO below to point at a sample clip.
"""

import argparse
import logging
from pathlib import Path

try:
    from .blink_analyzer import analyze_blinks
except ImportError:
    from blink_analyzer import analyze_blinks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_VIDEO = Path(__file__).resolve().parents[3] / "sample_videos" / "sample.mp4"


def main():
    parser = argparse.ArgumentParser(description="Run eye-blink analysis on a video.")
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Path to video file.")
    parser.add_argument("--max-fps", type=float, default=None, help="Optional sampling fps cap.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return

    logger.info("Analyzing: %s", video_path)
    result = analyze_blinks(video_path, max_fps=args.max_fps)

    print(f"Blink count: {result['blink_count']}")
    print(f"Blink rate: {result['blink_rate_per_min']} per min")
    print(f"Blink irregularity score: {result['blink_irregularity_score']}")
    print(f"Blink status: {result['blink_status']}")


if __name__ == "__main__":
    main()

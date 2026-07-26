import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_FOLDER = PROJECT_ROOT / "datasets" / "FakeAVCeleb_v1.2"
OUTPUT_FOLDER = PROJECT_ROOT / "visual" / "data" / "features"
DEBUG_FRAME_PATH = PROJECT_ROOT / "visual" / "debug" / "extracted_frames" / "frame_000.jpg"
DEFAULT_FPS_TARGET = 2


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

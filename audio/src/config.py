import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = PROJECT_ROOT / "sample_videos"
AUDIO_FOLDER = PROJECT_ROOT / "temp_audio"
OUTPUT_FOLDER = PROJECT_ROOT / "audio" / "data" / "features"
MODEL_NAME = "facebook/wav2vec2-base-960h"
DEVICE = "cpu"
DEFAULT_SAMPLE_RATE = 16000


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

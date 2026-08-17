import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset
INPUT_FOLDER = PROJECT_ROOT / "FakeAVCeleb_v1.2"

# Temporary extracted audio
AUDIO_FOLDER = PROJECT_ROOT / "temp_audio"

# Saved embeddings
OUTPUT_FOLDER = PROJECT_ROOT / "audio" / "data" / "features"

MODEL_NAME = "facebook/wav2vec2-base-960h"

DEVICE = "cpu"

DEFAULT_SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

SKIP_EXISTING = True


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
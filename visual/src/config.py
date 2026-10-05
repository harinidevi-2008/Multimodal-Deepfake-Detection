import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = os.environ.get("DATASET_NAME", "fakeavceleb").strip().lower()
FFPP_SOURCES = (
    ("original_sequences/youtube", 0),
    ("original_sequences/actors", 0),
    ("manipulated_sequences/Deepfakes", 1),
    ("manipulated_sequences/Face2Face", 1),
    ("manipulated_sequences/FaceSwap", 1),
    ("manipulated_sequences/NeuralTextures", 1),
)
_DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Datasets" / "FaceForensics++" / "data"
    if DATASET_NAME == "ffpp"
    else PROJECT_ROOT / "datasets" / "FakeAVCeleb_v1.2"
)
VIDEO_FOLDER = Path(os.environ.get("DATASET_ROOT", _DEFAULT_DATASET_ROOT))
OUTPUT_FOLDER = PROJECT_ROOT / "visual" / "data" / "features_aligned"
DEBUG_FRAME_PATH = PROJECT_ROOT / "visual" / "debug" / "extracted_frames" / "frame_000.jpg"
DEFAULT_FPS_TARGET = 2


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

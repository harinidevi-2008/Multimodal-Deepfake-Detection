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

# Temporary WAV files under AUDIO_FOLDER are only a means to compute
# the saved Wav2Vec2 feature (.npy) - once that feature has been
# written successfully, the WAV is no longer needed for this project's
# pipeline. Deleting it afterward keeps temp_audio from silently
# accumulating a duplicate, uncompressed copy of the entire dataset's
# audio (tens of GB across ~21,544 videos) forever. Set to False to
# keep temp WAVs around (e.g. for manual debugging/inspection) - the
# "reuse an already-extracted WAV after a crashed run" optimization in
# process_video() keeps working either way, since it only matters for
# WAVs that are still on disk when a run resumes.
DELETE_TEMP_AUDIO_AFTER_EXTRACTION = True


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
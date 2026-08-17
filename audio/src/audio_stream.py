import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from .config import (
        INPUT_FOLDER,
        AUDIO_FOLDER,
        OUTPUT_FOLDER,
        MODEL_NAME,
        DEFAULT_SAMPLE_RATE,
        SUPPORTED_EXTENSIONS,
        SKIP_EXISTING,
        configure_logging,
    )
except ImportError:
    from config import (
        INPUT_FOLDER,
        AUDIO_FOLDER,
        OUTPUT_FOLDER,
        MODEL_NAME,
        DEFAULT_SAMPLE_RATE,
        SUPPORTED_EXTENSIONS,
        SKIP_EXISTING,
        configure_logging,
    )

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Load Wav2Vec2 once
# ------------------------------------------------------------

logger.info("Loading Wav2Vec2 Processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

logger.info("Loading Wav2Vec2 Model...")
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

# Freeze Wav2Vec2 parameters
for param in model.parameters():
    param.requires_grad = False

logger.info("Model loaded successfully.")


# ------------------------------------------------------------
# Process one video
# ------------------------------------------------------------

def process_video(video_path: Path, output_path: Path):

    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_FOLDER / (video_path.stem + ".wav")

    # -----------------------
    # Extract Audio
    # -----------------------

    if not audio_path.exists():

        clip = VideoFileClip(str(video_path))

        clip.audio.write_audiofile(
            str(audio_path),
            fps=DEFAULT_SAMPLE_RATE,
            verbose=False,
            logger=None,
        )

        clip.close()

    # -----------------------
    # Load Audio
    # -----------------------

    waveform, sr = librosa.load(
        str(audio_path),
        sr=DEFAULT_SAMPLE_RATE,
    )

    # -----------------------
    # Mel Spectrogram
    # -----------------------

    _ = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=80,
    )

    # -----------------------
    # Wav2Vec2
    # -----------------------

    inputs = processor(
        waveform,
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state

    # -----------------------
    # Mean Pooling
    # -----------------------

    feature_vector = hidden_states.mean(dim=1)

    np.save(
        str(output_path),
        feature_vector.squeeze().cpu().numpy(),
    )


# ------------------------------------------------------------
# Process complete dataset
# ------------------------------------------------------------

def process_directory():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    video_files = []

    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(INPUT_FOLDER.rglob(f"*{ext}"))

    video_files = sorted(video_files)

    logger.info("Found %d videos", len(video_files))

    for video_path in tqdm(video_files):

        # Preserve folder structure
        relative_path = video_path.relative_to(INPUT_FOLDER)
        output_path = OUTPUT_FOLDER / relative_path.with_suffix(".npy")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if SKIP_EXISTING and output_path.exists():
            continue

        process_video(video_path, output_path)

    logger.info("Finished processing all videos.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    configure_logging()

    process_directory()


if __name__ == "__main__":
    main()
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from .config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, INPUT_FOLDER, MODEL_NAME, OUTPUT_FOLDER, configure_logging
except ImportError:
    from config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, INPUT_FOLDER, MODEL_NAME, OUTPUT_FOLDER, configure_logging

logger = logging.getLogger(__name__)


def process_video(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    audio_path = AUDIO_FOLDER / (video_path.stem + ".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Wav2Vec2 Processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    logger.info("Processor Loaded Successfully")

    logger.info("Loading Wav2Vec2 Model...")
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("Model Loaded Successfully")

    if not audio_path.exists():
        logger.info("Extracting Audio...")
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path), fps=DEFAULT_SAMPLE_RATE, verbose=False, logger=None)
        clip.close()
    else:
        logger.info("Audio already exists.")

    logger.info("Loading Audio...")
    waveform, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)

    logger.info("Sample Rate : %s", sr)
    logger.info("Waveform Shape : %s", waveform.shape)

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)
    logger.info("Mel Spectrogram Shape : %s", mel.shape)

    inputs = processor(waveform, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state
    logger.info("Hidden State Shape : %s", hidden_states.shape)

    feature_vector = hidden_states.mean(dim=1)
    logger.info("Final Feature Shape : %s", feature_vector.shape)

    np.save(str(output_path), feature_vector.squeeze().cpu().numpy())
    logger.info("Saved Feature : %s", output_path)

    return feature_vector.squeeze().cpu().numpy()


def process_directory(video_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    output_folder.mkdir(parents=True, exist_ok=True)
    video_files = sorted(video_folder.iterdir()) if video_folder.exists() else []

    for video_path in video_files:
        if video_path.is_file() and video_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
            output_file = video_path.stem + ".npy"
            output_path = output_folder / output_file
            process_video(str(video_path), str(output_path))

    return output_folder


def main():
    configure_logging()
    process_directory()


if __name__ == "__main__":
    main()

import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from .config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, MODEL_NAME, OUTPUT_FOLDER, configure_logging
except ImportError:
    from config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, MODEL_NAME, OUTPUT_FOLDER, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    audio_path = AUDIO_FOLDER / "WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"
    waveform, sample_rate = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    inputs = processor(waveform, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state
    logger.info("Original Shape : %s", hidden_states.shape)

    feature_vector = hidden_states.mean(dim=1)
    logger.info("Final Feature Shape : %s", feature_vector.shape)

    output_path = OUTPUT_FOLDER / "audio_feature.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), feature_vector.squeeze().numpy())
    logger.info("Feature Vector Saved Successfully")


if __name__ == "__main__":
    main()

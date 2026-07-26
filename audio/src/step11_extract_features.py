import logging
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from .config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, MODEL_NAME, configure_logging
except ImportError:
    from config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, MODEL_NAME, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    audio_path = AUDIO_FOLDER / "WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"
    waveform, sample_rate = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)

    logger.info("Audio Loaded")
    logger.info("Sample Rate: %s", sample_rate)
    logger.info("Waveform Shape: %s", waveform.shape)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    logger.info("Model Loaded")

    inputs = processor(waveform, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt", padding=True)
    logger.info("Processor Finished")

    with torch.no_grad():
        outputs = model(**inputs)

    logger.info("Feature Extraction Completed")
    hidden_states = outputs.last_hidden_state
    logger.info("Hidden State Shape: %s", hidden_states.shape)


if __name__ == "__main__":
    main()

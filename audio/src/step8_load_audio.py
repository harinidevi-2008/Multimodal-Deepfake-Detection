import logging
from pathlib import Path

import librosa

try:
    from .config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, configure_logging
except ImportError:
    from config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    audio_path = AUDIO_FOLDER / "WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"
    waveform, sample_rate = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)

    logger.info("Sample Rate: %s", sample_rate)
    logger.info("Waveform Shape: %s", waveform.shape)
    logger.info("First 20 Values:")
    logger.info("%s", waveform[:20])


if __name__ == "__main__":
    main()

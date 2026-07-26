import logging
from pathlib import Path

import librosa
import matplotlib.pyplot as plt

try:
    from .config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, configure_logging
except ImportError:
    from config import AUDIO_FOLDER, DEFAULT_SAMPLE_RATE, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    audio_path = AUDIO_FOLDER / "WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"
    waveform, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)
    logger.info("%s", mel.shape)


if __name__ == "__main__":
    main()

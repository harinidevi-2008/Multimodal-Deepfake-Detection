import hashlib
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

# librosa / moviepy / torch / transformers are intentionally NOT imported
# at module level. This module used to import all of them here plus
# eagerly load the Wav2Vec2 processor/model below (a real network
# download on first run) just by being imported - so anything that
# merely wanted `_unique_temp_audio_path()` (pure Path logic, no ML
# dependencies at all) paid for the entire audio ML stack, and any
# environment without librosa/moviepy/transformers installed (or without
# network access, for the Wav2Vec2 download) failed on import rather
# than only when audio actually needs to be processed. They are imported
# lazily inside the functions that need them instead - see
# _get_wav2vec2() and process_video() below.

try:
    from .config import (
        INPUT_FOLDER,
        AUDIO_FOLDER,
        OUTPUT_FOLDER,
        MODEL_NAME,
        DEFAULT_SAMPLE_RATE,
        SUPPORTED_EXTENSIONS,
        SKIP_EXISTING,
        DELETE_TEMP_AUDIO_AFTER_EXTRACTION,
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
        DELETE_TEMP_AUDIO_AFTER_EXTRACTION,
        configure_logging,
    )

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Load Wav2Vec2 lazily, on first actual use
# ------------------------------------------------------------

_wav2vec2_processor = None
_wav2vec2_model = None


def _get_wav2vec2():
    """
    Builds (and caches) the Wav2Vec2 processor/model on first use instead
    of at import time. `.from_pretrained()` downloads model weights over
    the network the first time it runs for a given cache - loading it
    eagerly at module level meant simply importing this module downloaded
    those weights, which broke pytest collection on an offline/restricted
    checkout and is wrong even with network access.
    """
    global _wav2vec2_processor, _wav2vec2_model
    if _wav2vec2_model is None:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        logger.info("Loading Wav2Vec2 Processor...")
        _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

        logger.info("Loading Wav2Vec2 Model...")
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        model.eval()

        # Freeze Wav2Vec2 parameters
        for param in model.parameters():
            param.requires_grad = False

        _wav2vec2_model = model
        logger.info("Model loaded successfully.")
    return _wav2vec2_processor, _wav2vec2_model


# ------------------------------------------------------------
# Process one video
# ------------------------------------------------------------

def _unique_temp_audio_path(video_path: Path) -> Path:
    """
    Collision-safe path under AUDIO_FOLDER for one video's temporary
    extracted WAV file.

    The previous naming (`AUDIO_FOLDER / (video_path.stem + ".wav")`)
    was flat, so two videos with the same filename stem in different
    dataset categories (e.g. category_A/foo/001.mp4 and
    category_B/bar/001.mp4) both mapped to "temp_audio/001.wav" and
    would silently overwrite each other's temp audio mid-run.

    Preferred fix: mirror the video's path relative to INPUT_FOLDER
    under AUDIO_FOLDER, exactly like OUTPUT_FOLDER already mirrors it
    for the saved .npy features in process_directory() below. Two
    distinct videos under INPUT_FOLDER always have distinct relative
    paths, so this is collision-safe by construction, deterministic
    (reruns land on the same temp path, preserving SKIP_EXISTING-style
    reuse), and human-readable.

    Fallback: process_video() is also called directly with a single,
    arbitrary video path that is not necessarily under INPUT_FOLDER
    (see audio/src/extractor.py, used for one-off/application-style
    extraction outside the batch dataset walk). relative_to() raises
    ValueError in that case; fall back to a flat name built from the
    stem plus a short hash of the resolved absolute path, which stays
    deterministic and collision-safe without assuming any directory
    structure.
    """
    try:
        relative_path = video_path.relative_to(INPUT_FOLDER)
        return (AUDIO_FOLDER / relative_path).with_suffix(".wav")
    except ValueError:
        digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return AUDIO_FOLDER / f"{video_path.stem}_{digest}.wav"


def process_video(video_path: Path, output_path: Path):
    # Heavy ML/media dependencies are imported here, not at module level -
    # see the module-level comment above _get_wav2vec2(). This keeps
    # `_unique_temp_audio_path()` (and anything else that doesn't actually
    # extract audio) importable without librosa/moviepy/torch/transformers
    # installed at all.
    import torch
    import librosa
    from moviepy.editor import VideoFileClip

    video_path = Path(video_path)
    output_path = Path(output_path)

    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_path = _unique_temp_audio_path(video_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Extract Audio
    # -----------------------

    # SKIP_EXISTING's own effect is on OUTPUT_FOLDER (see
    # process_directory) - this separate existence check is an
    # independent optimization that lets a run resumed after a crash
    # reuse audio already extracted for a video whose feature (.npy)
    # was not yet saved, instead of re-decoding the video. It keeps
    # working unchanged: the path it checks is now unique per video
    # instead of colliding across categories.
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

    processor, model = _get_wav2vec2()

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

    # -----------------------
    # Clean up temp audio
    # -----------------------

    # Only remove it after the feature has been saved successfully -
    # if anything above raised, the WAV stays on disk so a resumed run
    # can reuse it instead of re-decoding the video from scratch.
    if DELETE_TEMP_AUDIO_AFTER_EXTRACTION:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete temp audio file %s", audio_path, exc_info=True)


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
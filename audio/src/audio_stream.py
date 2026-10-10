import hashlib
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Heavy ML/media dependencies are imported lazily inside the functions
# that need them. Importing this module alone should not load the model.

try:
    from .config import (
        INPUT_FOLDER,
        DATASET_NAME,
        FFPP_SOURCES,
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
        DATASET_NAME,
        FFPP_SOURCES,
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
    Load and cache the Wav2Vec2 processor and model on first use.

    The checkpoint uses legacy weight-normalization keys for the
    positional convolution. These are converted to the parameter names
    expected by the installed Transformers/PyTorch model implementation.
    """
    global _wav2vec2_processor, _wav2vec2_model

    if _wav2vec2_model is None:
        from safetensors.torch import load_file
        from transformers import (
            Wav2Vec2Config,
            Wav2Vec2Model,
            Wav2Vec2Processor,
        )

        logger.info("Loading Wav2Vec2 Processor...")
        _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        logger.info("Loading Wav2Vec2 Model...")

        # Locate the checkpoint already cached on this computer.
        cache_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--facebook--wav2vec2-base-960h"
            / "snapshots"
        )

        checkpoint_files = sorted(cache_dir.glob("*/model.safetensors"))

        if not checkpoint_files:
            raise FileNotFoundError(
                "Could not find the cached Wav2Vec2 model.safetensors "
                f"checkpoint under: {cache_dir}"
            )

        checkpoint = checkpoint_files[0]
        logger.info("Using Wav2Vec2 checkpoint: %s", checkpoint)

        # Read checkpoint tensors on the CPU.
        raw_state_dict = load_file(
            str(checkpoint),
            device="cpu",
        )

        # Retain the Wav2Vec2 backbone, excluding the CTC prediction head.
        state_dict = {
            key.removeprefix("wav2vec2."): value
            for key, value in raw_state_dict.items()
            if key.startswith("wav2vec2.")
        }

        # Convert legacy weight-normalization names to the names expected
        # by the current model implementation.
        converted_state_dict = {}

        for key, value in state_dict.items():
            new_key = key.replace(
                "encoder.pos_conv_embed.conv.weight_g",
                "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
            ).replace(
                "encoder.pos_conv_embed.conv.weight_v",
                "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
            )

            converted_state_dict[new_key] = value

        # Construct the model from the locally cached configuration.
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
        model = Wav2Vec2Model(config)

        # Load the checkpoint and validate all expected model weights.
        load_result = model.load_state_dict(
            converted_state_dict,
            strict=False,
        )

        # The checkpoint test established that masked_spec_embed is the
        # only missing key. It is used for masked pretraining, not normal
        # inference. Any other missing or unexpected keys must stop loading.
        if load_result.missing_keys != ["masked_spec_embed"]:
            raise RuntimeError(
                "Unexpected missing Wav2Vec2 weights: "
                f"{load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected Wav2Vec2 weights: "
                f"{load_result.unexpected_keys}"
            )

        # Freeze the pretrained model for feature extraction.
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        _wav2vec2_model = model

        logger.info(
            "Wav2Vec2 model loaded successfully with converted checkpoint weights."
        )

    return _wav2vec2_processor, _wav2vec2_model


# ------------------------------------------------------------
# Temporary audio path
# ------------------------------------------------------------

def _unique_temp_audio_path(video_path: Path) -> Path:
    """
    Generate a deterministic, collision-safe temporary WAV path.

    When the video is inside INPUT_FOLDER, mirror its relative path
    under AUDIO_FOLDER. For arbitrary external video paths, use a
    short hash of the absolute path.
    """
    video_path = Path(video_path)

    try:
        relative_path = video_path.relative_to(INPUT_FOLDER)
        return (AUDIO_FOLDER / relative_path).with_suffix(".wav")

    except ValueError:
        digest = hashlib.sha1(
            str(video_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]

        return AUDIO_FOLDER / f"{video_path.stem}_{digest}.wav"


# ------------------------------------------------------------
# Process one video
# ------------------------------------------------------------

def process_video(video_path: Path, output_path: Path):
    """
    Extract audio from one video and save its Wav2Vec2 feature vector.
    """
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
    # Extract audio
    # -----------------------

    if not audio_path.exists():
        clip = VideoFileClip(str(video_path))

        try:
            if clip.audio is None:
                raise ValueError(
                    f"Video contains no audio stream: {video_path}"
                )

            clip.audio.write_audiofile(
                str(audio_path),
                fps=DEFAULT_SAMPLE_RATE,
                verbose=False,
                logger=None,
            )
        finally:
            clip.close()

    # -----------------------
    # Load audio
    # -----------------------

    waveform, sr = librosa.load(
        str(audio_path),
        sr=DEFAULT_SAMPLE_RATE,
    )

    if waveform.size == 0:
        raise ValueError(f"Extracted audio is empty: {video_path}")

    # -----------------------
    # Mel spectrogram
    # -----------------------
    # Preserved from the original pipeline. This result is currently
    # not used as the saved feature representation.

    _ = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=80,
    )

    # -----------------------
    # Wav2Vec2 feature extraction
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
    # Mean pooling
    # -----------------------

    feature_vector = hidden_states.mean(dim=1)

    np.save(
        str(output_path),
        feature_vector.squeeze().cpu().numpy(),
    )

    # -----------------------
    # Clean up temporary audio
    # -----------------------

    if DELETE_TEMP_AUDIO_AFTER_EXTRACTION:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not delete temporary audio file %s",
                audio_path,
                exc_info=True,
            )


# ------------------------------------------------------------
# Process complete dataset
# ------------------------------------------------------------

def process_directory():
    """Extract audio features for every supported video in the dataset."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    video_files = []

    scan_roots = (
        [
            INPUT_FOLDER / source / "c23" / "videos"
            for source, _label in FFPP_SOURCES
        ]
        if DATASET_NAME == "ffpp"
        else [INPUT_FOLDER]
    )

    for scan_root in scan_roots:
        for ext in SUPPORTED_EXTENSIONS:
            video_files.extend(scan_root.rglob(f"*{ext}"))

    video_files = sorted(video_files)

    logger.info("Found %d videos", len(video_files))

    for video_path in tqdm(video_files):
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
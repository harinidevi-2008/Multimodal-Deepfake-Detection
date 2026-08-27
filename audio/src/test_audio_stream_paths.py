"""
Pytest-based regression test for audio_stream.py's
_unique_temp_audio_path().

Synthetic Path objects only - no real video is read or written, and no
audio/model dependency (librosa, moviepy, torch, transformers) is
exercised. _unique_temp_audio_path() is pure Path arithmetic (no
filesystem I/O of its own), and audio_stream.py's heavy ML/media
imports are lazy (see the module-level comment above _get_wav2vec2()
in audio_stream.py) - so importing this test file, and running it,
never triggers a Wav2Vec2 download or requires librosa/moviepy to be
installed.

Covers the exact collision this function was introduced to fix (two
videos with the same filename stem under different dataset categories
previously mapped to the same flat temp_audio/<stem>.wav path), plus
determinism and the outside-INPUT_FOLDER fallback path.

Run directly or via `python -m pytest -q`:
    python audio/src/test_audio_stream_paths.py
"""

from pathlib import Path

try:
    from .audio_stream import _unique_temp_audio_path
    from .config import AUDIO_FOLDER, INPUT_FOLDER
except ImportError:
    from audio_stream import _unique_temp_audio_path
    from config import AUDIO_FOLDER, INPUT_FOLDER


def test_unique_temp_audio_path_avoids_cross_category_collision():
    """The bug this function fixes: two videos that share a filename
    stem but live under different dataset categories must not resolve
    to the same temp WAV path."""
    video_a = INPUT_FOLDER / "category_A" / "foo" / "001.mp4"
    video_b = INPUT_FOLDER / "category_B" / "bar" / "001.mp4"

    path_a = _unique_temp_audio_path(video_a)
    path_b = _unique_temp_audio_path(video_b)

    assert path_a != path_b
    # Both stay under AUDIO_FOLDER and mirror the video's relative
    # structure (this is the "preferred fix" branch, not the fallback).
    assert path_a == AUDIO_FOLDER / "category_A" / "foo" / "001.wav"
    assert path_b == AUDIO_FOLDER / "category_B" / "bar" / "001.wav"


def test_unique_temp_audio_path_is_deterministic_for_the_same_video():
    """Two calls for the SAME video (e.g. a resumed/re-run extraction)
    must land on the exact same path, so the existing
    'reuse an already-extracted WAV' optimization keeps working."""
    video = INPUT_FOLDER / "category_A" / "foo" / "001.mp4"

    first_call = _unique_temp_audio_path(video)
    second_call = _unique_temp_audio_path(video)

    assert first_call == second_call


def test_unique_temp_audio_path_fallback_outside_input_folder_is_deterministic():
    """process_video() can be called directly with a video that is not
    under INPUT_FOLDER at all (see audio/src/extractor.py) - the hashed
    fallback path must still be deterministic across repeated calls."""
    outside_video = Path("/some/external/place/clip.mp4")

    first_call = _unique_temp_audio_path(outside_video)
    second_call = _unique_temp_audio_path(outside_video)

    assert first_call == second_call
    assert first_call.parent == AUDIO_FOLDER
    assert first_call.suffix == ".wav"
    # Falls back to <stem>_<hash>.wav, not a bare "clip.wav" - otherwise
    # this fallback would just reintroduce the same flat-name collision
    # risk it exists to avoid.
    assert first_call.stem != "clip"
    assert first_call.stem.startswith("clip_")


def test_unique_temp_audio_path_fallback_is_collision_resistant():
    """Two different videos outside INPUT_FOLDER that happen to share a
    filename stem must still resolve to different fallback paths - the
    fallback hashes the full resolved path, not just the stem."""
    outside_video_1 = Path("/some/external/place/clip.mp4")
    outside_video_2 = Path("/a/totally/different/location/clip.mp4")

    path_1 = _unique_temp_audio_path(outside_video_1)
    path_2 = _unique_temp_audio_path(outside_video_2)

    assert path_1 != path_2


if __name__ == "__main__":
    test_unique_temp_audio_path_avoids_cross_category_collision()
    test_unique_temp_audio_path_is_deterministic_for_the_same_video()
    test_unique_temp_audio_path_fallback_outside_input_folder_is_deterministic()
    test_unique_temp_audio_path_fallback_is_collision_resistant()
    print("All _unique_temp_audio_path() regression checks passed.")

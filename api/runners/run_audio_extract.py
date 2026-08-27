"""
Subprocess entrypoint for the audio extraction step, run in its own
clean Python process with ONLY audio/src on sys.path.

Why this exists (this is a real, pre-existing constraint in the repo,
not something introduced by the API layer): visual/src/config.py and
audio/src/config.py are both importable under the same flat module
name "config" (each file's own fallback import path, used when it
isn't loaded as part of a package - see the `except ImportError:`
blocks in visual/src/process_video.py and audio/src/audio_stream.py).
If both visual/src and audio/src are ever added to sys.path in the
same process, whichever config.py's import wins the race silently
shadows the other, and the loser fails with e.g. "cannot import name
'VIDEO_FOLDER' from 'config'" or "cannot import name 'INPUT_FOLDER'
from 'config'" - reproduced directly while building this API layer.
The API server's main process needs visual/src on sys.path (for
process_video / blink_lipsync_precompute / the eye_blink and lipsync
analyzers), so audio extraction - the only stream that would collide -
is isolated here instead, in a subprocess whose sys.path never
contains visual/src at all.

This does not reimplement audio_stream.process_video() - it just calls
it, exactly as audio/src/extractor.py's extract() does for a single
video, with an explicit output path instead of the dataset-relative
one extractor.py derives.

Usage:
    python run_audio_extract.py <video_path> <output_npy_path>

Exit code 0 + "OK" on stdout: <output_npy_path> was written.
Exit code 1 + a one-line reason on stderr: extraction failed (the
    parent process is responsible for translating this into an
    AnalysisAPIError - see api/pipeline_adapter.py - never for
    silently treating a failed extraction as "no fake signal here").
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_SRC = REPO_ROOT / "audio" / "src"


def main():
    if len(sys.argv) != 3:
        print("Usage: run_audio_extract.py <video_path> <output_npy_path>", file=sys.stderr)
        sys.exit(2)

    video_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # ONLY audio/src goes on sys.path in this process - see module
    # docstring. Intentionally not adding REPO_ROOT or visual/src.
    sys.path.insert(0, str(AUDIO_SRC))

    try:
        from audio_stream import process_video  # noqa: E402 (path set up above)
    except Exception as exc:  # noqa: BLE001 - report, let parent classify
        print(f"IMPORT_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        process_video(video_path, output_path)
    except Exception as exc:  # noqa: BLE001 - report, let parent classify
        print(f"EXTRACTION_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    if not output_path.exists():
        print("EXTRACTION_FAILED: process_video() returned without writing an output file "
              "(this happens if e.g. audio decoding silently produced no samples).", file=sys.stderr)
        sys.exit(1)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

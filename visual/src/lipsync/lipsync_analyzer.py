"""
Lip-sync consistency analysis module.

Pipeline: mouth movement (from video, via face-mesh landmarks) + audio
speech-energy envelope -> time-aligned correlation -> a single
lip-sync consistency score.

This is a lightweight, correlation-based module (no trainable weights).
It is independent of the existing EfficientNet visual pipeline and the
Wav2Vec2 audio pipeline - it does not import or modify either. If an
already-extracted audio file (e.g. from the audio stream's temp_audio/
folder) is available, pass it in directly via audio_path; otherwise
this module extracts a mono 16kHz wav from the video itself using
ffmpeg, independently.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# MediaPipe Face Mesh mouth landmark indices used for a simple
# vertical-opening / horizontal-width Mouth Aspect Ratio (MAR).
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER_INNER = 13
MOUTH_LOWER_INNER = 14

AUDIO_SAMPLE_RATE = 16000
# Below this, the two signals are considered not meaningfully correlated.
CONSISTENCY_THRESHOLD = 0.40
# Search a small window of frame offsets to tolerate imperfect start
# alignment between the two signals (in frames, at the video's sampling fps).
MAX_LAG_FRAMES = 5


def _euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _mouth_aspect_ratio(landmarks, frame_w, frame_h):
    left = landmarks[MOUTH_LEFT_CORNER]
    right = landmarks[MOUTH_RIGHT_CORNER]
    upper = landmarks[MOUTH_UPPER_INNER]
    lower = landmarks[MOUTH_LOWER_INNER]

    left_pt = (left.x * frame_w, left.y * frame_h)
    right_pt = (right.x * frame_w, right.y * frame_h)
    upper_pt = (upper.x * frame_w, upper.y * frame_h)
    lower_pt = (lower.x * frame_w, lower.y * frame_h)

    horizontal = _euclidean(left_pt, right_pt)
    if horizontal == 0:
        return None

    vertical = _euclidean(upper_pt, lower_pt)
    return vertical / horizontal


def _read_frames(video_path):
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required to read video frames.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def compute_mouth_series(video_path):
    """
    Run face-mesh landmark detection across a video and return a
    per-frame mouth-opening (MAR) time series.

    Returns:
        dict with mar_values (list[float or None]), fps (float),
        frame_count (int).
    """

    if mp is None:
        raise ModuleNotFoundError("mediapipe is required for mouth landmark tracking.")

    frames, fps = _read_frames(video_path)
    mar_values = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for frame_bgr in frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_bgr.shape[:2]
            result = face_mesh.process(frame_rgb)

            if not result.multi_face_landmarks:
                mar_values.append(None)
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            mar_values.append(_mouth_aspect_ratio(landmarks, w, h))

    return {"mar_values": mar_values, "fps": fps, "frame_count": len(frames)}


def _extract_audio_ffmpeg(video_path, out_wav_path):
    """Extract mono 16kHz audio from a video using ffmpeg (self-contained,
    does not depend on or modify the audio stream's extraction code)."""

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-vn", str(out_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")


def compute_audio_energy_series(audio_path, target_frame_count, target_fps):
    """
    Load a wav file and return an RMS energy envelope resampled to
    target_frame_count points, aligned to target_fps (the video's
    sampling rate) so it lines up 1:1 with the mouth series.
    """

    if librosa is None:
        raise ModuleNotFoundError("librosa is required for audio energy extraction.")

    waveform, sr = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE)

    # Hop length chosen so RMS frames land roughly at the video's fps,
    # then linearly resampled to exactly match frame count.
    hop_length = max(int(sr / target_fps), 1)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]

    if len(rms) == 0:
        return [0.0] * target_frame_count

    x_original = np.linspace(0, 1, num=len(rms))
    x_target = np.linspace(0, 1, num=target_frame_count)
    resampled = np.interp(x_target, x_original, rms)
    return resampled.tolist()


def _best_lag_correlation(mouth_series, audio_series, max_lag):
    """
    Compute Pearson correlation at zero lag and at small shifts, and
    return the best (highest) absolute correlation found. This tolerates
    a small constant timing offset between the two signals without
    treating it as a genuine sync inconsistency.
    """

    mouth = np.array(mouth_series, dtype=float)
    audio = np.array(audio_series, dtype=float)
    n = len(mouth)

    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            m = mouth[lag:]
            a = audio[: n - lag] if lag > 0 else audio
        else:
            m = mouth[: n + lag]
            a = audio[-lag:]

        if len(m) < 5 or np.std(m) == 0 or np.std(a) == 0:
            continue

        corr = float(np.corrcoef(m, a)[0, 1])
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr

    return best_corr


def analyze_lipsync(video_path, audio_path=None):
    """
    End-to-end lip-sync consistency analysis for a single video.

    Parameters:
        video_path (str or Path): path to the video file.
        audio_path (str or Path or None): path to an already-extracted
            wav file (e.g. from the audio stream's temp_audio/ folder).
            If None, audio is extracted from the video with ffmpeg into
            a temporary file.

    Returns:
        dict with:
            lipsync_consistency (float, 0-1, higher = more consistent)
            lipsync_status (str): "Consistent" or "Inconsistent"
            valid_frame_ratio (float): fraction of frames where a mouth
                was actually detected (low values mean the score is
                less reliable, e.g. face turned away or occluded)
    """

    mouth_data = compute_mouth_series(video_path)
    mar_values = mouth_data["mar_values"]
    fps = mouth_data["fps"]
    frame_count = mouth_data["frame_count"]

    valid_indices = [i for i, v in enumerate(mar_values) if v is not None]
    valid_frame_ratio = len(valid_indices) / frame_count if frame_count else 0.0

    if len(valid_indices) < 10:
        logger.warning("Too few frames with a detected mouth (%d) to score lip-sync.", len(valid_indices))
        return {
            "lipsync_consistency": 0.0,
            "sync_score": 0.0,
            "mismatch_score": 1.0,
            "lipsync_status": "Inconsistent",
            "sync_status": "Inconsistent",
            "valid_frame_ratio": round(valid_frame_ratio, 3),
        }

    cleanup_wav = None
    try:
        if audio_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            audio_path = tmp.name
            cleanup_wav = audio_path
            _extract_audio_ffmpeg(video_path, audio_path)

        audio_series = compute_audio_energy_series(audio_path, frame_count, fps)

        mouth_series = [mar_values[i] if mar_values[i] is not None else 0.0 for i in range(frame_count)]

        correlation = _best_lag_correlation(mouth_series, audio_series, MAX_LAG_FRAMES)
        consistency = round((correlation + 1.0) / 2.0, 4)  # map [-1,1] -> [0,1]
        mismatch = round(1.0 - consistency, 4)

        status = "Consistent" if consistency >= CONSISTENCY_THRESHOLD else "Inconsistent"

        return {
            "lipsync_consistency": consistency,
            "sync_score": consistency,
            "mismatch_score": mismatch,
            "lipsync_status": status,
            "sync_status": status,
            "valid_frame_ratio": round(valid_frame_ratio, 3),
        }
    finally:
        if cleanup_wav is not None:
            Path(cleanup_wav).unlink(missing_ok=True)

"""
Eye-blink behavioural analysis module.

Pipeline: video frames -> face landmarks -> eye aspect ratio (EAR) ->
blink detection -> blink count / rate / irregularity score.

This is a lightweight, rule-based module (no trainable weights). It is
independent of the existing EfficientNet visual feature extraction
pipeline (face_detector.py / feature_extractor.py) and does not import
or modify anything from it.

Note on sampling rate: a blink typically lasts ~100-400ms. The main
visual pipeline samples at DEFAULT_FPS_TARGET (2 fps in config.py),
which is far too sparse to see individual blinks - most would fall
between sampled frames. This module therefore reads the video at its
own, higher frame rate (default: every frame, or a fps cap) instead of
reusing extract_frames().
"""

import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# MediaPipe Face Mesh landmark indices for eye contour points, in the
# standard 6-point order used for the Eye Aspect Ratio formula:
# [left corner, top-1, top-2, right corner, bottom-2, bottom-1]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# EAR drops sharply during a blink; below this the eye is considered closed.
EAR_CLOSED_THRESHOLD = 0.21
# Minimum consecutive closed-eye frames to count as a real blink (filters
# out single-frame landmark noise / detection jitter).
MIN_BLINK_FRAMES = 2
# Typical human blink rate range (blinks per minute) used to score
# irregularity - values well outside this range are treated as abnormal.
NORMAL_BLINK_RATE_RANGE = (10, 30)


def _euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _eye_aspect_ratio(landmarks, eye_idx, frame_w, frame_h):
    """Compute EAR for one eye given 6 (x, y) landmark points."""

    pts = []
    for idx in eye_idx:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _read_frames(video_path, max_fps=None):
    """
    Read frames from a video, optionally capping the sampling rate.

    Parameters:
        video_path (str or Path): path to the video file.
        max_fps (float or None): if set, subsample down to this fps.
            If None, every frame is read (best for blink detection on
            already-short clips).

    Returns:
        (list of BGR frames, float frames_per_second_used)
    """

    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required to read video frames.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = 1
    used_fps = source_fps

    if max_fps is not None and max_fps > 0 and source_fps > max_fps:
        frame_interval = max(int(round(source_fps / max_fps)), 1)
        used_fps = source_fps / frame_interval

    frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frames.append(frame)
        frame_index += 1

    cap.release()
    return frames, used_fps


def compute_ear_series(video_path, max_fps=None):
    """
    Run face-mesh landmark detection across a video and return a
    per-frame EAR time series (averaged across both eyes).

    Parameters:
        video_path (str or Path): path to the video file.
        max_fps (float or None): cap on sampling rate (see _read_frames).

    Returns:
        dict with:
            ear_values (list[float or None]): EAR per sampled frame,
                None where no face/landmarks were found.
            fps (float): effective sampling rate used.
            frame_count (int): number of frames read.
    """

    if mp is None:
        raise ModuleNotFoundError("mediapipe is required for landmark-based blink analysis.")

    frames, fps = _read_frames(video_path, max_fps=max_fps)
    ear_values = []

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
                ear_values.append(None)
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)

            if left_ear is None or right_ear is None:
                ear_values.append(None)
            else:
                ear_values.append((left_ear + right_ear) / 2.0)

    return {"ear_values": ear_values, "fps": fps, "frame_count": len(frames)}


def detect_blinks(ear_values, ear_threshold=EAR_CLOSED_THRESHOLD, min_blink_frames=MIN_BLINK_FRAMES):
    """
    Convert a per-frame EAR series into discrete blink events.

    Returns:
        list[tuple[int, int]]: (start_frame_idx, end_frame_idx) for each
            detected blink (inclusive range of closed-eye frames).
    """

    blinks = []
    closed_start = None

    for i, ear in enumerate(ear_values):
        is_closed = ear is not None and ear < ear_threshold

        if is_closed and closed_start is None:
            closed_start = i
        elif not is_closed and closed_start is not None:
            duration = i - closed_start
            if duration >= min_blink_frames:
                blinks.append((closed_start, i - 1))
            closed_start = None

    if closed_start is not None:
        duration = len(ear_values) - closed_start
        if duration >= min_blink_frames:
            blinks.append((closed_start, len(ear_values) - 1))

    return blinks


def _blink_irregularity_score(blinks, fps, total_frames):
    """
    Combine blink rate deviation and inter-blink interval variability
    into a single 0-1 abnormality score. 0 = normal, 1 = highly abnormal.
    """

    if total_frames == 0 or fps == 0:
        return 1.0

    duration_minutes = (total_frames / fps) / 60.0
    if duration_minutes <= 0:
        return 1.0

    blink_rate = len(blinks) / duration_minutes
    low, high = NORMAL_BLINK_RATE_RANGE

    if low <= blink_rate <= high:
        rate_penalty = 0.0
    elif blink_rate < low:
        rate_penalty = min((low - blink_rate) / low, 1.0)
    else:
        rate_penalty = min((blink_rate - high) / high, 1.0)

    if len(blinks) >= 3:
        centers = [(s + e) / 2.0 for s, e in blinks]
        intervals = [(centers[i + 1] - centers[i]) / fps for i in range(len(centers) - 1)]
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval > 0:
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            std = variance ** 0.5
            coeff_of_variation = std / mean_interval
            # Very regular (low CoV) or very erratic (high CoV) blinking
            # both diverge from natural human blink timing.
            interval_penalty = min(abs(coeff_of_variation - 0.5) / 0.5, 1.0)
        else:
            interval_penalty = 1.0
    else:
        # Too few blinks to assess rhythm - rely on rate penalty only.
        interval_penalty = 0.0

    score = 0.6 * rate_penalty + 0.4 * interval_penalty
    return round(min(max(score, 0.0), 1.0), 4)


def analyze_blinks(video_path, max_fps=None):
    """
    End-to-end eye-blink behavioural analysis for a single video.

    Parameters:
        video_path (str or Path): path to the video file.
        max_fps (float or None): optional cap on sampling rate. Leave
            None to read every frame (recommended for short clips).

    Returns:
        dict with:
            blink_count (int)
            blink_rate_per_min (float)
            average_blink_duration_sec (float)
            blink_irregularity_score (float, 0-1) - rule-based anomaly score,
                not a learned probability (see evidence/evidence_builder.py's
                terminology notes)
            blink_status (str): "Normal" or "Abnormal"

    Note: an earlier version of this function also returned
    "blink_anomaly_score" as a duplicate alias of blink_irregularity_score.
    It has been removed - nothing in this repository reads it by that
    key (confirmed via eye_blink_manual.py), and evidence_builder.py now
    reads blink_irregularity_score directly.
    """

    ear_data = compute_ear_series(video_path, max_fps=max_fps)
    ear_values = ear_data["ear_values"]
    fps = ear_data["fps"]
    total_frames = ear_data["frame_count"]

    blinks = detect_blinks(ear_values)
    duration_minutes = (total_frames / fps) / 60.0 if fps else 0.0
    blink_rate = len(blinks) / duration_minutes if duration_minutes > 0 else 0.0
    irregularity_score = _blink_irregularity_score(blinks, fps, total_frames)

    if blinks and fps:
        avg_duration_sec = sum((e - s + 1) for s, e in blinks) / len(blinks) / fps
    else:
        avg_duration_sec = 0.0

    status = "Abnormal" if irregularity_score >= 0.5 else "Normal"

    return {
        "blink_count": len(blinks),
        "blink_rate_per_min": round(blink_rate, 2),
        "average_blink_duration_sec": round(avg_duration_sec, 4),
        "blink_irregularity_score": irregularity_score,
        "blink_status": status,
    }

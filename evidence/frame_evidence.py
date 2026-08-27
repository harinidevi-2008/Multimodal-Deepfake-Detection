"""
Frame-level evidence extraction.

Selects a small set of representative frames from a video to show
alongside the numeric evidence report - "here's a frame, and here's
why it was picked" rather than only a bare probability. Two selection
strategies, chosen automatically based on what real data is actually
available - this module never invents timestamps:

1. Blink-event frames: if eye_blink/blink_analyzer.py's detect_blinks()
   found real blink events for this video (genuine EAR-based timing
   data), one frame from the middle of each of the first few events is
   selected and tagged "eye_blink_event".
2. Uniform fallback: wherever blink-event frames don't fill the quota
   (no blink data was passed in, or fewer events than requested
   frames), frames are added at evenly spaced points across the clip
   and tagged "uniform_sample" - explicitly NOT claimed to correspond
   to any detected anomaly.

This module never claims a selected frame "shows" manipulation; it
only records why that frame was chosen. Pair its output with the
numeric blink/lip-sync/fusion evidence for the same video if you want
to say more.

Schema note: each selection's "frame_selection_reason" ("eye_blink_event"
or "uniform_sample") answers ONLY "why was this frame picked" - it is
deliberately kept separate from "detected_artifact", which would
answer "what does an actual visual-artifact detector see in this
frame". There is no such detector in this repository, so
"detected_artifact" is always None here - it exists as a schema slot
for a real detector to fill in later, not as an implicit claim. Do not
populate it with anything derived from frame_selection_reason,
blink/lip-sync scores, or the fusion model's probability - none of
those are a visual-artifact detection.
"""

import logging
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAMES = 6


def select_evidence_frame_indices(total_frames, blink_events=None, max_frames=DEFAULT_MAX_FRAMES):
    """
    Pure selection logic - no video I/O - kept separate from
    extract_evidence_frames() so it can be unit tested without a real
    video file or OpenCV.

    Parameters:
        total_frames (int): total frame count of the video.
        blink_events (list[tuple[int, int]] or None): (start, end)
            frame-index pairs, as returned by
            eye_blink/blink_analyzer.py's detect_blinks(). None or an
            empty list means no blink-event data is available.
        max_frames (int): maximum number of frames to select in total.

    Returns:
        list[dict]: each {"frame_index": int, "frame_selection_reason": str,
        "blink_event": [start, end] or None, "detected_artifact": None}
        (see module docstring for why detected_artifact is always None
        here), sorted by frame_index.
    """
    if total_frames <= 0:
        return []

    selections = []
    if blink_events:
        for start, end in blink_events[:max_frames]:
            mid = (start + end) // 2
            mid = max(0, min(mid, total_frames - 1))
            selections.append({
                "frame_index": mid,
                "frame_selection_reason": "eye_blink_event",
                "blink_event": [start, end],
                "detected_artifact": None,
            })

    remaining = max_frames - len(selections)
    if remaining > 0:
        used = {s["frame_index"] for s in selections}
        step = max(total_frames // (remaining + 1), 1)
        for i in range(1, remaining + 1):
            idx = min(i * step, total_frames - 1)
            if idx in used:
                continue
            selections.append({
                "frame_index": idx,
                "frame_selection_reason": "uniform_sample",
                "blink_event": None,
                "detected_artifact": None,
            })
            used.add(idx)

    selections.sort(key=lambda s: s["frame_index"])
    return selections[:max_frames]


def extract_evidence_frames(video_path, output_dir, blink_events=None, max_frames=DEFAULT_MAX_FRAMES):
    """
    Reads `video_path` with OpenCV, selects frames via
    select_evidence_frame_indices(), and saves each as a .jpg under
    output_dir.

    Returns the same list select_evidence_frame_indices() produced,
    with an added "saved_path" key per entry (None if that frame could
    not be read, e.g. an off-by-one at the very end of a variable-
    frame-rate container).
    """
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required to extract evidence frames.")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selections = select_evidence_frame_indices(total_frames, blink_events, max_frames)

    for sel in selections:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sel["frame_index"])
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d from %s", sel["frame_index"], video_path)
            sel["saved_path"] = None
            continue
        out_path = output_dir / f"frame_{sel['frame_index']:06d}_{sel['frame_selection_reason']}.jpg"
        cv2.imwrite(str(out_path), frame)
        sel["saved_path"] = str(out_path)

    cap.release()
    return selections

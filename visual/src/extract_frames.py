import logging
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from .config import DEFAULT_FPS_TARGET, VIDEO_FOLDER, configure_logging
except ImportError:
    from config import DEFAULT_FPS_TARGET, VIDEO_FOLDER, configure_logging

logger = logging.getLogger(__name__)


def extract_frames(video_path, fps_target=DEFAULT_FPS_TARGET):
    """
    Extract frames from a video at the desired FPS.

    Parameters:
        video_path (str): Path to input video.
        fps_target (int): Number of frames sampled per second.

    Returns:
        list: List of OpenCV frames (BGR).
    """

    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required to extract frames.")

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        video_fps = fps_target

    frame_interval = max(int(video_fps / fps_target), 1)

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
    return frames


def main():
    configure_logging()
    video_path = VIDEO_FOLDER / "real_harini_001.mp4"
    frames = extract_frames(video_path)
    logger.info("Frames extracted: %s", len(frames))


if __name__ == "__main__":
    main()
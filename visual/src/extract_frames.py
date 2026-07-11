import cv2


def extract_frames(video_path, fps_target=2):
    """
    Extract frames from a video at the desired FPS.

    Parameters:
        video_path (str): Path to input video.
        fps_target (int): Number of frames sampled per second.

    Returns:
        list: List of OpenCV frames (BGR).
    """

    cap = cv2.VideoCapture(video_path)

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


if __name__ == "__main__":

    video_path = "data/raw_videos/real_harini_001.mp4"

    frames = extract_frames(video_path)

    print(f"Frames extracted: {len(frames)}")
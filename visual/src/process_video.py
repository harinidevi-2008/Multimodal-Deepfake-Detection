import logging
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .config import OUTPUT_FOLDER, VIDEO_FOLDER, DEFAULT_FPS_TARGET, configure_logging
    from .extract_frames import extract_frames
    from .face_detector import get_aligned_face
    from .feature_extractor import get_feature_vector
except ImportError:
    from config import OUTPUT_FOLDER, VIDEO_FOLDER, DEFAULT_FPS_TARGET, configure_logging
    from extract_frames import extract_frames
    from face_detector import get_aligned_face
    from feature_extractor import get_feature_vector

logger = logging.getLogger(__name__)


def process_video(video_path, output_path):
    start_time = time.time()

    frames = extract_frames(video_path, fps_target=DEFAULT_FPS_TARGET)

    total_frames = len(frames)
    skipped_frames = 0

    features = []

    for frame in frames:
        face_tensor = get_aligned_face(frame)

        if face_tensor is None:
            skipped_frames += 1
            continue

        feature_vector = get_feature_vector(face_tensor)
        features.append(feature_vector)

    if len(features) == 0:
        logger.info("No faces detected in %s", os.path.basename(video_path))
        return None

    pooled_feature = np.mean(features, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), pooled_feature)

    end_time = time.time()
    processing_time = end_time - start_time
    detected_faces = len(features)
    skip_rate = (skipped_frames / total_frames) * 100 if total_frames > 0 else 0

    logger.info("------------------------------------------")
    logger.info("Video : %s", os.path.basename(video_path))
    logger.info("Frames Sampled : %s", total_frames)
    logger.info("Faces Detected : %s", detected_faces)
    logger.info("Frames Skipped : %s", skipped_frames)
    logger.info("Skip Rate : %.2f%%", skip_rate)
    logger.info("Output Shape : %s", pooled_feature.shape)
    logger.info("Processing Time : %.2f sec", processing_time)
    logger.info("Saved : %s", output_path)
    logger.info("------------------------------------------")

    return pooled_feature


def process_directory(video_folder=VIDEO_FOLDER, output_folder=OUTPUT_FOLDER):
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_folder.rglob("*.mp4"))

    logger.info("Found %d videos", len(video_files))

    for video_path in tqdm(video_files):
        relative_path = video_path.relative_to(video_folder)
        output_path = output_folder / relative_path.with_suffix(".npy")

        process_video(str(video_path), str(output_path))

    return output_folder


def main():
    configure_logging()
    process_directory()


if __name__ == "__main__":
    main()
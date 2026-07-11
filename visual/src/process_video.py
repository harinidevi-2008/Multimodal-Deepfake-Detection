import os
import numpy as np
from tqdm import tqdm

from extract_frames import extract_frames
from face_detector import get_aligned_face
from feature_extractor import get_feature_vector


def process_video(video_path, output_folder, fps_target=2):
    """
    Process one video and save its visual features.

    Parameters
    ----------
    video_path : str
    output_folder : str
    fps_target : int

    Returns
    -------
    numpy.ndarray
    """

    print(f"\nProcessing {os.path.basename(video_path)}")

    frames = extract_frames(video_path, fps_target)

    features = []

    skipped = 0

    for frame in tqdm(frames, desc="Processing Frames"):

        face = get_aligned_face(frame)

        if face is None:
            skipped += 1
            continue

        feature = get_feature_vector(face)

        features.append(feature)

    features = np.array(features)

    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    save_path = os.path.join(output_folder, video_name + ".npy")

    np.save(save_path, features)

    print("\nProcessing Complete")
    print("-------------------------")
    print("Frames Sampled :", len(frames))
    print("Faces Detected :", len(features))
    print("Frames Skipped :", skipped)
    print("Feature Shape  :", features.shape)
    print("Saved File     :", save_path)

    return features


if __name__ == "__main__":

    process_video(
        video_path="data/raw_videos/real_harini_001.mp4",
        output_folder="visual/data/visual_features",
        fps_target=2
    )
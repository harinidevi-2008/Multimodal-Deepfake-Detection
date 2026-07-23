import os
import time
import numpy as np
from tqdm import tqdm

from extract_frames import extract_frames
from face_detector import get_aligned_face
from feature_extractor import get_feature_vector

# -----------------------------
# Folder Paths
# -----------------------------
VIDEO_FOLDER = "data/raw_videos"
OUTPUT_FOLDER = "visual/data/features"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Process One Video
# -----------------------------
def process_video(video_path, output_path):

    start_time = time.time()

    frames = extract_frames(video_path, fps_target=2)

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

    # ------------------------------------
    # No faces detected
    # ------------------------------------
    if len(features) == 0:

        print(f"\n❌ No faces detected in {os.path.basename(video_path)}")

        return

    # ------------------------------------
    # Mean Pooling
    # ------------------------------------
    pooled_feature = np.mean(features, axis=0)

    # shape -> (1280,)
    np.save(output_path, pooled_feature)

    end_time = time.time()

    processing_time = end_time - start_time

    detected_faces = len(features)

    skip_rate = (skipped_frames / total_frames) * 100 if total_frames > 0 else 0

    print("\n------------------------------------------")
    print("Video :", os.path.basename(video_path))
    print("Frames Sampled :", total_frames)
    print("Faces Detected :", detected_faces)
    print("Frames Skipped :", skipped_frames)
    print(f"Skip Rate : {skip_rate:.2f}%")
    print("Output Shape :", pooled_feature.shape)
    print(f"Processing Time : {processing_time:.2f} sec")
    print("Saved :", output_path)
    print("------------------------------------------")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    for filename in tqdm(os.listdir(VIDEO_FOLDER)):

        if filename.lower().endswith((".mp4", ".avi", ".mov")):

            video_path = os.path.join(VIDEO_FOLDER, filename)

            output_file = os.path.splitext(filename)[0] + ".npy"

            output_path = os.path.join(OUTPUT_FOLDER, output_file)

            process_video(video_path, output_path)
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from visual.src.process_video import process_video
from visual.src.config import configure_logging

configure_logging()

FFPP_ROOT = PROJECT_ROOT / "Datasets" / "FaceForensics++" / "data"

VIDEO_ROOTS = [
    FFPP_ROOT / "original_sequences" / "youtube" / "c23" / "videos",
    FFPP_ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos",
]

OUTPUT_ROOT = PROJECT_ROOT / "visual" / "data" / "features_ffpp"

videos = []

for root in VIDEO_ROOTS:
    if root.exists():
        videos.extend(sorted(root.glob("*.mp4")))

print(f"Found {len(videos)} FF++ videos")

if len(videos) == 0:
    raise RuntimeError("No FF++ videos found.")

success = 0

for video_path in videos:
    relative_name = video_path.stem + ".npy"
    output_path = OUTPUT_ROOT / relative_name

    print(f"\nProcessing: {video_path.name}")

    result = process_video(
        str(video_path),
        str(output_path)
    )

    if result is not None and result.shape == (1280,):
        success += 1
        print(f"SUCCESS: {result.shape} -> {output_path}")
    else:
        print(f"FAILED: {video_path.name}")

print("\n===================================")
print(f"Successful: {success}/{len(videos)}")
print(f"Output directory: {OUTPUT_ROOT}")
print("===================================")

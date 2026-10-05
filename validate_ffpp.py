"""Read-only inventory of core FF++ C23 videos and feature output folders."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get(
    "DATASET_ROOT", PROJECT_ROOT / "Datasets" / "FaceForensics++" / "data"
))
SOURCES = (
    ("original_sequences/youtube", 0),
    ("original_sequences/actors", 0),
    ("manipulated_sequences/Deepfakes", 1),
    ("manipulated_sequences/Face2Face", 1),
    ("manipulated_sequences/FaceSwap", 1),
    ("manipulated_sequences/NeuralTextures", 1),
)


def main():
    counts = {}
    print(f"FF++ root: {DATASET_ROOT}")
    for source, label in SOURCES:
        folder = DATASET_ROOT / source / "c23" / "videos"
        count = sum(1 for _ in folder.rglob("*.mp4")) if folder.is_dir() else 0
        counts[source] = count
        print(f"{source:<42} {count:>8} MP4 (label {label})")
    real = sum(count for source, label in SOURCES if label == 0 for count in [counts[source]])
    fake = sum(count for source, label in SOURCES if label == 1 for count in [counts[source]])
    print(f"Total videos: {real + fake}")
    print(f"Expected real/fake counts: {real} / {fake}")
    print(f"Visual output:   {PROJECT_ROOT / 'visual' / 'data' / 'features_aligned'}")
    print(f"Audio output:    {PROJECT_ROOT / 'audio' / 'data' / 'features'}")
    print(f"Semantic output: {PROJECT_ROOT / 'semantic' / 'data' / 'features' / 'ffpp'}")


if __name__ == "__main__":
    main()

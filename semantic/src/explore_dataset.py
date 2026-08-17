import os
from pathlib import Path

# ── CHANGE THIS to wherever you extracted FakeAVCeleb_v1 ──────
DATASET_PATH = r"C:\Users\ASUS\semantic_stream\FakeAVCeleb_v1.2"

dataset = Path(DATASET_PATH)

CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
    "RealVideo-FakeAudio": 1,
}

print("=" * 60)
print("FAKEAVCELEB DATASET EXPLORER")
print("=" * 60)
print()

total_videos = 0
total_real   = 0
total_fake   = 0

for category, label in CATEGORIES.items():
    cat_path = dataset / category
    if not cat_path.exists():
        print(f"NOT FOUND: {category}")
        continue

    videos = list(cat_path.rglob("*.mp4"))
    count  = len(videos)
    total_videos += count

    if label == 0:
        total_real += count
    else:
        total_fake += count

    label_str = "REAL (0)" if label == 0 else "FAKE (1)"
    print(f"{category:<30} {count:>6} videos   label={label_str}")

    for eth in sorted(cat_path.iterdir()):
        if eth.is_dir():
            eth_vids = list(eth.rglob("*.mp4"))
            print(f"    {eth.name:<28} {len(eth_vids):>5} videos")

    print()

print("=" * 60)
print(f"Total videos    : {total_videos:,}")
print(f"Real videos     : {total_real:,}  (label 0)")
print(f"Fake videos     : {total_fake:,}  (label 1)")
print()

sample_shown = 0
print("SAMPLE VIDEOS:")
for category, label in CATEGORIES.items():
    cat_path = dataset / category
    if not cat_path.exists():
        continue
    videos = list(cat_path.rglob("*.mp4"))
    if videos:
        v = videos[0]
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  [{category}]")
        print(f"    File : {v.name}")
        print(f"    Size : {size_mb:.2f} MB")
        print()
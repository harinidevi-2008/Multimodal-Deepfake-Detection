from pathlib import Path
import numpy as np

folder = Path("visual/data/features_ffpp")
files = sorted(folder.glob("*.npy"))

print(f"Found {len(files)} feature files")

bad = []

for f in files:
    x = np.load(f)

    print(f"{f.name}: shape={x.shape}, dtype={x.dtype}")

    if x.shape != (1280,):
        bad.append((f.name, x.shape))

print("\n==============================")
print(f"Total: {len(files)}")
print(f"Valid 1280-D: {len(files)-len(bad)}")
print(f"Invalid: {len(bad)}")

if bad:
    print("\nInvalid files:")
    for name, shape in bad:
        print(f"  {name}: {shape}")
else:
    print("All features are valid (1280,)")
print("==============================")

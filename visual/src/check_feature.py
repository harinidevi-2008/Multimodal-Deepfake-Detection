import numpy as np

feature_path = "visual/data/features/real_harini_001.npy"

feature = np.load(feature_path)

print("Shape :", feature.shape)
print()

print("First 10 values:")
print(feature[:10])

print()

print("Data type:", feature.dtype)
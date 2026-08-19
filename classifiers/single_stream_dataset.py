"""
Single-modality feature dataset.

Loads precomputed .npy feature files from one modality's feature
directory (visual, audio, or semantic) and labels them the same way
FusionDataset does: RealVideo* -> 0, FakeVideo* -> 1, based on the
first path component under the feature root.

This intentionally does not require the other two modalities to be
present, unlike FusionDataset - each single-modality classifier trains
and evaluates independently.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleStreamDataset(Dataset):

    def __init__(self, feature_root):
        self.feature_root = Path(feature_root)
        self.samples = []

        feature_files = sorted(self.feature_root.rglob("*.npy"))

        for feature_file in feature_files:
            relative_path = feature_file.relative_to(self.feature_root)
            first_folder = relative_path.parts[0]

            if first_folder.startswith("RealVideo"):
                label = 0
            elif first_folder.startswith("FakeVideo"):
                label = 1
            else:
                continue

            self.samples.append((feature_file, label))

        print(f"{self.feature_root.name}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        feature_file, label = self.samples[index]
        feature = np.load(feature_file).astype(np.float32)

        return (
            torch.from_numpy(feature),
            torch.tensor(label, dtype=torch.long),
        )

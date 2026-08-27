"""
Single-modality feature dataset.

Loads precomputed .npy feature files from one modality's feature
directory (visual, audio, or semantic) and labels them the same way
FusionDataset does: RealVideo* -> 0, FakeVideo* -> 1, based on the
first path component under the feature root.

This intentionally does not require the other two modalities to be
present, unlike FusionDataset - each single-modality classifier trains
and evaluates independently.

Optional split_path/split_name restrict the dataset to one persisted
split from fusion/data_split.json (see fusion/split_utils.py), so a
classifier's evaluation can be compared fairly against the fusion
models on the exact same held-out samples. Both default to "do
nothing" (split_path=None), so existing callers are unaffected.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_FUSION_DIR = Path(__file__).resolve().parent.parent / "fusion"
if str(_FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(_FUSION_DIR))
from split_utils import load_split, split_of  # noqa: E402


class SingleStreamDataset(Dataset):

    def __init__(self, feature_root, split_path=None, split_name="all"):
        self.feature_root = Path(feature_root)
        self.split_name = split_name
        self.samples = []

        split_data = load_split(split_path) if split_path is not None else None

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

            if split_data is not None and split_name != "all":
                if split_of(relative_path, split_data) != split_name:
                    continue

            self.samples.append((feature_file, label))

        split_suffix = f", split='{split_name}'" if split_data is not None else ""
        print(f"{self.feature_root.name}: {len(self.samples)} samples{split_suffix}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        feature_file, label = self.samples[index]
        feature = np.load(feature_file).astype(np.float32)

        return (
            torch.from_numpy(feature),
            torch.tensor(label, dtype=torch.long),
        )

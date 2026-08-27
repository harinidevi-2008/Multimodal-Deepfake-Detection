"""
Dataset for the enhanced (5-modality) fusion model. Same alignment
convention as fusion_dataset.py's FusionDataset (align by relative
path under each root, label from the RealVideo*/FakeVideo* folder
prefix) - extended to also require blink and lipsync feature files.

This is a NEW file, not a modification of fusion_dataset.py - the
original 3-modality dataset stays available and usable on its own.

Optional split_path/split_name restrict the dataset to one persisted
split (fusion/create_split.py / fusion/data_split.json); optional
normalization_stats applies train-only blink/lipsync normalization
(fusion/feature_normalization.py) at __getitem__ time. Both default to
"do nothing", so existing callers that only pass the five roots are
unaffected.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .split_utils import load_split, split_of
    from .feature_normalization import apply_normalization
except ImportError:
    from split_utils import load_split, split_of
    from feature_normalization import apply_normalization


class EnhancedFusionDataset(Dataset):

    def __init__(
        self,
        visual_root,
        audio_root,
        semantic_root,
        blink_root,
        lipsync_root,
        split_path=None,
        split_name="all",
        normalization_stats=None,
    ):
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root)
        self.semantic_root = Path(semantic_root)
        self.blink_root = Path(blink_root)
        self.lipsync_root = Path(lipsync_root)
        self.split_name = split_name
        self.normalization_stats = normalization_stats

        split_data = load_split(split_path) if split_path is not None else None

        self.samples = []

        visual_files = sorted(self.visual_root.rglob("*.npy"))

        for visual_file in visual_files:
            relative_path = visual_file.relative_to(self.visual_root)

            audio_file = self.audio_root / relative_path
            semantic_file = self.semantic_root / relative_path
            blink_file = self.blink_root / relative_path
            lipsync_file = self.lipsync_root / relative_path

            if not (audio_file.exists() and semantic_file.exists()
                    and blink_file.exists() and lipsync_file.exists()):
                continue

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

            self.samples.append(
                (visual_file, audio_file, semantic_file, blink_file, lipsync_file, label)
            )

        split_suffix = f", split='{split_name}'" if split_data is not None else ""
        norm_suffix = ", normalized blink/lipsync" if normalization_stats is not None else ""
        print(f"Enhanced fusion samples: {len(self.samples)}{split_suffix}{norm_suffix}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        visual_file, audio_file, semantic_file, blink_file, lipsync_file, label = self.samples[index]

        visual = np.load(visual_file).astype(np.float32)
        audio = np.load(audio_file).astype(np.float32)
        semantic = np.load(semantic_file).astype(np.float32)
        blink = np.load(blink_file).astype(np.float32)
        lipsync = np.load(lipsync_file).astype(np.float32)

        if self.normalization_stats is not None:
            blink = apply_normalization(blink, self.normalization_stats["blink"])
            lipsync = apply_normalization(lipsync, self.normalization_stats["lipsync"])

        return (
            torch.from_numpy(visual),
            torch.from_numpy(audio),
            torch.from_numpy(semantic),
            torch.from_numpy(blink),
            torch.from_numpy(lipsync),
            torch.tensor(label, dtype=torch.long),
        )

"""
Dataset for the enhanced (5-modality) fusion model. Same alignment
convention as fusion_dataset.py's FusionDataset (align by relative
path under each root, label from the RealVideo*/FakeVideo* folder
prefix) - extended to also require blink and lipsync feature files.

This is a NEW file, not a modification of fusion_dataset.py - the
original 3-modality dataset stays available and usable on its own.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EnhancedFusionDataset(Dataset):

    def __init__(self, visual_root, audio_root, semantic_root, blink_root, lipsync_root):
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root)
        self.semantic_root = Path(semantic_root)
        self.blink_root = Path(blink_root)
        self.lipsync_root = Path(lipsync_root)

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

            self.samples.append(
                (visual_file, audio_file, semantic_file, blink_file, lipsync_file, label)
            )

        print(f"Enhanced fusion samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        visual_file, audio_file, semantic_file, blink_file, lipsync_file, label = self.samples[index]

        visual = np.load(visual_file).astype(np.float32)
        audio = np.load(audio_file).astype(np.float32)
        semantic = np.load(semantic_file).astype(np.float32)
        blink = np.load(blink_file).astype(np.float32)
        lipsync = np.load(lipsync_file).astype(np.float32)

        return (
            torch.from_numpy(visual),
            torch.from_numpy(audio),
            torch.from_numpy(semantic),
            torch.from_numpy(blink),
            torch.from_numpy(lipsync),
            torch.tensor(label, dtype=torch.long),
        )

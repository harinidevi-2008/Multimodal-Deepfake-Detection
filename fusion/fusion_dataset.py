from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .split_utils import load_split, split_of
except ImportError:
    from split_utils import load_split, split_of


class FusionDataset(Dataset):

    def __init__(
        self,
        visual_root,
        audio_root,
        semantic_root,
        split_path=None,
        split_name="all",
    ):
        """
        split_path / split_name: OPTIONAL, default to None / "all" so
        existing callers that only pass the three roots keep their
        original behavior (every aligned sample included) unchanged.
        Pass split_path=<path to fusion/data_split.json> and
        split_name in {"train", "validation", "test"} to restrict this
        dataset to one persisted split (see fusion/create_split.py and
        fusion/split_utils.py).
        """
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root)
        self.semantic_root = Path(semantic_root)
        self.split_name = split_name

        split_data = load_split(split_path) if split_path is not None else None

        self.samples = []

        visual_files = sorted(
            self.visual_root.rglob("*.npy")
        )

        for visual_file in visual_files:

            relative_path = visual_file.relative_to(
                self.visual_root
            )

            audio_file = self.audio_root / relative_path
            semantic_file = self.semantic_root / relative_path

            if not audio_file.exists():
                continue

            if not semantic_file.exists():
                continue

            # Determine label from video type
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
                (
                    visual_file,
                    audio_file,
                    semantic_file,
                    label
                )
            )

        split_suffix = f", split='{split_name}'" if split_data is not None else ""
        print(f"Fusion samples: {len(self.samples)}{split_suffix}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        visual_file, audio_file, semantic_file, label = \
            self.samples[index]

        visual = np.load(visual_file).astype(np.float32)
        audio = np.load(audio_file).astype(np.float32)
        semantic = np.load(semantic_file).astype(np.float32)

        return (
            torch.from_numpy(visual),
            torch.from_numpy(audio),
            torch.from_numpy(semantic),
            torch.tensor(label, dtype=torch.long)
        )

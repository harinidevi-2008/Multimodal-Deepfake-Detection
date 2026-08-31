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
    from .split_utils import load_split, split_of, to_key
    from .feature_normalization import apply_normalization
except ImportError:
    from split_utils import load_split, split_of, to_key
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
        cache_features=True,
    ):
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root)
        self.semantic_root = Path(semantic_root)
        self.blink_root = Path(blink_root)
        self.lipsync_root = Path(lipsync_root)
        self.split_name = split_name
        self.normalization_stats = normalization_stats
        self.cache_features = cache_features

        split_data = load_split(split_path) if split_path is not None else None

        if split_data is not None and split_name != "all":
            configured_roots = {
                "visual": self.visual_root,
                "audio": self.audio_root,
                "semantic": self.semantic_root,
                "blink": self.blink_root,
                "lipsync": self.lipsync_root,
            }
            recorded_roots = split_data.get("metadata", {})
            split_keys = [
                to_key(key) for key, entry in split_data.get("samples", {}).items()
                if entry.get("split") == split_name
            ]
            for name, configured_root in configured_roots.items():
                recorded = recorded_roots.get(f"{name}_root")
                if not recorded:
                    continue
                recorded_root = Path(recorded)
                if not recorded_root.is_absolute():
                    recorded_root = Path(split_path).resolve().parent.parent / recorded_root
                configured_matches = sum((configured_root / key).is_file() for key in split_keys)
                recorded_matches = sum((recorded_root / key).is_file() for key in split_keys)
                if recorded_matches > configured_matches:
                    print(
                        f"Enhanced fusion root '{name}' resolved from split metadata: "
                        f"{configured_root} -> {recorded_root} "
                        f"({configured_matches} -> {recorded_matches} matching '{split_name}' keys)"
                    )
                    configured_roots[name] = recorded_root
            self.visual_root = configured_roots["visual"]
            self.audio_root = configured_roots["audio"]
            self.semantic_root = configured_roots["semantic"]
            self.blink_root = configured_roots["blink"]
            self.lipsync_root = configured_roots["lipsync"]

        self.samples = []

        files_by_key = {
            name: {to_key(path.relative_to(root)): path for path in sorted(root.rglob("*.npy"))}
            for name, root in {
                "visual": self.visual_root,
                "audio": self.audio_root,
                "semantic": self.semantic_root,
                "blink": self.blink_root,
                "lipsync": self.lipsync_root,
            }.items()
        }
        if split_data is not None and split_name != "all":
            candidate_keys = [
                to_key(key) for key, entry in split_data.get("samples", {}).items()
                if entry.get("split") == split_name
            ]
        else:
            candidate_keys = sorted(files_by_key["visual"])

        stage_counts = {"split IDs": len(candidate_keys), "visual": 0, "audio": 0,
                        "semantic": 0, "blink": 0, "lipsync": 0}
        for key in candidate_keys:
            if key not in files_by_key["visual"]:
                continue
            stage_counts["visual"] += 1
            if key not in files_by_key["audio"]:
                continue
            stage_counts["audio"] += 1
            if key not in files_by_key["semantic"]:
                continue
            stage_counts["semantic"] += 1
            if key not in files_by_key["blink"]:
                continue
            stage_counts["blink"] += 1
            if key not in files_by_key["lipsync"]:
                continue
            stage_counts["lipsync"] += 1

            relative_path = Path(key)
            first_folder = relative_path.parts[0]
            if first_folder.startswith("RealVideo"):
                label = 0
            elif first_folder.startswith("FakeVideo"):
                label = 1
            else:
                continue

            self.samples.append(
                (files_by_key["visual"][key], files_by_key["audio"][key],
                 files_by_key["semantic"][key], files_by_key["blink"][key],
                 files_by_key["lipsync"][key], label)
            )

        split_suffix = f", split='{split_name}'" if split_data is not None else ""
        norm_suffix = ", normalized blink/lipsync" if normalization_stats is not None else ""
        print("Enhanced fusion matching: " + " -> ".join(
            f"{name}={count}" for name, count in stage_counts.items()
        ))
        print(f"Enhanced fusion samples: {len(self.samples)}{split_suffix}{norm_suffix}")
        self._cached_features = None
        if self.cache_features:
            self._cached_features = [
                tuple(np.load(path).astype(np.float32) for path in sample[:5])
                for sample in self.samples
            ]
            print(f"Enhanced fusion feature cache: {len(self._cached_features)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        visual_file, audio_file, semantic_file, blink_file, lipsync_file, label = self.samples[index]

        if self._cached_features is None:
            visual = np.load(visual_file).astype(np.float32)
            audio = np.load(audio_file).astype(np.float32)
            semantic = np.load(semantic_file).astype(np.float32)
            blink = np.load(blink_file).astype(np.float32)
            lipsync = np.load(lipsync_file).astype(np.float32)
        else:
            visual, audio, semantic, blink, lipsync = self._cached_features[index]

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

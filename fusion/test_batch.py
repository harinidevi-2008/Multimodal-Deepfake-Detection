"""
Manual real-data integration script exercising FusionDataset +
FusionModel through a real DataLoader batch (3-modal baseline).

NOT a synthetic pytest unit test: it points at real, machine-specific
feature directories (including a hardcoded Windows semantic-features
path). Guarded under `if __name__ == "__main__":`, so pytest can
safely import this module (0 tests collected) without executing it.

Run directly, after adjusting the paths below for your machine:
    python fusion/test_batch.py
"""

import torch
from torch.utils.data import DataLoader

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    dataset = FusionDataset(
        visual_root=r"visual/data/features_aligned",
        audio_root=r"audio/data/features",
        semantic_root=r"C:\Deepfake_Features\semantic_features"
    )

    # -----------------------------
    # 2. Create DataLoader
    # -----------------------------
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # -----------------------------
    # 3. Get one batch
    # -----------------------------
    visual, audio, semantic, labels = next(iter(loader))

    print("\nBatch shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)
    print("Labels   :", labels.shape)

    # -----------------------------
    # 4. Create fusion model
    # -----------------------------
    model = FusionModel()
    model.eval()

    # -----------------------------
    # 5. Test forward pass
    # -----------------------------
    with torch.no_grad():

        logits, attention = model(
            visual,
            audio,
            semantic
        )

    print("\nFusion output:")
    print("Logits    :", logits.shape)
    print("Attention :", attention.shape)

    print("\nBatch fusion test successful!")


if __name__ == "__main__":
    main()
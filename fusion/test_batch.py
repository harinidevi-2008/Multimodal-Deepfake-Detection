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
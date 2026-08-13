import numpy as np
import torch

from fusion_model import FusionModel


def main():

    # -----------------------------
    # 1. Load precomputed features
    # -----------------------------
    visual = np.load("fusion/test_data/visual.npy")
    audio = np.load("fusion/test_data/audio.npy")
    semantic = np.load("fusion/test_data/semantic.npy")

    print("Loaded feature shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # -----------------------------
    # 2. Convert NumPy → PyTorch
    # -----------------------------
    visual = torch.from_numpy(visual)
    audio = torch.from_numpy(audio)
    semantic = torch.from_numpy(semantic)

    # Add batch dimension
    visual = visual.unsqueeze(0)
    audio = audio.unsqueeze(0)
    semantic = semantic.unsqueeze(0)

    print("\nTensor shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # -----------------------------
    # 3. Create Fusion Model
    # -----------------------------
    model = FusionModel()

    model.eval()

    # -----------------------------
    # 4. Forward pass
    # -----------------------------
    with torch.no_grad():
        logits, attention = model(
            visual,
            audio,
            semantic
        )

    # -----------------------------
    # 5. Display results
    # -----------------------------
    print("\nFusion output:")
    print("Logits    :", logits.shape)
    print("Attention :", attention.shape)

    print("\nLogits:")
    print(logits)

    print("\nAttention:")
    print(attention)

    print("\nReal feature fusion successful!")


if __name__ == "__main__":
    main()
"""
Manual real-data integration script that pulls one real, aligned
sample straight off disk (3-modal baseline) and runs it through
FusionModel.

NOT a synthetic pytest unit test: VISUAL_ROOT/AUDIO_ROOT/SEMANTIC_ROOT
below point at real, machine-specific feature directories (including
a hardcoded Windows semantic-features path) and this script requires
at least one real extracted visual feature file to exist. Guarded
under `if __name__ == "__main__":`, so pytest can safely import this
module (0 tests collected) without executing it.

Run directly, after adjusting the paths below for your machine:
    python fusion/test_aligned_fusion.py
"""

import numpy as np
import torch
from pathlib import Path

from fusion_model import FusionModel


VISUAL_ROOT = Path("visual/data/features_aligned")
AUDIO_ROOT = Path("audio/data/features")
SEMANTIC_ROOT = Path(r"C:\Deepfake_Features\semantic_features")


def main():

    # Find one visual feature
    visual_file = next(VISUAL_ROOT.rglob("*.npy"))

    # Get its relative path
    relative_path = visual_file.relative_to(VISUAL_ROOT)

    # Find corresponding audio and semantic features
    audio_file = AUDIO_ROOT / relative_path
    semantic_file = SEMANTIC_ROOT / relative_path

    print("Visual   :", visual_file)
    print("Audio    :", audio_file)
    print("Semantic :", semantic_file)

    # Verify all three exist
    assert audio_file.exists(), f"Missing audio: {audio_file}"
    assert semantic_file.exists(), f"Missing semantic: {semantic_file}"

    # Load features
    visual = np.load(visual_file)
    audio = np.load(audio_file)
    semantic = np.load(semantic_file)

    print("\nFeature shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # Convert to PyTorch tensors
    visual = torch.from_numpy(visual).unsqueeze(0)
    audio = torch.from_numpy(audio).unsqueeze(0)
    semantic = torch.from_numpy(semantic).unsqueeze(0)

    print("\nBatch shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # Create model
    model = FusionModel()
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits, attention = model(
            visual,
            audio,
            semantic
        )

    print("\nFusion output:")
    print("Logits    :", logits.shape)
    print("Attention :", attention.shape)

    print("\nLogits:")
    print(logits)

    print("\nAttention:")
    print(attention)

    print("\nREAL ALIGNED FUSION TEST SUCCESSFUL!")


if __name__ == "__main__":
    main()
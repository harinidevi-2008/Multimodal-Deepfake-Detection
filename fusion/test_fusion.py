"""
Manual synthetic smoke script for the 3-modal baseline FusionModel:
dummy random tensors through a forward pass, checking output shape.

Wrapped in main() behind the `if __name__ == "__main__":` guard so
pytest can safely import this module (0 tests collected, matching its
sibling fusion/test_*.py scripts) without running a full model forward
pass and printing output as a side effect of test collection. No
behavior change when run directly.

Run directly:
    python fusion/test_fusion.py
"""

import torch

from fusion_model import FusionModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 4

VISUAL_DIM = 1280
AUDIO_DIM = 768
SEMANTIC_DIM = 384


def main():
    # --------------------------------------------------
    # Create dummy features
    # --------------------------------------------------

    visual = torch.randn(BATCH_SIZE, VISUAL_DIM)
    audio = torch.randn(BATCH_SIZE, AUDIO_DIM)
    semantic = torch.randn(BATCH_SIZE, SEMANTIC_DIM)

    print("Input shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = FusionModel()

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    logits, attention = model(
        visual,
        audio,
        semantic
    )

    print("\nOutput shapes:")
    print("Logits    :", logits.shape)
    print("Attention :", attention.shape)

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------

    assert logits.shape == (BATCH_SIZE, 2)

    print("\nFusion forward pass successful!")


if __name__ == "__main__":
    main()
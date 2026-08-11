import torch

from fusion_model import FusionModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 4

VISUAL_DIM = 1280
AUDIO_DIM = 768
SEMANTIC_DIM = 384


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
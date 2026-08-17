import numpy as np
import torch

from fusion_model import FusionModel
from fusion_dataset import FusionDataset


def main():

    # --------------------------------------------------
    # 1. Load aligned dataset
    # --------------------------------------------------

    dataset = FusionDataset(
        visual_root=r"visual/data/features_aligned",
        audio_root=r"audio/data/features",
        semantic_root=r"C:\Deepfake_Features\semantic_features"
    )

    # Pick one sample
    visual_path, audio_path, semantic_path, true_label = dataset.samples[0]

    print("Test sample:")
    print("Visual  :", visual_path)
    print("Audio   :", audio_path)
    print("Semantic:", semantic_path)
    print("True label:", true_label)

    # --------------------------------------------------
    # 2. Load features
    # --------------------------------------------------

    visual = np.load(visual_path)
    audio = np.load(audio_path)
    semantic = np.load(semantic_path)

    print("\nFeature shapes:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)

    # --------------------------------------------------
    # 3. Convert to tensors
    # --------------------------------------------------

    visual = torch.from_numpy(visual).float().unsqueeze(0)
    audio = torch.from_numpy(audio).float().unsqueeze(0)
    semantic = torch.from_numpy(semantic).float().unsqueeze(0)

    # --------------------------------------------------
    # 4. Load TRAINED fusion model
    # --------------------------------------------------

    model = FusionModel()

    model.load_state_dict(
        torch.load(
            "fusion/best_fusion_model.pt",
            map_location="cpu"
        )
    )

    model.eval()

    print("\nTrained fusion model loaded successfully.")

    # --------------------------------------------------
    # 5. Fusion inference
    # --------------------------------------------------

    with torch.no_grad():

        logits, attention = model(
            visual,
            audio,
            semantic
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # --------------------------------------------------
    # 6. Results
    # --------------------------------------------------

    print("\n===================================")
    print("TRAINED FUSION INFERENCE")
    print("===================================")

    print("Logits:")
    print(logits)

    print("\nProbabilities:")
    print(probabilities)

    print("\nPredicted class:", prediction)
    print("True class     :", true_label)

    print("\nAttention matrix:")
    print(attention)

    print("\nAttention shape:", attention.shape)

    print("\n===================================")
    print("TRAINED FUSION TEST SUCCESSFUL")
    print("===================================")


if __name__ == "__main__":
    main()
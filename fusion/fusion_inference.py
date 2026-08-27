import numpy as np
import torch

from fusion_model import FusionModel


VISUAL_ROOT = r"visual/data/features_aligned"
AUDIO_ROOT = r"audio/data/features"
SEMANTIC_ROOT = r"C:\Deepfake_Features\semantic_features"

MODEL_PATH = r"fusion/best_fusion_model.pt"


def load_features(visual_path, audio_path, semantic_path):

    visual = np.load(visual_path).astype(np.float32)
    audio = np.load(audio_path).astype(np.float32)
    semantic = np.load(semantic_path).astype(np.float32)

    visual = torch.from_numpy(visual).unsqueeze(0)
    audio = torch.from_numpy(audio).unsqueeze(0)
    semantic = torch.from_numpy(semantic).unsqueeze(0)

    return visual, audio, semantic


def load_model():

    model = FusionModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


def predict(visual_path, audio_path, semantic_path):

    visual, audio, semantic = load_features(
        visual_path,
        audio_path,
        semantic_path
    )

    model = load_model()

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

        fake_probability = probabilities[0, 1].item()
        real_probability = probabilities[0, 0].item()

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return {
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "prediction": prediction,
        "attention": attention
    }


def main():

    # --------------------------------------------------
    # Example aligned sample
    # --------------------------------------------------

    relative_path = (
        r"FakeVideo-FakeAudio\African\men\id00076"
        r"\00109_10_id00476_wavtolip.npy"
    )

    visual_path = f"{VISUAL_ROOT}/{relative_path}"
    audio_path = f"{AUDIO_ROOT}/{relative_path}"
    semantic_path = f"{SEMANTIC_ROOT}/{relative_path}"

    print("========================================")
    print("MULTIMODAL FUSION INFERENCE")
    print("========================================")

    print("\nInput:")
    print("Visual   :", visual_path)
    print("Audio    :", audio_path)
    print("Semantic :", semantic_path)

    result = predict(
        visual_path,
        audio_path,
        semantic_path
    )

    fake_percentage = result["fake_probability"] * 100
    real_percentage = result["real_probability"] * 100

    print("\n----------------------------------------")
    print("FINAL MULTIMODAL RESULT")
    print("----------------------------------------")

    print(f"Real probability : {real_percentage:.2f}%")
    print(f"Fake probability : {fake_percentage:.2f}%")

    if result["prediction"] == 1:
        print("Prediction       : DEEPFAKE")
    else:
        print("Prediction       : REAL")

    print("\nAttention matrix:")
    print(result["attention"])

    print("\n========================================")


if __name__ == "__main__":
    main()
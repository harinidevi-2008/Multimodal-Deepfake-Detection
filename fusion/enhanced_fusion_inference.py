"""
5-modality inference using EnhancedFusionModel (visual + audio +
semantic + blink + lipsync).

This is a NEW, separate script - fusion/fusion_inference.py
(3-modality) is untouched and still works standalone for the original
baseline model.

Applies the same train-only blink/lipsync normalization used during
training (fusion/feature_normalization.py) if a normalization file is
found, so predictions match what the model was actually trained on;
pass --no-normalization to skip it (only correct if the checkpoint
itself was trained without normalization).

Usage:
    python fusion/enhanced_fusion_inference.py \
        --relative-path "FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.npy"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_fusion_model import EnhancedFusionModel  # noqa: E402
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_ENHANCED_FUSION_WEIGHTS,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_ROOT,
)
from feature_normalization import DEFAULT_NORMALIZATION_PATH, apply_normalization, load_normalization  # noqa: E402


def load_features(visual_path, audio_path, semantic_path, blink_path, lipsync_path, normalization=None):
    visual = np.load(visual_path).astype(np.float32)
    audio = np.load(audio_path).astype(np.float32)
    semantic = np.load(semantic_path).astype(np.float32)
    blink = np.load(blink_path).astype(np.float32)
    lipsync = np.load(lipsync_path).astype(np.float32)

    if normalization is not None:
        blink = apply_normalization(blink, normalization["blink"])
        lipsync = apply_normalization(lipsync, normalization["lipsync"])

    visual_t = torch.from_numpy(visual).unsqueeze(0)
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    semantic_t = torch.from_numpy(semantic).unsqueeze(0)
    blink_t = torch.from_numpy(blink).unsqueeze(0)
    lipsync_t = torch.from_numpy(lipsync).unsqueeze(0)

    return visual_t, audio_t, semantic_t, blink_t, lipsync_t


def load_model(weights_path):
    model = EnhancedFusionModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def predict(visual_path, audio_path, semantic_path, blink_path, lipsync_path, weights_path, normalization=None):
    visual, audio, semantic, blink, lipsync = load_features(
        visual_path, audio_path, semantic_path, blink_path, lipsync_path, normalization=normalization
    )
    model = load_model(weights_path)

    with torch.no_grad():
        logits, attention = model(visual, audio, semantic, blink, lipsync)
        probabilities = torch.softmax(logits, dim=1)

        fake_probability = probabilities[0, 1].item()
        real_probability = probabilities[0, 0].item()
        prediction = torch.argmax(probabilities, dim=1).item()

    return {
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "prediction": prediction,
        "attention": attention,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 5-modality enhanced fusion inference on one sample.")
    parser.add_argument(
        "--relative-path", required=True,
        help=r"Path relative to each feature root, e.g. "
             r"FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.npy",
    )
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    parser.add_argument("--weights", default=DEFAULT_ENHANCED_FUSION_WEIGHTS)
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION_PATH))
    parser.add_argument("--no-normalization", action="store_true",
                         help="Skip blink/lipsync normalization even if a normalization file exists.")
    args = parser.parse_args()

    normalization = None
    if not args.no_normalization:
        try:
            normalization = load_normalization(args.normalization_path)
        except FileNotFoundError:
            print(f"[info] No normalization file at {args.normalization_path} - running WITHOUT "
                  "blink/lipsync normalization. This must match how the model was trained.")

    rel = Path(args.relative_path)
    visual_path = Path(args.visual_root) / rel
    audio_path = Path(args.audio_root) / rel
    semantic_path = Path(args.semantic_root) / rel
    blink_path = Path(args.blink_root) / rel
    lipsync_path = Path(args.lipsync_root) / rel

    print("========================================")
    print("5-MODALITY ENHANCED FUSION INFERENCE")
    print("========================================")
    print("\nInput:")
    print("Visual   :", visual_path)
    print("Audio    :", audio_path)
    print("Semantic :", semantic_path)
    print("Blink    :", blink_path)
    print("Lip-sync :", lipsync_path)
    print("Normalization:", args.normalization_path if normalization is not None else "NONE (raw features)")

    result = predict(visual_path, audio_path, semantic_path, blink_path, lipsync_path, args.weights, normalization)

    fake_percentage = result["fake_probability"] * 100
    real_percentage = result["real_probability"] * 100

    print("\n----------------------------------------")
    print("FINAL MULTIMODAL RESULT")
    print("----------------------------------------")
    print(f"Real probability : {real_percentage:.2f}%")
    print(f"Fake probability : {fake_percentage:.2f}%")
    print("Prediction       :", "DEEPFAKE" if result["prediction"] == 1 else "REAL")

    print("\nAttention matrix (descriptive only - see fusion/attention_utils.py's docstring; "
          "not a causal explanation):")
    print(result["attention"])

    print("\n========================================")


if __name__ == "__main__":
    main()

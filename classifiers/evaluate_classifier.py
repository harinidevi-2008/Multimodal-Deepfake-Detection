"""
Evaluate a trained single-modality classifier on its full feature set.

Usage:
    python classifiers/evaluate_classifier.py --modality visual \
        --feature-root visual/data/features_aligned --input-dim 1280 \
        --weights classifiers/best_visual_classifier.pt

Prints accuracy, precision, recall, F1, ROC-AUC, and the confusion
matrix - not just a handful of samples. Also saves per-sample
probabilities to a .npy file so this modality's scores can later be
combined with blink/lip-sync/fusion results in the evidence layer.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from .single_stream_classifier import SingleStreamClassifier
    from .single_stream_dataset import SingleStreamDataset
except ImportError:
    from single_stream_classifier import SingleStreamClassifier
    from single_stream_dataset import SingleStreamDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_probs, all_preds = [], [], []

    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single-modality deepfake classifier.")
    parser.add_argument("--modality", required=True, choices=["visual", "audio", "semantic"])
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--weights", required=True, help="Path to a trained best_<modality>_classifier.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-probs", default=None, help="Optional path to save per-sample fake probabilities as .npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SingleStreamDataset(args.feature_root)
    if len(dataset) == 0:
        logger.error("No samples found under %s", args.feature_root)
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = SingleStreamClassifier(input_dim=args.input_dim).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    labels, probs, preds = run_inference(model, loader, device)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        # Only one class present in this split - AUC is undefined.
        roc_auc = float("nan")

    cm = confusion_matrix(labels, preds)

    print(f"\n=== {args.modality.upper()} classifier evaluation ({len(dataset)} samples) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(cm)

    if args.save_probs:
        np.save(args.save_probs, probs)
        logger.info("Saved per-sample fake-probabilities to %s", args.save_probs)


if __name__ == "__main__":
    main()

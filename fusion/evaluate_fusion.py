"""
Full-dataset evaluation for the original 3-modality FusionModel.

Unlike test_trained_fusion.py (single sample) or the val-split metrics
printed during train_fusion.py, this runs over the whole dataset you
point it at and reports the full metric set: accuracy, precision,
recall, F1, ROC-AUC, confusion matrix.

Usage:
    python fusion/evaluate_fusion.py \
        --visual-root visual/data/features_aligned \
        --audio-root audio/data/features \
        --semantic-root semantic/data/features \
        --weights fusion/best_fusion_model.pt
"""

import argparse

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

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_probs, all_preds = [], [], []

    for visual, audio, semantic, labels in loader:
        visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
        logits, _ = model(visual, audio, semantic)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained fusion model on the full dataset.")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--weights", default="fusion/best_fusion_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FusionDataset(args.visual_root, args.audio_root, args.semantic_root)
    if len(dataset) == 0:
        print("No aligned samples found - check the three root paths.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = FusionModel().to(device)
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
        roc_auc = float("nan")
    cm = confusion_matrix(labels, preds)

    print(f"\n=== FUSION evaluation ({len(dataset)} samples) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(cm)


if __name__ == "__main__":
    main()

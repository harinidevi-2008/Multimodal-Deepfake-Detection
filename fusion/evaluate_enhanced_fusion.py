"""
Full-dataset evaluation for the enhanced (5-modality) fusion model.
Same metric set as evaluate_fusion.py, so the two can be compared
directly in the results table (original fusion vs enhanced fusion).

Usage:
    python fusion/evaluate_enhanced_fusion.py \
        --visual-root visual/data/features_aligned \
        --audio-root audio/data/features \
        --semantic-root semantic/data/features \
        --blink-root visual/data/blink_features \
        --lipsync-root visual/data/lipsync_features \
        --weights fusion/best_enhanced_fusion_model.pt
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

from enhanced_fusion_dataset import EnhancedFusionDataset
from enhanced_fusion_model import EnhancedFusionModel


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_probs, all_preds = [], [], []

    for visual, audio, semantic, blink, lipsync, labels in loader:
        visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
        blink, lipsync = blink.to(device), lipsync.to(device)

        logits, _ = model(visual, audio, semantic, blink, lipsync)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained enhanced fusion model on the full dataset.")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--blink-root", required=True)
    parser.add_argument("--lipsync-root", required=True)
    parser.add_argument("--weights", default="fusion/best_enhanced_fusion_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = EnhancedFusionDataset(
        args.visual_root, args.audio_root, args.semantic_root, args.blink_root, args.lipsync_root
    )
    if len(dataset) == 0:
        print("No aligned samples found - check the five root paths.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = EnhancedFusionModel().to(device)
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

    print(f"\n=== ENHANCED FUSION evaluation ({len(dataset)} samples) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(cm)


if __name__ == "__main__":
    main()

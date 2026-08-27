"""
Evaluation for the original 3-modality FusionModel.

Unlike test_trained_fusion.py (single sample), this runs over a whole
split and reports the full metric set: accuracy, precision, recall,
F1, ROC-AUC, confusion matrix.

IMPORTANT (leakage): --split defaults to "test" - the held-out split
from fusion/data_split.json that is never used for training or model
selection. "all" evaluates on every aligned sample regardless of
split membership and is NOT a valid final metric - it exists only for
debugging. NOTE: fusion/train_fusion.py now also trains against this
same persistent split (train-only training, validation-only model
selection, test never touched), so --split test is a genuinely
held-out number for the 3-modal baseline PROVIDED the checkpoint being
evaluated was produced by the current train_fusion.py. An older
checkpoint trained before this change (with the old random split) may
have already seen some of these "test" samples during its own
training - retrain with train_fusion.py to get a checkpoint that is
strictly comparable to the 5-modal model under this split.

Usage:
    python fusion/evaluate_fusion.py \
        --visual-root visual/data/features_aligned \
        --audio-root audio/data/features \
        --semantic-root semantic/data/features \
        --weights fusion/best_fusion_model.pt \
        --split test
"""

import argparse
import sys
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

from fusion_dataset import FusionDataset
from fusion_model import FusionModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import DEFAULT_SPLIT_PATH  # noqa: E402


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


def evaluate(visual_root, audio_root, semantic_root, weights, split="test",
             split_path=DEFAULT_SPLIT_PATH, batch_size=64):
    """
    Returns a metrics dict, or None if no samples were found for this
    split (caller decides how to treat that - main() below exits 1).
    Callable directly (e.g. from eval/generate_report.py) as well as
    via the CLI.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_path_arg = split_path if split != "all" else None
    dataset = FusionDataset(visual_root, audio_root, semantic_root,
                             split_path=split_path_arg, split_name=split)
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
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

    return {
        "split": split,
        "n_samples": len(dataset),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained fusion model.")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--weights", default="fusion/best_fusion_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"],
                         help="Defaults to 'test' (held-out). 'all' is NOT a final metric - debugging only.")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    args = parser.parse_args()

    if args.split == "all":
        print("WARNING: --split all includes every aligned sample regardless of train/val/test "
              "membership. This is NOT a valid held-out metric - use only for debugging.\n")

    metrics = evaluate(
        args.visual_root, args.audio_root, args.semantic_root, args.weights,
        split=args.split, split_path=args.split_path, batch_size=args.batch_size,
    )

    if metrics is None:
        print(f"No samples found for split='{args.split}' under the given roots. "
              f"Check the root paths, and that fusion/data_split.json exists "
              f"(run fusion/create_split.py first) if using a named split.")
        sys.exit(1)

    print(f"\n=== FUSION evaluation ({metrics['n_samples']} samples, split='{metrics['split']}', "
          f"real={metrics['n_real']} fake={metrics['n_fake']}) ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()

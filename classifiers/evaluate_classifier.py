"""
Evaluate a trained single-modality classifier on one split.

Usage:
    python classifiers/evaluate_classifier.py --modality visual \
        --feature-root visual/data/features_aligned --input-dim 1280 \
        --weights classifiers/best_visual_classifier.pt --split test

Prints accuracy, precision, recall, F1, ROC-AUC, and the confusion
matrix. Also saves per-sample probabilities to a .npy file so this
modality's scores can later be combined with blink/lip-sync/fusion
results in the evidence layer.

IMPORTANT (leakage / fair comparison): --split defaults to "test" -
the same held-out split fusion/data_split.json defines for the fusion
models, so a single-modality classifier's number is comparable to
fusion's on the exact same samples. "all" ignores split membership
entirely and is NOT a valid final metric - debugging only.
"""

import argparse
import logging
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

try:
    from .single_stream_classifier import SingleStreamClassifier
    from .single_stream_dataset import SingleStreamDataset
except ImportError:
    from single_stream_classifier import SingleStreamClassifier
    from single_stream_dataset import SingleStreamDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion"))
from split_utils import DEFAULT_SPLIT_PATH  # noqa: E402

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


def evaluate(modality, feature_root, input_dim, weights, split="test",
             split_path=DEFAULT_SPLIT_PATH, batch_size=32):
    """Returns a metrics dict (including per-sample 'probs'), or None
    if no samples were found for this split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_path_arg = split_path if split != "all" else None
    dataset = SingleStreamDataset(feature_root, split_path=split_path_arg, split_name=split)
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SingleStreamClassifier(input_dim=input_dim).to(device)
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
        # Only one class present in this split - AUC is undefined.
        roc_auc = float("nan")
    cm = confusion_matrix(labels, preds)

    return {
        "modality": modality,
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
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single-modality deepfake classifier.")
    parser.add_argument("--modality", required=True, choices=["visual", "audio", "semantic"])
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--weights", required=True, help="Path to a trained best_<modality>_classifier.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-probs", default=None, help="Optional path to save per-sample fake probabilities as .npy")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"],
                         help="Defaults to 'test' (held-out). 'all' is NOT a final metric - debugging only.")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    args = parser.parse_args()

    if args.split == "all":
        print("WARNING: --split all includes every sample regardless of train/val/test membership. "
              "This is NOT a valid held-out metric - use only for debugging.\n")

    result = evaluate(
        args.modality, args.feature_root, args.input_dim, args.weights,
        split=args.split, split_path=args.split_path, batch_size=args.batch_size,
    )

    if result is None:
        logger.error("No samples found under %s for split='%s'", args.feature_root, args.split)
        sys.exit(1)

    print(f"\n=== {args.modality.upper()} classifier evaluation ({result['n_samples']} samples, "
          f"split='{result['split']}', real={result['n_real']} fake={result['n_fake']}) ===")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(np.array(result["confusion_matrix"]))

    if args.save_probs:
        np.save(args.save_probs, result["probs"])
        logger.info("Saved per-sample fake-probabilities to %s", args.save_probs)


if __name__ == "__main__":
    main()

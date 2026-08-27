"""
Evaluation for the enhanced (5-modality) fusion model. Same metric set
as evaluate_fusion.py so the two can be compared directly.

IMPORTANT (leakage): --split defaults to "test" - the held-out split
from fusion/data_split.json, never used for training or model
selection when train_enhanced_fusion.py is used as updated. "all"
evaluates on every aligned sample regardless of split and is NOT a
valid final metric - debugging only.

IMPORTANT (normalization): if fusion/feature_normalization.json exists
it is applied to blink/lipsync features automatically, matching
training - pass --no-normalization to force raw features (only useful
if the checkpoint you're evaluating was itself trained without
normalization; mismatching train/eval normalization will silently
produce wrong numbers).

Usage:
    python fusion/evaluate_enhanced_fusion.py \
        --visual-root visual/data/features_aligned \
        --audio-root audio/data/features \
        --semantic-root semantic/data/features \
        --blink-root visual/data/blink_features \
        --lipsync-root visual/data/lipsync_features \
        --weights fusion/best_enhanced_fusion_model.pt \
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

from enhanced_fusion_dataset import EnhancedFusionDataset
from enhanced_fusion_model import EnhancedFusionModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import DEFAULT_SPLIT_PATH  # noqa: E402
from feature_normalization import (  # noqa: E402
    DEFAULT_NORMALIZATION_PATH,
    check_normalization_consistency,
    load_normalization,
)


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


def evaluate(visual_root, audio_root, semantic_root, blink_root, lipsync_root, weights,
             split="test", split_path=DEFAULT_SPLIT_PATH, batch_size=64,
             normalization_path=DEFAULT_NORMALIZATION_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalization = None
    normalization_used = False
    if normalization_path is not None and Path(normalization_path).exists():
        normalization = load_normalization(normalization_path)
        normalization_used = True

    split_path_arg = split_path if split != "all" else None
    dataset = EnhancedFusionDataset(
        visual_root, audio_root, semantic_root, blink_root, lipsync_root,
        split_path=split_path_arg, split_name=split, normalization_stats=normalization,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedFusionModel().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    normalization_warning = check_normalization_consistency(weights, normalization_used)

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
        "normalization_applied": normalization_used,
        "normalization_warning": normalization_warning,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained enhanced fusion model.")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--blink-root", required=True)
    parser.add_argument("--lipsync-root", required=True)
    parser.add_argument("--weights", default="fusion/best_enhanced_fusion_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"],
                         help="Defaults to 'test' (held-out). 'all' is NOT a final metric - debugging only.")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION_PATH))
    parser.add_argument("--no-normalization", action="store_true",
                         help="Evaluate on raw (unnormalized) blink/lipsync features.")
    args = parser.parse_args()

    if args.split == "all":
        print("WARNING: --split all includes every aligned sample regardless of train/val/test "
              "membership. This is NOT a valid held-out metric - use only for debugging.\n")

    normalization_path = None if args.no_normalization else args.normalization_path

    metrics = evaluate(
        args.visual_root, args.audio_root, args.semantic_root, args.blink_root, args.lipsync_root,
        args.weights, split=args.split, split_path=args.split_path, batch_size=args.batch_size,
        normalization_path=normalization_path,
    )

    if metrics is None:
        print(f"No samples found for split='{args.split}' under the given roots. "
              f"Check the root paths, and that fusion/data_split.json exists "
              f"(run fusion/create_split.py first) if using a named split.")
        sys.exit(1)

    norm_note = ("fusion/feature_normalization.json found and applied" if metrics["normalization_applied"]
                 else "NO normalization file used - features evaluated raw; make sure this matches "
                      "how the checkpoint was trained")
    print(f"\n=== ENHANCED FUSION evaluation ({metrics['n_samples']} samples, split='{metrics['split']}', "
          f"real={metrics['n_real']} fake={metrics['n_fake']}) ===")
    print(f"Normalization: {norm_note}")
    if metrics["normalization_warning"]:
        print(f"\n{'!' * 70}\n{metrics['normalization_warning']}\n{'!' * 70}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()

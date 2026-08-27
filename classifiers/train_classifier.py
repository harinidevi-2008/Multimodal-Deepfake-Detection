"""
Train a single-modality classifier (visual, audio, or semantic).

Updated to use the persistent, group-aware split (fusion/data_split.json,
produced by fusion/create_split.py) instead of drawing a fresh random
85/15 random_split() every run: TRAINING now only ever sees the
"train" split, model selection/checkpointing only ever sees the
"validation" split, and the "test" split is never touched by this
script at all - a classifier's test-split number
(classifiers/evaluate_classifier.py --split test) is therefore
genuinely held-out, not partly seen during this script's own training.

Usage:
    python classifiers/train_classifier.py --modality visual \
        --feature-root visual/data/features_aligned --input-dim 1280

    python classifiers/train_classifier.py --modality audio \
        --feature-root audio/data/features --input-dim 768

    python classifiers/train_classifier.py --modality semantic \
        --feature-root semantic/data/features --input-dim 384

Saves the trained weights to <output-dir>/best_<modality>_classifier.pt
and a per-epoch history to <output-dir>/<modality>_training_history.json.
This trains each modality independently of the fusion model - it
reuses the same precomputed .npy features, no re-extraction needed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        total += features.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss / total, accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train a single-modality deepfake classifier.")
    parser.add_argument("--modality", required=True, choices=["visual", "audio", "semantic"])
    parser.add_argument("--feature-root", required=True, help="Path to that modality's .npy feature directory.")
    parser.add_argument("--input-dim", type=int, required=True, help="Feature vector dimension (1280/768/384).")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH),
                         help="fusion/data_split.json - run fusion/create_split.py first.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                         help="Max epochs; early stopping (--patience) usually ends training sooner.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                         help="Stop after this many consecutive epochs with no validation-F1 improvement.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    train_dataset = SingleStreamDataset(args.feature_root, split_path=args.split_path, split_name="train")
    val_dataset = SingleStreamDataset(args.feature_root, split_path=args.split_path, split_name="validation")

    train_labels = [label for _, label in train_dataset.samples]
    val_labels = [label for _, label in val_dataset.samples]
    train_real, train_fake = train_labels.count(0), train_labels.count(1)
    val_real, val_fake = val_labels.count(0), val_labels.count(1)

    print(f"\n[{args.modality}] Train samples:      {len(train_dataset)} (real={train_real}, fake={train_fake})")
    print(f"[{args.modality}] Validation samples: {len(val_dataset)} (real={val_real}, fake={val_fake})")
    print(f"[{args.modality}] Test split is NOT loaded and will NOT be touched by this script.")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Train or validation split is empty under %s. Run fusion/create_split.py first, "
                     "and check the feature root path.", args.feature_root)
        sys.exit(1)
    if train_real == 0 or train_fake == 0:
        logger.error("Training split has only one class present (real=%d, fake=%d) - cannot train.",
                     train_real, train_fake)
        sys.exit(1)
    if val_real == 0 or val_fake == 0:
        logger.error("Validation split has only one class present (real=%d, fake=%d) - cannot reliably "
                     "select/checkpoint a model. This can happen with an old, hand-edited, or otherwise "
                     "invalid split JSON - the shared split generator's own checks "
                     "(fusion/create_split.py) should normally prevent this.", val_real, val_fake)
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SingleStreamClassifier(input_dim=args.input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class-weighted loss, consistent with fusion/train_fusion.py and
    # fusion/train_enhanced_fusion.py, for robustness to class imbalance.
    total = train_real + train_fake
    class_weights = torch.tensor(
        [total / (2 * train_real), total / (2 * train_fake)], dtype=torch.float32
    ).to(device)
    print(f"[{args.modality}] Class weights: real={class_weights[0].item():.4f} fake={class_weights[1].item():.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Start below any achievable F1 (not 0.0) so that a first validation
    # epoch which happens to score exactly F1=0.0 still saves a
    # checkpoint (0.0 > 0.0 is False, so a 0.0 starting point would
    # silently produce a completed training run with no checkpoint at
    # all - a real failure mode, not a hypothetical one).
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    stopped_early = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"best_{args.modality}_classifier.pt"
    history_path = Path(args.output_dir) / f"{args.modality}_training_history.json"

    def save_history():
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "modality": args.modality,
                    "input_dim": args.input_dim,
                    "epochs_requested": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "split_path": args.split_path,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(val_dataset),
                },
                "history": history,
                "best_val_f1": best_val_f1,
                "stopped_early": stopped_early,
            }, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_one_epoch(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, val_f1,
        )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_path)
            logger.info("Saved new best model (val_f1=%.4f) to %s", val_f1, output_path)
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1": val_f1,
            "checkpoint_saved": improved,
        })
        save_history()

        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping: no val F1 improvement for %d consecutive epochs.", args.patience)
            stopped_early = True
            save_history()
            break

    logger.info("Training complete. Best val_f1=%.4f, saved at %s", best_val_f1, output_path)
    print(f"[{args.modality}] Training history saved to {history_path}")
    print(f"[{args.modality}] Test split was NOT touched during this run.")


if __name__ == "__main__":
    main()

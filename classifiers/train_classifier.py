"""
Train a single-modality classifier (visual, audio, or semantic).

Usage:
    python classifiers/train_classifier.py --modality visual \
        --feature-root visual/data/features_aligned --input-dim 1280

    python classifiers/train_classifier.py --modality audio \
        --feature-root audio/data/features --input-dim 768

    python classifiers/train_classifier.py --modality semantic \
        --feature-root semantic/data/features --input-dim 384

Saves the trained weights to classifiers/best_<modality>_classifier.pt
and prints a train/val loss and accuracy curve per epoch. This trains
each modality independently of the fusion model - it reuses the same
precomputed .npy features, no re-extraction needed.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from .single_stream_classifier import SingleStreamClassifier
    from .single_stream_dataset import SingleStreamDataset
except ImportError:
    from single_stream_classifier import SingleStreamClassifier
    from single_stream_dataset import SingleStreamDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_VAL_SPLIT = 0.15


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
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a single-modality deepfake classifier.")
    parser.add_argument("--modality", required=True, choices=["visual", "audio", "semantic"])
    parser.add_argument("--feature-root", required=True, help="Path to that modality's .npy feature directory.")
    parser.add_argument("--input-dim", type=int, required=True, help="Feature vector dimension (1280/768/384).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    dataset = SingleStreamDataset(args.feature_root)
    if len(dataset) == 0:
        logger.error("No samples found under %s - check the feature root path.", args.feature_root)
        return

    val_size = max(int(len(dataset) * args.val_split), 1)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = SingleStreamClassifier(input_dim=args.input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    output_path = Path(args.output_dir) / f"best_{args.modality}_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info("Saved new best model (val_acc=%.4f) to %s", val_acc, output_path)

    logger.info("Training complete. Best val_acc=%.4f, saved at %s", best_val_acc, output_path)


if __name__ == "__main__":
    main()

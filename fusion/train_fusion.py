"""
Train the original 3-modality FusionModel (visual + audio + semantic).

Updated to use the persistent, group-aware split (fusion/data_split.json,
produced by fusion/create_split.py) instead of drawing a fresh random
train_test_split() every run - the same change already made to
fusion/train_enhanced_fusion.py. TRAINING now only ever sees the
"train" split, model selection/checkpointing only ever sees the
"validation" split, and the "test" split is never touched by this
script - so the 3-modal baseline can finally be compared against the
5-modal model under the same strict held-out protocol.

FusionModel itself (fusion/fusion_model.py) is UNCHANGED - same
architecture: visual+audio+semantic -> projection -> cross-attention
-> fusion -> classifier.

IMPORTANT: this does not retroactively fix the existing
fusion/best_fusion_model.pt checkpoint - that file was trained with
the OLD random split and is left in place (not deleted). To get a
3-modal baseline that is genuinely comparable to the 5-modal model
under the shared split, you need to re-run this script; the new
checkpoint will overwrite fusion/best_fusion_model.pt only if/when
validation F1 improves during that run, exactly like before.

Usage:
    python fusion/train_fusion.py \
        --visual-root visual/data/features_aligned --audio-root audio/data/features \
        --semantic-root semantic/data/features
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from fusion_dataset import FusionDataset
from fusion_model import FusionModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_defaults import DEFAULT_AUDIO_ROOT, DEFAULT_SEMANTIC_ROOT, DEFAULT_VISUAL_ROOT  # noqa: E402
from split_utils import DEFAULT_SPLIT_PATH  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the original 3-modality fusion model.")
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT,
                         help="Defaults to $DFD_SEMANTIC_ROOT if set, else semantic/data/features.")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH),
                         help="fusion/data_split.json - run fusion/create_split.py first.")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Max epochs; early stopping (--patience) usually ends training sooner.")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop after this many consecutive epochs with no validation-F1 improvement.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="fusion/best_fusion_model.pt")
    parser.add_argument("--history-output", default="fusion/fusion_training_history.json")
    args = parser.parse_args()

    train_dataset = FusionDataset(
        visual_root=args.visual_root, audio_root=args.audio_root, semantic_root=args.semantic_root,
        split_path=args.split_path, split_name="train",
    )
    val_dataset = FusionDataset(
        visual_root=args.visual_root, audio_root=args.audio_root, semantic_root=args.semantic_root,
        split_path=args.split_path, split_name="validation",
    )
    print(f"\nTraining samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nTrain or validation split is empty. Run `python fusion/create_split.py` first, and "
              "double-check the three feature root paths above.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FusionModel().to(device)

    train_labels = [sample[3] for sample in train_dataset.samples]
    real_count = train_labels.count(0)
    fake_count = train_labels.count(1)
    if real_count == 0 or fake_count == 0:
        print("\nThe training split has only one class present - check create_split.py's group "
              "assignment / the feature roots.")
        sys.exit(1)

    val_labels = [sample[3] for sample in val_dataset.samples]
    val_real_count = val_labels.count(0)
    val_fake_count = val_labels.count(1)
    if val_real_count == 0 or val_fake_count == 0:
        print("\nThe validation split has only one class present - cannot reliably select/checkpoint a "
              "model. This can happen with an old, hand-edited, or otherwise invalid split JSON - the "
              "shared split generator's own checks (fusion/create_split.py) should normally prevent this.")
        sys.exit(1)

    total = real_count + fake_count
    class_weights = torch.tensor(
        [total / (2 * real_count), total / (2 * fake_count)], dtype=torch.float32
    ).to(device)
    print("\nClass weights:", "Real:", class_weights[0].item(), "Fake:", class_weights[1].item())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_output).resolve().parent.mkdir(parents=True, exist_ok=True)

    # Start below any achievable F1 (not 0.0) so the first validation
    # epoch always establishes a checkpoint, even if it scores exactly
    # F1=0.0 (0.0 > 0.0 is False - a 0.0 starting point can finish
    # training with no checkpoint saved at all).
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    stopped_early = False

    def save_history():
        with open(args.history_output, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "epochs_requested": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "split_path": args.split_path,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(val_dataset),
                },
                "history": history,
                "best_val_f1": best_f1,
                "stopped_early": stopped_early,
            }, f, indent=2)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        for batch_idx, (visual, audio, semantic, labels_batch) in enumerate(train_loader):
            visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(visual, audio, semantic)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        average_loss = running_loss / len(train_loader)

        model.eval()
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for visual, audio, semantic, labels_batch in val_loader:
                visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
                logits, _ = model(visual, audio, semantic)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_batch.numpy())

        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, zero_division=0)
        val_recall = recall_score(all_labels, all_predictions, zero_division=0)
        val_f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(f"\nEpoch {epoch + 1}/{args.epochs} Results (validation split - never trained on)")
        print(f"Train loss:    {average_loss:.4f}")
        print(f"Val Accuracy:  {val_accuracy:.4f}")
        print(f"Val Precision: {val_precision:.4f}")
        print(f"Val Recall:    {val_recall:.4f}")
        print(f"Val F1:        {val_f1:.4f}")

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output)
            print(f"\nSaved best fusion model to {args.output} (val F1={val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"\nNo val F1 improvement for {epochs_without_improvement}/{args.patience} epoch(s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "checkpoint_saved": improved,
        })
        save_history()

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no val F1 improvement for {args.patience} consecutive epochs.")
            stopped_early = True
            save_history()
            break

    print("\n================================")
    print("Fusion training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Training history saved to {args.history_output}")
    print("\nThe test split was NOT touched during this run. Evaluate on it separately with:")
    print(f"  python fusion/evaluate_fusion.py --visual-root {args.visual_root} "
          f"--audio-root {args.audio_root} --semantic-root {args.semantic_root} "
          f"--weights {args.output} --split test")
    print("================================")


if __name__ == "__main__":
    main()

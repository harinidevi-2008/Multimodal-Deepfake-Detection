"""
Train the enhanced (5-modality) fusion model.

Mirrors train_fusion.py's conventions (class-weighted loss, stratified
train/val split, F1-based checkpointing) but trains EnhancedFusionModel
on all five modalities via EnhancedFusionDataset. Does not touch or
import train_fusion.py - kept fully separate so the original 3-modality
training path still works standalone.

Edit the five root paths below (or pass them as CLI flags) to point at
wherever visual/audio/semantic/blink/lipsync features actually live on
your machine.
"""

import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from enhanced_fusion_dataset import EnhancedFusionDataset
from enhanced_fusion_model import EnhancedFusionModel


def main():
    parser = argparse.ArgumentParser(description="Train the enhanced 5-modality fusion model.")
    parser.add_argument("--visual-root", default=r"visual/data/features_aligned")
    parser.add_argument("--audio-root", default=r"audio/data/features")
    parser.add_argument("--semantic-root", default=r"C:\Deepfake_Features\semantic_features")
    parser.add_argument("--blink-root", default=r"visual/data/blink_features")
    parser.add_argument("--lipsync-root", default=r"visual/data/lipsync_features")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="fusion/best_enhanced_fusion_model.pt")
    args = parser.parse_args()

    dataset = EnhancedFusionDataset(
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        semantic_root=args.semantic_root,
        blink_root=args.blink_root,
        lipsync_root=args.lipsync_root,
    )
    print(f"\nTotal samples: {len(dataset)}")

    labels = [sample[5] for sample in dataset.samples]
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=labels
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EnhancedFusionModel().to(device)

    train_labels = [labels[i] for i in train_indices]
    real_count = train_labels.count(0)
    fake_count = train_labels.count(1)
    total = real_count + fake_count
    class_weights = torch.tensor(
        [total / (2 * real_count), total / (2 * fake_count)], dtype=torch.float32
    ).to(device)
    print("\nClass weights:", "Real:", class_weights[0].item(), "Fake:", class_weights[1].item())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        for batch_idx, (visual, audio, semantic, blink, lipsync, labels_batch) in enumerate(train_loader):
            visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
            blink, lipsync = blink.to(device), lipsync.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(visual, audio, semantic, blink, lipsync)
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
            for visual, audio, semantic, blink, lipsync, labels_batch in val_loader:
                visual, audio, semantic = visual.to(device), audio.to(device), semantic.to(device)
                blink, lipsync = blink.to(device), lipsync.to(device)

                logits, _ = model(visual, audio, semantic, blink, lipsync)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_batch.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(f"\nEpoch {epoch + 1}/{args.epochs} Results")
        print(f"Loss:      {average_loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.output)
            print(f"\nSaved best enhanced fusion model to {args.output}")

    print("\n================================")
    print("Enhanced fusion training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print("================================")


if __name__ == "__main__":
    main()

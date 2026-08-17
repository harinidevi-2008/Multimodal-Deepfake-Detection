import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


def main():

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    dataset = FusionDataset(
        visual_root=r"visual/data/features_aligned",
        audio_root=r"audio/data/features",
        semantic_root=r"C:\Deepfake_Features\semantic_features"
    )

    print(f"\nTotal samples: {len(dataset)}")

    # --------------------------------------------------
    # 2. Stratified train/validation split
    # --------------------------------------------------

    labels = [sample[3] for sample in dataset.samples]
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --------------------------------------------------
    # 3. DataLoaders
    # --------------------------------------------------

    # Larger batch = fewer batches and faster CPU training
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------
    # 4. Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # 5. Model
    # --------------------------------------------------

    model = FusionModel().to(device)

    # --------------------------------------------------
    # 6. Class-weighted loss
    # --------------------------------------------------

    train_labels = [labels[i] for i in train_indices]

    real_count = train_labels.count(0)
    fake_count = train_labels.count(1)

    total = real_count + fake_count

    real_weight = total / (2 * real_count)
    fake_weight = total / (2 * fake_count)

    class_weights = torch.tensor(
        [real_weight, fake_weight],
        dtype=torch.float32
    ).to(device)

    print("\nClass weights:")
    print("Real:", real_weight)
    print("Fake:", fake_weight)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------
    # 7. Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # --------------------------------------------------
    # 8. Training
    # --------------------------------------------------

    # Start with 3 epochs for the initial fusion test.
    epochs = 3

    best_f1 = 0.0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

        for batch_idx, (
            visual,
            audio,
            semantic,
            labels_batch
        ) in enumerate(train_loader):

            visual = visual.to(device)
            audio = audio.to(device)
            semantic = semantic.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            logits, _ = model(
                visual,
                audio,
                semantic
            )

            loss = criterion(
                logits,
                labels_batch
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}",
                    flush=True
                )

        average_loss = running_loss / len(train_loader)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for visual, audio, semantic, labels_batch in val_loader:

                visual = visual.to(device)
                audio = audio.to(device)
                semantic = semantic.to(device)

                logits, _ = model(
                    visual,
                    audio,
                    semantic
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_labels.extend(
                    labels_batch.numpy()
                )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        precision = precision_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs} Results"
        )

        print(
            f"Loss:      {average_loss:.4f}"
        )

        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1:        {f1:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                "fusion/best_fusion_model.pt"
            )

            print(
                "\nSaved best fusion model."
            )

    print("\n================================")
    print("Fusion training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print("================================")


if __name__ == "__main__":
    main()
from fusion_dataset import FusionDataset


def main():

    dataset = FusionDataset(
        visual_root=r"visual/data/features_aligned",
        audio_root=r"audio/data/features",
        semantic_root=r"C:\Deepfake_Features\semantic_features"
    )

    print("\nDataset length:", len(dataset))

    visual, audio, semantic, label = dataset[0]

    print("\nFirst sample:")
    print("Visual   :", visual.shape)
    print("Audio    :", audio.shape)
    print("Semantic :", semantic.shape)
    print("Label    :", label.item())


if __name__ == "__main__":
    main()
from enhanced_fusion_dataset import EnhancedFusionDataset

dataset = EnhancedFusionDataset(
    visual_root=r"visual/data/features_aligned",
    audio_root=r"audio/data/features",
    semantic_root=r"C:\Deepfake_Features\semantic_features",
    blink_root=r"visual/data/blink_features",
    lipsync_root=r"visual/data/lipsync_features",
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nNumber of returned elements:", len(sample))

visual, audio, semantic, blink, lipsync, label = sample

print("Visual shape:", visual.shape)
print("Audio shape:", audio.shape)
print("Semantic shape:", semantic.shape)
print("Blink shape:", blink.shape)
print("Lip-sync shape:", lipsync.shape)
print("Label:", label)
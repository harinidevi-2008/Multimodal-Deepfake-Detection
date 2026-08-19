"""
Enhanced fusion model: extends the original 3-modality FusionModel to
also attend over eye-blink and lip-sync evidence.

This is a NEW class, not a modification of FusionModel in
fusion_model.py - the original stays untouched and usable. Once this
is trained and evaluated against the original, whichever performs
better (or both, reported side by side) becomes the one actually used.

Blink and lip-sync are projected into the same shared 256-d space as
the other three modalities and treated as two additional tokens in the
cross-attention block (approach 1 from the integration discussion,
rather than concatenating them only at the final decision layer).
Small input dims by design - these are compact statistic vectors, not
deep embeddings:

    blink_dim   -> 4  ([blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score])
    lipsync_dim -> 2  ([sync_score, mismatch_score])

which is exactly what visual/src/blink_lipsync_precompute.py saves.
"""

import torch
import torch.nn as nn


class EnhancedFusionModel(nn.Module):

    def __init__(
        self,
        visual_dim=1280,
        audio_dim=768,
        semantic_dim=384,
        blink_dim=4,
        lipsync_dim=2,
        fusion_dim=256,
        num_heads=8,
        num_classes=2,
    ):
        super().__init__()

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.blink_projection = nn.Sequential(
            nn.Linear(blink_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.lipsync_projection = nn.Sequential(
            nn.Linear(lipsync_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )

        self.norm = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, visual, audio, semantic, blink, lipsync):
        visual = self.visual_projection(visual)
        audio = self.audio_projection(audio)
        semantic = self.semantic_projection(semantic)
        blink = self.blink_projection(blink)
        lipsync = self.lipsync_projection(lipsync)

        # (B, 5, fusion_dim) - order matters only for reading attention
        # weights back out; the model itself is permutation-symmetric.
        modalities = torch.stack([visual, audio, semantic, blink, lipsync], dim=1)

        attended, attention_weights = self.cross_attention(modalities, modalities, modalities)

        fused = self.norm(modalities + attended)
        fused = fused.mean(dim=1)

        logits = self.classifier(fused)
        return logits, attention_weights

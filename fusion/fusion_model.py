import torch
import torch.nn as nn


class FusionModel(nn.Module):

    def __init__(
        self,
        visual_dim=1280,
        audio_dim=768,
        semantic_dim=384,
        fusion_dim=256,
        num_heads=8,
        num_classes=2
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. Projection layers
        # --------------------------------------------------

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # --------------------------------------------------
        # 2. Cross-modal attention
        # --------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # --------------------------------------------------
        # 3. Normalization
        # --------------------------------------------------

        self.norm = nn.LayerNorm(fusion_dim)

        # --------------------------------------------------
        # 4. Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, visual, audio, semantic):

        # --------------------------------------------------
        # Project each modality to 256 dimensions
        # --------------------------------------------------

        visual = self.visual_projection(visual)
        audio = self.audio_projection(audio)
        semantic = self.semantic_projection(semantic)

        # Shapes:
        # visual   = (B, 256)
        # audio    = (B, 256)
        # semantic = (B, 256)

        # --------------------------------------------------
        # Stack the three modalities
        # --------------------------------------------------

        modalities = torch.stack(
            [visual, audio, semantic],
            dim=1
        )

        # Shape:
        # (B, 3, 256)

        # --------------------------------------------------
        # Cross-modal attention
        # --------------------------------------------------

        attended, attention_weights = self.cross_attention(
            modalities,
            modalities,
            modalities
        )

        # Shape:
        # attended = (B, 3, 256)

        # Residual connection + normalization
        fused = self.norm(modalities + attended)

        # --------------------------------------------------
        # Pool across the three modalities
        # --------------------------------------------------

        fused = fused.mean(dim=1)

        # Shape:
        # (B, 256)

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(fused)

        # Shape:
        # (B, 2)

        return logits, attention_weights
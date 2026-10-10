"""
Legacy five-stream model: extends the original 3-modality FusionModel to
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

This model is retained only for old checkpoint inspection and its legacy
regression tests. The production API and active train/evaluate aliases use
fusion_model.FusionModel with visual, audio, and semantic inputs only.
"""

import torch
import torch.nn as nn


MODALITY_ORDER = ["visual", "audio", "semantic", "blink", "lipsync"]

# Parameters that the Innovation Day "gated" variant added to the module.
# They are not part of the documented architecture. Checkpoints saved while
# that code was in place still carry them (untrained, unused in "attention"
# mode), so the loader below drops them instead of failing a strict load.
_LEGACY_GATE_KEY_PREFIXES = ("reliability_gates.", "invalid_retention_logits")


def semantic_validity(semantic, eps=1e-8):
    """
    (B, 1) flag: 1.0 when the semantic vector is not the all-zero sentinel
    that the semantic extractor writes for empty or unreliable transcripts.
    Used for reporting only. It does not change the fusion forward pass.
    """
    if semantic.ndim == 1:
        semantic = semantic.unsqueeze(0)
    valid = torch.amax(torch.abs(semantic), dim=1, keepdim=True) > eps
    return valid.to(dtype=semantic.dtype)


def default_reliability(semantic):
    """
    (B, 5) reporting vector in MODALITY_ORDER. Only the semantic stream has
    a degraded sentinel (the zero vector), so the other entries are 1.0.
    """
    sem_valid = semantic_validity(semantic)
    ones = torch.ones_like(sem_valid)
    return torch.cat([ones, ones, sem_valid, ones, ones], dim=1)


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


def load_enhanced_state_dict(model, state_dict, strict=True):
    """
    Load a checkpoint into the attention-only EnhancedFusionModel.

    Gate parameters left over from the removed gated variant are dropped
    (they were never used by attention checkpoints). A checkpoint of the
    gated architecture itself cannot be loaded, because its token mixing is
    a different computation.
    """
    cleaned = {
        key: value for key, value in state_dict.items()
        if not key.startswith(_LEGACY_GATE_KEY_PREFIXES)
    }
    dropped = sorted(set(state_dict) - set(cleaned))
    result = model.load_state_dict(cleaned, strict=strict)
    return result, dropped


def check_checkpoint_architecture(weights_path):
    """
    Refuse checkpoints that a sidecar *.model_meta.json marks as the removed
    gated architecture. Checkpoints with no sidecar are attention models.
    """
    import json
    from pathlib import Path

    meta_path = Path(str(weights_path) + ".model_meta.json")
    if not meta_path.exists():
        return "attention"
    with open(meta_path, "r", encoding="utf-8") as f:
        architecture = json.load(f).get("fusion_architecture", "attention")
    if architecture != "attention":
        raise ValueError(
            f"{weights_path} was trained with the '{architecture}' fusion architecture, which is not part "
            "of the documented design and is no longer supported. Use an attention checkpoint."
        )
    return architecture

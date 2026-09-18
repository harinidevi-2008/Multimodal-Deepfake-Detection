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


MODALITY_ORDER = ["visual", "audio", "semantic", "blink", "lipsync"]


def semantic_validity(semantic, eps=1e-8):
    """
    Returns a (B, 1) reliability signal for semantic features.

    The existing semantic extractor can produce an all-zero 384-d vector
    when transcript/embedding generation is degraded. That vector is still
    a valid tensor shape, but it is not valid semantic evidence. We preserve
    the raw vector and expose a bounded reliability bit instead of replacing
    it with fabricated content.
    """
    if semantic.ndim == 1:
        semantic = semantic.unsqueeze(0)
    valid = torch.amax(torch.abs(semantic), dim=1, keepdim=True) > eps
    return valid.to(dtype=semantic.dtype)


def default_reliability(semantic):
    """
    Reliability layout matches MODALITY_ORDER.

    Only semantic currently has a reliable missing/degraded sentinel in the
    feature pipeline: an all-zero embedding. The other modalities remain
    available unless callers pass explicit reliability values in the future.
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
        fusion_architecture="attention",
        use_reliability_gates=None,
    ):
        super().__init__()
        if fusion_architecture not in ("attention", "gated"):
            raise ValueError("fusion_architecture must be 'attention' or 'gated'")
        self.fusion_architecture = fusion_architecture
        self.use_reliability_gates = (
            fusion_architecture == "gated" if use_reliability_gates is None else bool(use_reliability_gates)
        )
        self.modality_order = list(MODALITY_ORDER)

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

        # Each gate receives one projected modality token plus that
        # modality's reliability scalar. The learned sigmoid decides how
        # much of the token should reach fusion. For semantic=invalid, an
        # additional trainable retention factor starts low, so the semantic
        # token is reduced but not hard-deleted; training can learn a
        # different retention if the data supports it.
        self.reliability_gates = nn.ModuleList(
            [nn.Linear(fusion_dim + 1, 1) for _ in MODALITY_ORDER]
        )
        for gate in self.reliability_gates:
            nn.init.zeros_(gate.weight[:, -1])
        self.invalid_retention_logits = nn.Parameter(torch.full((len(MODALITY_ORDER),), -2.0))

        self.norm = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _project_modalities(self, visual, audio, semantic, blink, lipsync):
        visual = self.visual_projection(visual)
        audio = self.audio_projection(audio)
        semantic = self.semantic_projection(semantic)
        blink = self.blink_projection(blink)
        lipsync = self.lipsync_projection(lipsync)
        return torch.stack([visual, audio, semantic, blink, lipsync], dim=1)

    def _reliability(self, semantic, reliability):
        if reliability is None:
            reliability = default_reliability(semantic)
        if reliability.ndim == 1:
            reliability = reliability.unsqueeze(0)
        if reliability.shape[1] != len(MODALITY_ORDER):
            raise ValueError(
                f"reliability must have {len(MODALITY_ORDER)} values in order {MODALITY_ORDER}, "
                f"got shape {tuple(reliability.shape)}"
            )
        return reliability.to(device=semantic.device, dtype=semantic.dtype).clamp(0.0, 1.0)

    def _apply_reliability_gates(self, modalities, reliability):
        gate_values = []
        gated_tokens = []
        invalid_retention = torch.sigmoid(self.invalid_retention_logits).view(1, -1, 1)
        reliability_3d = reliability.unsqueeze(-1)
        reliability_factor = reliability_3d + (1.0 - reliability_3d) * invalid_retention

        for index, gate in enumerate(self.reliability_gates):
            token = modalities[:, index, :]
            rel = reliability[:, index:index + 1]
            learned_gate = torch.sigmoid(gate(torch.cat([token, rel], dim=1)))
            effective_gate = learned_gate * reliability_factor[:, index, :]
            gate_values.append(effective_gate)
            gated_tokens.append(token * effective_gate)

        return torch.stack(gated_tokens, dim=1), torch.cat(gate_values, dim=1)

    def forward(self, visual, audio, semantic, blink, lipsync, reliability=None, return_diagnostics=False):
        reliability = self._reliability(semantic, reliability)
        modalities = self._project_modalities(visual, audio, semantic, blink, lipsync)
        if self.use_reliability_gates:
            fused_modalities, gate_values = self._apply_reliability_gates(modalities, reliability)
        else:
            fused_modalities = modalities
            gate_values = torch.ones(
                modalities.shape[0], len(MODALITY_ORDER), device=modalities.device, dtype=modalities.dtype
            )

        # (B, 5, fusion_dim) - order matters only for reading attention
        # weights back out; the model itself is permutation-symmetric.
        if self.fusion_architecture == "attention":
            attended, attention_weights = self.cross_attention(
                fused_modalities, fused_modalities, fused_modalities
            )
            fused = self.norm(fused_modalities + attended)
            fused = fused.mean(dim=1)
        else:
            attention_weights = None
            fused = self.norm(fused_modalities.mean(dim=1))

        logits = self.classifier(fused)
        if return_diagnostics:
            return logits, attention_weights, {
                "reliability": reliability,
                "gate_values": gate_values,
                "modality_order": list(MODALITY_ORDER),
                "fusion_architecture": self.fusion_architecture,
                "use_reliability_gates": self.use_reliability_gates,
            }
        return logits, attention_weights

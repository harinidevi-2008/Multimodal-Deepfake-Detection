"""
Lightweight classifier head for a single modality's frozen features.

Same shape of network the fusion model's projection layers use
(Linear -> LayerNorm -> ReLU), just followed by a small classification
head instead of feeding a cross-attention block. Kept small on purpose:
the heavy lifting was already done by the frozen upstream encoders
(EfficientNet-B0 / Wav2Vec 2.0 / Sentence-BERT); this only needs to
learn a decision boundary over their fixed features.

Input dims by modality (must match what the feature-extraction stream
actually saves):
    visual   -> 1280  (EfficientNet-B0)
    audio    -> 768   (Wav2Vec 2.0)
    semantic -> 384   (Sentence-BERT all-MiniLM-L6-v2)
"""

import torch.nn as nn


class SingleStreamClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

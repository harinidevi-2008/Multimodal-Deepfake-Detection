import torch
import torch.nn as nn

class AudioProjection(nn.Module):

    def __init__(self):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

    def forward(self, x):
        return self.projection(x)
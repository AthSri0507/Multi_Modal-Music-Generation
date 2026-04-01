"""Audio feature extraction network for tabular Spotify audio attributes.

Input: 11 normalized audio features
Output: 256-dim audio feature vector for multimodal fusion
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    """MLP block: 11 -> 64 -> 128 -> 256 with norm/dropout."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: tuple[int, int, int] = (64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        h1, h2, h3 = hidden_dims
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

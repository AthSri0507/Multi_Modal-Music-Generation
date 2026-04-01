"""Text feature extraction network for lyrics embeddings.

Input: 768-dim BERT [CLS] embedding
Output: 256-dim text feature vector for multimodal fusion
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextFeatureExtractor(nn.Module):
    """MLP block: 768 -> 512 -> 256 with norm/dropout."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

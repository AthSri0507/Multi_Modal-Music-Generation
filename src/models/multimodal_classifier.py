"""Multimodal emotion classification model.

This module combines:
- TextFeatureExtractor: 768 -> 256
- AudioFeatureExtractor: 11 -> 256
- Fusion MLP: (256 + 256) -> 512 -> 256 -> 128
- Classification head: 128 -> num_classes logits
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .audio_feature_extractor import AudioFeatureExtractor
from .text_feature_extractor import TextFeatureExtractor


@dataclass
class FusionOutput:
    """Container for forward-pass outputs used in training/debugging."""

    logits: torch.Tensor
    text_features: torch.Tensor
    audio_features: torch.Tensor
    fused_features: torch.Tensor


class FusionLayer(nn.Module):
    """Fusion block: 512 -> 512 -> 256 -> 128 with norm/relu/dropout."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: tuple[int, int, int] = (512, 256, 128),
        dropout: float = 0.4,
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


class EmotionClassifier(nn.Module):
    """Classification head that returns logits for CrossEntropyLoss."""

    def __init__(self, input_dim: int = 128, num_classes: int = 8) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultimodalEmotionClassifier(nn.Module):
    """End-to-end multimodal classifier for emotion prediction."""

    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 11,
        text_feature_dim: int = 256,
        audio_feature_dim: int = 256,
        fusion_hidden_dims: tuple[int, int, int] = (512, 256, 128),
        num_classes: int = 8,
        text_dropout: float = 0.3,
        audio_dropout: float = 0.3,
        fusion_dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.text_branch = TextFeatureExtractor(
            input_dim=text_input_dim,
            hidden_dim=512,
            output_dim=text_feature_dim,
            dropout=text_dropout,
        )
        self.audio_branch = AudioFeatureExtractor(
            input_dim=audio_input_dim,
            hidden_dims=(64, 128, audio_feature_dim),
            dropout=audio_dropout,
        )

        self.fusion_layer = FusionLayer(
            input_dim=text_feature_dim + audio_feature_dim,
            hidden_dims=fusion_hidden_dims,
            dropout=fusion_dropout,
        )
        self.classifier = EmotionClassifier(input_dim=fusion_hidden_dims[-1], num_classes=num_classes)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_features: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | FusionOutput:
        text_z = self.text_branch(text_embeddings)
        audio_z = self.audio_branch(audio_features)
        fused_in = torch.cat([text_z, audio_z], dim=1)
        fused_z = self.fusion_layer(fused_in)
        logits = self.classifier(fused_z)

        if return_dict:
            return FusionOutput(
                logits=logits,
                text_features=text_z,
                audio_features=audio_z,
                fused_features=fused_z,
            )
        return logits
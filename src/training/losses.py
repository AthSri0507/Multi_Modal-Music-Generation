"""Loss functions used by multimodal classifier training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: Focusing parameter.
        alpha: Optional class weighting tensor of shape [num_classes].
        reduction: "mean", "sum", or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(
    name: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory for selecting the training loss."""
    key = name.strip().lower()
    if key in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if key in {"focal", "focal_loss"}:
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss name: {name}")

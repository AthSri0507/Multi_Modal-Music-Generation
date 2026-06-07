"""Conditional GAN architectures for emotion-conditioned music generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(layer) if enabled else layer


class GeneratorBlock(nn.Module):
    """Generator MLP block with batch norm for stable feature scaling."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualGeneratorBlock(nn.Module):
    """Residual MLP block used by deeper generator variants."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DiscriminatorBlock(nn.Module):
    """Discriminator block with optional spectral norm and leaky activation."""

    def __init__(self, in_dim: int, out_dim: int, spectral_norm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(in_dim, out_dim), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class MusicGanOutput:
    """Container for discriminator outputs."""

    adv_logits: torch.Tensor
    emotion_logits: torch.Tensor


class MusicGenerator(nn.Module):
    """Emotion-conditioned generator.

    Input: noise z + learned emotion embedding.
    Output: flattened normalized sequence vector in [-1, 1].
    """

    def __init__(
        self,
        noise_dim: int = 100,
        num_emotions: int = 8,
        emotion_embedding_dim: int = 8,
        control_dim: int = 0,
        control_embedding_dim: int = 16,
        output_dim: int = 480,
        hidden_dims: tuple[int, int, int] = (256, 512, 1024),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.control_dim = int(control_dim)

        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        self.control_projection: nn.Module | None = None
        self.control_embedding_dim = 0
        if self.control_dim > 0:
            self.control_embedding_dim = int(control_embedding_dim)
            self.control_projection = nn.Sequential(
                nn.Linear(self.control_dim, self.control_embedding_dim),
                nn.LayerNorm(self.control_embedding_dim),
                nn.ReLU(inplace=True),
            )

        input_dim = noise_dim + emotion_embedding_dim + self.control_embedding_dim
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            GeneratorBlock(input_dim, h1, dropout=dropout),
            GeneratorBlock(h1, h2, dropout=dropout),
            GeneratorBlock(h2, h3, dropout=dropout),
            nn.Linear(h3, output_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        noise: torch.Tensor,
        emotion_ids: torch.Tensor,
        control_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"Expected noise shape [B, {self.noise_dim}], got {tuple(noise.shape)}")

        cond = self.emotion_embedding(emotion_ids)
        parts = [noise, cond]
        if self.control_projection is not None:
            if control_vector is None:
                control_vector = torch.zeros(noise.shape[0], self.control_dim, device=noise.device, dtype=noise.dtype)
            if control_vector.ndim != 2 or control_vector.shape[1] != self.control_dim:
                raise ValueError(f"Expected control vector [B, {self.control_dim}], got {tuple(control_vector.shape)}")
            parts.append(self.control_projection(control_vector))
        x = torch.cat(parts, dim=1)
        return self.net(x)


class ResidualMusicGenerator(nn.Module):
    """Generator with residual hidden processing for stronger expressivity."""

    def __init__(
        self,
        noise_dim: int = 100,
        num_emotions: int = 8,
        emotion_embedding_dim: int = 8,
        control_dim: int = 0,
        control_embedding_dim: int = 16,
        output_dim: int = 480,
        hidden_dims: tuple[int, int, int] = (256, 512, 1024),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.control_dim = int(control_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        self.control_projection: nn.Module | None = None
        self.control_embedding_dim = 0
        if self.control_dim > 0:
            self.control_embedding_dim = int(control_embedding_dim)
            self.control_projection = nn.Sequential(
                nn.Linear(self.control_dim, self.control_embedding_dim),
                nn.LayerNorm(self.control_embedding_dim),
                nn.ReLU(inplace=True),
            )

        input_dim = noise_dim + emotion_embedding_dim + self.control_embedding_dim
        h1, h2, h3 = hidden_dims
        self.proj = nn.Sequential(
            GeneratorBlock(input_dim, h1, dropout=dropout),
            GeneratorBlock(h1, h2, dropout=dropout),
            GeneratorBlock(h2, h3, dropout=dropout),
        )
        self.res_stack = nn.Sequential(
            ResidualGeneratorBlock(h3, dropout=dropout),
            ResidualGeneratorBlock(h3, dropout=dropout),
        )
        self.out = nn.Sequential(nn.Linear(h3, output_dim), nn.Tanh())

    def forward(
        self,
        noise: torch.Tensor,
        emotion_ids: torch.Tensor,
        control_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"Expected noise shape [B, {self.noise_dim}], got {tuple(noise.shape)}")
        cond = self.emotion_embedding(emotion_ids)
        parts = [noise, cond]
        if self.control_projection is not None:
            if control_vector is None:
                control_vector = torch.zeros(noise.shape[0], self.control_dim, device=noise.device, dtype=noise.dtype)
            if control_vector.ndim != 2 or control_vector.shape[1] != self.control_dim:
                raise ValueError(f"Expected control vector [B, {self.control_dim}], got {tuple(control_vector.shape)}")
            parts.append(self.control_projection(control_vector))
        x = torch.cat(parts, dim=1)
        h = self.proj(x)
        h = self.res_stack(h)
        return self.out(h)


class AttentionMusicGenerator(nn.Module):
    """Generator with token-wise self-attention over sequence features."""

    def __init__(
        self,
        noise_dim: int = 100,
        num_emotions: int = 8,
        emotion_embedding_dim: int = 8,
        control_dim: int = 0,
        control_embedding_dim: int = 16,
        output_dim: int = 480,
        hidden_dims: tuple[int, int, int] = (256, 512, 1024),
        dropout: float = 0.1,
        seq_len: int = 120,
        feature_dim: int = 4,
    ) -> None:
        super().__init__()
        if output_dim != seq_len * feature_dim:
            raise ValueError("For attention generator, output_dim must equal seq_len * feature_dim")

        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.control_dim = int(control_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        self.control_projection: nn.Module | None = None
        self.control_embedding_dim = 0
        if self.control_dim > 0:
            self.control_embedding_dim = int(control_embedding_dim)
            self.control_projection = nn.Sequential(
                nn.Linear(self.control_dim, self.control_embedding_dim),
                nn.LayerNorm(self.control_embedding_dim),
                nn.ReLU(inplace=True),
            )

        input_dim = noise_dim + emotion_embedding_dim + self.control_embedding_dim
        h1, h2, h3 = hidden_dims
        self.base = nn.Sequential(
            GeneratorBlock(input_dim, h1, dropout=dropout),
            GeneratorBlock(h1, h2, dropout=dropout),
            GeneratorBlock(h2, h3, dropout=dropout),
            nn.Linear(h3, output_dim),
        )
        self.token_proj = nn.Linear(feature_dim, 32)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, dropout=dropout, batch_first=True)
        self.token_out = nn.Linear(32, feature_dim)

    def forward(
        self,
        noise: torch.Tensor,
        emotion_ids: torch.Tensor,
        control_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"Expected noise shape [B, {self.noise_dim}], got {tuple(noise.shape)}")
        cond = self.emotion_embedding(emotion_ids)
        parts = [noise, cond]
        if self.control_projection is not None:
            if control_vector is None:
                control_vector = torch.zeros(noise.shape[0], self.control_dim, device=noise.device, dtype=noise.dtype)
            if control_vector.ndim != 2 or control_vector.shape[1] != self.control_dim:
                raise ValueError(f"Expected control vector [B, {self.control_dim}], got {tuple(control_vector.shape)}")
            parts.append(self.control_projection(control_vector))
        x = torch.cat(parts, dim=1)
        flat = self.base(x)

        seq = flat.view(flat.shape[0], self.seq_len, self.feature_dim)
        tokens = self.token_proj(seq)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        seq_out = self.token_out(attn_out)
        return torch.tanh(seq_out.reshape(flat.shape[0], -1))


class MusicDiscriminator(nn.Module):
    """Conditional discriminator with adversarial and auxiliary emotion heads."""

    def __init__(
        self,
        input_dim: int = 480,
        num_emotions: int = 8,
        emotion_embedding_dim: int = 32,
        control_dim: int = 0,
        control_embedding_dim: int = 16,
        hidden_dims: tuple[int, int, int] = (512, 256, 128),
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        self.control_dim = int(control_dim)
        self.control_projection: nn.Module | None = None
        self.control_embedding_dim = 0
        if self.control_dim > 0:
            self.control_embedding_dim = int(control_embedding_dim)
            self.control_projection = nn.Sequential(
                nn.Linear(self.control_dim, self.control_embedding_dim),
                nn.LayerNorm(self.control_embedding_dim),
                nn.ReLU(inplace=True),
            )

        in_dim = input_dim + emotion_embedding_dim + self.control_embedding_dim
        h1, h2, h3 = hidden_dims

        self.backbone = nn.Sequential(
            DiscriminatorBlock(in_dim, h1, spectral_norm=spectral_norm),
            DiscriminatorBlock(h1, h2, spectral_norm=spectral_norm),
            DiscriminatorBlock(h2, h3, spectral_norm=spectral_norm),
        )
        self.adv_head = _maybe_spectral_norm(nn.Linear(h3, 1), spectral_norm)
        self.emotion_head = _maybe_spectral_norm(nn.Linear(h3, num_emotions), spectral_norm)

    def forward(
        self,
        sequence_vectors: torch.Tensor,
        emotion_ids: torch.Tensor,
        control_vector: Optional[torch.Tensor] = None,
    ) -> MusicGanOutput:
        if sequence_vectors.ndim != 2:
            raise ValueError(
                f"Expected sequence vectors rank 2 [B, D], got shape {tuple(sequence_vectors.shape)}"
            )

        cond = self.emotion_embedding(emotion_ids)
        parts = [sequence_vectors, cond]
        if self.control_projection is not None:
            if control_vector is None:
                control_vector = torch.zeros(
                    sequence_vectors.shape[0], self.control_dim, device=sequence_vectors.device, dtype=sequence_vectors.dtype
                )
            if control_vector.ndim != 2 or control_vector.shape[1] != self.control_dim:
                raise ValueError(f"Expected control vector [B, {self.control_dim}], got {tuple(control_vector.shape)}")
            parts.append(self.control_projection(control_vector))
        x = torch.cat(parts, dim=1)
        z = self.backbone(x)
        adv_logits = self.adv_head(z)
        emotion_logits = self.emotion_head(z)
        return MusicGanOutput(adv_logits=adv_logits, emotion_logits=emotion_logits)


def build_generator(
    variant: str,
    noise_dim: int,
    num_emotions: int,
    emotion_embedding_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, int, int],
    dropout: float = 0.1,
    seq_len: int = 120,
    feature_dim: int = 4,
    control_dim: int = 0,
    control_embedding_dim: int = 16,
) -> nn.Module:
    """Factory for generator architecture variants used in A/B experiments."""
    key = variant.strip().lower()
    if key == "mlp":
        return MusicGenerator(
            noise_dim=noise_dim,
            num_emotions=num_emotions,
            emotion_embedding_dim=emotion_embedding_dim,
            control_dim=control_dim,
            control_embedding_dim=control_embedding_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if key == "residual":
        return ResidualMusicGenerator(
            noise_dim=noise_dim,
            num_emotions=num_emotions,
            emotion_embedding_dim=emotion_embedding_dim,
            control_dim=control_dim,
            control_embedding_dim=control_embedding_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if key == "attention":
        return AttentionMusicGenerator(
            noise_dim=noise_dim,
            num_emotions=num_emotions,
            emotion_embedding_dim=emotion_embedding_dim,
            control_dim=control_dim,
            control_embedding_dim=control_embedding_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            seq_len=seq_len,
            feature_dim=feature_dim,
        )
    raise ValueError(f"Unsupported generator variant: {variant}")

"""Dataset utilities for emotion-conditioned music generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MusicLabelEncoder:
    """Bi-directional mapping between emotion label strings and class ids."""

    label_to_id: Dict[str, int]

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.label_to_id.items()}


class EmotionConditionedMusicDataset(Dataset):
    """Dataset for flattened sequence vectors with emotion labels.

    Expected input file format: NPZ with keys
    - vectors: shape [N, vector_dim], float32 in [-1, 1]
    - labels: shape [N], string or bytes labels
    """

    def __init__(
        self,
        npz_path: str | Path,
        label_encoder: Optional[MusicLabelEncoder] = None,
    ) -> None:
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset artifact: {p}")

        data = np.load(p, allow_pickle=True)
        if "vectors" not in data or "labels" not in data:
            raise ValueError("NPZ must contain keys: vectors and labels")

        vectors = data["vectors"].astype(np.float32)
        raw_labels = data["labels"]
        labels = [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in raw_labels
        ]

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be rank-2, got shape {vectors.shape}")
        if len(labels) != vectors.shape[0]:
            raise ValueError("vectors row count must match labels count")

        if label_encoder is None:
            unique = sorted(set(labels))
            label_encoder = MusicLabelEncoder({name: i for i, name in enumerate(unique)})

        self.label_encoder = label_encoder
        self.vectors = vectors
        self.labels = np.asarray([self.label_encoder.label_to_id[name] for name in labels], dtype=np.int64)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.vectors[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def class_counts(self) -> Dict[int, int]:
        ids, counts = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}


def build_music_dataloader(
    dataset: EmotionConditionedMusicDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build basic DataLoader for pre-training sanity checks and training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def save_music_npz(
    out_path: str | Path,
    vectors: np.ndarray,
    labels: Iterable[str],
) -> None:
    """Persist normalized vectors and labels as NPZ artifact."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(list(labels), dtype=object)
    np.savez_compressed(out, vectors=vectors, labels=labels)

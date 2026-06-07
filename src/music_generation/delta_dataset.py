"""Dataset utilities for emotion-conditioned music generation stored in Delta Lake."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from deltalake import DeltaTable
except ImportError:  # pragma: no cover - optional dependency at import time
    DeltaTable = None


@dataclass
class MusicLabelEncoder:
    """Bi-directional mapping between emotion label strings and class ids."""

    label_to_id: Dict[str, int]

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.label_to_id.items()}


def _decode_labels(raw_labels: Iterable[object]) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in raw_labels
    ]


def _build_label_encoder(labels: Iterable[str], label_encoder: Optional[MusicLabelEncoder]) -> MusicLabelEncoder:
    if label_encoder is not None:
        return label_encoder

    unique = sorted(set(labels))
    return MusicLabelEncoder({name: i for i, name in enumerate(unique)})


def _build_storage_options(delta_path: str) -> dict[str, str] | None:
    if not delta_path.lower().startswith("s3://"):
        return None

    options: dict[str, str] = {}
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_ALLOW_HTTP",
    ]:
        value = os.getenv(key)
        if value:
            options[key] = value

    if "AWS_REGION" not in options and "AWS_DEFAULT_REGION" in options:
        options["AWS_REGION"] = options["AWS_DEFAULT_REGION"]

    return options or None


class DeltaEmotionConditionedMusicDataset(Dataset):
    """Dataset for flattened sequence vectors with emotion labels stored in Delta."""

    def __init__(self, delta_path: str | Path, label_encoder: Optional[MusicLabelEncoder] = None) -> None:
        if DeltaTable is None:
            raise ImportError("deltalake is not installed; cannot read Delta tables")

        path_str = str(delta_path)
        if "://" not in path_str:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Missing Delta dataset directory: {path}")

        storage_options = _build_storage_options(path_str)
        table = DeltaTable(path_str, storage_options=storage_options).to_pyarrow_table(
            columns=["vector", "emotion"]
        )
        vectors = np.asarray(table.column("vector").to_pylist(), dtype=np.float32)
        labels = _decode_labels(table.column("emotion").to_pylist())

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be rank-2, got shape {vectors.shape}")
        if len(labels) != vectors.shape[0]:
            raise ValueError("vectors row count must match labels count")

        label_encoder = _build_label_encoder(labels, label_encoder)
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
    dataset: Dataset,
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

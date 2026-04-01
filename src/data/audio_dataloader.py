"""PyTorch dataset and dataloader utilities for normalized audio features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class AudioLabelEncoder:
    """Bi-directional mapping between string labels and class ids."""

    label_to_id: Dict[str, int]

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.label_to_id.items()}


class HDF5AudioFeatureDataset(Dataset):
    """Dataset backed by split groups from audio feature HDF5 file."""

    def __init__(
        self,
        h5_path: str,
        split: str,
        label_encoder: Optional[AudioLabelEncoder] = None,
    ) -> None:
        self.h5_path = h5_path
        self.split = split

        with h5py.File(self.h5_path, "r") as h5f:
            if split not in h5f:
                raise ValueError(f"Split '{split}' not found in {h5_path}")
            grp = h5f[split]
            self.size = int(grp["features"].shape[0])
            has_label = "label" in grp

            if has_label:
                raw_labels = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in grp["label"][:]
                ]
            else:
                raw_labels = []

        if raw_labels and label_encoder is None:
            unique = sorted(set(raw_labels))
            label_encoder = AudioLabelEncoder(label_to_id={label: i for i, label in enumerate(unique)})

        self.label_encoder = label_encoder
        if raw_labels and self.label_encoder is not None:
            self.encoded_labels = np.array(
                [self.label_encoder.label_to_id[label] for label in raw_labels], dtype=np.int64
            )
        else:
            self.encoded_labels = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(self.h5_path, "r") as h5f:
            feats = h5f[self.split]["features"][idx]

        x = torch.tensor(feats, dtype=torch.float32)
        if self.encoded_labels is None:
            return x, None

        y = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return x, y

    def get_class_counts(self) -> Optional[Dict[int, int]]:
        if self.encoded_labels is None:
            return None
        ids, counts = np.unique(self.encoded_labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}


def build_audio_dataloader(
    dataset: HDF5AudioFeatureDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    weighted_sampling: bool = False,
) -> DataLoader:
    """Build a DataLoader with optional weighted class sampling."""
    sampler = None
    effective_shuffle = shuffle

    if weighted_sampling and dataset.encoded_labels is not None:
        class_counts = dataset.get_class_counts()
        if class_counts:
            class_weights = {
                class_id: 1.0 / max(count, 1) for class_id, count in class_counts.items()
            }
            sample_weights = np.array(
                [class_weights[int(label)] for label in dataset.encoded_labels], dtype=np.float32
            )
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            effective_shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

"""Training-time augmentations for flattened MIDI vectors."""

from __future__ import annotations

import torch


def augment_music_batch(
    batch: torch.Tensor,
    pitch_shift_range: int = 0,
    velocity_jitter: float = 0.0,
    augment_prob: float = 0.0,
) -> torch.Tensor:
    """Apply safe pitch transposition / velocity jitter to a batch of vectors.

    The input is a flattened [B, seq_len * 4] tensor in normalized [-1, 1]
    format. Padded rows are preserved because only rows with any nonzero value
    are altered.
    """

    if batch.ndim != 2 or batch.shape[1] % 4 != 0:
        raise ValueError(f"Expected flattened MIDI batch [B, 4k], got {tuple(batch.shape)}")
    if pitch_shift_range <= 0 and velocity_jitter <= 0:
        return batch

    out = batch.clone()
    seq_len = out.shape[1] // 4
    reshaped = out.view(out.shape[0], seq_len, 4)
    valid_mask = reshaped.abs().sum(dim=2) > 1e-6

    if augment_prob <= 0:
        return batch

    selected = torch.nonzero(torch.rand(out.shape[0], device=out.device) < augment_prob, as_tuple=False).flatten()

    if selected.numel() == 0:
        return batch

    for index in selected.tolist():
        sample_mask = valid_mask[index]
        if not bool(sample_mask.any()):
            continue

        if pitch_shift_range > 0:
            shift = int(torch.randint(-pitch_shift_range, pitch_shift_range + 1, (1,), device=out.device).item())
            pitch_norm = reshaped[index, sample_mask, 0]
            pitch_midi = ((pitch_norm + 1.0) * 0.5) * 127.0
            pitch_midi = torch.clamp(pitch_midi + float(shift), 0.0, 127.0)
            reshaped[index, sample_mask, 0] = (pitch_midi / 63.5) - 1.0

        if velocity_jitter > 0:
            jitter = torch.randn_like(reshaped[index, sample_mask, 1]) * float(velocity_jitter)
            reshaped[index, sample_mask, 1] = torch.clamp(reshaped[index, sample_mask, 1] + jitter, -1.0, 1.0)

    return reshaped.reshape_as(out)

"""MIDI representation and conversion utilities.

Representation order per event: [pitch, velocity, duration, delta_time].
- pitch: MIDI note number [0, 127]
- velocity: note velocity [0, 127]
- duration: note duration in beats [0.0, max_duration_beats]
- delta_time: time since previous note in beats [0.0, max_delta_beats]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MidiRepresentationConfig:
    """Configuration for fixed-length flattened MIDI sequence vectors."""

    seq_len: int = 120
    feature_dim: int = 4
    max_duration_beats: float = 8.0
    max_delta_beats: float = 8.0
    pitch_min: int = 0
    pitch_max: int = 127
    velocity_min: int = 0
    velocity_max: int = 127

    @property
    def vector_dim(self) -> int:
        return self.seq_len * self.feature_dim


class MidiSequenceConverter:
    """Convert between event matrices and flattened normalized vectors."""

    def __init__(self, config: MidiRepresentationConfig | None = None) -> None:
        self.config = config or MidiRepresentationConfig()

    def normalize_events(self, events: np.ndarray) -> np.ndarray:
        """Normalize events from natural ranges into [-1, 1]."""
        if events.ndim != 2 or events.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"Expected shape [N, {self.config.feature_dim}], got {tuple(events.shape)}"
            )

        events = events.astype(np.float32).copy()

        # Clamp to valid value ranges before mapping to [-1, 1].
        events[:, 0] = np.clip(events[:, 0], self.config.pitch_min, self.config.pitch_max)
        events[:, 1] = np.clip(events[:, 1], self.config.velocity_min, self.config.velocity_max)
        events[:, 2] = np.clip(events[:, 2], 0.0, self.config.max_duration_beats)
        events[:, 3] = np.clip(events[:, 3], 0.0, self.config.max_delta_beats)

        events[:, 0] = 2.0 * (events[:, 0] - self.config.pitch_min) / (
            self.config.pitch_max - self.config.pitch_min
        ) - 1.0
        events[:, 1] = 2.0 * (events[:, 1] - self.config.velocity_min) / (
            self.config.velocity_max - self.config.velocity_min
        ) - 1.0
        events[:, 2] = 2.0 * events[:, 2] / self.config.max_duration_beats - 1.0
        events[:, 3] = 2.0 * events[:, 3] / self.config.max_delta_beats - 1.0
        return events

    def denormalize_events(self, normalized_events: np.ndarray) -> np.ndarray:
        """Convert normalized [-1, 1] events back to natural MIDI ranges."""
        if normalized_events.ndim != 2 or normalized_events.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"Expected shape [N, {self.config.feature_dim}], got {tuple(normalized_events.shape)}"
            )

        x = np.clip(normalized_events.astype(np.float32), -1.0, 1.0).copy()

        x[:, 0] = ((x[:, 0] + 1.0) * 0.5) * (self.config.pitch_max - self.config.pitch_min) + self.config.pitch_min
        x[:, 1] = ((x[:, 1] + 1.0) * 0.5) * (self.config.velocity_max - self.config.velocity_min) + self.config.velocity_min
        x[:, 2] = ((x[:, 2] + 1.0) * 0.5) * self.config.max_duration_beats
        x[:, 3] = ((x[:, 3] + 1.0) * 0.5) * self.config.max_delta_beats

        x[:, 0] = np.rint(x[:, 0])
        x[:, 1] = np.rint(x[:, 1])
        return x

    def events_to_vector(self, events: np.ndarray) -> np.ndarray:
        """Pad/truncate event matrix and flatten into a fixed-size vector."""
        normalized = self.normalize_events(events)
        out = np.zeros((self.config.seq_len, self.config.feature_dim), dtype=np.float32)
        n = min(self.config.seq_len, normalized.shape[0])
        if n > 0:
            out[:n, :] = normalized[:n, :]
        return out.reshape(-1)

    def vector_to_events(self, vector: np.ndarray, remove_padding: bool = True) -> np.ndarray:
        """Unflatten a vector back to event matrix and optionally strip zero-padded rows."""
        if vector.ndim != 1 or vector.shape[0] != self.config.vector_dim:
            raise ValueError(f"Expected shape [{self.config.vector_dim}], got {tuple(vector.shape)}")

        mat = vector.reshape(self.config.seq_len, self.config.feature_dim)
        if remove_padding:
            nonzero_mask = np.any(np.abs(mat) > 1e-6, axis=1)
            mat = mat[nonzero_mask]
            if mat.size == 0:
                return np.zeros((0, self.config.feature_dim), dtype=np.float32)
        return self.denormalize_events(mat)

    def events_iterable_to_array(self, events: Iterable[Iterable[float]]) -> np.ndarray:
        """Convert generic iterable of note events into shape [N, 4] float array."""
        arr = np.asarray(list(events), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"Expected iterable of [pitch, velocity, duration, delta_time] rows; got {arr.shape}"
            )
        return arr

    def midi_file_to_events(self, midi_path: str | Path) -> np.ndarray:
        """Parse a MIDI file into event rows.

        Requires pretty_midi. This is optional and only needed when preparing real MIDI data.
        """
        try:
            import pretty_midi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pretty_midi is required for MIDI parsing. Install with: pip install pretty_midi"
            ) from exc

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        events: list[list[float]] = []

        # Use first non-drum instrument to keep representation deterministic.
        instrument = None
        for inst in pm.instruments:
            if not inst.is_drum:
                instrument = inst
                break
        if instrument is None:
            return np.zeros((0, self.config.feature_dim), dtype=np.float32)

        notes = sorted(instrument.notes, key=lambda n: n.start)
        prev_start = 0.0
        beat_times = pm.get_beats()
        if len(beat_times) >= 2:
            sec_per_beat = float(np.median(np.diff(beat_times)))
        else:
            sec_per_beat = 0.5  # fallback for malformed files

        for note in notes:
            duration_beats = max(0.0, (note.end - note.start) / sec_per_beat)
            delta_beats = max(0.0, (note.start - prev_start) / sec_per_beat)
            events.append([
                float(note.pitch),
                float(note.velocity),
                float(duration_beats),
                float(delta_beats),
            ])
            prev_start = note.start

        return np.asarray(events, dtype=np.float32)

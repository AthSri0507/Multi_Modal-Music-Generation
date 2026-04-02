"""Post-processing utilities for converting generated vectors to MIDI files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .midi_representation import MidiRepresentationConfig, MidiSequenceConverter


@dataclass
class MidiPostprocessConfig:
    """Rules for cleaning generated note events before MIDI export."""

    quantize_step_beats: float = 0.25
    min_duration_beats: float = 0.125
    max_duration_beats: float = 8.0
    min_velocity: int = 1
    max_velocity: int = 127
    max_notes: int = 120
    remove_duplicate_notes: bool = True


class MidiPostprocessor:
    """Converts normalized vectors to clean note events and MIDI files."""

    def __init__(
        self,
        repr_config: MidiRepresentationConfig | None = None,
        post_config: MidiPostprocessConfig | None = None,
    ) -> None:
        self.repr_config = repr_config or MidiRepresentationConfig()
        self.post_config = post_config or MidiPostprocessConfig(
            max_duration_beats=self.repr_config.max_duration_beats,
            max_notes=self.repr_config.seq_len,
        )
        self.converter = MidiSequenceConverter(self.repr_config)

    def vector_to_clean_events(self, vector: np.ndarray) -> np.ndarray:
        """Convert generated vector to cleaned events in natural MIDI ranges."""
        events = self.converter.vector_to_events(vector, remove_padding=True)
        if events.size == 0:
            return np.zeros((0, self.repr_config.feature_dim), dtype=np.float32)

        # Clamp and quantize.
        pitch = np.clip(np.rint(events[:, 0]), self.repr_config.pitch_min, self.repr_config.pitch_max)
        velocity = np.clip(
            np.rint(events[:, 1]),
            self.post_config.min_velocity,
            self.post_config.max_velocity,
        )
        duration = np.clip(
            events[:, 2],
            self.post_config.min_duration_beats,
            self.post_config.max_duration_beats,
        )
        delta = np.clip(events[:, 3], 0.0, self.repr_config.max_delta_beats)

        q = max(1e-6, self.post_config.quantize_step_beats)
        duration = np.maximum(self.post_config.min_duration_beats, np.round(duration / q) * q)
        delta = np.round(delta / q) * q

        cleaned = np.stack([pitch, velocity, duration, delta], axis=1).astype(np.float32)

        # Remove duplicate events that share same quantized pitch/start/duration tuple.
        if self.post_config.remove_duplicate_notes and cleaned.shape[0] > 1:
            starts = np.cumsum(cleaned[:, 3])
            keys = np.stack([cleaned[:, 0], starts, cleaned[:, 2]], axis=1)
            _, uniq_idx = np.unique(keys, axis=0, return_index=True)
            cleaned = cleaned[np.sort(uniq_idx)]

        # Normalize sequence length.
        if cleaned.shape[0] > self.post_config.max_notes:
            cleaned = cleaned[: self.post_config.max_notes]

        return cleaned

    def write_midi(self, events: np.ndarray, out_path: str | Path, tempo_bpm: float = 120.0) -> None:
        """Write cleaned events to MIDI file (pretty_midi backend)."""
        try:
            import pretty_midi  # type: ignore
        except ImportError as exc:
            raise ImportError("pretty_midi is required for MIDI export") from exc

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        pm = pretty_midi.PrettyMIDI(initial_tempo=float(max(30.0, tempo_bpm)))
        instrument = pretty_midi.Instrument(program=0)

        sec_per_beat = 60.0 / float(max(1e-6, tempo_bpm))
        current_time = 0.0
        for row in events:
            pitch = int(np.clip(round(row[0]), 0, 127))
            velocity = int(np.clip(round(row[1]), 1, 127))
            duration_beats = float(max(self.post_config.min_duration_beats, row[2]))
            delta_beats = float(max(0.0, row[3]))

            current_time += delta_beats * sec_per_beat
            end_time = current_time + duration_beats * sec_per_beat
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=current_time, end=end_time)
            instrument.notes.append(note)

        pm.instruments.append(instrument)
        pm.write(str(out))

    def vectors_to_midi_batch(
        self,
        vectors: np.ndarray,
        out_dir: str | Path,
        file_prefix: str,
        tempo_bpm: float = 120.0,
    ) -> list[str]:
        """Convert a batch of vectors to MIDI files and return file paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: list[str] = []
        for i, vec in enumerate(vectors):
            events = self.vector_to_clean_events(vec)
            midi_path = out / f"{file_prefix}_{i:04d}.mid"
            self.write_midi(events, midi_path, tempo_bpm=tempo_bpm)
            paths.append(str(midi_path.as_posix()))
        return paths

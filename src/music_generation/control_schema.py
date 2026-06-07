"""Structured control schema for prompt-to-music conditioning.

The schema keeps the product-facing interface stable even when the underlying
training data only provides emotion labels. The generator can consume a fixed
control vector composed of emotion, style, instrumentation, tempo, and
intensity signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch


DEFAULT_STYLE_LABELS: tuple[str, ...] = ("generic", "jazz", "ambient", "cinematic", "piano", "electronic")
DEFAULT_INSTRUMENTATION_LABELS: tuple[str, ...] = (
    "ensemble",
    "jazz_combo",
    "solo_piano",
    "strings",
    "synth_pad",
)


@dataclass(frozen=True)
class MusicControlSpec:
    """High-level music controls used by the generator."""

    emotion: str
    style: str = "generic"
    tempo_bpm: float = 120.0
    intensity: float = 0.5
    instrumentation: str = "ensemble"


class MusicControlSchema:
    """Encode prompt-friendly controls into a fixed numeric vector."""

    def __init__(
        self,
        emotion_to_id: Dict[str, int],
        style_labels: Sequence[str] = DEFAULT_STYLE_LABELS,
        instrumentation_labels: Sequence[str] = DEFAULT_INSTRUMENTATION_LABELS,
        tempo_min: float = 40.0,
        tempo_max: float = 180.0,
    ) -> None:
        self.emotion_to_id = dict(emotion_to_id)
        self.id_to_emotion = {idx: label for label, idx in self.emotion_to_id.items()}
        self.style_labels = tuple(style_labels)
        self.instrumentation_labels = tuple(instrumentation_labels)
        self.tempo_min = float(tempo_min)
        self.tempo_max = float(tempo_max)
        self._default_by_emotion = self._build_default_presets()

    @classmethod
    def from_label_encoder(
        cls,
        emotion_to_id: Dict[str, int],
        style_labels: Sequence[str] = DEFAULT_STYLE_LABELS,
        instrumentation_labels: Sequence[str] = DEFAULT_INSTRUMENTATION_LABELS,
    ) -> "MusicControlSchema":
        return cls(
            emotion_to_id=emotion_to_id,
            style_labels=style_labels,
            instrumentation_labels=instrumentation_labels,
        )

    def _build_default_presets(self) -> Dict[str, MusicControlSpec]:
        presets: Dict[str, MusicControlSpec] = {}
        for emotion in self.emotion_to_id:
            key = emotion.strip().lower()
            if key in {"q1", "happy", "joyful", "energetic"}:
                presets[emotion] = MusicControlSpec(
                    emotion=emotion,
                    style="jazz",
                    tempo_bpm=132.0,
                    intensity=0.78,
                    instrumentation="jazz_combo",
                )
            elif key in {"q2", "angry", "tense", "aggressive"}:
                presets[emotion] = MusicControlSpec(
                    emotion=emotion,
                    style="cinematic",
                    tempo_bpm=126.0,
                    intensity=0.86,
                    instrumentation="ensemble",
                )
            elif key in {"q3", "sad", "melancholy", "dark"}:
                presets[emotion] = MusicControlSpec(
                    emotion=emotion,
                    style="ambient",
                    tempo_bpm=72.0,
                    intensity=0.30,
                    instrumentation="strings",
                )
            else:
                presets[emotion] = MusicControlSpec(
                    emotion=emotion,
                    style="piano",
                    tempo_bpm=92.0,
                    intensity=0.38,
                    instrumentation="solo_piano",
                )
        return presets

    @property
    def emotion_dim(self) -> int:
        return len(self.emotion_to_id)

    @property
    def style_dim(self) -> int:
        return len(self.style_labels)

    @property
    def instrumentation_dim(self) -> int:
        return len(self.instrumentation_labels)

    @property
    def control_dim(self) -> int:
        return self.emotion_dim + self.style_dim + self.instrumentation_dim + 2

    def default_spec(self, emotion: str) -> MusicControlSpec:
        key = emotion if emotion in self._default_by_emotion else emotion.strip().lower()
        if key in self._default_by_emotion:
            return self._default_by_emotion[key]
        if emotion in self.emotion_to_id:
            return MusicControlSpec(emotion=emotion)
        raise KeyError(f"Unknown emotion label: {emotion}")

    def _one_hot(self, label: str, labels: Sequence[str]) -> np.ndarray:
        vec = np.zeros(len(labels), dtype=np.float32)
        key = label.strip().lower()
        if key not in labels:
            return vec
        vec[labels.index(key)] = 1.0
        return vec

    def encode_spec(self, spec: MusicControlSpec) -> np.ndarray:
        emotion_vec = np.zeros(self.emotion_dim, dtype=np.float32)
        emotion_key = spec.emotion if spec.emotion in self.emotion_to_id else spec.emotion.strip().lower()
        if emotion_key not in self.emotion_to_id:
            raise KeyError(f"Unknown emotion label: {spec.emotion}")
        emotion_vec[self.emotion_to_id[emotion_key]] = 1.0

        style_vec = self._one_hot(spec.style, self.style_labels)
        instr_vec = self._one_hot(spec.instrumentation, self.instrumentation_labels)

        tempo_norm = np.float32((float(spec.tempo_bpm) - self.tempo_min) / max(1e-6, self.tempo_max - self.tempo_min))
        tempo_norm = np.clip(tempo_norm, 0.0, 1.0)
        intensity_norm = np.clip(np.float32(spec.intensity), 0.0, 1.0)

        return np.concatenate([emotion_vec, style_vec, instr_vec, np.asarray([tempo_norm, intensity_norm], dtype=np.float32)])

    def encode_controls(self, specs: Iterable[MusicControlSpec]) -> np.ndarray:
        return np.stack([self.encode_spec(spec) for spec in specs], axis=0)

    def batch_from_emotions(self, emotion_ids: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        if emotion_ids.ndim != 1:
            raise ValueError(f"Expected 1D emotion ids tensor, got shape {tuple(emotion_ids.shape)}")

        specs = []
        for emotion_id in emotion_ids.detach().cpu().tolist():
            emotion = self.id_to_emotion[int(emotion_id)]
            specs.append(self.default_spec(emotion))
        arr = self.encode_controls(specs)
        tensor = torch.tensor(arr, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def spec_to_prompt(self, spec: MusicControlSpec) -> str:
        return f"{spec.emotion} {spec.style} {spec.instrumentation} tempo={spec.tempo_bpm:.0f} intensity={spec.intensity:.2f}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "emotion_to_id": dict(self.emotion_to_id),
            "style_labels": list(self.style_labels),
            "instrumentation_labels": list(self.instrumentation_labels),
            "tempo_min": self.tempo_min,
            "tempo_max": self.tempo_max,
            "default_presets": {name: asdict(spec) for name, spec in self._default_by_emotion.items()},
        }

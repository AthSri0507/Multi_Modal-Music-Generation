"""Generation presets: bundles of prompt-enhancement bias + sampling settings.

Each preset influences both *what* prompt we build (genre bias / extra descriptors)
and *how* MusicGen samples (guidance, temperature, top-k/p, candidate count). Higher
candidate counts + guidance trade CPU time for quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.config.config import (
    MUSICGEN_DEFAULT_PRESET,
    MUSICGEN_MAX_DURATION_S,
)


@dataclass(frozen=True)
class GenerationPreset:
    name: str
    display: str
    guidance_scale: float
    temperature: float
    top_k: int
    top_p: float
    num_candidates: int      # multi-candidate: render N, keep best
    max_duration_s: float
    quantize: bool           # int8 dynamic quant (CPU speed; minor quality loss)
    genre_bias: Optional[str] = None   # nudge genre when the prompt names none
    extra_descriptor: Optional[str] = None
    description: str = ""


PRESETS: Dict[str, GenerationPreset] = {
    "fast": GenerationPreset(
        name="fast", display="Fast",
        guidance_scale=1.0, temperature=1.0, top_k=250, top_p=0.0,
        num_candidates=1, max_duration_s=20.0, quantize=True,
        description="Quickest; single take, no classifier-free guidance. Lowest fidelity.",
    ),
    "balanced": GenerationPreset(
        name="balanced", display="Balanced",
        guidance_scale=3.0, temperature=1.0, top_k=250, top_p=0.0,
        num_candidates=3, max_duration_s=30.0, quantize=False,
        description="Full guidance, 3 candidates, best of. Recommended default.",
    ),
    "cinematic": GenerationPreset(
        name="cinematic", display="Cinematic",
        guidance_scale=4.0, temperature=0.95, top_k=200, top_p=0.0,
        num_candidates=3, max_duration_s=30.0, quantize=False,
        genre_bias="cinematic", extra_descriptor="epic, widescreen and emotive",
        description="Strong guidance + orchestral bias for film-score style.",
    ),
    "experimental": GenerationPreset(
        name="experimental", display="Experimental",
        guidance_scale=3.0, temperature=1.2, top_k=320, top_p=0.0,
        num_candidates=4, max_duration_s=30.0, quantize=False,
        extra_descriptor="unconventional textures and bold sound design",
        description="Higher temperature + more candidates for surprising results.",
    ),
}


def get_preset(name: Optional[str]) -> GenerationPreset:
    """Return a preset by name, falling back to the configured default."""
    key = (name or MUSICGEN_DEFAULT_PRESET).strip().lower()
    return PRESETS.get(key, PRESETS.get(MUSICGEN_DEFAULT_PRESET, PRESETS["balanced"]))


def clamp_duration_for_preset(duration_s: float, preset: GenerationPreset) -> float:
    return float(min(preset.max_duration_s, min(MUSICGEN_MAX_DURATION_S, duration_s)))

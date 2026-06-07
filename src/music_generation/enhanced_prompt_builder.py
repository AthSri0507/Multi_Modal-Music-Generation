"""Compose rich MusicGen prompts from a structured MusicRequest + taxonomy.

Turns the analysed attributes (mood, genre, instrumentation, purpose, atmosphere,
energy, complexity, tempo, dynamics) into a single descriptive instrumental prompt,
e.g.:

    "uplifting jazz featuring piano, upright bass and brushed drums, warm smoky club
     atmosphere, medium tempo around 120 BPM, expressive swinging dynamics, richly
     interplaying, instrumental only, no vocals"
"""

from __future__ import annotations

from typing import List, Optional

from src.music_generation.prompt_builder import _tempo_word  # reuse tempo wording
from src.music_generation.prompt_understanding import MusicRequest
from src.music_generation.taxonomy import Taxonomy, get_taxonomy

# Short lead adjective per mood (kept brief so the lead reads "uplifting jazz").
_MOOD_ADJ = {
    "happy": "uplifting",
    "calm": "calm",
    "sad": "melancholic",
    "tense": "dark, intense",
    "romantic": "romantic",
    "neutral": "",
}

_COMPLEXITY_PHRASE = {
    "minimal": "sparse and uncluttered",
    "rich": "richly layered",
    "moderate": "",
}

_DYNAMICS_BY_INTENSITY = (
    (0.33, "gentle, soft dynamics"),
    (0.66, "balanced dynamics"),
    (1.01, "powerful, intense dynamics"),
)


def _dynamics_word(intensity: float) -> str:
    for hi, word in _DYNAMICS_BY_INTENSITY:
        if intensity < hi:
            return word
    return "balanced dynamics"


def _join(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def build(request: MusicRequest, taxonomy: Optional[Taxonomy] = None) -> str:
    """Build a rich instrumental MusicGen prompt from a MusicRequest."""
    taxo = taxonomy or get_taxonomy()
    g = taxo.genre(request.genre)

    genre_display = g.display if g else request.genre
    mood_adj = _MOOD_ADJ.get(request.mood, "")
    lead = " ".join(p for p in (mood_adj, genre_display) if p).strip()

    parts: List[str] = [lead]

    # Purpose modifiers (e.g. "low distraction, steady" for study).
    if request.purpose and (ps := taxo.purpose(request.purpose)):
        parts.append(", ".join(ps.modifiers[:2]))

    # Instruments.
    instruments = _join(request.instrumentation[:4])
    if instruments:
        parts.append(f"featuring {instruments}")

    # Atmosphere.
    if request.atmosphere:
        parts.append(f"{request.atmosphere} atmosphere")

    # Tempo.
    parts.append(f"{_tempo_word(request.tempo_bpm)} tempo around {int(round(request.tempo_bpm))} BPM")

    # Dynamics + complexity + one genre flavour descriptor.
    parts.append(_dynamics_word(request.intensity))
    cx = _COMPLEXITY_PHRASE.get(request.complexity, "")
    if cx:
        parts.append(cx)
    if g and g.descriptors:
        parts.append(g.descriptors[0])

    prompt = ", ".join(p for p in parts if p)
    prompt += ". Instrumental only, no vocals, no lyrics, no singing."
    return prompt

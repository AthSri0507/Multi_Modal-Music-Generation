"""Rich prompt understanding: text -> structured MusicRequest.

Extends the old mood/tempo/intensity extraction with genre, purpose,
instrumentation, energy, atmosphere, complexity and duration, grounded by the
music-style taxonomy. Deterministic + dependency-light (keyword matching + reuse of
the keyword helpers in :mod:`prompt_to_control`); no BERT download required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.music_generation.mood_map import detect_mood, quadrant_to_mood
from src.music_generation.prompt_builder import parse_duration_seconds
from src.music_generation.prompt_to_control import PromptToControlMapper
from src.music_generation.taxonomy import Taxonomy, get_taxonomy

# Instruments we can recognise by name in a prompt (canonical -> keywords).
_INSTRUMENTS: dict[str, tuple[str, ...]] = {
    "solo piano": ("solo piano", "piano solo", "just piano"),
    "electric piano": ("electric piano", "rhodes"),
    "piano": ("piano",),
    "acoustic guitar": ("acoustic guitar", "nylon guitar"),
    "electric guitar": ("electric guitar", "guitar riff"),
    "guitar": ("guitar",),
    "violin": ("violin",),
    "cello": ("cello",),
    "string section": ("strings", "string section", "orchestral strings"),
    "saxophone": ("saxophone", "sax"),
    "trumpet": ("trumpet",),
    "brass": ("brass", "horns", "french horn"),
    "flute": ("flute",),
    "drums": ("drums", "percussion", "drum kit"),
    "synth pads": ("synth pad", "pads", "warm pads"),
    "synthesizer": ("synth", "synthesizer"),
    "upright bass": ("upright bass", "double bass"),
    "bass": ("bass",),
    "harp": ("harp",),
    "choir": ("choir", "choral"),
}

# mood -> a sensible default genre when the prompt names no style.
_MOOD_DEFAULT_GENRE = {
    "tense": "cinematic",
    "calm": "ambient",
    "sad": "piano",
    "happy": "acoustic",
    "romantic": "classical",
    "neutral": "acoustic",
}
# purpose -> default genre (takes priority over mood default).
_PURPOSE_DEFAULT_GENRE = {
    "study": "lofi",
    "meditation": "ambient",
    "sleep": "ambient",
    "relaxation": "ambient",
    "workout": "electronic",
    "gaming": "cinematic",
    "party": "electronic",
}

_ENERGY_BY_MOOD = {"tense": "high", "happy": "high", "calm": "low", "sad": "low", "romantic": "medium", "neutral": "medium"}
_ENERGY_INTENSITY = {"low": 0.3, "medium": 0.55, "high": 0.82}


@dataclass
class MusicRequest:
    """Structured understanding of a user's music prompt."""

    raw_prompt: str
    mood: str = "neutral"
    genre: str = "acoustic"
    genre_explicit: bool = False  # True if the prompt named the genre/style
    purpose: Optional[str] = None
    instrumentation: List[str] = field(default_factory=list)
    energy: str = "medium"          # low | medium | high
    atmosphere: str = ""
    complexity: str = "moderate"    # minimal | moderate | rich
    tempo_bpm: float = 110.0
    intensity: float = 0.55
    duration_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "raw_prompt": self.raw_prompt,
            "mood": self.mood,
            "genre": self.genre,
            "purpose": self.purpose,
            "instrumentation": list(self.instrumentation),
            "energy": self.energy,
            "atmosphere": self.atmosphere,
            "complexity": self.complexity,
            "tempo_bpm": self.tempo_bpm,
            "intensity": self.intensity,
            "duration_s": self.duration_s,
        }


def _detect_instruments(prompt: str) -> List[str]:
    p = prompt.lower()
    found: List[str] = []
    for canonical, keywords in _INSTRUMENTS.items():
        if any(kw in p for kw in keywords):
            found.append(canonical)
    # Collapse redundant generic vs specific (e.g. drop bare "guitar"/"piano"/"bass"
    # when a more specific variant matched).
    def _drop_if_specific(generic: str, specifics: tuple[str, ...]) -> None:
        if generic in found and any(s in found for s in specifics):
            found.remove(generic)

    _drop_if_specific("guitar", ("acoustic guitar", "electric guitar"))
    _drop_if_specific("piano", ("solo piano", "electric piano"))
    _drop_if_specific("bass", ("upright bass",))
    return found


def _detect_complexity(prompt: str, taxo: Taxonomy, genre: str, purpose: Optional[str]) -> str:
    p = prompt.lower()
    if any(w in p for w in ("minimal", "simple", "sparse", "stripped")):
        return "minimal"
    if any(w in p for w in ("complex", "layered", "rich", "lush", "epic", "full")):
        return "rich"
    if purpose and (ps := taxo.purpose(purpose)):
        return ps.complexity
    if genre in ("orchestral", "cinematic"):
        return "rich"
    if genre in ("ambient", "piano", "lofi"):
        return "minimal"
    return "moderate"


def _detect_atmosphere(prompt: str, taxo: Taxonomy, genre: str, mood: str) -> str:
    p = prompt.lower()
    explicit = [a for a in taxo.atmospheres if a in p]
    if explicit:
        return ", ".join(explicit[:2])
    g = taxo.genre(genre)
    if g and g.atmosphere:
        return g.atmosphere
    return {
        "tense": "dark and dramatic",
        "calm": "serene and spacious",
        "sad": "melancholic and intimate",
        "happy": "bright and warm",
        "romantic": "warm and tender",
    }.get(mood, "balanced")


def analyze_prompt(prompt: str, taxonomy: Optional[Taxonomy] = None) -> MusicRequest:
    """Parse a free-text prompt into a structured MusicRequest."""
    taxo = taxonomy or get_taxonomy()
    M = PromptToControlMapper  # static keyword helpers only

    # Mood.
    mood = detect_mood(prompt)
    if mood is None:
        scores = M._keyword_bonus(prompt, "emotion")
        quad = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        mood = quadrant_to_mood(quad) if quad and scores.get(quad, 0) > 0 else "neutral"

    # Purpose + genre (purpose default genre wins over mood default).
    purpose = taxo.detect_purpose(prompt)
    detected_genre = taxo.detect_genre(prompt)
    genre_explicit = detected_genre is not None
    genre = detected_genre or (
        _PURPOSE_DEFAULT_GENRE.get(purpose or "")
        or _MOOD_DEFAULT_GENRE.get(mood, "acoustic")
    )

    # Instrumentation: explicit names, else the genre's template.
    instruments = _detect_instruments(prompt)
    if not instruments:
        g = taxo.genre(genre)
        instruments = list(g.instruments) if g else ["piano", "strings"]

    # Energy.
    if purpose and (ps := taxo.purpose(purpose)):
        energy = ps.energy
    else:
        energy = _ENERGY_BY_MOOD.get(mood, "medium")
    pl = prompt.lower()
    if any(w in pl for w in ("energetic", "intense", "powerful", "driving", "high energy", "hype")):
        energy = "high"
    elif any(w in pl for w in ("gentle", "soft", "mellow", "calm", "quiet", "low energy")):
        energy = "low"

    # Atmosphere + complexity.
    atmosphere = _detect_atmosphere(prompt, taxo, genre, mood)
    complexity = _detect_complexity(prompt, taxo, genre, purpose)

    # Tempo: explicit BPM > tempo words > genre range midpoint.
    tempo = M._extract_bpm(prompt) or M._keyword_tempo(prompt)
    if tempo is None:
        g = taxo.genre(genre)
        tempo = float(sum(g.tempo_bpm) / 2) if g else 110.0

    # Intensity (dynamics): explicit > energy-derived.
    intensity = M._keyword_intensity(prompt)
    if intensity is None:
        intensity = _ENERGY_INTENSITY.get(energy, 0.55)

    duration = parse_duration_seconds(prompt)

    return MusicRequest(
        raw_prompt=prompt,
        mood=mood,
        genre=genre,
        genre_explicit=genre_explicit,
        purpose=purpose,
        instrumentation=instruments,
        energy=energy,
        atmosphere=atmosphere,
        complexity=complexity,
        tempo_bpm=float(tempo),
        intensity=float(intensity),
        duration_s=duration,
    )

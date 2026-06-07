"""Build rich MusicGen text prompts from extracted controls.

This module turns a :class:`MusicControlSpec` (produced by
:class:`~src.music_generation.prompt_to_control.PromptToControlMapper`) into a
single descriptive sentence that a text-to-music model (MusicGen) can render as
instrumental audio.

The descriptors are grounded by data mined from the Spotify corpus via Spark/Hive
(see ``scripts/bd_build_emotion_profiles.py``):

* ``emotion_music_profiles.json`` -- per-emotion tempo/energy/valence/acousticness
  statistics that refine the wording.
* ``emotion_instrumentation.json`` -- per-emotion ranked instrument palette.

Both files are optional: if absent, curated fallbacks keep the builder working out
of the box (e.g. before the Spark jobs have been run).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.config.config import (
    EMOTION_INSTRUMENTATION_JSON,
    EMOTION_PROFILES_JSON,
    MUSICGEN_DEFAULT_DURATION_S,
    MUSICGEN_MAX_DURATION_S,
    MUSICGEN_MIN_DURATION_S,
)
from src.music_generation.control_schema import MusicControlSpec


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
}


def parse_duration_seconds(text: str) -> Optional[float]:
    """Extract a requested duration in seconds from free text.

    Handles forms like "30 sec", "30 seconds", "90s", "1 min", "2 minutes",
    "1.5 minutes", and "a minute and a half". Returns ``None`` if no duration is
    mentioned. The value is NOT clamped here -- use :func:`resolve_duration`.
    """
    if not text:
        return None
    p = text.lower()

    # "a minute and a half" / "one and a half minutes"
    if re.search(r"\bminute(?:s)?\s+and\s+a\s+half\b", p) or re.search(
        r"\band\s+a\s+half\s+minute", p
    ):
        return 90.0

    # Minutes: "1 min", "2 minutes", "1.5 min", "two minutes"
    m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min|m)\b", p)
    if m:
        return float(m.group("val")) * 60.0
    for word, val in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:minutes|minute|mins|min)\b", p):
            return val * 60.0

    # Seconds: "30 sec", "30 seconds", "90s", "45 secs"
    m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec|s)\b", p)
    if m:
        return float(m.group("val"))

    return None


def resolve_duration(
    explicit: Optional[float],
    prompt_text: str = "",
    default: float = MUSICGEN_DEFAULT_DURATION_S,
    min_s: float = MUSICGEN_MIN_DURATION_S,
    max_s: float = MUSICGEN_MAX_DURATION_S,
) -> float:
    """Resolve the final duration: explicit > parsed-from-text > default.

    The result is clamped to ``[min_s, max_s]``.
    """
    chosen: Optional[float] = None
    if explicit is not None:
        chosen = float(explicit)
    if chosen is None:
        chosen = parse_duration_seconds(prompt_text)
    if chosen is None:
        chosen = float(default)
    return float(min(max_s, max(min_s, chosen)))


# ---------------------------------------------------------------------------
# Curated fallbacks (used when the Spark/Hive JSON artifacts are absent)
# ---------------------------------------------------------------------------

# Map both the Spotify word labels and the GAN quadrant labels to a mood family.
_FALLBACK_PALETTE: Dict[str, List[str]] = {
    "joy": ["bright acoustic guitar", "piano", "light drums", "warm brass"],
    "happy": ["bright acoustic guitar", "piano", "light drums", "warm brass"],
    "q1": ["bright acoustic guitar", "piano", "light drums"],
    "love": ["warm piano", "soft strings", "acoustic guitar"],
    "calm": ["soft piano", "warm strings", "ambient pads"],
    "q4": ["soft piano", "warm strings", "ambient pads"],
    "neutral": ["piano", "strings", "soft synth pad"],
    "surprise": ["pizzicato strings", "marimba", "playful piano"],
    "sadness": ["soft piano", "cello", "warm strings"],
    "sad": ["soft piano", "cello", "warm strings"],
    "q3": ["soft piano", "cello", "warm strings"],
    "romantic": ["warm piano", "soft strings", "nylon guitar"],
    "tense": ["staccato strings", "low brass", "timpani", "taiko drums"],
    "anger": ["distorted synth", "driving drums", "low brass"],
    "q2": ["cinematic strings", "timpani", "low brass"],
    "fear": ["dark ambient pads", "tense strings", "low drone"],
    "disgust": ["dissonant strings", "muted brass"],
}

# Style keyword -> instrument palette. When the prompt names a style (e.g.
# "cinematic"), it overrides the data-mined pop palette, which otherwise pushes
# orchestral requests toward electric piano / synth / drum kit.
_STYLE_PALETTE: Dict[str, List[str]] = {
    "cinematic": ["epic orchestral strings", "low brass", "timpani", "french horns"],
    "ambient": ["warm synth pads", "soft drones", "gentle piano"],
    "jazz": ["upright bass", "brushed drums", "piano", "muted trumpet"],
    "piano": ["expressive solo piano"],
    "electronic": ["analog synths", "electronic drums", "bass synth"],
}

_FALLBACK_MOOD: Dict[str, str] = {
    "joy": "bright, uplifting and warm",
    "happy": "bright, uplifting and warm",
    "q1": "bright and energetic",
    "love": "tender, warm and romantic",
    "calm": "calm, soothing and gentle",
    "q4": "calm, soothing and peaceful",
    "neutral": "balanced and easygoing",
    "surprise": "playful and whimsical",
    "sadness": "melancholic, somber and reflective",
    "sad": "melancholic, somber and reflective",
    "q3": "melancholic and introspective",
    "romantic": "tender, warm and romantic",
    "tense": "dark, dramatic and intense",
    "anger": "intense, aggressive and driving",
    "q2": "dramatic, tense and powerful",
    "fear": "dark, suspenseful and uneasy",
    "disgust": "uneasy and dissonant",
}


def _norm_unit(value: float) -> float:
    """Coerce a feature to the 0..1 range (handles 0-100 percentage scales)."""
    v = float(value)
    if v > 1.5:  # likely a 0-100 percentage
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _tempo_word(bpm: float) -> str:
    if bpm < 70:
        return "very slow"
    if bpm < 95:
        return "slow"
    if bpm < 120:
        return "moderate"
    if bpm < 145:
        return "upbeat"
    return "fast"


def _dynamics_word(intensity: float) -> str:
    if intensity < 0.33:
        return "gentle, soft dynamics"
    if intensity < 0.66:
        return "moderate dynamics"
    return "powerful, intense dynamics"


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


class MusicPromptBuilder:
    """Compose instrumental text-to-music prompts from controls + mined data."""

    def __init__(
        self,
        profiles_path: Path = EMOTION_PROFILES_JSON,
        instrumentation_path: Path = EMOTION_INSTRUMENTATION_JSON,
    ) -> None:
        self.profiles: Dict[str, dict] = _load_json(Path(profiles_path))
        self.instrumentation: Dict[str, List[str]] = _load_json(Path(instrumentation_path))

    # -- palette / mood lookups (data-driven, with curated fallback) --------

    def _palette_for(self, emotion: str) -> List[str]:
        key = emotion.strip().lower()
        if key in self.instrumentation and self.instrumentation[key]:
            return list(self.instrumentation[key])
        return _FALLBACK_PALETTE.get(key, ["piano", "strings", "soft synth pad"])

    def _mood_for(self, emotion: str) -> str:
        # Prefer the hand-tuned per-mood lexicon for adjectives (more musically
        # meaningful than valence buckets); mined data drives tempo / instruments /
        # texture instead. Fall back to a valence-derived phrase for unknown keys.
        key = emotion.strip().lower()
        if key in _FALLBACK_MOOD:
            return _FALLBACK_MOOD[key]
        valence = self.profiles.get(key, {}).get("valence", self.profiles.get(key, {}).get("positiveness"))
        if valence is not None:
            v = _norm_unit(valence)
            if v >= 0.6:
                return "bright, uplifting and warm"
            if v <= 0.4:
                return "melancholic, somber and reflective"
            return "balanced and expressive"
        return "expressive"

    def _texture_for(self, emotion: str) -> Optional[str]:
        """Acoustic vs electronic hint from mined acousticness, if available."""
        profile = self.profiles.get(emotion.strip().lower(), {})
        acoustic = profile.get("acousticness")
        if acoustic is None:
            return None
        return "acoustic" if _norm_unit(acoustic) >= 0.5 else "electronic"

    @staticmethod
    def _user_named_instrument(spec: MusicControlSpec) -> bool:
        """True when the prompt explicitly requested a non-default instrumentation."""
        return bool(spec.instrumentation) and spec.instrumentation.strip().lower() not in {
            "ensemble",
            "",
            "generic",
        }

    @staticmethod
    def _instrumentation_phrase(instrumentation: str) -> str:
        return {
            "solo_piano": "solo piano",
            "jazz_combo": "jazz combo with piano, upright bass and brushed drums",
            "strings": "string ensemble with violin and cello",
            "synth_pad": "lush synth pads",
            "guitar_trio": "acoustic guitar trio",
        }.get(instrumentation.strip().lower(), instrumentation.replace("_", " "))

    # -- main entry point ----------------------------------------------------

    def build(self, spec: MusicControlSpec, raw_prompt: str = "") -> str:
        """Return a single instrumental text-to-music prompt string."""
        mood = self._mood_for(spec.emotion)
        tempo = _tempo_word(spec.tempo_bpm)
        dynamics = _dynamics_word(spec.intensity)

        # Instrument selection priority:
        #   1. an instrument the user explicitly named,
        #   2. a named style's palette (e.g. cinematic -> orchestral),
        #   3. the data-mined per-mood palette.
        style_key = (spec.style or "").strip().lower()
        if self._user_named_instrument(spec):
            instruments = self._instrumentation_phrase(spec.instrumentation)
        elif style_key in _STYLE_PALETTE:
            palette = list(_STYLE_PALETTE[style_key])
            # For tense + cinematic, lean into percussion that suits the mood.
            if spec.emotion.strip().lower() == "tense" and "taiko drums" not in palette:
                palette.append("taiko drums")
            instruments = ", ".join(palette[:4])
        else:
            palette = self._palette_for(spec.emotion)
            instruments = ", ".join(palette[:4])

        parts: List[str] = []
        parts.append(f"A {mood} instrumental piece")
        if spec.style and spec.style.strip().lower() not in {"generic", ""}:
            parts.append(f"in a {spec.style.replace('_', ' ')} style")

        # The instrument palette already conveys timbre (acoustic vs electric), so we
        # only add an explicit texture word when the user named a bare instrumentation.
        if self._user_named_instrument(spec):
            texture = self._texture_for(spec.emotion)
            parts.append(f"featuring {texture + ' ' if texture else ''}{instruments}")
        else:
            parts.append(f"featuring {instruments}")

        parts.append(f"{tempo} tempo around {int(round(spec.tempo_bpm))} BPM")
        parts.append(dynamics)

        prompt = ", ".join(parts)
        # Hard constraint: instrumental only.
        prompt += ". Instrumental only, no vocals, no lyrics, no singing."
        return prompt

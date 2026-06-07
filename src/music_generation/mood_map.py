"""Canonical 'mood' vocabulary shared across extraction, mining, and prompting.

The pipeline has three emotion vocabularies that must agree:

* the prompt extractor outputs Russell quadrants ``q1..q4``;
* the Spotify dataset is labelled with words (joy, sadness, anger, ...);
* the prompt builder + mined profiles need one stable key set.

This module defines that stable key set (MOODS) and the two mappings into it, so
the data mined by Spark/Hive (keyed by mood) lines up with what the extractor
produces at generation time.
"""

from __future__ import annotations

# Canonical mood keys. The prompt builder has descriptors + curated palettes for
# each of these (see prompt_builder._FALLBACK_*).
MOODS = ("happy", "calm", "sad", "tense", "romantic", "neutral")

# Russell quadrant -> mood. q4 = low-arousal / positive = calm.
QUADRANT_TO_MOOD = {
    "q1": "happy",
    "q2": "tense",
    "q3": "sad",
    "q4": "calm",
}

# Spotify dataset word labels -> mood.
SPOTIFY_EMOTION_TO_MOOD = {
    "joy": "happy",
    "happy": "happy",
    "surprise": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "tense",
    "angry": "tense",
    "fear": "tense",
    "disgust": "tense",
    "love": "romantic",
    "neutral": "neutral",
}


# Direct mood keywords (aligned to the canonical vocab). Checked before the coarse
# Russell-quadrant fallback so e.g. "romantic" isn't collapsed into calm/neutral.
MOOD_KEYWORDS = {
    "romantic": ("romantic", "romance", "love", "tender", "intimate", "sensual", "wedding", "heartfelt"),
    "tense": (
        "tense", "epic", "battle", "intense", "dramatic", "suspense", "suspenseful",
        "aggressive", "dark", "angry", "anger", "scary", "horror", "thriller", "action", "war", "menacing",
    ),
    "sad": ("sad", "melancholy", "melancholic", "sorrow", "grief", "lonely", "heartbreak", "somber", "mournful", "tragic"),
    "calm": ("calm", "soothing", "peaceful", "relax", "relaxing", "serene", "gentle", "ambient", "sleep", "meditation", "chill", "mellow", "dreamy"),
    "happy": ("happy", "joyful", "joy", "upbeat", "cheerful", "bright", "uplifting", "fun", "sunny", "playful", "festive", "energetic"),
}


def detect_mood(prompt: str) -> str | None:
    """Pick a mood from explicit keywords (highest hit count wins), else None."""
    p = (prompt or "").lower()
    best: str | None = None
    best_n = 0
    for mood, words in MOOD_KEYWORDS.items():
        n = sum(1 for w in words if w in p)
        if n > best_n:
            best, best_n = mood, n
    return best


def quadrant_to_mood(emotion: str) -> str:
    """Map an extractor emotion (q1..q4 or already a mood) to a mood key."""
    key = (emotion or "").strip().lower()
    if key in MOODS:
        return key
    return QUADRANT_TO_MOOD.get(key, "neutral")


def spotify_to_mood(emotion: str) -> str:
    """Map a Spotify word label to a mood key."""
    return SPOTIFY_EMOTION_TO_MOOD.get((emotion or "").strip().lower(), "neutral")

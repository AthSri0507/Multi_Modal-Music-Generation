"""Music-style taxonomy: genres, instrumentation templates, atmospheres, purposes.

This is the structured catalog that turns vague prompts ("study music", "epic
cinematic battle") into concrete, differentiated MusicGen prompts. The canonical
catalog lives here as Python defaults; ``scripts/bd_build_taxonomy.py`` materializes
it into Hive tables and exports ``data/processed/music_taxonomy.json``. At runtime
:func:`get_taxonomy` overlays that JSON (when present) on these defaults, so the
catalog can be curated through the big-data stack without code changes — and still
works with no Spark at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from src.config.config import MUSIC_TAXONOMY_JSON


@dataclass(frozen=True)
class GenreSpec:
    name: str
    display: str
    instruments: Tuple[str, ...]
    atmosphere: str
    tempo_bpm: Tuple[int, int]
    dynamics: str
    descriptors: Tuple[str, ...]
    keywords: Tuple[str, ...]


@dataclass(frozen=True)
class PurposeSpec:
    name: str
    display: str
    modifiers: Tuple[str, ...]
    keywords: Tuple[str, ...]
    energy: str  # low | medium | high
    complexity: str  # minimal | moderate | rich


# ---------------------------------------------------------------------------
# Canonical catalog (source of truth)
# ---------------------------------------------------------------------------

_GENRES: Tuple[GenreSpec, ...] = (
    GenreSpec(
        "lofi", "lo-fi hip-hop",
        ("mellow electric piano", "boom-bap drums", "warm sub bass", "vinyl crackle"),
        "warm, nostalgic and hazy", (60, 90), "soft, relaxed dynamics",
        ("chill", "dusty", "laid-back", "downtempo"),
        ("lofi", "lo-fi", "lo fi", "lofi hip hop", "chillhop", "study beats", "chill beats"),
    ),
    GenreSpec(
        "cinematic", "cinematic film score",
        ("epic orchestral strings", "low brass", "timpani", "french horns", "taiko drums"),
        "epic, dramatic and sweeping", (90, 140), "powerful, building dynamics",
        ("trailer", "heroic", "soaring", "orchestral"),
        ("cinematic", "film score", "soundtrack", "trailer", "epic", "battle", "movie", "score"),
    ),
    GenreSpec(
        "orchestral", "classical orchestra",
        ("full string section", "woodwinds", "brass", "orchestral percussion", "harp"),
        "grand and refined", (70, 130), "wide orchestral dynamics",
        ("symphonic", "majestic", "elegant"),
        ("orchestral", "orchestra", "symphony", "symphonic", "classical orchestra"),
    ),
    GenreSpec(
        "jazz", "jazz",
        ("piano", "upright bass", "brushed drums", "muted trumpet", "saxophone"),
        "warm, smoky club", (90, 160), "expressive, swinging dynamics",
        ("swing", "improvisation", "bluesy", "syncopated"),
        ("jazz", "swing", "bebop", "trio", "saxophone", "sax", "blues"),
    ),
    GenreSpec(
        "ambient", "ambient",
        ("lush synth pads", "evolving drones", "soft textures", "distant piano"),
        "ethereal, spacious and weightless", (50, 80), "gentle, sustained dynamics",
        ("atmospheric", "dreamy", "floating", "minimal"),
        ("ambient", "drone", "atmospheric", "soundscape", "pads", "textural"),
    ),
    GenreSpec(
        "classical", "classical",
        ("solo piano", "string quartet", "chamber strings"),
        "intimate and expressive", (60, 120), "nuanced classical dynamics",
        ("baroque", "romantic", "graceful"),
        ("classical", "baroque", "romantic era", "chamber", "string quartet", "sonata"),
    ),
    GenreSpec(
        "electronic", "electronic",
        ("analog synths", "punchy electronic drums", "bass synth", "arpeggios"),
        "energetic and modern", (110, 140), "tight, driving dynamics",
        ("synth", "pulsing", "club"),
        ("electronic", "edm", "techno", "house", "synth", "dance", "club"),
    ),
    GenreSpec(
        "synthwave", "synthwave / retrowave",
        ("retro analog synths", "gated drums", "fm bass", "neon arpeggios"),
        "neon-lit, retro-futuristic", (80, 118), "steady, nostalgic dynamics",
        ("retro", "80s", "outrun", "neon"),
        ("synthwave", "retrowave", "outrun", "80s", "vaporwave", "retro synth"),
    ),
    GenreSpec(
        "rock", "rock",
        ("electric guitars", "driving drums", "bass guitar"),
        "raw and energetic", (110, 160), "loud, punchy dynamics",
        ("distorted", "anthemic", "gritty"),
        ("rock", "guitar riff", "metal", "punk", "grunge"),
    ),
    GenreSpec(
        "acoustic", "acoustic / folk",
        ("acoustic guitar", "soft strings", "light percussion", "piano"),
        "warm and organic", (80, 120), "gentle, natural dynamics",
        ("folk", "earthy", "heartfelt"),
        ("acoustic", "folk", "singer-songwriter", "unplugged"),
    ),
    GenreSpec(
        "piano", "solo piano",
        ("expressive solo piano",),
        "intimate and reflective", (60, 110), "delicate, expressive dynamics",
        ("minimalist", "emotive", "contemplative"),
        ("solo piano", "piano only", "just piano", "piano solo"),
    ),
    GenreSpec(
        "world", "world / cinematic ethnic",
        ("ethnic flutes", "hand percussion", "strings", "plucked instruments"),
        "exotic and evocative", (80, 130), "dynamic, rhythmic",
        ("ethnic", "tribal", "folkloric"),
        ("world music", "ethnic", "tribal", "celtic", "oriental", "middle eastern"),
    ),
)

_PURPOSES: Tuple[PurposeSpec, ...] = (
    PurposeSpec(
        "study", "study / focus",
        ("low distraction", "steady and repetitive", "no sudden changes", "background-friendly"),
        ("study", "focus", "concentration", "work", "reading", "coding"),
        "low", "minimal",
    ),
    PurposeSpec(
        "meditation", "meditation",
        ("minimal percussion", "very slow", "long sustained notes", "calming"),
        ("meditation", "meditate", "mindfulness", "yoga", "zen", "spa"),
        "low", "minimal",
    ),
    PurposeSpec(
        "sleep", "sleep",
        ("extremely gentle", "no percussion", "soft and slow", "soothing"),
        ("sleep", "bedtime", "lullaby", "insomnia", "deep sleep"),
        "low", "minimal",
    ),
    PurposeSpec(
        "relaxation", "relaxation",
        ("soothing", "unhurried", "warm and pleasant"),
        ("relax", "relaxation", "calm down", "unwind", "chill out"),
        "low", "moderate",
    ),
    PurposeSpec(
        "workout", "workout / energy",
        ("high energy", "strong steady beat", "motivating", "driving"),
        ("workout", "gym", "running", "exercise", "training", "cardio", "hype"),
        "high", "moderate",
    ),
    PurposeSpec(
        "gaming", "gaming",
        ("immersive", "looping", "adventurous"),
        ("gaming", "game", "video game", "rpg", "boss fight", "8-bit"),
        "high", "rich",
    ),
    PurposeSpec(
        "background", "background",
        ("unobtrusive", "pleasant", "easy listening"),
        ("background", "ambience", "cafe", "lobby", "waiting"),
        "low", "moderate",
    ),
    PurposeSpec(
        "party", "party",
        ("upbeat", "danceable", "fun"),
        ("party", "celebration", "dance floor", "festive"),
        "high", "rich",
    ),
)

# Free atmosphere descriptors used to enrich prompts.
_ATMOSPHERES: Tuple[str, ...] = (
    "warm", "dark", "dreamy", "ethereal", "nostalgic", "epic", "intimate", "tense",
    "bright", "melancholic", "mysterious", "uplifting", "serene", "triumphant",
)


# ---------------------------------------------------------------------------
# Taxonomy container + loader
# ---------------------------------------------------------------------------


@dataclass
class Taxonomy:
    genres: Dict[str, GenreSpec] = field(default_factory=dict)
    purposes: Dict[str, PurposeSpec] = field(default_factory=dict)
    atmospheres: Tuple[str, ...] = ()

    def genre(self, name: Optional[str]) -> Optional[GenreSpec]:
        return self.genres.get((name or "").strip().lower())

    def purpose(self, name: Optional[str]) -> Optional[PurposeSpec]:
        return self.purposes.get((name or "").strip().lower())

    def detect_genre(self, text: str) -> Optional[str]:
        return _best_keyword_match(text, {g.name: g.keywords for g in self.genres.values()})

    def detect_purpose(self, text: str) -> Optional[str]:
        return _best_keyword_match(text, {p.name: p.keywords for p in self.purposes.values()})

    def to_dict(self) -> dict:
        return {
            "genres": {k: asdict(v) for k, v in self.genres.items()},
            "purposes": {k: asdict(v) for k, v in self.purposes.items()},
            "atmospheres": list(self.atmospheres),
        }


def _best_keyword_match(text: str, table: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    p = (text or "").lower()
    best: Optional[str] = None
    best_n = 0
    for name, keywords in table.items():
        n = sum(1 for kw in keywords if kw in p)
        if n > best_n:
            best, best_n = name, n
    return best


def _default_taxonomy() -> Taxonomy:
    return Taxonomy(
        genres={g.name: g for g in _GENRES},
        purposes={p.name: p for p in _PURPOSES},
        atmospheres=_ATMOSPHERES,
    )


def _overlay_json(base: Taxonomy, path) -> Taxonomy:
    """Overlay a Hive-exported JSON catalog onto the defaults (if present/valid)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return base

    genres = dict(base.genres)
    for name, g in data.get("genres", {}).items():
        try:
            genres[name] = GenreSpec(
                name=g["name"],
                display=g.get("display", g["name"]),
                instruments=tuple(g["instruments"]),
                atmosphere=g.get("atmosphere", ""),
                tempo_bpm=tuple(g.get("tempo_bpm", (90, 120))),  # type: ignore[arg-type]
                dynamics=g.get("dynamics", "moderate dynamics"),
                descriptors=tuple(g.get("descriptors", ())),
                keywords=tuple(g.get("keywords", ())),
            )
        except (KeyError, TypeError):
            continue

    purposes = dict(base.purposes)
    for name, p in data.get("purposes", {}).items():
        try:
            purposes[name] = PurposeSpec(
                name=p["name"],
                display=p.get("display", p["name"]),
                modifiers=tuple(p["modifiers"]),
                keywords=tuple(p.get("keywords", ())),
                energy=p.get("energy", "medium"),
                complexity=p.get("complexity", "moderate"),
            )
        except (KeyError, TypeError):
            continue

    atmospheres = tuple(data.get("atmospheres", base.atmospheres)) or base.atmospheres
    return Taxonomy(genres=genres, purposes=purposes, atmospheres=atmospheres)


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    """Return the catalog (defaults overlaid with the Hive-exported JSON if any)."""
    return _overlay_json(_default_taxonomy(), MUSIC_TAXONOMY_JSON)


def default_taxonomy_dict() -> dict:
    """Canonical catalog as plain dicts (used by the Spark/Hive export job)."""
    return _default_taxonomy().to_dict()


# Convenience module-level accessors.
def list_genres() -> List[str]:
    return list(get_taxonomy().genres.keys())


def list_purposes() -> List[str]:
    return list(get_taxonomy().purposes.keys())

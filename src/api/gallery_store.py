"""SQLite-backed gallery store for browsable generations.

Holds one row of metadata per published generation; the audio itself lives on disk
under ``GENERATED_AUDIO_DIR`` and is served by the API. Kept deliberately small and
dependency-free (stdlib ``sqlite3``) so it runs unchanged locally and on a single
Hugging Face Space.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from src.config.config import GALLERY_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    raw_prompt    TEXT NOT NULL,
    music_prompt  TEXT NOT NULL,
    mood          TEXT,
    style         TEXT,
    instrumentation TEXT,
    tempo_bpm     REAL,
    intensity     REAL,
    duration_s    REAL,
    sample_rate   INTEGER,
    audio_filename TEXT NOT NULL,
    latency_s     REAL,
    source        TEXT,
    genre         TEXT,
    purpose       TEXT,
    preset        TEXT,
    score         REAL
);
CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generations_mood ON generations(mood);
"""

# Columns added after the first release; applied to pre-existing DBs in-place.
_MIGRATIONS = (("genre", "TEXT"), ("purpose", "TEXT"), ("preset", "TEXT"), ("score", "REAL"))


class GalleryStore:
    """Thin SQLite wrapper for gallery metadata."""

    def __init__(self, db_path: Path = GALLERY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(generations)")}
            for col, sql_type in _MIGRATIONS:
                if col not in existing:
                    conn.execute(f"ALTER TABLE generations ADD COLUMN {col} {sql_type}")
            # Genre index created after the column is guaranteed to exist.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_genre ON generations(genre)")

    # -- writes --------------------------------------------------------------

    def add(
        self,
        *,
        raw_prompt: str,
        music_prompt: str,
        mood: Optional[str],
        style: Optional[str],
        instrumentation: Optional[str],
        tempo_bpm: Optional[float],
        intensity: Optional[float],
        duration_s: Optional[float],
        sample_rate: Optional[int],
        audio_filename: str,
        latency_s: Optional[float],
        source: str = "api",
        genre: Optional[str] = None,
        purpose: Optional[str] = None,
        preset: Optional[str] = None,
        score: Optional[float] = None,
    ) -> str:
        gen_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO generations (
                    id, created_at, raw_prompt, music_prompt, mood, style,
                    instrumentation, tempo_bpm, intensity, duration_s, sample_rate,
                    audio_filename, latency_s, source, genre, purpose, preset, score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    gen_id,
                    time.time(),
                    raw_prompt,
                    music_prompt,
                    mood,
                    style,
                    instrumentation,
                    tempo_bpm,
                    intensity,
                    duration_s,
                    sample_rate,
                    audio_filename,
                    latency_s,
                    source,
                    genre,
                    purpose,
                    preset,
                    score,
                ),
            )
        return gen_id

    # -- reads ---------------------------------------------------------------

    def list(self, limit: int = 24, offset: int = 0, mood: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM generations"
        params: list = []
        if mood:
            query += " WHERE mood = ?"
            params.append(mood)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get(self, gen_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
        return dict(row) if row else None

    def count(self, mood: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS n FROM generations"
        params: list = []
        if mood:
            query += " WHERE mood = ?"
            params.append(mood)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()["n"])

    def moods(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT mood, COUNT(*) AS n FROM generations "
                "WHERE mood IS NOT NULL GROUP BY mood ORDER BY n DESC"
            ).fetchall()
        return [r["mood"] for r in rows]

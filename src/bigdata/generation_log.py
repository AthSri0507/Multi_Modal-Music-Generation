"""Generation event log: a durable JSONL landing zone for analytics.

The CLI, FastAPI endpoint, and Spark batch job all append one flat record per
generation here (cheap, no JVM startup). ``scripts/bd_generation_analytics.py``
later materializes these records into the Hive table ``music.generation_logs`` and
runs Spark SQL analytics over them.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from src.config.config import GENERATION_LOG_JSONL
from src.music_generation.text_to_music import GenerationResult

_LOCK = threading.Lock()


def build_record(
    result: GenerationResult,
    output_path: str,
    latency_s: float,
    source: str = "cli",
) -> dict:
    """Flatten a GenerationResult into a log record (Hive/JSON friendly)."""
    spec = result.spec
    return {
        "ts": time.time(),
        "source": source,
        "raw_prompt": result.raw_prompt,
        "music_prompt": result.music_prompt,
        "mood": spec.emotion,
        "style": spec.style,
        "instrumentation": spec.instrumentation,
        "tempo_bpm": float(spec.tempo_bpm),
        "intensity": float(spec.intensity),
        "duration_s": float(result.duration_s),
        "sample_rate": int(result.audio.sample_rate),
        "output_path": output_path,
        "latency_s": float(latency_s),
    }


def append_log(record: dict, path: Path = GENERATION_LOG_JSONL) -> None:
    """Append one record as a JSON line (thread-safe, best-effort)."""
    path = Path(path)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_generation(
    result: GenerationResult,
    output_path: str,
    latency_s: float,
    source: str = "cli",
    path: Path = GENERATION_LOG_JSONL,
) -> Optional[dict]:
    """Build and append a generation record. Returns the record, or None on error."""
    try:
        record = build_record(result, output_path, latency_s, source=source)
        append_log(record, path=path)
        return record
    except OSError:
        return None

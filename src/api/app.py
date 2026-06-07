"""FastAPI backend for text-to-music generation + a browsable gallery.

Endpoints
---------
* ``GET  /api/v1/health``          -- liveness + device/model info.
* ``POST /api/v1/emotion/detect``  -- extract controls from a prompt (no audio).
* ``POST /api/v1/music/generate``  -- render audio, persist it, add to the gallery.
* ``GET  /api/v1/gallery``         -- list generations (newest first, mood filter).
* ``GET  /api/v1/gallery/{id}``    -- one generation's metadata.
* ``GET  /api/v1/audio/{id}``      -- stream a generation's WAV.
* ``GET  /api/v1/moods``           -- distinct moods present in the gallery.

If a built frontend exists at ``frontend/dist`` it is served at ``/`` so the whole
app (UI + API + model) deploys unchanged as a single Hugging Face Docker Space.

Run:
    uvicorn src.api.app:app --reload          # API only (use Vite dev server for UI)
    # or build the frontend first, then this also serves the UI at /.

Set ``MUSICGEN_FAST=1`` for faster (lower-fidelity) CPU generation. On a GPU host
the model uses CUDA automatically.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.gallery_store import GalleryStore
from src.bigdata.generation_log import log_generation
from src.config.config import (
    GENERATED_AUDIO_DIR,
    MUSICGEN_DEFAULT_DURATION_S,
    MUSICGEN_MODEL_NAME,
    PROJECT_ROOT,
)
from src.music_generation.presets import PRESETS
from src.music_generation.prompt_builder import resolve_duration
from src.music_generation.prompt_understanding import analyze_prompt
from src.music_generation.text_to_music import TextToMusicPipeline, keyword_extract_spec

app = FastAPI(title="Emotion-to-Music Studio", version="2.0.0")

# CORS so the Vite dev server (http://localhost:5173) can call the API in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared pipeline; the MusicGen model loads lazily and is reused. The web
# app defaults to QUALITY (full classifier-free guidance); callers opt into faster,
# lower-fidelity generation per request. MUSICGEN_FAST=1 makes fast the default.
_DEFAULT_FAST = os.environ.get("MUSICGEN_FAST") == "1"
_pipeline = TextToMusicPipeline(use_bert=False, fast=False)
_gallery = GalleryStore()

GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Free-text description of the desired music")
    duration: Optional[float] = Field(None, description="Length in seconds; overrides the prompt")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    fast: Optional[bool] = Field(None, description="Faster, lower-fidelity generation (legacy)")
    preset: Optional[str] = Field(None, description="fast | balanced | cinematic | experimental")
    candidates: Optional[int] = Field(None, description="Override number of candidates to render")


class DetectRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(waveform: np.ndarray, sample_rate: int, path: Path) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate)


def _row_get(row: dict, key: str):
    """Tolerant column access (old gallery rows may lack new columns)."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _item_payload(row: dict) -> dict:
    """Shape a gallery row for the frontend (adds the audio URL)."""
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "raw_prompt": row["raw_prompt"],
        "music_prompt": row["music_prompt"],
        "mood": row["mood"],
        "style": row["style"],
        "genre": _row_get(row, "genre"),
        "purpose": _row_get(row, "purpose"),
        "preset": _row_get(row, "preset"),
        "instrumentation": row["instrumentation"],
        "tempo_bpm": row["tempo_bpm"],
        "intensity": row["intensity"],
        "duration_s": row["duration_s"],
        "audio_url": f"/api/v1/audio/{row['id']}",
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
def health() -> dict:
    device = "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return {"status": "ok", "model": MUSICGEN_MODEL_NAME, "device": device, "default_fast": _DEFAULT_FAST}


@app.post("/api/v1/emotion/detect")
def detect(req: DetectRequest) -> dict:
    spec = keyword_extract_spec(req.prompt)
    duration = resolve_duration(None, prompt_text=req.prompt, default=MUSICGEN_DEFAULT_DURATION_S)
    return {
        "prompt": req.prompt,
        "mood": spec.emotion,
        "style": spec.style,
        "instrumentation": spec.instrumentation,
        "tempo_bpm": spec.tempo_bpm,
        "intensity": spec.intensity,
        "resolved_duration_s": duration,
    }


@app.post("/api/v1/analyze")
def analyze(req: DetectRequest) -> dict:
    """Full structured understanding of a prompt (genre/purpose/atmosphere/...)."""
    request = analyze_prompt(req.prompt)
    plan = _pipeline.plan(req.prompt)
    return {**request.to_dict(), "music_prompt": plan.music_prompt, "resolved_duration_s": plan.duration_s}


@app.get("/api/v1/presets")
def presets() -> dict:
    return {
        "presets": [
            {"name": p.name, "display": p.display, "description": p.description,
             "candidates": p.num_candidates}
            for p in PRESETS.values()
        ]
    }


@app.post("/api/v1/music/generate")
def generate(req: GenerateRequest) -> dict:
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    # Preset resolution: explicit preset > fast flag > server default.
    preset = req.preset
    fast = None if preset is not None else (_DEFAULT_FAST if req.fast is None else req.fast)

    start = time.time()
    result = _pipeline.generate(
        req.prompt, duration_s=req.duration, seed=req.seed,
        fast=fast, preset=preset, num_candidates=req.candidates,
    )
    latency = time.time() - start

    audio_filename = f"{uuid.uuid4().hex[:12]}.wav"
    _write_wav(result.audio.waveform, result.audio.sample_rate, GENERATED_AUDIO_DIR / audio_filename)

    req_obj = result.request
    chosen = next((c for c in result.candidate_scores if c.get("chosen")), {})
    gen_id = _gallery.add(
        raw_prompt=result.raw_prompt,
        music_prompt=result.music_prompt,
        mood=result.spec.emotion,
        style=result.spec.style,
        instrumentation=(", ".join(req_obj.instrumentation) if req_obj else result.spec.instrumentation),
        tempo_bpm=result.spec.tempo_bpm,
        intensity=result.spec.intensity,
        duration_s=result.duration_s,
        sample_rate=result.audio.sample_rate,
        audio_filename=audio_filename,
        latency_s=latency,
        source="web",
        genre=(req_obj.genre if req_obj else None),
        purpose=(req_obj.purpose if req_obj else None),
        preset=result.preset,
        score=chosen.get("clap_score") or chosen.get("spectral_score"),
    )
    # Keep the Spark/Hive analytics landing zone in sync too.
    log_generation(result, output_path=str((GENERATED_AUDIO_DIR / audio_filename).as_posix()),
                   latency_s=latency, source="web")

    row = _gallery.get(gen_id)
    assert row is not None
    return {
        **_item_payload(row),
        "latency_s": round(latency, 2),
        "preset": result.preset,
        "num_candidates": result.num_candidates,
        "candidate_scores": result.candidate_scores,
        "analysis": (req_obj.to_dict() if req_obj else None),
    }


@app.get("/api/v1/gallery")
def gallery(
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    mood: Optional[str] = None,
) -> dict:
    items = [_item_payload(r) for r in _gallery.list(limit=limit, offset=offset, mood=mood)]
    return {"items": items, "total": _gallery.count(mood=mood), "limit": limit, "offset": offset}


@app.get("/api/v1/gallery/{gen_id}")
def gallery_item(gen_id: str) -> dict:
    row = _gallery.get(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return _item_payload(row)


@app.get("/api/v1/moods")
def moods() -> dict:
    return {"moods": _gallery.moods()}


@app.get("/api/v1/audio/{gen_id}")
def audio(gen_id: str):
    row = _gallery.get(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    path = GENERATED_AUDIO_DIR / row["audio_filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file missing")
    return FileResponse(str(path), media_type="audio/wav", filename=f"{gen_id}.wav")


# ---------------------------------------------------------------------------
# Static frontend (served at / when built). Must be mounted last.
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(PROJECT_ROOT) / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

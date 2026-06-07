# Single-image deploy for the Emotion Music Studio (UI + API + MusicGen).
# Builds the React frontend, then serves it from FastAPI alongside the model.
# Targets a Hugging Face Docker Space (port 7860); GPU is used automatically when
# the host provides it (Linux `torch` ships CUDA wheels).

# ---- Stage 1: build the frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    MUSICGEN_FAST=0
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + data-grounding artifacts (generated offline by the Spark/Hive jobs;
# the prompt builder falls back to curated palettes if these are absent).
COPY src/ ./src/
COPY data/processed/emotion_music_profiles.json data/processed/emotion_instrumentation.json ./data/processed/
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Writable dirs for the gallery DB + generated audio.
RUN mkdir -p /app/artifacts/generated_audio

EXPOSE 7860
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]

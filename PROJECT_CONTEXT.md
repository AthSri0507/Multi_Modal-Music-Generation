# Project Context for AI

## What This Repository Is

This repository is a Python-based multimodal music AI system. It has two main phases:

1. **Phase 1: Emotion detection** from music-related inputs using lyrics/text embeddings plus tabular audio features.
2. **Phase 2 / Milestone 4: Emotion-conditioned music generation** using a conditional GAN that produces flattened musical sequences and converts them to MIDI/audio artifacts.

The codebase is centered on offline data processing and model training. It uses PyTorch for modeling, Hugging Face Transformers for text embeddings, HDF5 and Delta Lake for storage, and several utility scripts for validation, benchmarking, and generation workflows.

## One-Sentence Mental Model

Raw music metadata and lyrics are cleaned and split, text and audio features are transformed into model-ready tensors, a multimodal classifier predicts emotion labels, and those emotion labels can then condition a music generator that produces new music sequences.

## High-Level Architecture

### Phase 1: Multimodal Emotion Classification

- **Inputs**
  - Lyrics / text data
  - Spotify-style tabular audio features
- **Text branch**
  - BERT embeddings are extracted for lyrics
  - A dense feature extractor reduces 768-dim embeddings to 256 dims
- **Audio branch**
  - Normalized audio features go through a dense network
  - 11 input features are mapped to a 256-dim representation
- **Fusion layer**
  - Text and audio embeddings are concatenated
  - Dense fusion MLP produces a shared representation
- **Classification head**
  - Final logits predict emotion classes

### Phase 2: Emotion-Conditioned Music Generation

There are two generators in the repo:

**v1 (legacy): conditional GAN** -- kept as the Milestone-4 record.
- Inputs: noise vector + emotion condition + optional control vector.
- MLP generator -> flattened 480-dim sequence; discriminator with adversarial +
  auxiliary-emotion heads; vectors post-processed into MIDI events.
- Limitation: 4 coarse emotion quadrants, single flattened texture, piano-only
  training data -> low audio quality. Superseded by v2 for actual use.

**v2 (current): text -> rendered audio via MusicGen + a Spark/Hive backbone.**
- Pipeline (see `src/music_generation/text_to_music.py`):
  text prompt -> control extraction (`prompt_to_control` keyword helpers / optional
  BERT) -> canonical mood (`mood_map.py`) -> data-grounded MusicGen prompt
  (`prompt_builder.py`) -> instrumental audio (`audio_generator.py`,
  `facebook/musicgen-small`, no vocals) -> WAV.
- The descriptors (tempo, instrument palette, texture) are mined from the Spotify
  corpus by Spark + Hive (see Big Data Backbone below), so coarse emotion labels
  become nuanced, corpus-grounded prompts.
- User-controllable length: explicit duration > a duration parsed from the prompt
  ("a 30 second tune") > default; clamped to a safe range.

### Phase 2 (v3): Quality + controllability layer

Richer understanding, a style taxonomy, presets, and multi-candidate selection so
different prompts produce clearly different, on-style outputs.

- **Understanding** (`src/music_generation/prompt_understanding.py`): `analyze_prompt`
  → `MusicRequest` with mood, **genre**, **purpose**, instrumentation, energy,
  atmosphere, complexity, tempo, duration. Deterministic keyword matching, no BERT.
- **Taxonomy** (`src/music_generation/taxonomy.py`): genres (lofi, cinematic,
  orchestral, jazz, ambient, classical, electronic, synthwave, …) with instrument
  templates / tempo ranges / atmospheres, plus purposes (study, meditation, sleep,
  workout, …). Canonical defaults in code; **materialized to Hive** + exported JSON by
  `scripts/bd_build_taxonomy.py` (tables `music.{genre_taxonomy,
  instrumentation_templates, atmosphere_descriptors, use_cases}`); runtime overlays
  the JSON on defaults.
- **Enhanced prompt builder** (`enhanced_prompt_builder.py`): composes rich prompts,
  e.g. "uplifting jazz featuring piano, upright bass and brushed drums, warm smoky
  club atmosphere, upbeat tempo around 130 BPM …, no vocals". The old thin builder is
  retained for A/B (`TextToMusicPipeline(enhanced=False)`).
- **Presets** (`presets.py`): Fast / Balanced / Cinematic / Experimental — each sets
  sampling (guidance, temperature, top-k/p) **and** candidate count.
- **Multi-candidate + two-stage scorer** (`candidate_scorer.py`): non-Fast presets
  render N≥3 takes; Stage 1 spectral gate (librosa) rejects noisy takes, Stage 2 CLAP
  (optional, `MUSICGEN_USE_CLAP=1`) ranks by text↔audio adherence; best is kept.
- **Generator** sampling params + `generate_candidates`; RMS-target normalization
  (no longer amplifies noisy tails).
- **New API**: `POST /api/v1/analyze` (full understanding), `GET /api/v1/presets`;
  `generate` accepts `preset`/`candidates`, returns `analysis` + `candidate_scores`.
- **Eval/A-B**: `scripts/eval_generation.py` (diversity / adherence / extraction),
  `scripts/ab_prompt_builders.py` (old vs new builder). Reports in
  `artifacts/generation_eval/` and `artifacts/ab_prompt_builders/`.
- **Free GPU**: `deploy/colab_gpu_backend.ipynb` + `deploy/free_gpu_options.md` run
  the unchanged backend with `musicgen-medium` on a free Colab/Kaggle GPU.

### Big Data Backbone (Spark + Hive, optional HDFS) -- the graded core

- `src/bigdata/spark_hive.py` -- Hive-enabled SparkSession (embedded Derby
  metastore + warehouse; warehouse can point at HDFS via `HDFS_BASE_URI`). On
  Windows it auto-wires a project-local `hadoop/` (winutils) for `HADOOP_HOME`.
- `scripts/bd_build_spotify_warehouse.py` -- Spark ETL of the Spotify CSV into
  managed Hive tables `music.{tracks, track_features, track_emotions}`.
- `scripts/bd_build_emotion_profiles.py` -- Spark SQL aggregation ->
  `music.emotion_music_profiles` + `music.emotion_instrumentation` (per-mood
  instrument palette), exported to JSON for the prompt builder. Synthesizes a
  data-driven `calm` profile from the low-energy subset.
- `scripts/bd_batch_generate.py` -- Spark reads a prompt set
  (`music.generation_requests` or a CSV) and renders audio in bulk.
- `scripts/bd_generation_analytics.py` -- materializes the generation event log
  (`artifacts/generation_logs.jsonl`) into `music.generation_logs` and runs Spark
  SQL analytics (by mood / instrumentation).

### Serving + Web UI

- `src/api/app.py` -- FastAPI backend: `GET /api/v1/health`,
  `POST /api/v1/emotion/detect`, `POST /api/v1/music/generate` (renders + saves to
  the gallery), `GET /api/v1/gallery`, `GET /api/v1/gallery/{id}`,
  `GET /api/v1/audio/{id}`, `GET /api/v1/moods`. Serves the built frontend at `/`.
- `src/api/gallery_store.py` -- SQLite gallery store (metadata; audio on disk under
  `GENERATED_AUDIO_DIR`).
- `frontend/` -- React + Vite + Tailwind v4 single-page app (Impeccable design
  principles: OKLCH theme, tinted neutrals, Space Grotesk + Instrument Sans,
  restrained motion, real focus/empty/loading states). Two views: **Create**
  (prompt -> tune) and **Gallery** (browse everyone's tunes, filter by mood).
- Single-image deploy: `Dockerfile` builds the UI and serves UI+API+model from one
  `uvicorn` process on port 7860 (Hugging Face Docker Space; GPU auto-detected).
  See `deploy/huggingface_space.md`.

#### Run the web app

```bash
# Dev (two terminals): API + Vite dev server (proxies /api -> :8000)
uvicorn src.api.app:app --port 8000 --reload
cd frontend && npm install && npm run dev        # http://localhost:5173

# Single-process (build once, FastAPI serves the UI at http://localhost:8000)
cd frontend && npm run build && cd ..
uvicorn src.api.app:app --port 8000
# add MUSICGEN_FAST=1 for faster CPU generation
```

## Main Technologies

- **Python**
- **PyTorch** for models and training loops
- **Hugging Face Transformers** for BERT-based text embeddings and prompt encoding
- **NumPy / Pandas / SciPy** for data processing
- **scikit-learn** for splits, metrics, and evaluation helpers
- **PyArrow / HDF5 / h5py** for feature storage
- **Delta Lake / deltalake** for music generation datasets
- **kagglehub** for dataset acquisition
- **PySpark** for scalable or planned distributed processing paths
- **librosa / essentia / pretty_midi** for audio and MIDI-oriented workflows
- **pymongo** for database-oriented storage support

## Core Repository Layout

- `src/config/`
  - Runtime constants, paths, label sets, model names, and pipeline parameters
- `src/data/`
  - Data ingest, cleaning, validation, text processing, audio processing, and HDF5 export
- `src/models/`
  - Phase 1 neural network blocks and the multimodal classifier
- `src/training/`
  - Training loops, losses, and metrics for classification and GAN training
- `src/music_generation/`
  - Conditional generation dataset logic, control schema, GAN models, MIDI representation, augmentation, and post-processing
- `scripts/`
  - Smoke tests, benchmarking, training entrypoints, evaluation utilities, export tools, and data prep scripts
- `data/`
  - Raw and processed data artifacts
- `artifacts/`
  - Run outputs, experiment logs, and generated results
- `checkpoints/`
  - Saved model checkpoints
- `logs/`
  - Training and experiment logs
- `tests/`
  - Unit and smoke tests

## Key Files And What They Do

- `run_pipeline.py`
  - Orchestrates the base data pipeline: setup check, load, process, validate, and optional text/audio feature generation
- `check_setup.py`
  - Verifies the environment, expected folders, packages, disk space, and requirements file
- `src/config/config.py`
  - Central configuration for data paths, sample sizes, label sets, output locations, and model constants
- `src/models/multimodal_classifier.py`
  - End-to-end Phase 1 classifier combining text and audio branches
- `src/models/text_feature_extractor.py`
  - Dense projection network for BERT embeddings
- `src/models/audio_feature_extractor.py`
  - Dense projection network for tabular audio features
- `src/training/multimodal_trainer.py`
  - Phase 1 trainer with metrics, early stopping, checkpointing, and optional TensorBoard logging
- `src/music_generation/models.py`
  - Conditional GAN generator and discriminator architectures
- `src/music_generation/control_schema.py`
  - Stable structured control vector schema for emotion, style, tempo, intensity, and instrumentation
- `src/music_generation/prompt_to_control.py`
  - Maps free-form prompts into control specs using BERT plus lightweight priors
- `src/music_generation/dataset.py`
  - NPZ-based dataset utilities for emotion-conditioned generation
- `src/music_generation/delta_dataset.py`
  - Delta Lake-backed dataset utilities for generation training
- `src/training/music_gan_trainer.py`
  - GAN training loop for emotion-conditioned music generation

## Data And Artifact Flow

### Phase 1 Data Flow

1. Load raw Spotify-derived CSV data.
2. Clean, deduplicate, and split into train / validation / test.
3. Validate audio features and emotion labels.
4. Generate BERT embeddings for lyrics.
5. Normalize audio features.
6. Save model-ready artifacts in CSV, HDF5, and report files.
7. Feed embeddings and audio tensors into the multimodal classifier.

### Phase 2 Data Flow

1. Prepare music sequence data in NPZ or Delta Lake format.
2. Encode emotion labels and optional control signals.
3. Train the GAN generator/discriminator pair.
4. Convert generated vectors to MIDI-compatible outputs.
5. Optionally convert MIDI to audio/MP3 and save generated samples.

## What The Core Models Expect

### Multimodal classifier

- Text input: 768-dim BERT embedding
- Audio input: 11 normalized tabular features
- Text branch output: 256 dims
- Audio branch output: 256 dims
- Fusion input: 512 dims total
- Fusion output: 128 dims before classification
- Output: emotion logits

### Music generator

- Noise vector plus emotion embedding
- Optional structured control vector
- Output: flattened sequence representation, later post-processed into musical events

## Important Data Conventions

- Phase 1 uses a 50K Spotify sample as the baseline working set.
- Emotion labels are normalized and validated in the data pipeline.
- HDF5 is used for dense model inputs and embeddings.
- Delta Lake is used for some music-generation training workflows.
- Many scripts assume outputs live under `data/processed`, `artifacts`, or `checkpoints`.

## Common Run Commands

### Environment check

```bash
python check_setup.py
```

### Base data pipeline

```bash
python run_pipeline.py
```

### Base data pipeline with text and audio exports

```bash
python run_pipeline.py --with-text --with-audio
```

### Phase 1 model training scripts

```bash
python -m scripts.train_multimodal_phase1
python -m scripts.phase1_4class_cv
python -m scripts.phase1_4class_tune
```

### Music generation scripts (v1 GAN, legacy)

```bash
python -m scripts.train_music_gan_phase1
python -m scripts.m4_prepare_music_data
python -m scripts.m4_prepare_music_data_spark
```

### Text-to-audio (v2) + Spark/Hive backbone

```bash
# 1. Build the Hive warehouse from the Spotify CSV (Spark + Hive)
python scripts/bd_build_spotify_warehouse.py

# 2. Mine per-mood profiles + instrument palettes (Spark SQL)
python scripts/bd_build_emotion_profiles.py

# 2b. Materialize the music-style taxonomy into Hive + JSON
python scripts/bd_build_taxonomy.py

# 3. Generate a tune from text (presets: fast | balanced | cinematic | experimental)
python scripts/generate_music_from_text.py --prompt "happy upbeat jazz trio" --preset balanced
python scripts/generate_music_from_text.py --prompt "epic cinematic battle" --preset cinematic --dry-run

# 3b. Evaluate quality + A/B the prompt builders
python scripts/eval_generation.py --no-audio            # metadata-only (instant)
python scripts/eval_generation.py --preset fast --duration 5
python scripts/ab_prompt_builders.py                    # prompt-richness A/B (instant)

# 4. Batch generation (Spark) + analytics
python scripts/bd_batch_generate.py --prompts-csv data/prompts.csv
python scripts/bd_generation_analytics.py

# 5. Serve the API
uvicorn src.api.app:app --reload
```

> Windows note: Spark+Hive needs `winutils.exe`/`hadoop.dll`. A project-local
> `hadoop/` dir is auto-detected and used for `HADOOP_HOME` (no manual setup).
> CPU MusicGen downloads ~2 GB on first run.

**Generation speed (CPU).** Pass `--fast` to roughly 2.5-3x throughput at some
fidelity cost (measured: 170s -> 61s for a 5s clip). `--fast` drops classifier-free
guidance to 1.0 and int8-quantizes the transformer's Linear layers. Knobs live in
`config.py` (`MUSICGEN_GUIDANCE_SCALE`) and `MusicGenAudioGenerator`
(`guidance_scale`, `quantize`, `bf16` -- the latter is mutually exclusive with
`quantize`). The API honors `MUSICGEN_FAST=1`. A GPU (set up CUDA torch) is the
real fix: seconds per clip, and you can switch `MUSICGEN_MODEL_NAME` to
`facebook/musicgen-medium` for higher quality.

```bash
python scripts/generate_music_from_text.py --prompt "calm soothing piano" --fast
```

## Current Project Status

Based on the repository docs and task list, the current state is:

- The 50K baseline data pipeline is implemented and validated.
- Phase 1 multimodal emotion classification has a validated 4-class baseline.
- Text and audio feature pipelines are in place.
- The music GAN architecture, control schema, and Delta Lake data path are implemented.
- There are many experiment and smoke-test scripts for ablations, CV, tuning, OOD checks, inference benchmarking, and artifact generation.

## Practical Notes For Another AI

- Treat this as a two-stage system: classification first, generation second.
- The most important architectural boundary is between Phase 1 emotion prediction and Phase 2 emotion-conditioned generation.
- The repo is organized around scripts and trainers rather than a single monolithic app.
- Configuration is centralized, so inspect `src/config/config.py` before changing paths or class counts.
- When in doubt, follow the data artifacts: raw CSV -> cleaned splits -> embeddings/features -> training inputs -> checkpoints/artifacts.

## Suggested Prompt For Another AI

You are looking at a Python multimodal music AI repository. It contains a Phase 1 emotion classifier that fuses BERT text embeddings with normalized Spotify audio features, and a Phase 2 conditional GAN that generates emotion-conditioned music sequences. The repo uses PyTorch, Transformers, HDF5, Delta Lake, pandas, NumPy, scikit-learn, and optional Spark. Read the configuration first, then the data pipeline, then the model and training modules. The main entry points are `run_pipeline.py`, `check_setup.py`, the training scripts under `scripts/`, and the model/training code in `src/`. The expected flow is raw data ingestion -> cleaning/validation/splitting -> feature extraction -> classifier training -> control-conditioned generation -> MIDI/audio export.

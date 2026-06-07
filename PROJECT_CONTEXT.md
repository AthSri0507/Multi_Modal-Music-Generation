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

- **Inputs**
  - Noise vector
  - Emotion condition
  - Optional prompt-derived controls like style, tempo, intensity, and instrumentation
- **Generator**
  - MLP-based conditional GAN generator
  - Produces a flattened music sequence vector
- **Discriminator**
  - Distinguishes real vs fake sequences
  - Also predicts emotion with an auxiliary classification head
- **Post-processing**
  - Generated vectors are converted into valid MIDI-style musical events
  - Outputs can be exported or converted to MP3 in the artifact pipeline

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

### Music generation scripts

```bash
python -m scripts.train_music_gan_phase1
python -m scripts.m4_prepare_music_data
python -m scripts.m4_prepare_music_data_spark
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

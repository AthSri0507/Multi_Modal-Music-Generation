# Multimodal Emotion Detection and Music Generation

This repository contains an AI project that combines lyrics and audio features to:

1. detect emotion from music-related inputs
2. learn aligned text and audio representations
3. support emotion-aware music generation workflows

The codebase currently includes data ingestion, cleaning, validation, text embedding, audio feature processing, and model-ready dataloaders.

## What This Project Does

1. Ingests music datasets and prepares a clean working subset
2. Cleans and validates tabular metadata and labels
3. Generates text embeddings from lyrics using transformer models
4. Builds normalized audio feature tensors for training
5. Exports model-ready artifacts in CSV and HDF5 formats

## Repository Layout

1. `src/config/`: runtime configuration
2. `src/data/`: ingestion, preprocessing, validation, text and audio pipelines
3. `src/models/`: text and audio feature extraction models
4. `scripts/`: smoke checks and utility runners
5. `tests/`: unit tests
6. `data/`: generated raw and processed artifacts

## Clone and Set Up

### 1. Clone the repository

```bash
git clone https://github.com/AthSri0507/Multi_Modal-Music-Generation
cd music_gen
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project Pipeline

### 1. Verify environment

```bash
python check_setup.py
```

### 2. Run base data preparation

```bash
python -m src.data.data_loader
python -m src.data.data_processor
python -m src.data.data_validator
```

### 3. Run text processing

```bash
python -m src.data.text_processor --batch-size 16 --max-length 128
```

### 4. Run audio processing

```bash
python -m src.data.audio_processor
```

### 5. Optional: run orchestrated pipeline

```bash
python run_pipeline.py --with-text --with-audio
```

## Validate Outputs

After running the pipeline, check these artifacts:

1. `data/processed/spotify_50k_train.csv`
2. `data/processed/spotify_50k_val.csv`
3. `data/processed/spotify_50k_test.csv`
4. `data/processed/embeddings_full_50k.h5`
5. `data/processed/audio_features_full_50k.h5`

You can also run tests:

```bash
pytest -q
```

## Notes for First-Time Contributors

1. Use Python 3.10+
2. If transformer downloads are slow, rerun the same command after cache warm-up
3. If an HDF5 output file is locked on Windows, close any viewer/tool using it and rerun



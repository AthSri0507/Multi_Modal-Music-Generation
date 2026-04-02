"""
Configuration file for Multimodal Emotion Detection & Music Generation System
Pipeline Setup: Data Pipeline Setup
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PHASE 1: Data Configuration (50K Spotify Sample)
# ============================================================================

# Kaggle Dataset Configuration
KAGGLE_SPOTIFY_DATASET = "devdope/900k-spotify"
KAGGLE_SPOTIFY_FILE = "spotify_dataset.csv"

# Phase 1 Parameters
PHASE_1_SAMPLE_SIZE = 50000  # Start with 50k for fast iteration
PHASE_1_SEED = 42

# Data Paths
SPOTIFY_RAW_PATH = RAW_DATA_DIR / "spotify_50k_raw.csv"
SPOTIFY_PROCESSED_PATH = PROCESSED_DATA_DIR / "spotify_50k_processed.csv"
SPOTIFY_TRAIN_PATH = PROCESSED_DATA_DIR / "spotify_50k_train.csv"
SPOTIFY_VAL_PATH = PROCESSED_DATA_DIR / "spotify_50k_val.csv"
SPOTIFY_TEST_PATH = PROCESSED_DATA_DIR / "spotify_50k_test.csv"

# HDF5 Storage Configuration
HDF5_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "embeddings_clean_50k.h5"
HDF5_FEATURES_PATH = PROCESSED_DATA_DIR / "audio_features_clean_50k.h5"
TEXT_EMBEDDINGS_REPORT_PATH = PROCESSED_DATA_DIR / "text_embeddings_report_clean_50k.json"
AUDIO_FEATURES_REPORT_PATH = PROCESSED_DATA_DIR / "audio_features_report_clean_50k.json"

# Audio Features to Extract
AUDIO_FEATURES = [
    "Danceability",
    "Energy",
    "Positiveness",
    "Tempo",
    "Loudness (db)",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
]

# Canonical audio feature vector used by the audio branch (11-dim)
AUDIO_CANONICAL_FEATURES = [
    "danceability",
    "energy",
    "positiveness",
    "tempo",
    "loudness_db",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "key",
    "length",
]

# Candidate columns for robust schema matching across dataset variants.
AUDIO_FEATURE_COLUMN_CANDIDATES = {
    "danceability": ["Danceability", "danceability"],
    "energy": ["Energy", "energy"],
    "positiveness": ["Positiveness", "valence", "Valence"],
    "tempo": ["Tempo", "tempo"],
    "loudness_db": ["Loudness (db)", "loudness", "Loudness"],
    "speechiness": ["Speechiness", "speechiness"],
    "acousticness": ["Acousticness", "acousticness"],
    "instrumentalness": ["Instrumentalness", "instrumentalness"],
    "liveness": ["Liveness", "liveness"],
    "key": ["Key", "key"],
    "length": ["Length", "duration_ms", "Duration", "duration"],
}

AUDIO_LABEL_COLUMN_CANDIDATES = ["emotion", "label", "class"]
AUDIO_TRACK_COLUMN_CANDIDATES = ["song", "track_name", "title", "name"]
AUDIO_ARTIST_COLUMN_CANDIDATES = ["Artist(s)", "artist", "artists", "artist_name"]

# Audio processing defaults
AUDIO_BATCH_SIZE = 256
AUDIO_INCLUDE_METADATA = True
AUDIO_MAX_SAMPLES_PER_SPLIT = None

# Emotion Classes (8-class distribution found in Spotify dataset)
EMOTION_CLASSES = [
    "joy",
    "sadness", 
    "anger",
    "fear",
    "love",
    "surprise",
    "disgust",
    "neutral"
]

# Phase 1 canonical class set (clean 6-class setup used for training)
PHASE_1_CANONICAL_EMOTIONS = [
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise",
]

# Phase 1 reduced 4-class set (dropping fear F1=0.005 and surprise F1=0.067)
PHASE_1_4CLASS_EMOTIONS = [
    "anger",
    "joy",
    "love",
    "sadness",
]

# Keep only columns required for the current multimodal task in processed splits.
PHASE_1_KEEP_COLUMNS = [
    "Artist(s)",
    "song",
    "text",
    "emotion",
    "Length",
    "Key",
    "Tempo",
    "Loudness (db)",
    "Energy",
    "Danceability",
    "Positiveness",
    "Speechiness",
    "Liveness",
    "Acousticness",
    "Instrumentalness",
]

# Data Split Configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Text Processing Configuration
TEXT_SOURCE_COLUMN_CANDIDATES = ["text", "lyrics", "lyric", "song_lyrics"]
TEXT_TRACK_COLUMN_CANDIDATES = ["song", "track_name", "title", "name"]
TEXT_ARTIST_COLUMN_CANDIDATES = ["Artist(s)", "artist", "artists", "artist_name"]
TEXT_LABEL_COLUMN_CANDIDATES = ["emotion", "label", "class"]

BERT_MODEL_NAME = "bert-base-uncased"
TEXT_MAX_LENGTH = 256
TEXT_BATCH_SIZE = 32
TEXT_INFERENCE_DEVICE = "auto"  # auto, cpu, cuda
TEXT_INCLUDE_CLEANED_TEXT = True

# Set to an int for quick debug runs, or None for full split.
TEXT_MAX_SAMPLES_PER_SPLIT = None

# ============================================================================
# PHASE 2 & 3: Scaling Configuration (for future expansion)
# ============================================================================

PHASE_2_SAMPLE_SIZE = 300000  # Expand to 300k
PHASE_3_SAMPLE_SIZE = 900000  # Full dataset

# ============================================================================
# Spark Configuration (optional for Phase 1, required for Phases 2-3)
# ============================================================================

SPARK_MASTER = "local[*]"  # Single-node Spark for Phase 1
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "2g"

# ============================================================================
# Database Configuration (MongoDB local)
# ============================================================================

MONGODB_URI = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "music_emotion_db"
MONGODB_COLLECTION = "songs"

# ============================================================================
# Processing Parameters
# ============================================================================

# Text normalization
NORMALIZE_TEXT_LOWERCASE = True
NORMALIZE_REMOVE_SPECIAL_CHARS = True
REMOVE_PARENS_BRACKETS = True

# Deduplication
DEDUP_ON_ARTIST_TITLE = True
DEDUP_NORM_KEY_SEP = "|"

# Missing value handling
MISSING_VALUE_STRATEGY = "drop"  # "drop" or "impute"
IMPUTE_STRATEGY = "mean"  # For numeric features

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# ============================================================================
# Testing Configuration
# ============================================================================

TEST_SAMPLE_SIZE = 1000  # For quick tests
TEST_MODE = False  # Set to True for small dataset debugging

if TEST_MODE:
    PHASE_1_SAMPLE_SIZE = TEST_SAMPLE_SIZE


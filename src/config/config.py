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
# MILESTONE 4: Music Generation Configuration (pre-training foundation)
# ============================================================================

# Dataset artifact produced by scripts/m4_prepare_music_data.py
M4_MUSIC_NPZ_PATH = PROCESSED_DATA_DIR / "m4_music_sequences.npz"

# MIDI representation defaults: [pitch, velocity, duration, delta_time]
M4_MIDI_SEQ_LEN = 120
M4_MIDI_FEATURE_DIM = 4
M4_MIDI_MAX_DURATION_BEATS = 8.0
M4_MIDI_MAX_DELTA_BEATS = 8.0

# cGAN architecture defaults
M4_GAN_NUM_EMOTIONS = 8
M4_GAN_NOISE_DIM = 100
M4_GAN_EMOTION_EMBED_DIM_G = 8
M4_GAN_EMOTION_EMBED_DIM_D = 32
M4_GAN_GENERATOR_HIDDEN_DIMS = (256, 512, 1024)
M4_GAN_DISCRIMINATOR_HIDDEN_DIMS = (512, 256, 128)
M4_GAN_USE_SPECTRAL_NORM = True

# Future training defaults (kept here for reproducible startup values)
M4_GAN_BATCH_SIZE = 64
M4_GAN_G_LR = 1e-4
M4_GAN_D_LR = 2e-4
M4_GAN_BETA1 = 0.5
M4_GAN_BETA2 = 0.999
M4_GAN_R1_WEIGHT = 10.0
M4_GAN_AUX_CE_WEIGHT = 0.25

# ============================================================================
# Spark Configuration (optional for Phase 1, required for Phases 2-3)
# ============================================================================

SPARK_MASTER = "local[*]"  # Single-node Spark for Phase 1
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "2g"

# ============================================================================
# Big Data Warehouse: Spark + Hive (+ optional HDFS)
# ============================================================================

# Database / managed-table namespace inside the Hive metastore.
HIVE_DB_NAME = "music"

# Embedded Hive metastore (Derby) + warehouse directory. The warehouse can be
# pointed at HDFS by setting HDFS_BASE_URI; otherwise it lives on the local FS.
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
HIVE_WAREHOUSE_DIR = WAREHOUSE_DIR / "hive"
HIVE_METASTORE_DIR = WAREHOUSE_DIR / "metastore_db"

# Optional HDFS base URI (e.g. "hdfs://localhost:9000/music"). When set, the Hive
# warehouse and generated-audio artifacts are stored under HDFS instead of local FS.
# Read from env so the Docker cluster path can be enabled without code changes.
HDFS_BASE_URI = os.environ.get("HDFS_BASE_URI", "").strip() or None

# Data-driven mapping artifacts produced by scripts/bd_build_emotion_profiles.py
EMOTION_PROFILES_JSON = PROCESSED_DATA_DIR / "emotion_music_profiles.json"
EMOTION_INSTRUMENTATION_JSON = PROCESSED_DATA_DIR / "emotion_instrumentation.json"

# ============================================================================
# MILESTONE 4 (v2): Text-to-Audio Generation (MusicGen)
# ============================================================================

# Pretrained instrumental text-to-music model (no vocals/lyrics). CPU-friendly.
MUSICGEN_MODEL_NAME = "facebook/musicgen-small"
MUSICGEN_SAMPLE_RATE = 32000  # overridden at runtime by the model config
MUSICGEN_DEFAULT_DURATION_S = 10.0
MUSICGEN_MIN_DURATION_S = 2.0
MUSICGEN_MAX_DURATION_S = 60.0
# MusicGen EnCodec frame rate: ~50 audio tokens per second of output.
MUSICGEN_TOKENS_PER_SECOND = 50

# Inference speed / quality knobs (CPU).
# guidance_scale > 1 enables classifier-free guidance, which DOUBLES compute per
# step (runs the model conditionally + unconditionally). Lower = faster, less
# prompt adherence. "fast" mode forces 1.0 + int8 quantization + bf16 autocast.
MUSICGEN_GUIDANCE_SCALE = 3.0
MUSICGEN_FAST_GUIDANCE_SCALE = 1.0

# Sampling defaults (passed to MusicGen .generate). top_k=250/temperature=1.0 are
# the model's recommended values; presets may override these.
MUSICGEN_TEMPERATURE = 1.0
MUSICGEN_TOP_K = 250
MUSICGEN_TOP_P = 0.0  # 0 = disabled (use top_k)

# Default generation preset (see src/music_generation/presets.py).
MUSICGEN_DEFAULT_PRESET = "balanced"

# Mid-clip noise handling. `musicgen-small` drifts toward noise as a clip grows
# (autoregressive error accumulation); spectral flatness climbs over time. We detect
# that and trim the noisy tail of the chosen candidate.
MUSICGEN_NOISE_FLATNESS = 0.05       # per-window flatness above this ~= noise
MUSICGEN_AUTOTRIM = True             # trim the noisy tail of the delivered clip
MUSICGEN_MIN_CLEAN_S = 3.0           # never trim a clip shorter than this

# Stability-biased sampling for sparse / low-energy prompts (ambient, cinematic,
# thriller, ...), which drift fastest. Lower temperature / top_k = less wandering.
MUSICGEN_STABLE_TEMPERATURE = 0.9
MUSICGEN_STABLE_TOP_K = 150

# Music-style taxonomy catalog exported by scripts/bd_build_taxonomy.py (Hive ->
# JSON). The runtime overlays this on the built-in defaults in taxonomy.py.
MUSIC_TAXONOMY_JSON = PROCESSED_DATA_DIR / "music_taxonomy.json"

# Rendered audio output location.
GENERATED_AUDIO_DIR = PROJECT_ROOT / "artifacts" / "generated_audio"

# Durable append-only generation event log (landing zone). The CLI, API, and Spark
# batch job append JSONL rows here; bd_generation_analytics.py materializes it into
# the Hive table `music.generation_logs` and runs Spark SQL analytics over it.
GENERATION_LOG_JSONL = PROJECT_ROOT / "artifacts" / "generation_logs.jsonl"

# Web gallery: SQLite metadata store for generations browsable in the web UI.
GALLERY_DB_PATH = PROJECT_ROOT / "artifacts" / "gallery.db"

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


"""
Data Processor: Deduplication, cleaning, validation, and train/val/test splits
Pipeline Setup: Phase 1 - Process 50K Spotify sample
"""

import pandas as pd
import numpy as np
import logging
import re
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    SPOTIFY_RAW_PATH,
    SPOTIFY_PROCESSED_PATH,
    SPOTIFY_TRAIN_PATH,
    SPOTIFY_VAL_PATH,
    SPOTIFY_TEST_PATH,
    AUDIO_FEATURES,
    PHASE_1_CANONICAL_EMOTIONS,
    PHASE_1_KEEP_COLUMNS,
    EMOTION_CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PHASE_1_SEED,
    PROCESSED_DATA_DIR,
    LOG_FORMAT,
    LOG_LEVEL
)

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class SpotifyDataProcessor:
    """Process, clean, and validate Spotify data"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.report = {}
    
    def load_raw_data(self, csv_path=SPOTIFY_RAW_PATH):
        """Load raw CSV data"""
        logger.info(f"Loading raw data from: {csv_path}")
        self.raw_data = pd.read_csv(csv_path)
        logger.info(f"Loaded shape: {self.raw_data.shape}")
        self.report["raw_shape"] = list(self.raw_data.shape)
        return self.raw_data
    
    def normalize_text(self, s):
        """Normalize text for deduplication"""
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()
        s = re.sub(r"\(.*?\)", " ", s)  # Remove parentheses content
        s = re.sub(r"\[.*?\]", " ", s)  # Remove brackets content
        s = re.sub(r"[^a-z0-9\s]", " ", s)  # Keep only alphanumeric
        s = re.sub(r"\s+", " ", s).strip()  # Normalize whitespace
        return s

    def to_numeric_series(self, series):
        """Convert series to numeric, extracting numeric prefixes from strings like '-3.6db'."""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        cleaned = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        return pd.to_numeric(cleaned, errors="coerce")
    
    def find_column(self, candidates):
        """Find first available column from candidates"""
        def normalize_name(name):
            n = str(name).lower().strip()
            n = n.replace("_", " ")
            n = re.sub(r"[^a-z0-9\s]", "", n)
            n = re.sub(r"\s+", " ", n).strip()
            return n

        normalized = {normalize_name(c): c for c in self.raw_data.columns}
        for col in candidates:
            key = normalize_name(col)
            if key in normalized:
                return normalized[key]
        return None
    
    def deduplicate(self):
        """Deduplicate by normalized artist + title"""
        logger.info("Starting deduplication...")
        
        # Find key columns
        track_col = self.find_column(["track_name", "name", "song_name", "title", "track", "song"])
        artist_col = self.find_column(["artist_name", "artist(s)", "artists", "artist", "artist_names"])
        
        if not track_col or not artist_col:
            logger.warning("Could not find track/artist columns. Skipping dedeup.")
            self.processed_data = self.raw_data.copy()
            return self.processed_data
        
        logger.info(f"Using columns: track='{track_col}', artist='{artist_col}'")
        
        # Create dedup key
        df = self.raw_data.copy()
        df["_artist_norm"] = df[artist_col].map(self.normalize_text)
        df["_track_norm"] = df[track_col].map(self.normalize_text)
        df["_dedup_key"] = df["_artist_norm"] + "|" + df["_track_norm"]
        
        # Remove empty keys
        df = df[df["_dedup_key"] != "|"].copy()
        
        # Count duplicates
        dup_count = int(df.duplicated("_dedup_key").astype(int).sum())
        dup_rate = (dup_count / len(df)) * 100 if len(df) > 0 else 0
        
        logger.info(f"Found {dup_count} duplicate rows ({dup_rate:.2f}%)")
        
        # Deduplicate
        df = df.drop_duplicates("_dedup_key", keep="first")
        df = df.drop(columns=["_artist_norm", "_track_norm", "_dedup_key"])
        
        self.processed_data = df
        logger.info(f"After dedup shape: {self.processed_data.shape}")
        
        self.report["deduplication"] = {
            "rows_before": int(self.raw_data.shape[0]),
            "rows_after": int(self.processed_data.shape[0]),
            "duplicates_removed": int(dup_count),
            "duplicate_rate_pct": round(dup_rate, 3)
        }
        
        return self.processed_data
    
    def validate_audio_features(self):
        """Validate and document audio features"""
        logger.info("Validating audio features...")
        
        feature_stats = {}
        for feat in AUDIO_FEATURES:
            if feat in self.processed_data.columns:
                numeric_col = self.to_numeric_series(self.processed_data[feat])
                missing = numeric_col.isna().sum()
                missing_pct = (missing / len(self.processed_data)) * 100
                
                feature_stats[feat] = {
                    "present": int(self.processed_data.shape[0] - missing),
                    "missing": int(missing),
                    "missing_pct": round(missing_pct, 3),
                    "mean": round(float(numeric_col.mean()), 4) if numeric_col.notna().any() else None,
                    "min": round(float(numeric_col.min()), 4) if numeric_col.notna().any() else None,
                    "max": round(float(numeric_col.max()), 4) if numeric_col.notna().any() else None
                }
                logger.info(f"  {feat:20s}: {missing} missing ({missing_pct:5.2f}%)")
        
        self.report["audio_features"] = feature_stats
        return feature_stats
    
    def validate_emotion_labels(self):
        """Validate emotion class distribution"""
        logger.info("Validating emotion labels...")
        
        emotion_col = self.find_column(["emotion", "class", "label", "emotion_class"])
        
        if not emotion_col:
            logger.warning("Could not find emotion column")
            self.report["emotion_distribution"] = {}
            return {}
        
        dist = self.processed_data[emotion_col].value_counts().to_dict()
        dist_pct = self.processed_data[emotion_col].value_counts(normalize=True).to_dict()
        
        emotion_stats = {}
        for emotion, count in sorted(dist.items()):
            pct = dist_pct[emotion] * 100
            emotion_stats[str(emotion)] = {
                "count": int(count),
                "percentage": round(pct, 2)
            }
            logger.info(f"  {emotion:15s}: {count:6d} ({pct:5.2f}%)")
        
        self.report["emotion_distribution"] = emotion_stats
        return emotion_stats

    def clean_emotion_labels(self, allowed_labels=None):
        """Normalize emotion labels and drop rows outside the canonical Phase-1 class set."""
        logger.info("Cleaning emotion labels...")

        emotion_col = self.find_column(["emotion", "class", "label", "emotion_class"])
        if not emotion_col:
            logger.warning("Could not find emotion column; skipping label cleaning")
            self.report["label_cleaning"] = {
                "status": "skipped",
                "reason": "emotion column not found",
            }
            return self.processed_data

        if allowed_labels is None:
            allowed_labels = PHASE_1_CANONICAL_EMOTIONS

        allowed = {str(lbl).strip().lower() for lbl in allowed_labels}
        before = len(self.processed_data)

        labels_norm = self.processed_data[emotion_col].astype(str).str.strip().str.lower()
        keep_mask = labels_norm.isin(allowed)
        dropped = int((~keep_mask).sum())

        if dropped > 0:
            drop_counts = labels_norm[~keep_mask].value_counts().to_dict()
            logger.info(
                "Dropping %d rows with non-canonical labels: %s",
                dropped,
                drop_counts,
            )
        else:
            drop_counts = {}

        self.processed_data = self.processed_data.loc[keep_mask].copy()
        self.processed_data[emotion_col] = labels_norm[keep_mask]
        after = len(self.processed_data)

        self.report["label_cleaning"] = {
            "strategy": "drop_non_canonical",
            "emotion_column": emotion_col,
            "allowed_labels": sorted(allowed),
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_dropped": int(dropped),
            "dropped_label_counts": {str(k): int(v) for k, v in drop_counts.items()},
        }
        logger.info("After label cleaning shape: %s", self.processed_data.shape)
        return self.processed_data

    def filter_relevant_columns(self, keep_columns=None):
        """Keep only columns relevant to the current multimodal training task."""
        logger.info("Filtering to task-relevant columns...")

        if keep_columns is None:
            keep_columns = PHASE_1_KEEP_COLUMNS

        available_keep = [c for c in keep_columns if c in self.processed_data.columns]
        dropped = [c for c in self.processed_data.columns if c not in available_keep]

        if not available_keep:
            logger.warning("No configured keep-columns found in dataset. Skipping column filtering.")
            self.report["column_filtering"] = {
                "status": "skipped",
                "reason": "no keep columns matched",
            }
            return self.processed_data

        self.processed_data = self.processed_data[available_keep].copy()
        logger.info(
            "Column filtering complete: kept=%d dropped=%d",
            len(available_keep),
            len(dropped),
        )

        self.report["column_filtering"] = {
            "kept_columns": available_keep,
            "dropped_columns": dropped,
            "kept_count": int(len(available_keep)),
            "dropped_count": int(len(dropped)),
        }
        return self.processed_data
    
    def handle_missing_values(self, strategy="drop"):
        """Handle missing values"""
        logger.info(f"Handling missing values (strategy: {strategy})...")
        
        if strategy == "drop":
            before = len(self.processed_data)
            # Normalize configured audio features into numeric form before filtering.
            for feat in AUDIO_FEATURES:
                if feat in self.processed_data.columns:
                    self.processed_data[feat] = self.to_numeric_series(self.processed_data[feat])

            # Drop rows with missing audio features/emotion and missing core identifiers.
            drop_cols = [col for col in (AUDIO_FEATURES + ["emotion"]) if col in self.processed_data.columns]
            if "song" in self.processed_data.columns:
                drop_cols.append("song")
            self.processed_data = self.processed_data.dropna(subset=drop_cols)
            after = len(self.processed_data)
            removed = before - after
            logger.info(f"Dropped {removed} rows with missing values")
            self.report["missing_values_handling"] = {
                "strategy": strategy,
                "rows_dropped": int(removed)
            }
        
        return self.processed_data
    
    def create_train_val_test_splits(self, test_size=TEST_RATIO, val_size=VAL_RATIO, seed=PHASE_1_SEED):
        """Create stratified train/val/test splits"""
        logger.info("Creating train/val/test splits...")
        
        # Stratify by emotion if available
        emotion_col = self.find_column(["emotion", "class", "label"])
        stratify = self.processed_data[emotion_col] if emotion_col else None
        
        # Split 1: train + val vs test
        self.train_data, self.test_data = train_test_split(
            self.processed_data,
            test_size=test_size,
            random_state=seed,
            stratify=stratify
        )
        
        # Split 2: train vs val (from train subset)
        val_size_adj = val_size / (1 - test_size)  # Adjust for new size
        stratify_train = self.train_data[emotion_col] if emotion_col else None
        self.train_data, self.val_data = train_test_split(
            self.train_data,
            test_size=val_size_adj,
            random_state=seed,
            stratify=stratify_train
        )
        
        logger.info(f"Train set: {len(self.train_data)} rows ({len(self.train_data)/len(self.processed_data)*100:.1f}%)")
        logger.info(f"Val set:   {len(self.val_data)} rows ({len(self.val_data)/len(self.processed_data)*100:.1f}%)")
        logger.info(f"Test set:  {len(self.test_data)} rows ({len(self.test_data)/len(self.processed_data)*100:.1f}%)")
        
        self.report["splits"] = {
            "train": int(len(self.train_data)),
            "val": int(len(self.val_data)),
            "test": int(len(self.test_data)),
            "train_pct": round(len(self.train_data)/len(self.processed_data)*100, 2),
            "val_pct": round(len(self.val_data)/len(self.processed_data)*100, 2),
            "test_pct": round(len(self.test_data)/len(self.processed_data)*100, 2)
        }
        
        return self.train_data, self.val_data, self.test_data
    
    def save_splits(self):
        """Save train/val/test splits"""
        logger.info("Saving data splits...")
        
        for name, data, path in [
            ("train", self.train_data, SPOTIFY_TRAIN_PATH),
            ("val", self.val_data, SPOTIFY_VAL_PATH),
            ("test", self.test_data, SPOTIFY_TEST_PATH)
        ]:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False)
            logger.info(f"Saved {name} set to: {path}")
        
        # Also save processed (unsplit)
        self.processed_data.to_csv(SPOTIFY_PROCESSED_PATH, index=False)
        logger.info(f"Saved processed data to: {SPOTIFY_PROCESSED_PATH}")
    
    def save_report(self):
        """Save processing report as JSON"""
        report_path = PROCESSED_DATA_DIR / "data_processing_report.json"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        return report_path
    
    def process_pipeline(self):
        """Run complete processing pipeline"""
        logger.info("\n" + "="*80)
        logger.info("SPOTIFY DATA PROCESSING PIPELINE (PHASE 1)")
        logger.info("="*80)
        
        # Load
        self.load_raw_data()
        
        # Deduplicate
        self.deduplicate()

        # Clean labels to canonical Phase-1 6-class set before any validation/splitting.
        self.clean_emotion_labels()

        # Keep only model-relevant fields to avoid carrying unrelated metadata.
        self.filter_relevant_columns()
        
        # Validate
        self.validate_audio_features()
        self.validate_emotion_labels()
        
        # Handle missing
        self.handle_missing_values(strategy="drop")
        
        # Split
        self.create_train_val_test_splits()
        
        # Save
        self.save_splits()
        self.save_report()
        
        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Processed data shape: {self.processed_data.shape}")
        logger.info(f"Train: {len(self.train_data)} | Val: {len(self.val_data)} | Test: {len(self.test_data)}")
        logger.info(f"Report: {PROCESSED_DATA_DIR / 'data_processing_report.json'}")
        
        return self.report


def main():
    processor = SpotifyDataProcessor()
    processor.process_pipeline()


if __name__ == "__main__":
    main()


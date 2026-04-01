"""
Data Validator: Quality checks and statistics for Phase 1 dataset
Pipeline Setup - Data Validation Suite
"""

import pandas as pd
import numpy as np
import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    SPOTIFY_PROCESSED_PATH,
    AUDIO_FEATURES,
    LOG_FORMAT,
    LOG_LEVEL
)

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and generate statistics"""
    
    def __init__(self, data_path=SPOTIFY_PROCESSED_PATH):
        self.data = None
        self.validation_report = {}
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load processed data"""
        logger.info(f"Loading data from: {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded shape: {self.data.shape}")

    def to_numeric_series(self, series):
        """Convert series to numeric, extracting leading numbers from strings like '-3.6db'."""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        cleaned = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        return pd.to_numeric(cleaned, errors="coerce")
    
    def check_missing_values(self):
        """Check for missing values"""
        logger.info("Checking missing values...")
        
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        high_missing = missing[missing > 0]
        if len(high_missing) > 0:
            logger.warning(f"Found {len(high_missing)} columns with missing values:")
            for col, count in high_missing.items():
                pct = missing_pct[col]
                logger.warning(f"  {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("No missing values found âœ“")
        
        self.validation_report["missing_values"] = {
            str(col): {"count": int(count), "percentage": round(pct, 2)}
            for col, count, pct in zip(missing.index, missing.values, missing_pct.values)
            if count > 0
        }
        
        return len(high_missing) == 0
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        logger.info("Checking for duplicates...")
        
        dup_count = self.data.duplicated().sum()
        dup_pct = (dup_count / len(self.data)) * 100
        
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate rows ({dup_pct:.2f}%)")
        else:
            logger.info("No exact duplicates found âœ“")
        
        self.validation_report["duplicates"] = {
            "count": int(dup_count),
            "percentage": round(dup_pct, 2)
        }
        
        return dup_count == 0
    
    def check_audio_features_ranges(self):
        """Check if audio features are within expected ranges"""
        logger.info("Checking audio feature ranges...")
        
        ranges = {
            "Danceability": (0, 100),
            "Energy": (0, 100),
            "Positiveness": (0, 100),
            "Tempo": (0, 300),
            "Speechiness": (0, 100),
            "Acousticness": (0, 100),
            "Instrumentalness": (0, 100),
            "Liveness": (0, 100),
        }
        
        feature_status = {}
        all_valid = True
        
        for feat, (min_val, max_val) in ranges.items():
            if feat not in self.data.columns:
                continue
            
            col_data = self.to_numeric_series(self.data[feat]).dropna()
            if len(col_data) == 0:
                continue
            
            actual_min = col_data.min()
            actual_max = col_data.max()
            
            is_valid = (actual_min >= min_val) and (actual_max <= max_val)
            feature_status[feat] = {
                "expected_range": [min_val, max_val],
                "actual_min": round(float(actual_min), 4),
                "actual_max": round(float(actual_max), 4),
                "valid": bool(is_valid)
            }
            
            if is_valid:
                logger.info(f"  {feat:20s}: âœ“ [{actual_min:.2f}, {actual_max:.2f}]")
            else:
                logger.warning(f"  {feat:20s}: âœ— Out of range [{actual_min:.2f}, {actual_max:.2f}]")
                all_valid = False
        
        self.validation_report["audio_feature_ranges"] = feature_status
        return all_valid
    
    def check_emotion_distribution(self):
        """Check emotion class distribution"""
        logger.info("Checking emotion distribution...")
        
        emotion_col = None
        for col in ["emotion", "class", "label"]:
            if col in self.data.columns:
                emotion_col = col
                break
        
        if not emotion_col:
            logger.warning("No emotion column found")
            return False
        
        dist = self.data[emotion_col].value_counts()
        dist_pct = self.data[emotion_col].value_counts(normalize=True) * 100
        
        # Check class balance (warn if any class < 1%)
        imbalanced = dist_pct[dist_pct < 1]
        
        emotion_dist = {}
        for emotion, count in dist.items():
            pct = dist_pct[emotion]
            emotion_dist[str(emotion)] = {
                "count": int(count),
                "percentage": round(pct, 2)
            }
            logger.info(f"  {emotion:15s}: {count:6d} ({pct:5.2f}%)")
        
        self.validation_report["emotion_distribution"] = emotion_dist
        
        if len(imbalanced) > 0:
            logger.warning(f"Found {len(imbalanced)} rare classes (<1%)")
            return False
        
        return True
    
    def check_correlations(self):
        """Check correlations between audio features"""
        logger.info("Checking feature correlations...")

        existing_features = [c for c in AUDIO_FEATURES if c in self.data.columns]
        if not existing_features:
            logger.info("No configured audio feature columns found for correlation check")
            self.validation_report["high_correlations"] = []
            return True

        numeric_df = self.data[existing_features].copy()
        for col in existing_features:
            numeric_df[col] = self.to_numeric_series(numeric_df[col])
        numeric_cols = numeric_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if len(numeric_cols) < 2:
            logger.info("Not enough numeric features for correlation check")
            return True
        
        corr = numeric_cols.corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_val = abs(corr.iloc[i, j])
                if corr_val > 0.9:
                    high_corr_pairs.append({
                        "feature_1": corr.columns[i],
                        "feature_2": corr.columns[j],
                        "correlation": round(float(corr.iloc[i, j]), 3)
                    })
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs:")
            for pair in high_corr_pairs:
                logger.info(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']}")
        else:
            logger.info("No high correlations (>0.9) found âœ“")
        
        self.validation_report["high_correlations"] = high_corr_pairs
        return len(high_corr_pairs) < 5  # Warning if too many
    
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        logger.info("Generating summary statistics...")
        
        stats = {
            "total_rows": int(len(self.data)),
            "total_columns": int(len(self.data.columns)),
            "column_names": self.data.columns.tolist(),
            "dtypes": {str(col): str(dtype) for col, dtype in zip(self.data.columns, self.data.dtypes)},
            "memory_usage_mb": round(self.data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        self.validation_report["summary"] = stats
        
        logger.info(f"Total rows: {stats['total_rows']}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Memory usage: {stats['memory_usage_mb']} MB")
        
        return stats
    
    def validate_all(self):
        """Run all validation checks"""
        logger.info("\n" + "="*80)
        logger.info("DATA VALIDATION REPORT - PHASE 1 (50K SPOTIFY SAMPLE)")
        logger.info("="*80)
        
        checks = [
            ("Missing Values", self.check_missing_values()),
            ("Duplicates", self.check_duplicates()),
            ("Audio Feature Ranges", self.check_audio_features_ranges()),
            ("Emotion Distribution", self.check_emotion_distribution()),
            ("Correlations", self.check_correlations())
        ]
        
        self.generate_summary_statistics()
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        all_pass = True
        for check_name, passed in checks:
            status = "âœ“ PASS" if passed else "âœ— WARN"
            logger.info(f"{check_name:30s}: {status}")
            all_pass = all_pass and passed
        
        logger.info("\n" + "="*80)
        if all_pass:
            logger.info("âœ“ ALL CHECKS PASSED - DATA READY FOR TRAINING")
        else:
            logger.info("âš  SOME WARNINGS - REVIEW BEFORE TRAINING")
        logger.info("="*80)
        
        return self.validation_report
    
    def save_report(self, output_path=None):
        """Save validation report"""
        if output_path is None:
            output_path = Path(SPOTIFY_PROCESSED_PATH).parent / "validation_report.json"
        
        with open(output_path, "w") as f:
            json.dump(self.validation_report, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_path}")
        return output_path


def main():
    validator = DataValidator()
    validator.validate_all()
    validator.save_report()


if __name__ == "__main__":
    main()


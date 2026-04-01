"""Audio processing pipeline for normalized Spotify audio features.

Reads train/val/test CSV splits, resolves feature columns, imputes missing values,
normalizes with train-fit statistics, and stores split features in HDF5.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (  # noqa: E402
    AUDIO_ARTIST_COLUMN_CANDIDATES,
    AUDIO_BATCH_SIZE,
    AUDIO_CANONICAL_FEATURES,
    AUDIO_FEATURE_COLUMN_CANDIDATES,
    AUDIO_FEATURES_REPORT_PATH,
    AUDIO_INCLUDE_METADATA,
    AUDIO_LABEL_COLUMN_CANDIDATES,
    AUDIO_MAX_SAMPLES_PER_SPLIT,
    AUDIO_TRACK_COLUMN_CANDIDATES,
    HDF5_FEATURES_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    SPOTIFY_TEST_PATH,
    SPOTIFY_TRAIN_PATH,
    SPOTIFY_VAL_PATH,
)


logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class AudioFeatureProcessor:
    """Generate and persist normalized audio feature arrays per split."""

    def __init__(
        self,
        max_samples_per_split: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_h5_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> None:
        self.max_samples_per_split = (
            AUDIO_MAX_SAMPLES_PER_SPLIT if max_samples_per_split is None else max_samples_per_split
        )
        self.batch_size = AUDIO_BATCH_SIZE if batch_size is None else batch_size
        self.output_h5_path = HDF5_FEATURES_PATH if output_h5_path is None else Path(output_h5_path)
        self.report_path = AUDIO_FEATURES_REPORT_PATH if report_path is None else Path(report_path)

        self.feature_map: Dict[str, str] = {}
        self.impute_values: Optional[np.ndarray] = None
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None

        self.report: Dict[str, object] = {
            "canonical_features": AUDIO_CANONICAL_FEATURES,
            "batch_size": self.batch_size,
            "splits": {},
        }

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        normalized = {str(c).strip().lower(): c for c in df.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    @staticmethod
    def _to_numeric_series(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace("db", "", regex=False)
        s = s.str.replace("DB", "", regex=False)
        s = s.str.replace(r"[^0-9eE\.+\-]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _to_numeric_feature(series: pd.Series, canonical_name: str) -> pd.Series:
        if canonical_name != "key":
            return AudioFeatureProcessor._to_numeric_series(series)

        raw = series.astype(str).str.strip().str.upper()
        root = raw.str.extract(r"^([A-G](?:#|B)?)", expand=False)
        key_map = {
            "C": 0,
            "C#": 1,
            "DB": 1,
            "D": 2,
            "D#": 3,
            "EB": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "GB": 6,
            "G": 7,
            "G#": 8,
            "AB": 8,
            "A": 9,
            "A#": 10,
            "BB": 10,
            "B": 11,
        }
        mapped = root.map(key_map)
        numeric = pd.to_numeric(raw, errors="coerce")
        return mapped.where(mapped.notna(), numeric)

    @staticmethod
    def _to_utf8_np(values: List[str]) -> np.ndarray:
        return np.array([str(v).encode("utf-8") for v in values], dtype=h5py.string_dtype("utf-8"))

    def _resolve_feature_map(self, train_df: pd.DataFrame) -> Dict[str, str]:
        feature_map: Dict[str, str] = {}
        missing = []
        for canonical in AUDIO_CANONICAL_FEATURES:
            col = self._find_column(train_df, AUDIO_FEATURE_COLUMN_CANDIDATES[canonical])
            if col is None:
                missing.append(canonical)
            else:
                feature_map[canonical] = col

        if missing:
            raise ValueError(f"Missing required audio feature columns: {missing}")

        return feature_map

    def _read_split(
        self,
        split_name: str,
        path: Path,
    ) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str]]:
        logger.info("Loading split '%s' from %s", split_name, path)
        df = pd.read_csv(path, low_memory=False)

        if self.max_samples_per_split is not None:
            df = df.head(self.max_samples_per_split).copy()

        track_col = self._find_column(df, AUDIO_TRACK_COLUMN_CANDIDATES)
        artist_col = self._find_column(df, AUDIO_ARTIST_COLUMN_CANDIDATES)
        label_col = self._find_column(df, AUDIO_LABEL_COLUMN_CANDIDATES)

        return df, track_col, artist_col, label_col

    def _extract_matrix(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for name in AUDIO_CANONICAL_FEATURES:
            col = self.feature_map[name]
            arrays.append(self._to_numeric_feature(df[col], name).to_numpy(dtype=np.float32))
        mat = np.column_stack(arrays)
        return mat

    def _fit_train_stats(self, train_mat: np.ndarray) -> None:
        impute = []
        for i in range(train_mat.shape[1]):
            col = train_mat[:, i]
            valid = col[~np.isnan(col)]
            impute.append(float(valid.mean()) if valid.size > 0 else 0.0)
        self.impute_values = np.array(impute, dtype=np.float32)

        train_imputed = np.where(np.isnan(train_mat), self.impute_values, train_mat)
        self.train_mean = train_imputed.mean(axis=0)
        self.train_std = train_imputed.std(axis=0)
        self.train_std = np.where(self.train_std == 0, 1.0, self.train_std)

    def _transform(self, mat: np.ndarray) -> np.ndarray:
        if self.impute_values is None or self.train_mean is None or self.train_std is None:
            raise RuntimeError("Fit statistics must be computed before transform")

        imputed = np.where(np.isnan(mat), self.impute_values, mat)
        normalized = (imputed - self.train_mean) / self.train_std
        return normalized.astype(np.float32)

    def _split_missing_stats(self, raw_mat: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        total = max(raw_mat.shape[0], 1)
        for idx, name in enumerate(AUDIO_CANONICAL_FEATURES):
            missing = int(np.isnan(raw_mat[:, idx]).sum())
            out[name] = {
                "missing": missing,
                "missing_pct": round((missing / total) * 100, 4),
            }
        return out

    def _write_split(
        self,
        h5f: h5py.File,
        split_name: str,
        df: pd.DataFrame,
        track_col: Optional[str],
        artist_col: Optional[str],
        label_col: Optional[str],
    ) -> None:
        raw_mat = self._extract_matrix(df)
        norm_mat = self._transform(raw_mat)

        grp = h5f.create_group(split_name)
        grp.create_dataset(
            "features",
            data=norm_mat,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        if label_col and label_col in df.columns:
            grp.create_dataset("label", data=self._to_utf8_np(df[label_col].fillna("").astype(str).tolist()))

        if AUDIO_INCLUDE_METADATA:
            if track_col and track_col in df.columns:
                grp.create_dataset("track", data=self._to_utf8_np(df[track_col].fillna("").astype(str).tolist()))
            if artist_col and artist_col in df.columns:
                grp.create_dataset("artist", data=self._to_utf8_np(df[artist_col].fillna("").astype(str).tolist()))

        grp.attrs["rows"] = int(norm_mat.shape[0])
        grp.attrs["feature_dim"] = int(norm_mat.shape[1])

        self.report["splits"][split_name] = {
            "rows": int(df.shape[0]),
            "feature_shape": [int(x) for x in norm_mat.shape],
            "missing_before_impute": self._split_missing_stats(raw_mat),
            "label_column": label_col,
            "track_column": track_col,
            "artist_column": artist_col,
        }

    def run(self) -> Dict[str, object]:
        split_paths = {
            "train": SPOTIFY_TRAIN_PATH,
            "val": SPOTIFY_VAL_PATH,
            "test": SPOTIFY_TEST_PATH,
        }

        for split_name, split_path in split_paths.items():
            if not Path(split_path).exists():
                raise FileNotFoundError(
                    f"Missing split file for '{split_name}': {split_path}. Run processing first."
                )

        train_df, train_track_col, train_artist_col, train_label_col = self._read_split(
            "train", Path(split_paths["train"])
        )

        self.feature_map = self._resolve_feature_map(train_df)
        self.report["feature_map"] = self.feature_map

        train_raw = self._extract_matrix(train_df)
        self._fit_train_stats(train_raw)

        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing audio features to %s", self.output_h5_path)

        with h5py.File(self.output_h5_path, "w") as h5f:
            h5f.attrs["canonical_features"] = self._to_utf8_np(AUDIO_CANONICAL_FEATURES)
            h5f.attrs["batch_size"] = int(self.batch_size)
            h5f.attrs["impute_values"] = self.impute_values.astype(np.float32)
            h5f.attrs["train_mean"] = self.train_mean.astype(np.float32)
            h5f.attrs["train_std"] = self.train_std.astype(np.float32)

            self._write_split(h5f, "train", train_df, train_track_col, train_artist_col, train_label_col)

            for split_name in ["val", "test"]:
                split_df, track_col, artist_col, label_col = self._read_split(
                    split_name, Path(split_paths[split_name])
                )
                self._write_split(h5f, split_name, split_df, track_col, artist_col, label_col)

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

        logger.info("Audio feature report saved: %s", self.report_path)
        return self.report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized audio features")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap per split for faster debug runs",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Stored as metadata only")
    parser.add_argument(
        "--output-h5",
        type=str,
        default=None,
        help="Optional output HDF5 path (defaults to configured audio features path)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional output report JSON path (defaults to configured report path)",
    )
    args = parser.parse_args()

    processor = AudioFeatureProcessor(
        max_samples_per_split=args.max_samples_per_split,
        batch_size=args.batch_size,
        output_h5_path=Path(args.output_h5) if args.output_h5 else None,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    report = processor.run()

    logger.info("Audio processing complete")
    for split_name, stats in report.get("splits", {}).items():
        logger.info(
            "%s: rows=%s, shape=%s",
            split_name,
            stats.get("rows"),
            stats.get("feature_shape"),
        )


if __name__ == "__main__":
    main()

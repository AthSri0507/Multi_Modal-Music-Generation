"""
Text processing pipeline for lyrics embeddings.

Reads train/val/test CSV splits, cleans text, generates BERT embeddings,
and stores results in HDF5 for fast model training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (  # noqa: E402
    BERT_MODEL_NAME,
    HDF5_EMBEDDINGS_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    SPOTIFY_TEST_PATH,
    SPOTIFY_TRAIN_PATH,
    SPOTIFY_VAL_PATH,
    TEXT_ARTIST_COLUMN_CANDIDATES,
    TEXT_BATCH_SIZE,
    TEXT_EMBEDDINGS_REPORT_PATH,
    TEXT_INCLUDE_CLEANED_TEXT,
    TEXT_INFERENCE_DEVICE,
    TEXT_LABEL_COLUMN_CANDIDATES,
    TEXT_MAX_LENGTH,
    TEXT_MAX_SAMPLES_PER_SPLIT,
    TEXT_SOURCE_COLUMN_CANDIDATES,
    TEXT_TRACK_COLUMN_CANDIDATES,
)


logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class TextEmbeddingProcessor:
    """Generate and persist BERT text embeddings for each split."""

    def __init__(
        self,
        max_samples_per_split: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        output_h5_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> None:
        self.tokenizer = None
        self.model = None
        self.device = self._resolve_device(TEXT_INFERENCE_DEVICE)
        self.max_samples_per_split = (
            TEXT_MAX_SAMPLES_PER_SPLIT if max_samples_per_split is None else max_samples_per_split
        )
        self.batch_size = TEXT_BATCH_SIZE if batch_size is None else batch_size
        self.max_length = TEXT_MAX_LENGTH if max_length is None else max_length
        self.output_h5_path = HDF5_EMBEDDINGS_PATH if output_h5_path is None else Path(output_h5_path)
        self.report_path = (
            TEXT_EMBEDDINGS_REPORT_PATH if report_path is None else Path(report_path)
        )
        self.report: Dict[str, object] = {
            "model": BERT_MODEL_NAME,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "splits": {},
        }

    @staticmethod
    def _resolve_device(configured: str) -> str:
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def clean_text(text: object) -> str:
        """Apply lightweight text cleaning before tokenization."""
        if pd.isna(text):
            return ""

        t = str(text).lower().strip()
        t = re.sub(r"\[.*?\]", " ", t)
        t = re.sub(r"\(.*?\)", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        normalized = {str(c).strip().lower(): c for c in df.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("Loading tokenizer/model: %s", BERT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def _read_split(
        self, split_name: str, path: Path
    ) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], Optional[str]]:
        logger.info("Loading split '%s' from %s", split_name, path)
        df = pd.read_csv(path, low_memory=False)

        text_col = self._find_column(df, TEXT_SOURCE_COLUMN_CANDIDATES)
        if text_col is None:
            raise ValueError(f"No text column found in split '{split_name}'")

        track_col = self._find_column(df, TEXT_TRACK_COLUMN_CANDIDATES)
        artist_col = self._find_column(df, TEXT_ARTIST_COLUMN_CANDIDATES)
        label_col = self._find_column(df, TEXT_LABEL_COLUMN_CANDIDATES)

        if self.max_samples_per_split is not None:
            df = df.head(self.max_samples_per_split).copy()

        return df, text_col, track_col, artist_col, label_col

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        embeddings: List[np.ndarray] = []
        total = len(texts)
        if total == 0:
            return np.zeros((0, 768), dtype=np.float32)

        total_batches = (total + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        logger.info(
            "Embedding %d rows in %d batches (batch_size=%d)",
            total,
            total_batches,
            self.batch_size,
        )

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, total, self.batch_size), start=1):
                batch = texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded)
                cls_embeddings = output.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.detach().cpu().numpy().astype(np.float32))

                if batch_idx % 50 == 0 or batch_idx == total_batches:
                    elapsed = time.time() - start_time
                    rows_done = min(batch_idx * self.batch_size, total)
                    rows_per_sec = rows_done / max(elapsed, 1e-6)
                    eta_sec = (total - rows_done) / max(rows_per_sec, 1e-6)
                    logger.info(
                        "Progress: %d/%d batches (%d/%d rows, %.1f%%) | %.1f rows/s | ETA %.1fs",
                        batch_idx,
                        total_batches,
                        rows_done,
                        total,
                        (rows_done / total) * 100,
                        rows_per_sec,
                        eta_sec,
                    )

        return np.vstack(embeddings) if embeddings else np.zeros((0, 768), dtype=np.float32)

    @staticmethod
    def _to_utf8_np(values: List[str]) -> np.ndarray:
        return np.array([str(v).encode("utf-8") for v in values], dtype=h5py.string_dtype("utf-8"))

    def _write_split_to_hdf5(
        self,
        h5f: h5py.File,
        split_name: str,
        df: pd.DataFrame,
        text_col: str,
        track_col: Optional[str],
        artist_col: Optional[str],
        label_col: Optional[str],
    ) -> None:
        logger.info("Processing split: %s", split_name)
        split_group = h5f.create_group(split_name)

        cleaned_texts = [self.clean_text(v) for v in df[text_col].tolist()]
        embeddings = self._embed_texts(cleaned_texts)

        split_group.create_dataset(
            "embeddings",
            data=embeddings,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        if TEXT_INCLUDE_CLEANED_TEXT:
            split_group.create_dataset("cleaned_text", data=self._to_utf8_np(cleaned_texts))

        if track_col and track_col in df.columns:
            split_group.create_dataset("track", data=self._to_utf8_np(df[track_col].fillna("").astype(str).tolist()))
        if artist_col and artist_col in df.columns:
            split_group.create_dataset("artist", data=self._to_utf8_np(df[artist_col].fillna("").astype(str).tolist()))
        if label_col and label_col in df.columns:
            split_group.create_dataset("label", data=self._to_utf8_np(df[label_col].fillna("").astype(str).tolist()))

        split_group.attrs["text_column"] = text_col
        split_group.attrs["embedding_dim"] = embeddings.shape[1] if embeddings.shape[0] > 0 else 0
        split_group.attrs["rows"] = embeddings.shape[0]

        self.report["splits"][split_name] = {
            "rows": int(len(df)),
            "text_column": text_col,
            "track_column": track_col,
            "artist_column": artist_col,
            "label_column": label_col,
            "embedding_shape": [int(x) for x in embeddings.shape],
            "empty_text_rows": int(sum(1 for t in cleaned_texts if not t)),
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

        logger.info("Writing embeddings to %s", self.output_h5_path)
        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.output_h5_path, "w") as h5f:
            h5f.attrs["model"] = BERT_MODEL_NAME
            h5f.attrs["max_length"] = self.max_length
            h5f.attrs["batch_size"] = self.batch_size
            h5f.attrs["device"] = self.device

            for split_name, split_path in split_paths.items():
                df, text_col, track_col, artist_col, label_col = self._read_split(
                    split_name, Path(split_path)
                )
                self._write_split_to_hdf5(
                    h5f,
                    split_name,
                    df,
                    text_col,
                    track_col,
                    artist_col,
                    label_col,
                )

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

        logger.info("Text embedding report saved: %s", self.report_path)
        return self.report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BERT lyrics embeddings")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap per split for faster debug runs",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override embedding batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Override token max length")
    parser.add_argument(
        "--output-h5",
        type=str,
        default=None,
        help="Optional output HDF5 path (defaults to configured embeddings path)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional output report JSON path (defaults to configured report path)",
    )
    args = parser.parse_args()

    processor = TextEmbeddingProcessor(
        max_samples_per_split=args.max_samples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_h5_path=Path(args.output_h5) if args.output_h5 else None,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    report = processor.run()

    logger.info("Text processing complete")
    for split_name, stats in report.get("splits", {}).items():
        logger.info(
            "%s: rows=%s, shape=%s",
            split_name,
            stats.get("rows"),
            stats.get("embedding_shape"),
        )


if __name__ == "__main__":
    main()

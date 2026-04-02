"""Metric utilities for emotion classification."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Dict[int, str] | None = None,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float | dict]:
    """Compute aggregate and per-class metrics."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    acc = float(accuracy_score(true, pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true, pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        true, pred, average="micro", zero_division=0
    )

    labels_sorted = sorted(np.unique(true).tolist())
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        true, pred, labels=labels_sorted, average=None, zero_division=0
    )

    per_class = {}
    for i, cls_id in enumerate(labels_sorted):
        name = label_names.get(int(cls_id), str(cls_id)) if label_names else str(cls_id)
        per_class[name] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
        }

    roc_auc_macro = None
    roc_auc_per_class: Dict[str, float] = {}
    if y_score is not None:
        score = np.asarray(y_score)
        try:
            roc_auc_macro = float(roc_auc_score(true, score, multi_class="ovr", average="macro"))
            for i, cls_id in enumerate(labels_sorted):
                cls_name = label_names.get(int(cls_id), str(cls_id)) if label_names else str(cls_id)
                binary_true = (true == cls_id).astype(int)
                roc_auc_per_class[cls_name] = float(roc_auc_score(binary_true, score[:, i]))
        except ValueError:
            roc_auc_macro = None
            roc_auc_per_class = {}

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "per_class": per_class,
    }


def compute_confusion_matrix_payload(
    y_true: List[int],
    y_pred: List[int],
    label_names: Dict[int, str] | None = None,
) -> Dict[str, object]:
    """Return confusion matrix and normalized confusion matrix as JSON-safe payload."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    labels_sorted = sorted(np.unique(np.concatenate([true, pred])).tolist())
    cm = confusion_matrix(true, pred, labels=labels_sorted)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums != 0)

    display_labels = [
        label_names.get(int(cls_id), str(cls_id)) if label_names else str(cls_id)
        for cls_id in labels_sorted
    ]

    return {
        "label_ids": [int(v) for v in labels_sorted],
        "labels": display_labels,
        "matrix": cm.astype(int).tolist(),
        "matrix_normalized": cm_norm.astype(float).tolist(),
    }

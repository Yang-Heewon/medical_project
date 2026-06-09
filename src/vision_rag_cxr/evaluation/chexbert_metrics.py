"""CheXbert 14-label metric."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS


def labels_json_to_matrix(values: list[str], labels: list[str] | None = None) -> np.ndarray:
    labels = labels or CHEXBERT_LABELS
    rows = []
    for x in values:
        d = json.loads(x) if isinstance(x, str) else x
        rows.append([int(d.get(label, 0)) for label in labels])
    return np.asarray(rows, dtype=int)


def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "chexbert_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "chexbert_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "chexbert_micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "chexbert_micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

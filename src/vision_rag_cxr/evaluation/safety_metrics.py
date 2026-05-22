"""Clinical safety-oriented metrics.

normal collapse:
- abnormal GT인데 모델이 no finding/normal로 몰아가는 현상.
hallucination:
- GT에 없는 abnormal finding을 모델이 만들어내는 현상.
omission:
- GT abnormal finding을 모델이 누락하는 현상.
"""

from __future__ import annotations

import numpy as np


def normal_collapse_rate(y_true: np.ndarray, y_pred: np.ndarray, no_finding_idx: int = 0) -> float:
    abnormal_true = y_true.sum(axis=1) - y_true[:, no_finding_idx] > 0
    pred_no_finding = y_pred[:, no_finding_idx] == 1
    if abnormal_true.sum() == 0:
        return 0.0
    return float((abnormal_true & pred_no_finding).sum() / abnormal_true.sum())


def omission_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    omitted = (y_true == 1) & (y_pred == 0)
    return float(omitted.sum() / positives.sum())


def hallucination_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    hallucinated = (y_true == 0) & (y_pred == 1)
    return float(hallucinated.sum() / negatives.sum())

"""Localization metrics.

중요:
- IU-only 데이터에는 기본 bbox GT가 없으므로 strict IoU/mAP를 진짜 metric처럼 보고하면 안 된다.
- human audit mode에서는 사람이 검수한 CSV를 GT처럼 읽어서 correctness/agreement를 계산한다.
"""

from __future__ import annotations

import pandas as pd


def summarize_human_audit(audit_csv: str) -> dict:
    """human audit CSV에서 box correctness를 요약한다.

    예상 column:
    - uid
    - experiment_name
    - lesion_id
    - model_label
    - human_correct: 0/1
    - human_notes
    """
    df = pd.read_csv(audit_csv)
    return {
        "audit_count": int(len(df)),
        "box_correctness_rate": float(df["human_correct"].mean()) if len(df) else 0.0,
    }


def iou(box_a, box_b) -> float:
    """두 bbox의 IoU."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-8))

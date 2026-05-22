"""Hybrid RAG ranking utilities."""

from __future__ import annotations

import json
import numpy as np

from vision_rag_cxr.data.labeler_chexbert import CHEXBERT_LABELS


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def label_jaccard(a: dict, b: dict) -> float:
    """14-label positive set Jaccard similarity."""
    aset = {k for k, v in a.items() if int(v) == 1}
    bset = {k for k, v in b.items() if int(v) == 1}
    if not aset and not bset:
        return 1.0
    return len(aset & bset) / max(len(aset | bset), 1)


def parse_label_json(x) -> dict:
    if isinstance(x, dict):
        return x
    return json.loads(x)


def hybrid_score(img_sim: float, text_sim: float, label_sim: float, config: dict) -> float:
    """config weight 기반 최종 ranking score."""
    return (
        config.get("w_img", 0.5) * img_sim
        + config.get("w_text", 0.25) * text_sim
        + config.get("w_label", 0.25) * label_sim
    )

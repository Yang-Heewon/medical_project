"""Retrieval quality metrics."""

from __future__ import annotations

from vision_rag_cxr.inference.retrieval.hybrid_ranker import label_jaccard, parse_label_json


def mean_label_jaccard_at_k(query_labels: list, retrieved_labels: list[list]) -> float:
    vals = []
    for q, rs in zip(query_labels, retrieved_labels):
        qd = parse_label_json(q)
        vals.extend(label_jaccard(qd, parse_label_json(r)) for r in rs)
    return sum(vals) / max(len(vals), 1)

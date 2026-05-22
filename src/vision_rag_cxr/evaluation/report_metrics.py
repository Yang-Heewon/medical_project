"""Report generation metric wrapper skeleton."""

from __future__ import annotations


def compute_text_similarity_metrics(predictions: list[str], references: list[str]) -> dict:
    """BERTScore 등 자연어 metric을 계산하는 위치."""
    out = {}
    try:
        from bert_score import score
        _, _, f1 = score(predictions, references, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean().item())
    except Exception:
        out["bertscore_f1"] = None
    return out

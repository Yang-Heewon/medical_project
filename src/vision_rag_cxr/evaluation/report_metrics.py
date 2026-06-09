"""Report(impression) 텍스트 품질 metric.

생성된 impression이 데이터셋 GT impression의 스타일/내용과 얼마나 가까운지 측정한다.
TextGrad의 '텍스트(impression 스타일)' 최적화 목표로 사용된다 (병변 정확도는 별도 제약).
"""

from __future__ import annotations


def compute_text_similarity_metrics(predictions: list[str], references: list[str]) -> dict:
    """BERTScore-F1 + ROUGE-L(F)를 계산한다. 라이브러리 없으면 해당 항목은 None."""
    out: dict[str, float | None] = {"bertscore_f1": None, "rougeL_f": None}
    preds = [str(p or "") for p in predictions]
    refs = [str(r or "") for r in references]
    if not preds:
        return out

    try:
        from bert_score import score
        _, _, f1 = score(preds, refs, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean().item())
    except Exception:
        pass

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        out["rougeL_f"] = float(sum(vals) / max(len(vals), 1))
    except Exception:
        pass

    return out

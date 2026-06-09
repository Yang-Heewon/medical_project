"""Retrieval encoder benchmark.

데이터셋마다 어떤 vision encoder가 가장 좋은 related-retrieval을 주는지 자동으로 고른다.
각 후보 encoder로 support DB를 만들고, inference query를 image embedding만으로 검색한 뒤
(=leakage 없음), 검색된 support example이 query와 임상적으로 얼마나 맞는지 GT label로 채점한다.

metric (높을수록 좋음):
- same_label_rate@k: 검색된 example 중 query와 label을 1개 이상 공유하는 비율
- mean_label_jaccard@k: query vs 검색 example label Jaccard 평균

GT label은 채점에만 쓰고 검색에는 절대 쓰지 않는다(query_label_source: none 고정).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vision_rag_cxr.evaluation.retrieval_metrics import mean_label_jaccard_at_k
from vision_rag_cxr.inference.retrieval.build_database import build_support_database
from vision_rag_cxr.inference.retrieval.hybrid_ranker import label_jaccard, parse_label_json
from vision_rag_cxr.inference.retrieval.related_retriever import RelatedRetriever
from vision_rag_cxr.utils.io import ensure_dir


def _same_label_rate(query_labels: list, retrieved_labels: list[list]) -> float:
    hit = total = 0
    for q, rs in zip(query_labels, retrieved_labels):
        qd = parse_label_json(q)
        for r in rs:
            total += 1
            if label_jaccard(qd, parse_label_json(r)) > 0:
                hit += 1
    return hit / max(total, 1)


def benchmark_encoders(
    split_csv: str,
    encoder_configs: list[dict],
    top_k: int = 5,
    out_dir: str = "outputs/encoder_benchmark",
    label_space: str = "chexbert_14",
    max_inference: int | None = 100,
) -> pd.DataFrame:
    """후보 encoder들을 retrieval 품질로 비교하고 ranking을 저장/반환한다."""
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(split_csv)
    inference = df[df["split"] == "inference"].reset_index(drop=True)
    if max_inference is not None:
        inference = inference.head(int(max_inference))
    query_samples = [r.to_dict() for _, r in inference.iterrows()]

    results = []
    for enc_cfg in encoder_configs:
        cfg = dict(enc_cfg)
        name = cfg.get("vision_encoder_name", "unknown")
        tag = cfg.get("benchmark_tag", name)
        rag_dir = ensure_dir(Path(out_dir) / f"rag_{tag}")
        cfg["output_dir"] = str(rag_dir)
        cfg["label_space"] = label_space
        cfg["query_label_source"] = "none"  # leakage 방지: 검색은 image-only
        cfg["query_text_source"] = "none"

        try:
            build_support_database(split_csv, cfg)
            cfg["support_metadata_path"] = str(rag_dir / "support_metadata.parquet")
            cfg["image_embeddings_path"] = str(rag_dir / "image_embeddings.npy")
            cfg["text_embeddings_path"] = str(rag_dir / "text_embeddings.npy")
            cfg["label_vectors_path"] = str(rag_dir / "label_vectors.npy")
            retriever = RelatedRetriever.from_config(cfg)

            q_labels, r_labels = [], []
            for s in query_samples:
                retrieved = retriever.retrieve(s, top_k)
                q_labels.append(s.get("chexbert_labels_binary"))
                r_labels.append([ex.get("chexbert_labels_binary") for ex in retrieved])

            row = {
                "encoder": tag,
                "vision_encoder_name": name,
                "model_name_or_path": cfg.get("model_name_or_path", ""),
                "same_label_rate@k": round(_same_label_rate(q_labels, r_labels), 4),
                "mean_label_jaccard@k": round(mean_label_jaccard_at_k(q_labels, r_labels), 4),
                "n_queries": len(query_samples),
                "top_k": top_k,
                "status": "ok",
            }
        except Exception as e:  # 한 encoder가 실패해도 나머지는 계속 비교한다.
            row = {"encoder": tag, "vision_encoder_name": name, "status": f"failed: {type(e).__name__}: {e}"}
        results.append(row)
        print(f"  [{row.get('status')}] {tag}: same_label_rate@k={row.get('same_label_rate@k')}", flush=True)

    ranking = pd.DataFrame(results)
    ok = ranking[ranking["status"] == "ok"].sort_values("same_label_rate@k", ascending=False)
    ranking.to_csv(Path(out_dir) / "encoder_benchmark.csv", index=False)

    best = ok.iloc[0]["encoder"] if len(ok) else None
    summary = {
        "label_space": label_space,
        "top_k": top_k,
        "n_queries": len(query_samples),
        "ranking": ok[["encoder", "same_label_rate@k", "mean_label_jaccard@k"]].to_dict("records") if len(ok) else [],
        "best_encoder": best,
    }
    (Path(out_dir) / "encoder_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"BEST encoder for this dataset: {best}", flush=True)
    return ranking

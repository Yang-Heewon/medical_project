"""Experiment config에서 retriever를 생성하는 factory."""

from __future__ import annotations

from pathlib import Path

from vision_rag_cxr.inference.retrieval.related_retriever import RelatedRetriever
from vision_rag_cxr.inference.retrieval.unrelated_retriever import UnrelatedRetriever
from vision_rag_cxr.utils.io import load_yaml


def build_retriever_config(experiment_config: dict) -> dict:
    """retrieval yaml과 experiment override를 합친다."""
    cfg = {}
    retrieval_config_path = experiment_config.get("retrieval_config")
    if retrieval_config_path and Path(retrieval_config_path).exists():
        cfg.update(load_yaml(retrieval_config_path))

    for key in [
        "support_metadata_path",
        "image_embeddings_path",
        "text_embeddings_path",
        "label_vectors_path",
        "faiss_index_path",
        "top_k",
        "seed",
        "query_label_source",
        "query_text_source",
    ]:
        if key in experiment_config:
            cfg[key] = experiment_config[key]
    return cfg


def build_retriever_for_experiment(experiment_config: dict):
    """rag_mode에 맞는 retriever를 만든다. image_only면 None을 반환한다."""
    rag_mode = str(experiment_config.get("rag_mode", "image_only")).lower()
    if rag_mode in {"", "none", "image_only", "no_rag"}:
        return None

    cfg = build_retriever_config(experiment_config)
    if rag_mode == "related":
        return RelatedRetriever.from_config(cfg)
    if rag_mode == "unrelated":
        return UnrelatedRetriever.from_config(cfg)
    raise ValueError(f"Unknown rag_mode: {rag_mode}")

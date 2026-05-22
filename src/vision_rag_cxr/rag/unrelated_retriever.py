"""Unrelated RAG retriever."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from vision_rag_cxr.rag.related_retriever import _label_jaccard_scores, _read_metadata
from vision_rag_cxr.data.labeler_chexbert import CHEXBERT_LABELS


def _parse_query_labels(query_sample: dict) -> np.ndarray | None:
    import json

    value = query_sample.get("chexbert_labels_binary")
    if value is None:
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    return np.asarray([int(parsed.get(label, 0)) for label in CHEXBERT_LABELS], dtype="float32")


class UnrelatedRetriever:
    """negative control용 unrelated support examples를 반환한다.

    기본은 deterministic random sampling이다. config에서 ``query_label_source``를 GT 계열로 켠 경우에만
    low label-overlap 후보를 먼저 고른다. 이 설정은 leakage 위험이 있으므로 final setting에는 쓰지 않는다.
    """

    def __init__(self, metadata: pd.DataFrame, config: dict, label_vectors: np.ndarray | None = None):
        self.metadata = metadata.reset_index(drop=True)
        self.config = config
        self.label_vectors = label_vectors

    @classmethod
    def from_config(cls, config: dict) -> "UnrelatedRetriever":
        output_dir = Path(config.get("output_dir", "outputs/rag"))
        metadata_path = config.get("support_metadata_path", output_dir / "support_metadata.parquet")
        metadata = _read_metadata(metadata_path)
        label_path = Path(config.get("label_vectors_path", output_dir / "label_vectors.npy"))
        label_vectors = np.load(label_path) if label_path.exists() else None
        return cls(metadata, config, label_vectors=label_vectors)

    def retrieve(self, query_sample: dict, top_k: int) -> list[dict]:
        mask = np.ones(len(self.metadata), dtype=bool)
        query_uid = str(query_sample.get("uid", ""))
        if "uid" in self.metadata.columns and query_uid:
            mask &= self.metadata["uid"].astype(str).to_numpy() != query_uid

        candidate_indices = np.flatnonzero(mask)
        source = str(self.config.get("query_label_source", "none")).lower()
        if self.label_vectors is not None and source in {"gt", "gt_chexbert", "chexbert_labels_binary"}:
            query_labels = _parse_query_labels(query_sample)
            if query_labels is not None:
                label_scores = _label_jaccard_scores(self.label_vectors, query_labels)
                max_j = float(self.config.get("max_label_jaccard_for_unrelated", 0.05))
                low_overlap = np.flatnonzero(mask & (label_scores <= max_j))
                if len(low_overlap) >= top_k:
                    candidate_indices = low_overlap

        seed = int(self.config.get("seed", 0))
        uid_seed = int.from_bytes(hashlib.sha256(str(query_sample.get("uid", "")).encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed + uid_seed)
        chosen = rng.choice(candidate_indices, size=min(top_k, len(candidate_indices)), replace=False)

        rows: list[dict] = []
        for rank, idx in enumerate(chosen, start=1):
            row = self.metadata.iloc[int(idx)].to_dict()
            row["rank"] = rank
            row["retrieval_mode"] = "unrelated"
            row["retrieval_score"] = None
            row["image_similarity"] = None
            row["text_similarity"] = None
            row["label_similarity"] = None
            row["query_label_source"] = self.config.get("query_label_source", "none")
            rows.append(row)
        return rows

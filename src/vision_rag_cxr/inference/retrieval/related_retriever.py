"""Related RAG retriever.

이 모듈은 support DB에 저장된 embedding/metadata를 읽어 inference sample과 가장 관련 있는
k-shot example을 고른다. 기본 query는 image embedding만 사용한다. GT CheXbert label을 query에
사용하면 retrieval leakage가 생길 수 있으므로 config에서 명시적으로 켠 경우에만 label score를 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vision_rag_cxr.datasets.labeler_chexbert import CHEXBERT_LABELS
from vision_rag_cxr.datasets.label_spaces import read_label_space_sidecar, resolve_labels
from vision_rag_cxr.models.encoders.base import build_vision_encoder


def _resolve_retriever_labels(config: dict) -> list[str]:
    """retriever가 쓸 label space를 정한다. DB sidecar가 있으면 그걸 우선한다."""
    meta_path = config.get("support_metadata_path")
    if meta_path:
        return read_label_space_sidecar(meta_path, default=config.get("label_space", "chexbert_14"))
    return resolve_labels(config)


def _read_metadata(path: str | Path) -> pd.DataFrame:
    """parquet 우선, 실패하거나 파일이 없으면 같은 stem의 csv를 읽는다."""
    path = Path(path)
    csv_fallback = path.with_suffix(".csv") if path.suffix == ".parquet" else None
    if path.suffix == ".parquet":
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                if csv_fallback and csv_fallback.exists():
                    return pd.read_csv(csv_fallback)
                raise
        if csv_fallback and csv_fallback.exists():
            return pd.read_csv(csv_fallback)
    return pd.read_csv(path)


def _label_dict(value, labels: list[str] | None = None) -> dict[str, int]:
    labels = labels or CHEXBERT_LABELS
    if isinstance(value, dict):
        return {label: int(value.get(label, 0)) for label in labels}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {label: 0 for label in labels}
    parsed = json.loads(value)
    return {label: int(parsed.get(label, 0)) for label in labels}


def _label_matrix_from_metadata(metadata: pd.DataFrame, labels: list[str] | None = None) -> np.ndarray:
    labels = labels or CHEXBERT_LABELS
    return np.asarray(
        [[_label_dict(x, labels).get(label, 0) for label in labels] for x in metadata["chexbert_labels_binary"]],
        dtype="float32",
    )


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    query = np.asarray(query, dtype="float32").reshape(-1)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    return matrix_norm @ query_norm


def _label_jaccard_scores(label_vectors: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    label_vectors = np.asarray(label_vectors, dtype="float32") > 0
    query_labels = np.asarray(query_labels, dtype="float32") > 0
    inter = np.logical_and(label_vectors, query_labels).sum(axis=1)
    union = np.logical_or(label_vectors, query_labels).sum(axis=1)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1)).astype("float32")


class RelatedRetriever:
    """support DB에서 query image와 가장 가까운 related examples를 반환한다."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        config: dict,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray | None = None,
        label_vectors: np.ndarray | None = None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.config = config
        self.labels = _resolve_retriever_labels(config)
        self.image_embeddings = np.asarray(image_embeddings, dtype="float32")
        self.text_embeddings = None if text_embeddings is None else np.asarray(text_embeddings, dtype="float32")
        self.label_vectors = label_vectors if label_vectors is not None else _label_matrix_from_metadata(self.metadata, self.labels)
        self.encoder = build_vision_encoder(config)

    @classmethod
    def from_config(cls, config: dict) -> "RelatedRetriever":
        output_dir = Path(config.get("output_dir", "outputs/rag"))
        metadata_path = config.get("support_metadata_path", output_dir / "support_metadata.parquet")
        metadata = _read_metadata(metadata_path)
        image_embeddings = np.load(config.get("image_embeddings_path", output_dir / "image_embeddings.npy"))

        text_path = Path(config.get("text_embeddings_path", output_dir / "text_embeddings.npy"))
        text_embeddings = np.load(text_path) if text_path.exists() else None

        label_path = Path(config.get("label_vectors_path", output_dir / "label_vectors.npy"))
        label_vectors = np.load(label_path) if label_path.exists() else None
        return cls(metadata, config, image_embeddings, text_embeddings=text_embeddings, label_vectors=label_vectors)

    def _query_label_vector(self, query_sample: dict) -> np.ndarray | None:
        source = str(self.config.get("query_label_source", "none")).lower()
        if source in {"none", "", "image_only"}:
            return None
        if source in {"gt", "gt_chexbert", "chexbert_labels_binary"}:
            labels = _label_dict(query_sample.get("chexbert_labels_binary"), self.labels)
            return np.asarray([labels[label] for label in self.labels], dtype="float32")
        if source in query_sample:
            labels = _label_dict(query_sample.get(source), self.labels)
            return np.asarray([labels[label] for label in self.labels], dtype="float32")
        return None

    def _query_text_embedding(self, query_sample: dict) -> np.ndarray | None:
        source = self.config.get("query_text_source", "none")
        if source in {None, "", "none"} or self.text_embeddings is None:
            return None
        text = str(query_sample.get(source, "") or "")
        if not text.strip():
            return None
        return self.encoder.encode_text(text)

    def retrieve(self, query_sample: dict, top_k: int) -> list[dict]:
        query_img = self.encoder.encode_image(query_sample["frontal_path"], query_sample.get("lateral_path"))
        if query_img.shape[0] != self.image_embeddings.shape[1]:
            raise ValueError(
                "Query embedding dim does not match support DB. "
                f"query={query_img.shape[0]}, db={self.image_embeddings.shape[1]}. "
                "Rebuild DB with the same retrieval config."
            )

        img_scores = _cosine_scores(self.image_embeddings, query_img)
        active = [("image_similarity", img_scores, float(self.config.get("w_img", 0.5)))]

        query_text = self._query_text_embedding(query_sample)
        if query_text is not None and self.text_embeddings is not None:
            active.append(("text_similarity", _cosine_scores(self.text_embeddings, query_text), float(self.config.get("w_text", 0.25))))

        query_labels = self._query_label_vector(query_sample)
        label_scores = None
        if query_labels is not None:
            label_scores = _label_jaccard_scores(self.label_vectors, query_labels)
            active.append(("label_similarity", label_scores, float(self.config.get("w_label", 0.25))))

        weight_sum = sum(weight for _, _, weight in active) or 1.0
        total_scores = sum((weight / weight_sum) * scores for _, scores, weight in active)

        candidate_mask = np.ones(len(self.metadata), dtype=bool)
        query_uid = str(query_sample.get("uid", ""))
        if "uid" in self.metadata.columns and query_uid:
            candidate_mask &= self.metadata["uid"].astype(str).to_numpy() != query_uid

        if label_scores is not None and self.config.get("use_label_filter", False):
            min_j = float(self.config.get("min_label_jaccard_for_related", 0.1))
            label_mask = label_scores >= min_j
            if label_mask.any():
                candidate_mask &= label_mask

        candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(self.metadata))

        ranked = candidate_indices[np.argsort(total_scores[candidate_indices])[::-1]][:top_k]
        rows: list[dict] = []
        for rank, idx in enumerate(ranked, start=1):
            row = self.metadata.iloc[int(idx)].to_dict()
            row["rank"] = rank
            row["retrieval_mode"] = "related"
            row["retrieval_score"] = float(total_scores[idx])
            row["image_similarity"] = float(img_scores[idx])
            row["text_similarity"] = float(0.0)
            row["label_similarity"] = float(label_scores[idx]) if label_scores is not None else None
            row["query_label_source"] = self.config.get("query_label_source", "none")
            rows.append(row)
        return rows

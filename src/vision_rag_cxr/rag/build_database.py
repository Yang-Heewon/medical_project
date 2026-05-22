"""Support set RAG DB 구축.

DB key:
- image_embedding
- report_embedding
- chexbert_label_vector
- anatomy/pathology phrase
- metadata: uid, frontal_path, lateral_path, impression, findings, MeSH, Problems, labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vision_rag_cxr.data.labeler_chexbert import CHEXBERT_LABELS
from vision_rag_cxr.models.vision_encoder_base import build_vision_encoder
from vision_rag_cxr.rag.vector_store import VectorStore
from vision_rag_cxr.utils.io import ensure_dir


def label_dict_to_vec(label_json) -> np.ndarray:
    """label dict를 고정 순서 vector로 변환한다."""
    d = json.loads(label_json) if isinstance(label_json, str) else label_json
    return np.asarray([int(d.get(label, 0)) for label in CHEXBERT_LABELS], dtype="float32")


def extract_anatomy_pathology_phrase(row: pd.Series) -> str:
    """retrieval metadata에 넣을 anatomy/pathology phrase.

    초반 버전에서는 MeSH + Problems + impression 일부를 합쳐 저장한다.
    나중에 RadGraph/sectionizer/anatomy parser로 교체하면 된다.
    """
    return " | ".join(
        x for x in [
            str(row.get("mesh", row.get("MeSH", "")) or ""),
            str(row.get("problems", row.get("Problems", "")) or ""),
            str(row.get("impression", "") or "")[:300],
        ]
        if x.strip()
    )


def build_support_database(split_csv: str, config: dict) -> None:
    """support set으로 FAISS index와 metadata parquet/csv를 만든다."""
    df = pd.read_csv(split_csv)
    support = df[df["split"] == "support"].reset_index(drop=True)
    if len(support) == 0:
        raise ValueError("No support samples found in split CSV.")

    max_support_samples = config.get("max_support_samples")
    if max_support_samples is not None:
        support = support.head(int(max_support_samples)).reset_index(drop=True)
        print(f"Using support subset for DB build: {len(support)} samples", flush=True)

    encoder = build_vision_encoder(config)

    support_rows = [row.to_dict() for _, row in support.iterrows()]
    metadata_rows = []
    label_vectors = []

    for row in support_rows:
        label_vectors.append(label_dict_to_vec(row["chexbert_labels_binary"]))
        metadata_rows.append(
            {
                "uid": row["uid"],
                "frontal_path": row["frontal_path"],
                "lateral_path": row.get("lateral_path", ""),
                "impression": row.get("impression", ""),
                "findings": row.get("findings", ""),
                "MeSH": row.get("MeSH", row.get("mesh", "")),
                "Problems": row.get("Problems", row.get("problems", "")),
                "chexbert_labels_binary": row["chexbert_labels_binary"],
                "chexbert_labels_raw": row.get("chexbert_labels_raw", ""),
                "anatomy_pathology_phrase": row.get("anatomy_pathology_phrase", extract_anatomy_pathology_phrase(pd.Series(row))),
            }
        )

    batch_size = int(config.get("batch_size", 32))
    text_batch_size = int(config.get("text_batch_size", max(batch_size, 64)))
    if hasattr(encoder, "encode_image_batch") and hasattr(encoder, "encode_text_batch"):
        # BiomedCLIP처럼 batch inference가 가능한 encoder는 이 경로를 탄다.
        print(f"Encoding support set with batched encoder: n={len(support_rows)}, batch_size={batch_size}", flush=True)
        image_embeddings = encoder.encode_image_batch(support_rows, batch_size=batch_size)
        text_embeddings = encoder.encode_text_batch([row.get("impression", "") for row in support_rows], batch_size=text_batch_size)
    else:
        image_embeddings = []
        text_embeddings = []
        for row in tqdm(support_rows, total=len(support_rows), desc="Encoding support set"):
            emb = encoder.encode_sample(row)
            image_embeddings.append(emb["image_embedding"])
            text_embeddings.append(emb["text_embedding"])
        image_embeddings = np.vstack(image_embeddings).astype("float32")
        text_embeddings = np.vstack(text_embeddings).astype("float32")

    label_vectors = np.vstack(label_vectors).astype("float32")

    out_dir = ensure_dir(config.get("output_dir", "outputs/rag"))
    index_dir = ensure_dir(Path(out_dir) / "faiss_index")

    # 기본 retrieval index는 image embedding 기준으로 저장.
    store = VectorStore(dim=image_embeddings.shape[1])
    store.build(image_embeddings)
    store.save(index_dir / "support.index")

    np.save(Path(out_dir) / "image_embeddings.npy", image_embeddings)
    np.save(Path(out_dir) / "text_embeddings.npy", text_embeddings)
    np.save(Path(out_dir) / "label_vectors.npy", label_vectors)

    meta = pd.DataFrame(metadata_rows)
    parquet_path = Path(out_dir) / "support_metadata.parquet"
    csv_path = Path(out_dir) / "support_metadata.csv"
    try:
        meta.to_parquet(parquet_path, index=False)
        metadata_path = parquet_path
    except Exception:
        meta.to_csv(csv_path, index=False)
        metadata_path = csv_path
    meta.head(50).to_csv(Path(out_dir) / "support_metadata_preview.csv", index=False)

    summary = [
        "# Support RAG DB summary",
        "",
        f"- support_count: {len(support)}",
        f"- vision_encoder_name: {config.get('vision_encoder_name')}",
        f"- image_embedding_dim: {image_embeddings.shape[1] if len(image_embeddings) else 0}",
        f"- metadata_path: {metadata_path}",
        "",
        "## Leakage note",
        "Final retrieval should use query image embeddings only unless a predicted, non-GT label source is configured.",
    ]
    (Path(out_dir) / "db_summary.md").write_text("\n".join(summary), encoding="utf-8")

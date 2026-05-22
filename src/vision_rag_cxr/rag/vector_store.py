"""FAISS vector store wrapper.

FAISS가 설치되지 않았거나 GPU 환경이 맞지 않아도 최소한 numpy fallback으로 동작하게 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    """간단한 vector search interface."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None
        self.index = None
        self.use_faiss = False

    def build(self, vectors: np.ndarray) -> None:
        """vector index를 만든다."""
        vectors = np.asarray(vectors).astype("float32")
        self.vectors = vectors

        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.use_faiss = True
        except Exception:
            # fallback: search 시 numpy dot product 사용
            self.use_faiss = False

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """query에 대한 top_k score/index를 반환한다."""
        q = np.asarray(query).astype("float32").reshape(1, -1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

        if self.use_faiss:
            scores, idx = self.index.search(q, top_k)
            return scores[0], idx[0]

        vecs = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        scores = vecs @ q[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return scores[idx], idx

    def save(self, path: str | Path) -> None:
        """index를 저장한다. FAISS가 없으면 npy로 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_faiss:
            import faiss
            faiss.write_index(self.index, str(path))
        else:
            np.save(str(path) + ".npy", self.vectors)

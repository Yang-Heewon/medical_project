"""Vision encoder factory와 fallback encoder."""

from __future__ import annotations

import hashlib

import numpy as np

from vision_rag_cxr.models.base import BaseVisionEncoder


class DummyHashEncoder(BaseVisionEncoder):
    """설치 검증용 deterministic encoder.

    실제 연구 결과에 사용하면 안 된다.
    단, preprocessing/split/RAG DB build 코드가 끝까지 도는지 확인할 때 매우 유용하다.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.dim = int(config.get("embedding_dim", 384))

    def _hash_to_vec(self, text: str) -> np.ndarray:
        text = "" if text is None else str(text)
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype("float32")
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    def encode_image(self, frontal_path: str, lateral_path: str | None = None) -> np.ndarray:
        return self._hash_to_vec(f"{frontal_path}|{lateral_path}")

    def encode_text(self, text: str) -> np.ndarray:
        return self._hash_to_vec(text or "")


def build_vision_encoder(config: dict) -> BaseVisionEncoder:
    name = config.get("vision_encoder_name", "dummy_hash").lower()
    if name == "dummy_hash":
        return DummyHashEncoder(config)
    if name == "medsiglip":
        from vision_rag_cxr.models.medsiglip_encoder import MedSigLIPEncoder
        return MedSigLIPEncoder(config)
    if name == "medclip":
        from vision_rag_cxr.models.medclip_encoder import MedCLIPEncoder
        return MedCLIPEncoder(config)
    if name == "biovil":
        from vision_rag_cxr.models.biovil_encoder import BioViLEncoder
        return BioViLEncoder(config)
    if name == "chexzero":
        from vision_rag_cxr.models.chexzero_encoder import CheXzeroEncoder
        return CheXzeroEncoder(config)
    raise ValueError(f"지원하지 않는 vision encoder입니다: {name}")
